"""A `UI` that sends everything to the browser as events instead of printing it.

The pipeline, agent, builder and governor only know the small `UI` protocol from `ai_neo.ui`, so
the same code that drives the terminal chat drives the browser chat. Two things need care:

* Questions ("Allow this?", "Approve this plan?", "Send it?") block the worker thread until the
  browser answers through `respond`. Only the offered answers are accepted, a question nobody
  answers times out into "stop, run nothing", and the ids cannot be guessed.
* Stop: there is no Ctrl+C in a browser. `request_stop` raises `StopRequested` (a
  `KeyboardInterrupt`) inside the next stream event, tool call or pending question, which the
  Ollama client and the agent already treat exactly like Ctrl+C: the partial answer is kept.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_neo.types import GenStats, StreamEvent
from ai_neo.ui import format_stats, short_args

Sink = Callable[[dict[str, Any]], None]

MAX_OUTPUT_CHARS = 6000
# Larger than anything the governor can send (about 21,000 characters), so its "this is exactly
# what will be sent" preview is never cut short.
MAX_PREVIEW_CHARS = 60_000
PROMPT_TIMEOUT = 900.0  # seconds a question may wait before AI Neo gives up and runs nothing
_OPTION = re.compile(r"\[([A-Za-z])\]([A-Za-z]*)")
_CUT_MARKER = re.compile(r"\.\.\. \(\d+ more lines\)|\[truncated, \d+ more characters\]")
CUT_WARNING = (
    "\n\nPart of the text above is cut off, but ALL of it will be used. Deny if you are not sure."
)


class StopRequested(KeyboardInterrupt):
    """Raised inside the worker thread when the user pressed Stop in the browser."""


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[{len(text) - limit} more characters not shown]"


def visible_controls(text: str) -> str:
    """Show invisible and direction-changing characters as `<U+202E>` instead of acting on them.

    Without this, a right-to-left override could make code in an approval preview read as
    something harmless while it runs as something else.
    """
    text = text.replace("\r\n", "\n")
    if all(ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf") for ch in text):
        return text
    return "".join(
        ch if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf") else f"<U+{ord(ch):04X}>"
        for ch in text
    )


def options_from_prompt(prompt: str, letters: str) -> list[dict[str, str]]:
    """Turn 'Approve? [Y]es / [R]evise / [N]o:' and 'yrn' into button labels."""
    labels = {m.group(1).lower(): m.group(1).upper() + m.group(2) for m in _OPTION.finditer(prompt)}
    return [{"value": ch, "label": labels.get(ch, ch.upper())} for ch in letters]


@dataclass
class _Pending:
    default: str
    allowed: frozenset[str] | None = None  # None = free text
    event: threading.Event = field(default_factory=threading.Event)
    value: str | None = None


class _WebStream:
    """One streamed model reply. Raises StopRequested at the next event after Stop."""

    def __init__(self, ui: WebUI) -> None:
        self.ui = ui
        ui.emit({"type": "segment"})

    def event(self, event: StreamEvent) -> None:
        if self.ui.stop_requested:
            raise StopRequested
        self.ui.emit({"type": event.kind, "text": event.text})

    def close(self) -> None:
        self.ui.emit({"type": "segment_end"})


class _WebProgress:
    def __init__(self, ui: WebUI, description: str) -> None:
        self.ui = ui
        self.description = description
        self.last = -1

    def update(self, completed: int, total: int, label: str = "") -> None:
        percent = int(100 * completed / total) if total > 0 else 0
        if percent == self.last:
            return
        self.last = percent
        self.ui.emit({"type": "progress", "label": label or self.description, "percent": percent})

    def close(self) -> None:
        pass


class WebUI:
    interactive = True

    def __init__(self, prompt_timeout: float = PROMPT_TIMEOUT) -> None:
        self.prompt_timeout = prompt_timeout
        self._sink: Sink | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._last_preview_cut = False

    # -- job lifecycle (called by the web server, not by AI Neo's core) --------------------------

    def attach(self, sink: Sink, stop: threading.Event | None = None) -> None:
        """Start listening for one job. `stop` is that job's own Stop flag, so a Stop aimed at an
        earlier job can never cancel this one, and a Stop sent before this call is not lost."""
        self._stop = stop if stop is not None else threading.Event()
        self._last_preview_cut = False
        self._sink = sink

    def detach(self) -> None:
        self._sink = None
        self._release_all()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()
        self._release_all()

    def respond(self, prompt_id: str, value: str) -> bool:
        """Answer a waiting question. False if it is not waiting any more (answered, stopped).

        Raises ValueError for an answer that is not one of the offered choices; the question then
        keeps waiting, so a typo or a stray request can never approve anything by default.
        """
        with self._lock:  # check and answer in one step: two windows cannot both win
            pending = self._pending.get(prompt_id)
            if pending is None or pending.event.is_set():
                return False
            if pending.allowed is not None and value.strip().lower() not in pending.allowed:
                raise ValueError("that is not one of the offered choices")
            pending.value = value
            pending.event.set()
        return True

    def wake(self) -> None:
        """Release every waiting question as 'stopped' (its job's Stop flag is already set)."""
        self._release_all()

    def pending_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def _release_all(self) -> None:
        with self._lock:
            waiting = list(self._pending.values())
        for pending in waiting:
            pending.event.set()  # value stays None: the waiter sees "stopped"

    def emit(self, event: dict[str, Any]) -> None:
        sink = self._sink
        if sink is not None:
            sink(event)

    # -- questions -------------------------------------------------------------------------------

    def _ask(
        self,
        kind: str,
        text: str,
        *,
        default: str,
        options: list[dict[str, str]],
        allowed: frozenset[str] | None,
    ) -> str:
        if self._sink is None:
            return default
        pid = f"q{secrets.token_urlsafe(9)}"  # unguessable: only the page that got it can answer
        pending = _Pending(default, allowed)
        with self._lock:
            self._pending[pid] = pending
        try:
            self.emit(
                {
                    "type": "prompt",
                    "id": pid,
                    "kind": kind,
                    "text": visible_controls(text),
                    "default": default,
                    "options": options,
                }
            )
            deadline = time.monotonic() + self.prompt_timeout
            timed_out = False
            while not pending.event.wait(0.2):
                if self._stop.is_set():
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    break
            if pending.value is None:
                if timed_out:
                    self.tag(
                        "WARN",
                        f"No answer after {self.prompt_timeout / 60:g} minutes, so AI Neo stopped "
                        "and nothing was run or sent.",
                    )
                raise StopRequested
            return pending.value
        finally:
            with self._lock:
                self._pending.pop(pid, None)

    def read_line(self, prompt: str) -> str:
        raise EOFError("the browser chat has no console to read from")

    def ask(self, prompt: str, default: str | None = None) -> str:
        answer = self._ask("ask", prompt, default=default or "", options=[], allowed=None)
        return answer.strip() or (default or "")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        text = prompt + (CUT_WARNING if self._last_preview_cut else "")
        self._last_preview_cut = False
        answer = self._ask(
            "confirm",
            text,
            default="y" if default else "n",
            options=[{"value": "y", "label": "Approve"}, {"value": "n", "label": "Deny"}],
            allowed=frozenset({"y", "yes", "n", "no"}),
        )
        return answer.strip().lower() in ("y", "yes")

    def choose(self, prompt: str, letters: str, default: str | None = None) -> str:
        fallback = default if default is not None and default in letters else letters[:1]
        reply = self._ask(
            "choose",
            prompt,
            default=fallback,
            options=options_from_prompt(prompt, letters),
            allowed=frozenset(letters.lower()),
        )
        answer = reply.strip().lower()[:1]
        return answer if answer and answer in letters else fallback

    # -- output ----------------------------------------------------------------------------------

    def say(self, text: str = "", *, dim: bool = False) -> None:
        self.emit({"type": "say", "text": visible_controls(text), "dim": dim})

    def heading(self, text: str) -> None:
        self.emit({"type": "heading", "text": visible_controls(text)})

    def tag(self, status: str, text: str) -> None:
        self.emit({"type": "tag", "status": status, "text": visible_controls(text)})

    def preview(self, title: str, body: str) -> None:
        self._last_preview_cut = len(body) > MAX_PREVIEW_CHARS or bool(_CUT_MARKER.search(body))
        self.emit(
            {
                "type": "preview",
                "title": visible_controls(title),
                "body": visible_controls(clip(body, MAX_PREVIEW_CHARS)),
            }
        )

    def stream(self) -> _WebStream:
        return _WebStream(self)

    def progress(self, description: str) -> _WebProgress:
        return _WebProgress(self, description)

    def table(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]], title: str | None = None
    ) -> None:
        self.emit(
            {
                "type": "table",
                "title": visible_controls(title or ""),
                "headers": [visible_controls(str(h)) for h in headers],
                "rows": [[visible_controls(str(c)) for c in row] for row in rows],
            }
        )

    def tool_call(self, name: str, arguments: Any) -> None:
        if self.stop_requested:
            raise StopRequested
        self.emit(
            {
                "type": "tool_call",
                "name": visible_controls(name),
                "args": visible_controls(short_args(arguments, 400)),
            }
        )

    def tool_result(self, ok: bool, output: str) -> None:
        self.emit(
            {
                "type": "tool_result",
                "ok": ok,
                "output": visible_controls(clip(output, MAX_OUTPUT_CHARS)),
            }
        )

    def stats_line(self, stats: GenStats, extra: str = "") -> None:
        self.emit({"type": "stats", "text": format_stats(stats, extra)})
