"""Terminal UI: a Rich implementation with a plain-text fallback behind one small interface.

All text AI Neo prints itself is ASCII so legacy Windows consoles never raise encoding errors.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TextIO

from ai_neo.types import GenStats, StreamEvent

try:  # Rich is optional; everything works without it.
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    HAVE_RICH = True
except ImportError:  # pragma: no cover - exercised only when rich is missing
    HAVE_RICH = False


class StreamRenderer(Protocol):
    def event(self, event: StreamEvent) -> None: ...

    def close(self) -> None: ...


class ProgressBar(Protocol):
    def update(self, completed: int, total: int, label: str = "") -> None: ...

    def close(self) -> None: ...


class UI(Protocol):
    interactive: bool

    def say(self, text: str = "", *, dim: bool = False) -> None: ...

    def heading(self, text: str) -> None: ...

    def tag(self, status: str, text: str) -> None: ...

    def ask(self, prompt: str, default: str | None = None) -> str: ...

    def confirm(self, prompt: str, default: bool = True) -> bool: ...

    def choose(self, prompt: str, letters: str, default: str | None = None) -> str: ...

    def preview(self, title: str, body: str) -> None: ...

    def stream(self) -> StreamRenderer: ...

    def progress(self, description: str) -> ProgressBar: ...

    def table(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]], title: str | None = None
    ) -> None: ...

    def tool_call(self, name: str, arguments: Any) -> None: ...

    def tool_result(self, ok: bool, output: str) -> None: ...

    def stats_line(self, stats: GenStats, extra: str = "") -> None: ...


def human_bytes(count: float) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover


def format_stats(stats: GenStats, extra: str = "") -> str:
    parts = [
        f"{stats.tok_per_s:.1f} tok/s",
        f"{stats.eval_count} tokens",
        f"{stats.seconds:.1f}s",
    ]
    if stats.load_duration_ns > 2_000_000_000:
        parts.append(f"model load {stats.load_duration_ns / 1e9:.1f}s")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def short_args(arguments: Any, limit: int = 160) -> str:
    if isinstance(arguments, dict):
        text = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    else:
        text = repr(arguments)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _BaseUI:
    """Input handling shared by both implementations."""

    interactive: bool

    def _read(self, prompt: str) -> str:
        raise NotImplementedError

    def ask(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default else ""
        answer = self._read(f"{prompt}{suffix}: ").strip()
        return answer if answer else (default or "")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        while True:
            answer = self._read(f"{prompt} {hint} ").strip().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.say("Please answer y or n.")  # type: ignore[attr-defined]

    def choose(self, prompt: str, letters: str, default: str | None = None) -> str:
        while True:
            answer = self._read(f"{prompt} ").strip().lower()
            if not answer and default:
                return default
            if answer[:1] in letters:
                return answer[:1]
            self.say(f"Please type one of: {', '.join(letters.upper())}.")  # type: ignore[attr-defined]


class _PlainStream:
    def __init__(self, ui: PlainUI) -> None:
        self.ui = ui
        self.thinking_started = False
        self.answer_started = False

    def event(self, event: StreamEvent) -> None:
        if event.kind == "thinking":
            if not self.thinking_started:
                self.ui.write(self.ui.style("(thinking) ", dim=True))
                self.thinking_started = True
            self.ui.write(self.ui.style(event.text, dim=True))
            return
        if self.thinking_started and not self.answer_started:
            self.ui.write("\n--- answer ---\n")
        self.answer_started = True
        self.ui.write(event.text)

    def close(self) -> None:
        if self.thinking_started or self.answer_started:
            self.ui.write("\n")


class _PlainProgress:
    def __init__(self, ui: PlainUI, description: str) -> None:
        self.ui = ui
        self.description = description
        self.last_bucket = -1
        self.dirty = False

    def update(self, completed: int, total: int, label: str = "") -> None:
        if total <= 0:
            return
        percent = int(100 * completed / total)
        sizes = f"{human_bytes(completed)} of {human_bytes(total)}"
        text = f"{label or self.description}: {percent}% ({sizes})"
        if self.ui.tty:
            self.ui.write(f"\r{text}   ")
            self.dirty = True
        elif percent // 10 != self.last_bucket:
            self.last_bucket = percent // 10
            self.ui.say(text)

    def close(self) -> None:
        if self.dirty:
            self.ui.write("\n")
            self.dirty = False


class PlainUI(_BaseUI):
    def __init__(
        self,
        *,
        read: Callable[[str], str] | None = None,
        out: TextIO | None = None,
        color: bool | None = None,
        interactive: bool | None = None,
    ) -> None:
        self.out = out if out is not None else sys.stdout
        self._reader = read if read is not None else input
        self.tty = bool(getattr(self.out, "isatty", lambda: False)())
        self.color = (self.tty and "NO_COLOR" not in os.environ) if color is None else color
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        if self.color and os.name == "nt":  # pragma: no cover - Windows only
            os.system("")  # noqa: S605,S607 - turns on ANSI escape handling in the console

    def _read(self, prompt: str) -> str:
        return self._reader(prompt)

    def write(self, text: str) -> None:
        print(text, end="", file=self.out, flush=True)

    def style(self, text: str, *, dim: bool = False, color: str | None = None) -> str:
        if not self.color:
            return text
        codes = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "bold": "1"}
        if dim:
            return f"\x1b[2m{text}\x1b[0m"
        if color in codes:
            return f"\x1b[{codes[color]}m{text}\x1b[0m"
        return text

    def say(self, text: str = "", *, dim: bool = False) -> None:
        print(self.style(text, dim=dim) if dim else text, file=self.out, flush=True)

    def heading(self, text: str) -> None:
        self.say()
        self.say(self.style(text, color="bold"))
        self.say("-" * len(text))

    def tag(self, status: str, text: str) -> None:
        colors = {"OK": "green", "PASS": "green", "WARN": "yellow", "FAIL": "red"}
        label = self.style(f"[{status}]", color=colors.get(status))
        self.say(f"{label} {text}")

    def preview(self, title: str, body: str) -> None:
        self.say()
        self.say(self.style(f"== {title} ==", color="cyan"))
        self.say(body)
        self.say(self.style("=" * (len(title) + 6), color="cyan"))

    def stream(self) -> StreamRenderer:
        return _PlainStream(self)

    def progress(self, description: str) -> ProgressBar:
        return _PlainProgress(self, description)

    def table(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]], title: str | None = None
    ) -> None:
        widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows, strict=False)]
        if title:
            self.say(title)
        self.say("  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=False)))
        self.say("  ".join("-" * w for w in widths))
        for row in rows:
            self.say("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=False)))

    def tool_call(self, name: str, arguments: Any) -> None:
        self.say(f"  -> {name}({short_args(arguments)})", dim=True)

    def tool_result(self, ok: bool, output: str) -> None:
        first = output.strip().splitlines()[:6]
        text = "\n".join("     " + line[:200] for line in first) or "     (no output)"
        self.say(("  ok:\n" if ok else "  failed:\n") + text, dim=True)

    def stats_line(self, stats: GenStats, extra: str = "") -> None:
        self.say(f"[{format_stats(stats, extra)}]", dim=True)


class _RichStream:
    TAIL_LINES = 6

    def __init__(self, ui: RichUI) -> None:
        self.console = ui.console
        self.buffer = ""
        self.live: Live | None = None
        self.started = 0.0
        self.answer_started = False

    def event(self, event: StreamEvent) -> None:
        if event.kind == "thinking":
            if self.live is None and not self.answer_started:
                self.started = time.monotonic()
                self.live = Live(console=self.console, transient=True, refresh_per_second=8)
                self.live.start()
            self.buffer += event.text
            if self.live is not None:
                tail = "\n".join(self.buffer.splitlines()[-self.TAIL_LINES :])
                self.live.update(Text("thinking...\n" + tail, style="dim"))
            return
        self._collapse()
        self.answer_started = True
        self.console.print(event.text, end="", markup=False, highlight=False, soft_wrap=True)

    def _collapse(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        self.live = None
        seconds = time.monotonic() - self.started
        tokens = max(1, len(self.buffer) // 4)
        self.console.print(
            f"[thought for {seconds:.0f}s, ~{tokens} tokens]", style="dim", markup=False
        )

    def close(self) -> None:
        self._collapse()
        if self.answer_started:
            self.console.print()


class _RichProgress:
    def __init__(self, ui: RichUI, description: str) -> None:
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=ui.console,
            transient=False,
        )
        self.task = self.progress.add_task(description[:40], total=None)
        self.progress.start()

    def update(self, completed: int, total: int, label: str = "") -> None:
        fields: dict[str, Any] = {"completed": completed, "total": total or None}
        if label:
            fields["description"] = label[:40]
        self.progress.update(self.task, **fields)

    def close(self) -> None:
        self.progress.stop()


class RichUI(_BaseUI):
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.interactive = sys.stdin.isatty()

    def _read(self, prompt: str) -> str:
        return self.console.input(prompt)

    def say(self, text: str = "", *, dim: bool = False) -> None:
        self.console.print(text, style="dim" if dim else None, markup=False, highlight=False)

    def heading(self, text: str) -> None:
        self.console.print()
        self.console.rule(text, style="cyan", align="left")

    def tag(self, status: str, text: str) -> None:
        colors = {"OK": "green", "PASS": "green", "WARN": "yellow", "FAIL": "red"}
        line = Text()
        line.append(f"[{status}]", style=f"bold {colors.get(status, 'white')}")
        line.append(f" {text}")
        self.console.print(line)

    def preview(self, title: str, body: str) -> None:
        self.console.print(Panel(Text(body), title=title, title_align="left", border_style="cyan"))

    def stream(self) -> StreamRenderer:
        return _RichStream(self)

    def progress(self, description: str) -> ProgressBar:
        return _RichProgress(self, description)

    def table(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]], title: str | None = None
    ) -> None:
        grid = Table(title=title, title_justify="left")
        for header in headers:
            grid.add_column(str(header))
        for row in rows:
            grid.add_row(*[str(c) for c in row])
        self.console.print(grid)

    def tool_call(self, name: str, arguments: Any) -> None:
        self.say(f"  -> {name}({short_args(arguments)})", dim=True)

    def tool_result(self, ok: bool, output: str) -> None:
        first = output.strip().splitlines()[:6]
        text = "\n".join("     " + line[:200] for line in first) or "     (no output)"
        self.say(("  ok:\n" if ok else "  failed:\n") + text, dim=True)

    def stats_line(self, stats: GenStats, extra: str = "") -> None:
        self.say(f"[{format_stats(stats, extra)}]", dim=True)


def make_ui(*, plain: bool = False) -> UI:
    """Rich on an interactive terminal, plain text everywhere else (pipes, CI, NO_COLOR)."""
    if plain or not HAVE_RICH or not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return PlainUI()
    return RichUI()
