"""Streaming client for the Ollama HTTP API (/api/chat, /api/tags, /api/pull)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from ai_neo.config import is_local_url
from ai_neo.types import ChatResult, GenStats, Message, Sampling, StreamEvent, ToolCall

EventSink = Callable[[StreamEvent], None]


class OllamaError(Exception):
    """Any problem talking to Ollama. `partial` holds text received before the failure."""

    def __init__(self, message: str, *, partial: ChatResult | None = None) -> None:
        super().__init__(message)
        self.partial = partial


class OllamaUnavailable(OllamaError):
    pass


class ModelNotFound(OllamaError):
    pass


class ToolCallParseError(OllamaError):
    """The server could not parse a tool call the model wrote (malformed JSON)."""


class PullFailed(OllamaError):
    pass


class PullCancelled(OllamaError):
    pass


class _ThinkingUnsupported(Exception):
    pass


def unavailable_message(base_url: str) -> str:
    return (
        f"Cannot reach Ollama at {base_url}.\n"
        "  - Start the Ollama app (Start menu or system tray) or run `ollama serve` in another "
        "terminal.\n"
        "  - If Ollama runs on another machine, set ollama.base_url in config.toml "
        "(or pass --base-url)."
    )


def missing_model_message(model: str) -> str:
    return (
        f"The model '{model}' is not installed in Ollama.\n  Install it with:  ollama pull {model}"
    )


def normalize_model(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def model_installed(names: Sequence[str], model: str) -> bool:
    wanted = normalize_model(model)
    return any(normalize_model(n) == wanted for n in names)


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest proper prefix of `tag` that `text` ends with."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class ThinkTagSplitter:
    """Separates `<think>...</think>` from answer text, even when tags span chunks."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False
        self._strip_leading = True

    def feed(self, chunk: str) -> list[StreamEvent]:
        self._buffer += chunk
        events: list[StreamEvent] = []
        while True:
            tag = self.CLOSE if self._in_think else self.OPEN
            index = self._buffer.find(tag)
            if index >= 0:
                self._emit(events, self._buffer[:index])
                self._buffer = self._buffer[index + len(tag) :]
                self._in_think = not self._in_think
                if not self._in_think:
                    self._strip_leading = True
                continue
            hold = _partial_suffix(self._buffer, tag)
            cut = len(self._buffer) - hold
            self._emit(events, self._buffer[:cut])
            self._buffer = self._buffer[cut:]
            return events

    def flush(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        self._emit(events, self._buffer)
        self._buffer = ""
        return events

    def _emit(self, events: list[StreamEvent], text: str) -> None:
        if not text:
            return
        if self._in_think:
            events.append(StreamEvent("thinking", text))
            return
        if self._strip_leading:
            text = text.lstrip()
            if not text:
                return
            self._strip_leading = False
        events.append(StreamEvent("answer", text))


@dataclass
class _Accumulator:
    content: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stats: GenStats = field(default_factory=GenStats)
    done_reason: str | None = None
    started: float = field(default_factory=time.monotonic)
    saw_done: bool = False

    def result(self, *, interrupted: bool = False) -> ChatResult:
        text = "".join(self.content).strip()
        thinking = "".join(self.thinking)
        stats = self.stats
        stats.wall_seconds = time.monotonic() - self.started
        if not self.saw_done and stats.eval_count == 0:
            stats.eval_count = max(1, (len(text) + len(thinking)) // 4) if text or thinking else 0
        return ChatResult(
            content=text,
            thinking=thinking,
            tool_calls=self.tool_calls,
            stats=stats,
            interrupted=interrupted,
            done_reason=self.done_reason,
        )


@dataclass(frozen=True)
class PullProgress:
    status: str
    completed: int = 0
    total: int = 0
    digest: str = ""


def _keep_alive_value(raw: str) -> str | int:
    return int(raw) if raw.lstrip("-").isdigit() else raw


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        keep_alive: str = "30m",
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self._no_think: set[str] = set()
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
            trust_env=not is_local_url(self.base_url),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- discovery ---------------------------------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            response = self._http.get("/api/tags", timeout=httpx.Timeout(5.0))
        except httpx.TransportError as exc:
            raise OllamaUnavailable(unavailable_message(self.base_url)) from exc
        if response.status_code != 200:
            raise OllamaError(f"Ollama answered /api/tags with HTTP {response.status_code}.")
        try:
            models = response.json().get("models") or []
        except ValueError as exc:
            raise OllamaError("Ollama sent an unreadable /api/tags reply.") from exc
        return [str(m.get("name") or m.get("model")) for m in models]

    def health(self) -> list[str]:
        """Return installed model names, or raise OllamaUnavailable / ModelNotFound."""
        names = self.list_models()
        if not model_installed(names, self.model):
            raise ModelNotFound(missing_model_message(self.model))
        return names

    # -- chat --------------------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        sampling: Sampling,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_event: EventSink | None = None,
    ) -> ChatResult:
        """Stream one assistant reply. Ctrl+C returns the partial text with `interrupted=True`."""
        try:
            return self._chat_once(messages, sampling, think, tools, on_event)
        except _ThinkingUnsupported:
            self._no_think.add(self.model)
            return self._chat_once(messages, sampling, None, tools, on_event)

    def _payload(
        self,
        messages: Sequence[Message],
        sampling: Sampling,
        think: bool | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_ollama() for m in messages],
            "stream": True,
            "keep_alive": _keep_alive_value(self.keep_alive),
            "options": sampling.options(),
        }
        if think is not None and self.model not in self._no_think:
            payload["think"] = think
        if tools:
            payload["tools"] = tools
        return payload

    def _chat_once(
        self,
        messages: Sequence[Message],
        sampling: Sampling,
        think: bool | None,
        tools: list[dict[str, Any]] | None,
        on_event: EventSink | None,
    ) -> ChatResult:
        acc = _Accumulator()
        splitter = ThinkTagSplitter()
        payload = self._payload(messages, sampling, think, tools)
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    self._raise_for_status(response.status_code, response.text)
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    chunk = self._parse_line(line)
                    self._consume(chunk, acc, splitter, on_event)
                    if chunk.get("done"):
                        break
            self._drain(splitter, acc, on_event)
        except KeyboardInterrupt:
            self._drain(splitter, acc, None)
            return acc.result(interrupted=True)
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(unavailable_message(self.base_url)) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(
                "Timed out waiting for Ollama. The model may still be loading; "
                "raise ollama.request_timeout in config.toml if this keeps happening.",
                partial=acc.result(),
            ) from exc
        except httpx.TransportError as exc:
            raise OllamaError(
                f"The connection to Ollama was lost ({exc.__class__.__name__}).",
                partial=acc.result(),
            ) from exc
        return acc.result()

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any]:
        try:
            chunk = json.loads(line)
        except ValueError as exc:
            raise OllamaError(f"Ollama sent an unreadable stream line: {line[:80]!r}") from exc
        if not isinstance(chunk, dict):
            raise OllamaError(f"Ollama sent an unexpected stream line: {line[:80]!r}")
        return chunk

    def _raise_for_status(self, status: int, body: str) -> None:
        try:
            detail = str(json.loads(body).get("error", body))
        except (ValueError, AttributeError):
            detail = body.strip()
        lowered = detail.lower()
        if status == 400 and "does not support thinking" in lowered:
            raise _ThinkingUnsupported
        if status == 404:
            raise ModelNotFound(missing_model_message(self.model))
        if "error parsing tool call" in lowered:
            raise ToolCallParseError(detail)
        raise OllamaError(f"Ollama returned HTTP {status}: {detail[:300]}")

    def _consume(
        self,
        chunk: dict[str, Any],
        acc: _Accumulator,
        splitter: ThinkTagSplitter,
        on_event: EventSink | None,
    ) -> None:
        if "error" in chunk:
            detail = str(chunk["error"])
            if "does not support thinking" in detail.lower() and not acc.content:
                raise _ThinkingUnsupported
            if "error parsing tool call" in detail.lower():
                raise ToolCallParseError(detail)
            raise OllamaError(f"Ollama reported an error: {detail[:300]}", partial=acc.result())
        message = chunk.get("message") or {}
        thinking = message.get("thinking")
        if thinking:
            self._record(StreamEvent("thinking", thinking), acc, on_event)
        content = message.get("content")
        if content:
            for event in splitter.feed(content):
                self._record(event, acc, on_event)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            acc.tool_calls.append(
                ToolCall(str(function.get("name", "")), function.get("arguments"))
            )
        if chunk.get("done"):
            acc.saw_done = True
            acc.done_reason = chunk.get("done_reason")
            acc.stats = GenStats(
                eval_count=int(chunk.get("eval_count") or 0),
                eval_duration_ns=int(chunk.get("eval_duration") or 0),
                total_duration_ns=int(chunk.get("total_duration") or 0),
                load_duration_ns=int(chunk.get("load_duration") or 0),
                prompt_eval_count=int(chunk.get("prompt_eval_count") or 0),
            )

    def _drain(
        self, splitter: ThinkTagSplitter, acc: _Accumulator, on_event: EventSink | None
    ) -> None:
        for event in splitter.flush():
            self._record(event, acc, on_event)

    @staticmethod
    def _record(event: StreamEvent, acc: _Accumulator, on_event: EventSink | None) -> None:
        (acc.thinking if event.kind == "thinking" else acc.content).append(event.text)
        if on_event is not None:
            on_event(event)

    # -- model download ----------------------------------------------------------------------

    def pull(self, name: str, on_progress: Callable[[PullProgress], None]) -> None:
        """Download a model via /api/pull. Ctrl+C raises PullCancelled; re-running resumes."""
        payload = {"model": name, "name": name, "stream": True}
        timeout = httpx.Timeout(120.0, connect=5.0)
        try:
            with self._http.stream("POST", "/api/pull", json=payload, timeout=timeout) as response:
                if response.status_code != 200:
                    response.read()
                    raise PullFailed(
                        f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
                    )
                finished = False
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    chunk = self._parse_line(line)
                    if "error" in chunk:
                        raise PullFailed(str(chunk["error"]))
                    status = str(chunk.get("status", ""))
                    on_progress(
                        PullProgress(
                            status=status,
                            completed=int(chunk.get("completed") or 0),
                            total=int(chunk.get("total") or 0),
                            digest=str(chunk.get("digest", "")),
                        )
                    )
                    finished = finished or status == "success"
                if not finished:
                    raise PullFailed("The connection closed before the download finished.")
        except KeyboardInterrupt:
            raise PullCancelled(
                "Download cancelled. Run the same step again and Ollama resumes where it stopped."
            ) from None
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(unavailable_message(self.base_url)) from exc
        except httpx.TransportError as exc:
            raise PullFailed(
                f"The connection dropped during the download ({exc.__class__.__name__})."
            ) from exc
