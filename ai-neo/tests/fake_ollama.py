"""A scripted fake Ollama HTTP server bound to 127.0.0.1 (no external network is touched)."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class ChatReply:
    content: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    error: str | None = None  # non-200: JSON error body; 200 + stream_error: mid-stream error
    stream_error: str | None = None
    eval_count: int = 20
    eval_duration: int = 2_000_000_000
    total_duration: int = 2_500_000_000
    load_duration: int = 300_000_000
    prompt_eval_count: int = 12
    done_reason: str = "stop"
    delay: float = 0.0


@dataclass
class PullScript:
    lines: list[dict[str, Any]]
    add_model: bool = True


def text(*chunks: str, **kw: Any) -> ChatReply:
    return ChatReply(content=list(chunks), **kw)


def tool_call(name: str, arguments: Any, **kw: Any) -> ChatReply:
    return ChatReply(tool_calls=[{"function": {"name": name, "arguments": arguments}}], **kw)


def code_reply(code: str, note: str = "Here you go.") -> ChatReply:
    return text(f"{note}\n\n```python\n{code}\n```\n")


class FakeOllama:
    def __init__(self) -> None:
        self.models: list[str] = ["qwen3:30b"]
        self.chat_replies: deque[ChatReply] = deque()
        self.chat_handler: Callable[[dict[str, Any]], ChatReply] | None = None
        self.pull_scripts: deque[PullScript] = deque()
        self.tags_status = 200
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.disconnects = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def chat_payloads(self) -> list[dict[str, Any]]:
        return [body for _, path, body in self.requests if path == "/api/chat" and body]

    def queue(self, *replies: ChatReply) -> None:
        self.chat_replies.extend(replies)

    def _next_reply(self, payload: dict[str, Any]) -> ChatReply:
        if self.chat_handler is not None:
            return self.chat_handler(payload)
        if self.chat_replies:
            return self.chat_replies.popleft()
        return ChatReply(status=500, error="fake server: no scripted reply left")

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _body(self) -> dict[str, Any] | None:
                length = int(self.headers.get("Content-Length") or 0)
                if not length:
                    return None
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def _json(self, status: int, obj: Any) -> None:
                data = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _line(self, obj: Any) -> None:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append(("GET", self.path, None))
                if self.path == "/api/tags":
                    if outer.tags_status != 200:
                        self._json(outer.tags_status, {"error": "tags unavailable"})
                    else:
                        self._json(200, {"models": [{"name": m} for m in outer.models]})
                elif self.path == "/api/version":
                    self._json(200, {"version": "0.0-fake"})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                body = self._body()
                outer.requests.append(("POST", self.path, body))
                try:
                    if self.path == "/api/chat":
                        self._chat(body or {})
                    elif self.path == "/api/pull":
                        self._pull(body or {})
                    else:
                        self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError):
                    outer.disconnects += 1

            def _chat(self, payload: dict[str, Any]) -> None:
                model = payload.get("model", "")
                if model not in outer.models:
                    self._json(404, {"error": f"model '{model}' not found"})
                    return
                reply = outer._next_reply(payload)
                if reply.status != 200:
                    self._json(reply.status, {"error": reply.error or "error"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                base = {"model": model, "created_at": "2026-01-01T00:00:00Z"}
                for piece in reply.thinking:
                    self._line(
                        {
                            **base,
                            "message": {"role": "assistant", "content": "", "thinking": piece},
                            "done": False,
                        }
                    )
                    time.sleep(reply.delay)
                for piece in reply.content:
                    self._line(
                        {**base, "message": {"role": "assistant", "content": piece}, "done": False}
                    )
                    time.sleep(reply.delay)
                if reply.stream_error:
                    self._line({"error": reply.stream_error})
                    return
                if reply.tool_calls:
                    self._line(
                        {
                            **base,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": reply.tool_calls,
                            },
                            "done": False,
                        }
                    )
                self._line(
                    {
                        **base,
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "done_reason": reply.done_reason,
                        "total_duration": reply.total_duration,
                        "load_duration": reply.load_duration,
                        "prompt_eval_count": reply.prompt_eval_count,
                        "eval_count": reply.eval_count,
                        "eval_duration": reply.eval_duration,
                    }
                )

            def _pull(self, payload: dict[str, Any]) -> None:
                script = outer.pull_scripts.popleft() if outer.pull_scripts else None
                if script is None:
                    self._json(500, {"error": "fake server: no pull script"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for line in script.lines:
                    self._line(line)
                    time.sleep(0.01)
                if script.add_model:
                    name = payload.get("model") or payload.get("name")
                    if name and name not in outer.models:
                        outer.models.append(name)

        return Handler
