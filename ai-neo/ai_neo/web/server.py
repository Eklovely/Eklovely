"""The local web server behind `ai-neo web`.

It serves one page (`index.html`) and a small JSON API on top of the same `TaskPipeline` that the
terminal chat uses, so modes, tools, run-and-repair, build mode, the governor, sessions and logs all
behave exactly as they do in the terminal. The browser never talks to Ollama itself.

Because the tools can write files and run code, the server is deliberately locked down:

* it only binds to a loopback address, so other computers cannot connect;
* every API call needs a random per-launch token (printed in the URL on start-up);
* the Host header must be a loopback name and POSTs must be same-origin JSON, which blocks DNS
  rebinding and cross-site requests from other web pages;
* the page is served with a strict Content-Security-Policy and no CORS headers.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import os
import queue
import secrets
import select
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import replace
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ai_neo import __version__
from ai_neo.agent import resolve_think
from ai_neo.cli import EXIT_FAIL, EXIT_OK, EXIT_USAGE, App, build_stack
from ai_neo.ollama_client import (
    ModelNotFound,
    OllamaError,
    OllamaUnavailable,
    missing_model_message,
    model_installed,
)
from ai_neo.pipeline import TaskOptions, TaskOutcome
from ai_neo.prompts import (
    CODE_NO_BLOCK,
    CODE_REPAIR,
    COMPUTE_NUDGE,
    GOVERNOR_REVISE,
    MODES,
    TOOL_CALL_RETRY,
)
from ai_neo.sandbox import kill_running
from ai_neo.sessions import Session, SessionNotFound, SessionStore, export_markdown, safe_name
from ai_neo.ui import format_stats
from ai_neo.web.webui import WebUI, clip

# Messages the pipeline sends to the model on its own (repair feedback, nudges). They are not
# something the user typed, so the page does not show them as if they were.
_INTERNAL_PREFIXES = tuple(
    text.strip()
    for text in (
        COMPUTE_NUDGE[:40],
        CODE_NO_BLOCK[:40],
        CODE_REPAIR.split("\n")[0],
        TOOL_CALL_RETRY.split("$")[0],
        GOVERNOR_REVISE.split("\n")[0],
    )
)

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
TOKEN_HEADER = "X-AI-Neo-Token"
MAX_BODY_BYTES = 1_000_000
MAX_PROMPT_CHARS = 100_000
MAX_SYSTEM_CHARS = 20_000
DEFAULT_PORT = 8765
PORT_TRIES = 10
KEEPALIVE_SECONDS = 10.0
IDLE_POLL_SECONDS = 1.0  # how often a waiting stream checks that the browser is still there
SOCKET_TIMEOUT = 30.0  # a connection that stalls this long is dropped
MAX_CONNECTIONS = 64
LAUNCH_CODE_SECONDS = 120.0
THINK_CHOICES = {"auto": None, "on": True, "off": False}


class BadRequest(Exception):
    """A request the server refuses; the message is shown to the user."""

    def __init__(self, message: str, status: int = 400, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class LaunchCode:
    """A short-lived code for opening the page. Unlike the token it may sit in a browser's
    command line or history: it stops working after two minutes."""

    def __init__(self, ttl: float = LAUNCH_CODE_SECONDS) -> None:
        self.value = secrets.token_urlsafe(18)
        self._deadline = time.monotonic() + ttl

    def valid(self, supplied: str) -> bool:
        fresh = time.monotonic() < self._deadline
        return hmac.compare_digest(supplied.encode("utf-8"), self.value.encode("utf-8")) and fresh


_IDLE = object()  # returned by Job.next when nothing happened for a moment


class Job:
    """One running task: the worker thread puts events here, the HTTP thread streams them."""

    def __init__(self) -> None:
        self._events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.stopped = threading.Event()  # this job's own Stop flag

    def emit(self, event: dict[str, Any]) -> None:
        self._events.put(event)

    def close(self) -> None:
        self._events.put(None)

    def next(self, timeout: float) -> Any:
        """The next event, None when the job is over, or _IDLE if nothing happened in `timeout`."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return _IDLE


class WebApp:
    """The state behind the page: one conversation, its settings and at most one running task."""

    def __init__(self, app: App) -> None:
        self.ui = WebUI()
        self.app = replace(app, ui=self.ui)
        self.cfg = app.cfg
        self.stack = build_stack(self.app, auto_yes=False)  # the governor must always ask first
        self.stack.workspace.root.mkdir(parents=True, exist_ok=True)
        self.auto_approve = bool(getattr(app.args, "yes", False))
        self.stack.agent.registry.set_auto_yes(self.auto_approve)
        self.sessions = SessionStore(app.paths.sessions)
        self.session = Session(mode=app.cfg.chat.mode, model=self.stack.client.model)
        self.think: bool | None = None  # None = config / the mode's own default
        self.tools_on = True
        self.verify_on = not bool(getattr(app.args, "no_verify", False))
        self._busy = threading.Lock()  # held while a reply runs, and while settings change
        self._job_lock = threading.Lock()
        self._job: Job | None = None

    @contextlib.contextmanager
    def _mutation(self) -> Any:
        """Hold the busy lock while changing settings or the conversation, so nothing can change
        underneath a running reply (and a reply cannot start in the middle of a change)."""
        if not self._busy.acquire(blocking=False):
            raise BadRequest("A reply is still running. Wait for it or press Stop.", 409)
        try:
            yield
        finally:
            self._busy.release()

    # -- state -----------------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def effective_think(self) -> bool:
        return resolve_think(self.cfg.chat.think, self.think, MODES[self.session.mode])

    def display_messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for message in self.session.messages:
            if message.role == "user":
                if message.content.lstrip().startswith(_INTERNAL_PREFIXES):
                    continue
                out.append({"role": "user", "content": message.content})
            elif message.role == "assistant" and message.content.strip():
                out.append({"role": "assistant", "content": message.content})
            elif message.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "name": message.tool_name or "tool",
                        "content": clip(message.content, 2000),
                    }
                )
        return out

    def state(self) -> dict[str, Any]:
        client, gov = self.stack.client, self.stack.governor
        models: list[str] = []
        ollama: dict[str, Any]
        try:
            models = client.list_models()
            ollama = {"ok": True, "model_installed": model_installed(models, client.model)}
        except OllamaError as exc:
            ollama = {"ok": False, "error": str(exc)}
        context = self.stack.agent.context
        session = self.session
        think = next(label for label, value in THINK_CHOICES.items() if value == self.think)
        return {
            "version": __version__,
            "busy": self.busy,
            "ollama": ollama,
            "model": client.model,
            "models": models,
            "modes": [
                {
                    "name": preset.name,
                    "summary": preset.summary,
                    "think_default": preset.think_default,
                    "tools": list(preset.tools),
                }
                for preset in MODES.values()
            ],
            "settings": {
                "mode": session.mode,
                "think": think,
                "effective_think": self.effective_think(),
                "tools": self.tools_on,
                "verify": self.verify_on,
                "auto_approve": self.auto_approve,
                "govern": gov.enabled,
                "system": session.system_override or "",
            },
            "governor": {
                "reason": gov.unavailable_reason(),
                "enabled": gov.enabled,
                "calls": gov.calls,
                "tokens": gov.tokens,
            },
            "session": {
                "id": session.id,
                "name": session.name,
                "turns": session.turns,
                "context_percent": context.percent(session),
                "num_ctx": self.cfg.ollama.num_ctx,
            },
            "workspace": str(self.stack.workspace.root),
            "messages": self.display_messages(),
        }

    # -- settings and sessions ---------------------------------------------------------------------

    def update_settings(self, data: dict[str, Any]) -> list[str]:
        """Apply the settings in `data`. Returns warnings (for example why the governor stayed off).

        Everything is validated before anything changes, so a bad request changes nothing.
        """
        if self.busy:
            raise BadRequest("A reply is still running. Wait for it or press Stop.", 409)
        mode = data.get("mode")
        if mode is not None and not (isinstance(mode, str) and mode in MODES):
            raise BadRequest(f"mode must be one of {', '.join(MODES)}")
        think = data.get("think")
        if think is not None and not (isinstance(think, str) and think in THINK_CHOICES):
            raise BadRequest("think must be auto, on or off")
        flags: dict[str, bool] = {}
        for key in ("tools", "verify", "auto_approve", "govern"):
            if key in data:
                if not isinstance(data[key], bool):
                    raise BadRequest(f"{key} must be true or false")
                flags[key] = data[key]
        system = data.get("system")
        if system is not None and (not isinstance(system, str) or len(system) > MAX_SYSTEM_CHARS):
            raise BadRequest(f"system must be text of at most {MAX_SYSTEM_CHARS} characters")
        model = data.get("model")
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise BadRequest("model must be a model name")
            model = model.strip()
            try:
                installed = self.stack.client.list_models()
            except OllamaError as exc:
                raise BadRequest(str(exc)) from exc
            if not model_installed(installed, model):
                raise BadRequest(missing_model_message(model))

        warnings: list[str] = []
        with self._mutation():
            if mode is not None:
                self.session.mode = mode
            if think is not None:
                self.think = THINK_CHOICES[think]
            if "tools" in flags:
                self.tools_on = flags["tools"]
            if "verify" in flags:
                self.verify_on = flags["verify"]
            if "auto_approve" in flags:
                self.auto_approve = flags["auto_approve"]
                self.stack.agent.registry.set_auto_yes(self.auto_approve)
            if "govern" in flags:
                self.stack.governor.reset_consent()  # turning it on or off always starts fresh
                if flags["govern"]:
                    reason = self.stack.governor.enable()
                    if reason:
                        warnings.append(f"The governor cannot be turned on: {reason}.")
                else:
                    self.stack.governor.disable()
            if system is not None:
                self.session.system_override = system.strip() or None
            if model is not None:
                self.stack.client.model = model
                self.session.model = model
        return warnings

    def new_chat(self) -> None:
        with self._mutation():
            self.session = Session(mode=self.session.mode, model=self.stack.client.model)
            self.stack.governor.reset_consent()  # a new chat asks again before anything is sent

    def save_session(self, name: str | None, overwrite: bool = False) -> str:
        with self._mutation():  # never save a half-finished turn, or two saves at once
            if name and not overwrite:
                path = self.sessions.path_for(name)
                if path.exists() and self._saved_id(path) != self.session.id:
                    raise BadRequest(
                        f"A saved chat called '{safe_name(name)}' already exists.", 409, "exists"
                    )
            self.sessions.save(self.session, name or None)
            return self.session.name or self.session.id

    @staticmethod
    def _saved_id(path: Any) -> str | None:
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("id"))
        except (OSError, ValueError, AttributeError):
            return None

    def load_session(self, name: str) -> None:
        with self._mutation():
            try:
                self.session = self.sessions.load(name)
            except SessionNotFound as exc:
                raise BadRequest(str(exc), 404) from exc
            self.stack.governor.reset_consent()

    def export(self) -> tuple[str, str]:
        name = safe_name(self.session.name or f"session-{self.session.id}")
        return f"{name}.md", export_markdown(self.session)

    # -- running a task --------------------------------------------------------------------------

    def start_job(self, prompt: str) -> Job | None:
        """Start the task in a worker thread. None means another task is already running."""
        if not self._busy.acquire(blocking=False):
            return None
        job = Job()
        with self._job_lock:
            self._job = job  # from here on a Stop reaches this job, even before its thread runs
        opts = TaskOptions(
            think=self.think,
            use_tools=self.tools_on,
            verify=self.verify_on,
            git=bool(getattr(self.app.args, "git", False)),
        )
        try:
            threading.Thread(target=self._run, args=(job, prompt, opts), daemon=True).start()
        except BaseException:
            with self._job_lock:
                self._job = None
            self._busy.release()
            raise
        return job

    def _run(self, job: Job, prompt: str, opts: TaskOptions) -> None:
        self.ui.attach(job.emit, job.stopped)
        try:
            job.emit(
                {
                    "type": "start",
                    "mode": self.session.mode,
                    "think": self.effective_think() if opts.think is None else opts.think,
                    "tools": opts.use_tools,
                }
            )
            try:
                outcome = self.stack.pipeline.run(self.session, prompt, opts)
            except KeyboardInterrupt:  # Stop pressed outside a model call
                job.emit(self._stopped_event())
            except OllamaError as exc:
                job.emit({"type": "error", "message": str(exc)})
            except Exception as exc:  # never leave the browser waiting on a dead worker
                job.emit(
                    {"type": "error", "message": f"Unexpected problem: {type(exc).__name__}: {exc}"}
                )
            else:
                job.emit(self._done_event(outcome, stopped=job.stopped.is_set()))
        finally:
            self.ui.detach()
            with self._job_lock:
                if self._job is job:
                    self._job = None
            self._busy.release()  # before the stream ends, so the next request is never refused
            job.close()

    def _context(self) -> dict[str, int]:
        return {
            "percent": self.stack.agent.context.percent(self.session),
            "num_ctx": self.cfg.ollama.num_ctx,
        }

    def _stopped_event(self) -> dict[str, Any]:
        return {
            "type": "done",
            "answer": "",
            "thinking": "",
            "interrupted": True,
            "error": None,
            "notes": [],
            "stats": {"text": ""},
            "context": self._context(),
            "verify": None,
            "governor": None,
            "build": None,
            "mode": self.session.mode,
        }

    def _done_event(self, outcome: TaskOutcome, stopped: bool = False) -> dict[str, Any]:
        context = self._context()
        stats: dict[str, Any] = {"text": ""}
        if outcome.stats.eval_count:
            extra = f"context {context['percent']}% of {context['num_ctx']}"
            stats = {**outcome.stats.to_dict(), "text": format_stats(outcome.stats, extra)}
        verify = None
        if outcome.verify is not None and outcome.verify.attempts:
            verify = {"passed": outcome.verify.passed, "summary": outcome.verify.summary_line()}
        governor = None
        if outcome.governor is not None and outcome.governor.used and outcome.governor.rounds:
            governor = (
                f"{len(outcome.governor.rounds)} round(s), last score "
                f"{outcome.governor.rounds[-1].critique.score:g}/10, "
                f"{outcome.governor.calls} Claude call(s) so far"
            )
        build = None
        if outcome.build is not None:
            build = {
                "passed": outcome.build.passed,
                "project_dir": str(outcome.build.project_dir or ""),
                "files": len(outcome.build.files_written),
            }
        return {
            "type": "done",
            "answer": outcome.answer,
            "thinking": clip(outcome.thinking, 20000),
            "interrupted": outcome.interrupted or stopped,
            "error": outcome.error,
            "notes": list(outcome.notes),
            "stats": stats,
            "context": context,
            "verify": verify,
            "governor": governor,
            "build": build,
            "mode": self.session.mode,
        }

    def stop(self, job: Job | None = None) -> None:
        """Stop the running reply. With `job`, only if that job is still the running one, so a
        late Stop from an old connection can never cancel a newer reply."""
        with self._job_lock:
            current = self._job
            if current is None or (job is not None and job is not current):
                return
            current.stopped.set()
        self.ui.wake()  # release a question that is waiting for an answer
        self.stack.client.cancel()  # also wakes a read still waiting for the first token
        kill_running()  # and kill any script or test run that is in progress

    def respond(self, prompt_id: str, value: str) -> bool:
        return self.ui.respond(prompt_id, value)

    def close(self) -> None:
        self.stop()
        if self._busy.acquire(timeout=5):  # let the worker finish unwinding
            self._busy.release()
        kill_running()  # nothing is left running after the server is gone
        self.stack.client.close()


# -- HTTP ----------------------------------------------------------------------------------------


def load_template() -> str:
    return resources.files("ai_neo.web").joinpath("index.html").read_text(encoding="utf-8")


def make_handler(
    web: WebApp, token: str, template: str, launch: LaunchCode
) -> type[BaseHTTPRequestHandler]:
    token_bytes = token.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the reply streams and ends when the connection closes
        server_version = "AINeoWeb"
        sys_version = ""
        timeout = SOCKET_TIMEOUT  # idle or half-sent connections are dropped, not kept forever

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        # -- plumbing ----------------------------------------------------------------------------

        def _allowed_hosts(self) -> set[str]:
            port = self.server.server_address[1]
            return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

        def _common_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")

        def _send(self, status: int, body: bytes, content_type: str, **extra: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._common_headers()
            for key, value in extra.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

        def _text(self, status: int, text: str) -> None:
            self._send(status, text.encode("utf-8"), "text/plain; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            if self.headers.get("Transfer-Encoding"):
                raise BadRequest("chunked request bodies are not supported", 411)
            raw = (self.headers.get("Content-Length") or "0").strip()
            if not (raw.isascii() and raw.isdigit()):
                raise BadRequest("bad Content-Length")
            length = int(raw)
            if length > MAX_BODY_BYTES:
                raise BadRequest("request too large", 413)
            if length == 0:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, RecursionError):
                raise BadRequest("the request body is not valid JSON") from None
            if not isinstance(data, dict):
                raise BadRequest("the request body must be a JSON object")
            return data

        # -- security ----------------------------------------------------------------------------

        def _host_ok(self) -> bool:
            return (self.headers.get("Host") or "").lower() in self._allowed_hosts()

        def _token_ok(self, supplied: str) -> bool:
            return hmac.compare_digest(supplied.encode("utf-8"), token_bytes)

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return origin.lower() in {f"http://{h}" for h in self._allowed_hosts()}

        # -- dispatch ----------------------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._text(405, "No cross-origin access.")

        def _dispatch(self, method: str) -> None:
            try:
                url = urlsplit(self.path)
                if not self._host_ok():
                    self._text(403, "Forbidden host name.")
                elif method == "GET" and url.path == "/":
                    self._index(url)
                elif method == "GET" and url.path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                elif url.path.startswith("/api/"):
                    self._api(method, url.path)
                else:
                    self._text(404, "Not found.")
            except BadRequest as exc:
                body: dict[str, Any] = {"error": str(exc)}
                if exc.code:
                    body["code"] = exc.code
                self._json(exc.status, body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                return
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)  # for whoever runs the server; not the page
                self._json(500, {"error": f"Internal error: {type(exc).__name__}"})

        def _cookie_name(self) -> str:
            return f"ai_neo_{self.server.server_address[1]}"

        def _cookie_ok(self) -> bool:
            try:
                jar: SimpleCookie = SimpleCookie(self.headers.get("Cookie") or "")
            except CookieError:
                return False
            morsel = jar.get(self._cookie_name())
            return morsel is not None and self._token_ok(morsel.value)

        def _index(self, url: Any) -> None:
            query = parse_qs(url.query)
            supplied = (query.get("token") or [""])[0]
            code = (query.get("code") or [""])[0]
            if self._token_ok(supplied) or (code and launch.valid(code)):
                # Remember the visit in a cookie the page script cannot read, so the page can
                # drop the secret from the address bar and still survive a reload.
                cookie = f"{self._cookie_name()}={token}; HttpOnly; SameSite=Strict; Path=/"
            elif self._cookie_ok():
                cookie = ""
            else:
                self._text(
                    403,
                    "Open the link that `ai-neo web` printed in your terminal "
                    "(it contains a secret token).",
                )
                return
            nonce = secrets.token_urlsafe(16)
            page = (
                template.replace("__NONCE__", nonce)
                .replace("__TOKEN__", token)
                .replace("__VERSION__", __version__)
            )
            policy = (
                "default-src 'none'; "
                f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                "connect-src 'self'; img-src 'none'; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'"
            )
            extra = {"Content_Security_Policy": policy}
            if cookie:
                extra["Set_Cookie"] = cookie
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8", **extra)

        def _api(self, method: str, path: str) -> None:
            if not self._token_ok(self.headers.get(TOKEN_HEADER) or ""):
                raise BadRequest("Missing or wrong token. Open the link `ai-neo web` printed.", 403)
            if method == "POST":
                if not self._origin_ok():
                    raise BadRequest("Cross-origin request refused.", 403)
                if (
                    not (self.headers.get("Content-Type") or "")
                    .lower()
                    .startswith("application/json")
                ):
                    raise BadRequest("Content-Type must be application/json.", 415)
            routes = {
                ("GET", "/api/state"): self._state,
                ("GET", "/api/sessions"): self._sessions,
                ("GET", "/api/export"): self._export,
                ("POST", "/api/settings"): self._settings,
                ("POST", "/api/chat"): self._chat,
                ("POST", "/api/respond"): self._respond,
                ("POST", "/api/stop"): self._stop,
                ("POST", "/api/new"): self._new,
                ("POST", "/api/save"): self._save,
                ("POST", "/api/load"): self._load,
            }
            route = routes.get((method, path))
            if route is None:
                raise BadRequest("Unknown endpoint.", 404)
            route()

        # -- endpoints ---------------------------------------------------------------------------

        def _state(self) -> None:
            self._json(200, web.state())

        def _sessions(self) -> None:
            names = [{"name": n, "mode": m, "messages": c} for n, m, c in web.sessions.list()]
            self._json(200, {"sessions": names})

        def _export(self) -> None:
            filename, text = web.export()
            self._send(
                200,
                text.encode("utf-8"),
                "text/markdown; charset=utf-8",
                Content_Disposition=f'attachment; filename="{filename}"',
            )

        def _settings(self) -> None:
            warnings = web.update_settings(self._read_json())
            self._json(200, {"warnings": warnings})

        def _new(self) -> None:
            web.new_chat()
            self._json(200, {"ok": True})

        def _save(self) -> None:
            data = self._read_json()
            name, overwrite = data.get("name"), data.get("overwrite", False)
            if name is not None and not isinstance(name, str):
                raise BadRequest("name must be text")
            if not isinstance(overwrite, bool):
                raise BadRequest("overwrite must be true or false")
            self._json(200, {"name": web.save_session(name, overwrite)})

        def _load(self) -> None:
            name = self._read_json().get("name")
            if not isinstance(name, str) or not name.strip():
                raise BadRequest("name must be the name of a saved chat")
            web.load_session(name)
            self._json(200, {"ok": True})

        def _stop(self) -> None:
            web.stop()
            self._json(200, {"ok": True})

        def _respond(self) -> None:
            data = self._read_json()
            prompt_id, value = data.get("id"), data.get("value")
            if not isinstance(prompt_id, str) or not isinstance(value, str):
                raise BadRequest("id and value must be text")
            try:
                self._json(200, {"ok": web.respond(prompt_id, value[:20_000])})
            except ValueError as exc:
                raise BadRequest(f"Not an offered answer: {exc}.") from exc

        def _chat(self) -> None:
            prompt = self._read_json().get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise BadRequest("Type a message first.")
            if len(prompt) > MAX_PROMPT_CHARS:
                raise BadRequest(f"The message is too long (limit {MAX_PROMPT_CHARS} characters).")
            job = web.start_job(prompt.strip())
            if job is None:
                raise BadRequest("A reply is already running. Wait for it or press Stop.", 409)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self._common_headers()
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                last_write = time.monotonic()
                while True:
                    event = job.next(IDLE_POLL_SECONDS)
                    if event is _IDLE:
                        if self._client_gone():
                            raise ConnectionResetError("the browser closed the connection")
                        if time.monotonic() - last_write < KEEPALIVE_SECONDS:
                            continue
                        event = {"type": "ping"}
                    if event is None:
                        break
                    line = json.dumps(event) + "\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                    last_write = time.monotonic()
            except OSError:  # includes resets, broken pipes and timeouts
                web.stop(job)  # the tab went away: stop this reply instead of finishing unseen

        def _client_gone(self) -> bool:
            """True if the browser has closed its end (checked without sending anything)."""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"  # on Windows this would let two servers share a port
    max_connections = MAX_CONNECTIONS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._slots = threading.BoundedSemaphore(self.max_connections)
        super().__init__(*args, **kwargs)

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # pragma: no cover
            # On Windows another program could otherwise bind the same port and share it.
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)  # too many open connections: refuse this one
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_server(
    web: WebApp,
    token: str,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    launch: LaunchCode | None = None,
) -> ThreadingHTTPServer:
    """Bind to `host`, trying the next few ports if `port` is taken (port 0 = any free port)."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"host must be one of {', '.join(LOOPBACK_HOSTS)}")
    handler = make_handler(web, token, load_template(), launch or LaunchCode())
    last_error: OSError | None = None
    for attempt in range(min(PORT_TRIES, 65536 - port) if port else 1):
        server_class = _Server
        if ":" in host:
            server_class = type("_Server6", (_Server,), {"address_family": socket.AF_INET6})
        try:
            return server_class((host, port + attempt if port else 0), handler)
        except (OSError, OverflowError) as exc:
            last_error = exc if isinstance(exc, OSError) else OSError(str(exc))
    raise OSError(f"could not open a port starting at {port}: {last_error}")


def serve(app: App, *, host: str, port: int, open_browser: bool) -> int:
    ui = app.ui
    if host not in LOOPBACK_HOSTS:
        ui.tag(
            "FAIL",
            f"--host must be one of {', '.join(LOOPBACK_HOSTS)}. The browser chat can write files "
            "and run code, so it only accepts connections from this computer.",
        )
        return EXIT_USAGE
    if not 0 <= port <= 65535:
        ui.tag("FAIL", f"--port must be between 0 and 65535 (got {port}).")
        return EXIT_USAGE
    web = WebApp(app)
    token = secrets.token_urlsafe(24)
    launch = LaunchCode()
    try:
        server = make_server(web, token, host, port, launch)
    except OSError as exc:
        ui.tag("FAIL", str(exc))
        web.close()
        return EXIT_FAIL
    shown = server.server_address[1]
    name = f"[{host}]" if ":" in host else host
    url = f"http://{name}:{shown}/?token={token}"
    browser_url = f"http://{name}:{shown}/?code={launch.value}"  # good for two minutes only
    try:
        web.stack.client.health()
    except (OllamaUnavailable, ModelNotFound, OllamaError) as exc:
        ui.tag("WARN", f"{exc}\nThe page will open anyway and show this status.")
    if getattr(app.args, "govern", False):
        reason = web.stack.governor.enable()
        if reason:
            ui.tag("WARN", f"--govern was ignored: {reason}.")
    ui.tag("OK", "AI Neo is running in your browser.")
    ui.say(f"Open this link (it contains a secret token, so keep it to yourself):\n  {url}")
    ui.say("Only this computer can connect. Press Ctrl+C here to stop the server.", dim=True)
    if open_browser:
        try:
            webbrowser.open(browser_url)
        except Exception:  # a missing browser must never stop the server
            ui.say("(Could not open a browser automatically; use the link above.)", dim=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        ui.say("\nStopping.")
    finally:
        web.close()
        server.server_close()
    return EXIT_OK
