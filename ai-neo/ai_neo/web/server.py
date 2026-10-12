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

import hmac
import json
import os
import queue
import secrets
import socket
import threading
import webbrowser
from dataclasses import replace
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
from ai_neo.prompts import MODES
from ai_neo.sessions import Session, SessionNotFound, SessionStore, export_markdown, safe_name
from ai_neo.ui import format_stats
from ai_neo.web.webui import WebUI, clip

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
TOKEN_HEADER = "X-AI-Neo-Token"
MAX_BODY_BYTES = 1_000_000
MAX_PROMPT_CHARS = 100_000
MAX_SYSTEM_CHARS = 20_000
DEFAULT_PORT = 8765
PORT_TRIES = 10
KEEPALIVE_SECONDS = 10.0
THINK_CHOICES = {"auto": None, "on": True, "off": False}


class BadRequest(Exception):
    """A request the server refuses; the message is shown to the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Job:
    """One running task: the worker thread puts events here, the HTTP thread streams them."""

    def __init__(self) -> None:
        self._events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def emit(self, event: dict[str, Any]) -> None:
        self._events.put(event)

    def close(self) -> None:
        self._events.put(None)

    def next(self, timeout: float) -> dict[str, Any] | None:
        """The next event, None when the job is over, or {"type": "ping"} after `timeout`."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return {"type": "ping"}


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
        self._busy = threading.Lock()

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
        if mode is not None and mode not in MODES:
            raise BadRequest(f"mode must be one of {', '.join(MODES)}")
        think = data.get("think")
        if think is not None and think not in THINK_CHOICES:
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
        if self.busy:
            raise BadRequest("A reply is still running. Wait for it or press Stop.", 409)
        self.session = Session(mode=self.session.mode, model=self.stack.client.model)

    def save_session(self, name: str | None) -> str:
        self.sessions.save(self.session, name or None)
        return self.session.name or self.session.id

    def load_session(self, name: str) -> None:
        if self.busy:
            raise BadRequest("A reply is still running. Wait for it or press Stop.", 409)
        try:
            self.session = self.sessions.load(name)
        except SessionNotFound as exc:
            raise BadRequest(str(exc), 404) from exc

    def export(self) -> tuple[str, str]:
        name = safe_name(self.session.name or f"session-{self.session.id}")
        return f"{name}.md", export_markdown(self.session)

    # -- running a task --------------------------------------------------------------------------

    def start_job(self, prompt: str) -> Job | None:
        """Start the task in a worker thread. None means another task is already running."""
        if not self._busy.acquire(blocking=False):
            return None
        job = Job()
        opts = TaskOptions(
            think=self.think,
            use_tools=self.tools_on,
            verify=self.verify_on,
            git=bool(getattr(self.app.args, "git", False)),
        )
        try:
            threading.Thread(target=self._run, args=(job, prompt, opts), daemon=True).start()
        except BaseException:
            self._busy.release()
            raise
        return job

    def _run(self, job: Job, prompt: str, opts: TaskOptions) -> None:
        self.ui.attach(job.emit)
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
                job.emit(self._done_event(outcome))
        finally:
            self.ui.detach()
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

    def _done_event(self, outcome: TaskOutcome) -> dict[str, Any]:
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
            "interrupted": outcome.interrupted,
            "error": outcome.error,
            "notes": list(outcome.notes),
            "stats": stats,
            "context": context,
            "verify": verify,
            "governor": governor,
            "build": build,
            "mode": self.session.mode,
        }

    def stop(self) -> None:
        if self.busy:
            self.ui.request_stop()

    def respond(self, prompt_id: str, value: str) -> bool:
        return self.ui.respond(prompt_id, value)

    def close(self) -> None:
        self.stop()
        self.stack.client.close()


# -- HTTP ----------------------------------------------------------------------------------------


def load_template() -> str:
    return resources.files("ai_neo.web").joinpath("index.html").read_text(encoding="utf-8")


def make_handler(web: WebApp, token: str, template: str) -> type[BaseHTTPRequestHandler]:
    token_bytes = token.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the reply streams and ends when the connection closes
        server_version = "AINeoWeb"
        sys_version = ""

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
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise BadRequest("bad Content-Length") from None
            if length > MAX_BODY_BYTES:
                raise BadRequest("request too large", 413)
            if length <= 0:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
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
                self._json(exc.status, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except Exception as exc:
                self._json(500, {"error": f"Internal error: {type(exc).__name__}"})

        def _index(self, url: Any) -> None:
            supplied = (parse_qs(url.query).get("token") or [""])[0]
            if not self._token_ok(supplied):
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
                "connect-src 'self'; img-src data:; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'"
            )
            self._send(
                200,
                page.encode("utf-8"),
                "text/html; charset=utf-8",
                Content_Security_Policy=policy,
            )

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
            name = self._read_json().get("name")
            if name is not None and not isinstance(name, str):
                raise BadRequest("name must be text")
            self._json(200, {"name": web.save_session(name)})

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
            self._json(200, {"ok": web.respond(prompt_id, value[:20_000])})

        def _chat(self) -> None:
            prompt = self._read_json().get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise BadRequest("Type a message first.")
            if len(prompt) > MAX_PROMPT_CHARS:
                raise BadRequest(f"The message is too long (limit {MAX_PROMPT_CHARS} characters).")
            job = web.start_job(prompt.strip())
            if job is None:
                raise BadRequest("A reply is already running. Wait for it or press Stop.", 409)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self._common_headers()
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    event = job.next(KEEPALIVE_SECONDS)
                    if event is None:
                        break
                    line = json.dumps(event) + "\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                web.stop()  # the tab went away: stop the model instead of finishing unseen

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"  # on Windows this would let two servers share a port


def make_server(
    web: WebApp, token: str, host: str = "127.0.0.1", port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind to `host`, trying the next few ports if `port` is taken (port 0 = any free port)."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"host must be one of {', '.join(LOOPBACK_HOSTS)}")
    handler = make_handler(web, token, load_template())
    last_error: OSError | None = None
    for attempt in range(PORT_TRIES if port else 1):
        server_class = _Server
        if ":" in host:
            server_class = type("_Server6", (_Server,), {"address_family": socket.AF_INET6})
        try:
            return server_class((host, port + attempt if port else 0), handler)
        except OSError as exc:
            last_error = exc
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
    web = WebApp(app)
    token = secrets.token_urlsafe(24)
    try:
        server = make_server(web, token, host, port)
    except OSError as exc:
        ui.tag("FAIL", str(exc))
        web.close()
        return EXIT_FAIL
    shown = server.server_address[1]
    name = f"[{host}]" if ":" in host else host
    url = f"http://{name}:{shown}/?token={token}"
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
            webbrowser.open(url)
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
