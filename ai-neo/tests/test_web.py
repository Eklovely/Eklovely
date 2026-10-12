"""The browser chat: security, streaming, approvals, stop, sessions and the UI adapter."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import ai_neo.web.server as server_module
from ai_neo.cli import EXIT_USAGE, build_parser, load_app, main
from ai_neo.types import GenStats, StreamEvent
from ai_neo.web.server import WebApp, load_template, make_server, serve
from ai_neo.web.webui import StopRequested, WebUI, clip, options_from_prompt
from tests.fake_ollama import ChatReply, FakeOllama, code_reply, text, tool_call
from tests.helpers import ScriptedUI

TOKEN = "test-token-123"
PLAN = {
    "name": "Calc Tool",
    "summary": "A tiny calculator module.",
    "files": [
        {"path": "tests/test_calc.py", "purpose": "unit tests"},
        {"path": "calc.py", "purpose": "add function"},
    ],
}


def make_web(
    server: FakeOllama, tmp_path: Path, extra: list[str] | None = None, env: dict | None = None
) -> WebApp:
    argv = ["web", "--base-url", server.url, "--workspace", str(tmp_path / "ws"), *(extra or [])]
    args = build_parser().parse_args(argv)
    environment = {"AI_NEO_HOME": str(tmp_path / "home"), **(env or {})}
    return WebApp(load_app(args, environment, ScriptedUI()))


class WebClient:
    """A real HTTP client talking to a real `make_server` on a free loopback port."""

    def __init__(self, web: WebApp, token: str = TOKEN) -> None:
        self.web = web
        self.token = token
        self.server = make_server(web, token, "127.0.0.1", 0)
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.port = self.server.server_address[1]
        self.http = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}", timeout=30.0, trust_env=False
        )

    def close(self) -> None:
        self.http.close()
        self.web.stop()
        deadline = time.monotonic() + 5
        while self.web.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        self.server.shutdown()
        self.server.server_close()
        self.web.close()

    def headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return {"X-AI-Neo-Token": self.token, **(extra or {})}

    def get(self, path: str, headers: dict[str, str] | None = None) -> httpx.Response:
        return self.http.get(path, headers=self.headers(headers))

    def post(
        self, path: str, data: Any = None, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        return self.http.post(
            path, json={} if data is None else data, headers=self.headers(headers)
        )

    def state(self) -> dict[str, Any]:
        response = self.get("/api/state")
        assert response.status_code == 200
        return response.json()

    def settings(self, **patch: Any) -> httpx.Response:
        return self.post("/api/settings", patch)

    def chat(
        self,
        prompt: str,
        answers: dict[str, str] | Callable[[dict], str] | None = None,
        into: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Stream /api/chat to the end; `answers` replies to questions ('prompt' events).

        Events are appended to `into` as they arrive, so another thread can watch a live reply.
        """
        events: list[dict[str, Any]] = [] if into is None else into
        with self.http.stream(
            "POST", "/api/chat", json={"prompt": prompt}, headers=self.headers()
        ) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                events.append(event)
                if event["type"] == "prompt" and answers is not None:
                    value = answers(event) if callable(answers) else answers[event["kind"]]
                    assert self.post("/api/respond", {"id": event["id"], "value": value}).json()[
                        "ok"
                    ]
        return events


@pytest.fixture
def make_client(fake_ollama: FakeOllama, tmp_path: Path) -> Iterator[Callable[..., WebClient]]:
    clients: list[WebClient] = []

    def factory(extra: list[str] | None = None, env: dict | None = None) -> WebClient:
        client = WebClient(make_web(fake_ollama, tmp_path, extra, env))
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


@pytest.fixture
def client(make_client: Callable[..., WebClient]) -> WebClient:
    return make_client()


def types(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


def of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == kind]


def done(events: list[dict[str, Any]]) -> dict[str, Any]:
    assert events[-1]["type"] == "done", events[-3:]
    return events[-1]


def wait_idle(web: WebApp, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while web.busy:
        assert time.monotonic() < deadline, "the worker never finished"
        time.sleep(0.02)


# -- security ------------------------------------------------------------------------------------


def test_page_needs_the_token_and_gets_a_strict_policy(client: WebClient) -> None:
    assert client.http.get("/").status_code == 403
    assert client.http.get("/", params={"token": "wrong"}).status_code == 403
    page = client.http.get("/", params={"token": TOKEN})
    assert page.status_code == 200
    assert TOKEN in page.text and "__TOKEN__" not in page.text
    assert "__NONCE__" not in page.text and "__VERSION__" not in page.text
    policy = page.headers["content-security-policy"]
    nonce = policy.split("script-src 'nonce-")[1].split("'")[0]
    assert f'<script nonce="{nonce}">' in page.text and f'<style nonce="{nonce}">' in page.text
    assert "default-src 'none'" in policy and "connect-src 'self'" in policy
    assert "'unsafe-inline'" not in policy and "unsafe-eval" not in policy
    assert page.headers["referrer-policy"] == "no-referrer"
    assert page.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in page.headers


def test_each_page_load_gets_a_fresh_nonce(client: WebClient) -> None:
    first = client.http.get("/", params={"token": TOKEN}).headers["content-security-policy"]
    second = client.http.get("/", params={"token": TOKEN}).headers["content-security-policy"]
    assert first != second


def test_api_needs_the_token_header(client: WebClient) -> None:
    assert client.http.get("/api/state").status_code == 403
    assert client.http.get("/api/state", headers={"X-AI-Neo-Token": "nope"}).status_code == 403
    assert client.http.post("/api/stop", json={}).status_code == 403
    assert client.get("/api/state").status_code == 200


def test_a_rebound_host_name_is_refused(client: WebClient) -> None:
    """DNS rebinding: a web page on evil.example resolving to 127.0.0.1 sends that Host header."""
    evil = {"Host": "evil.example"}
    assert client.http.get("/", params={"token": TOKEN}, headers=evil).status_code == 403
    assert client.get("/api/state", headers=evil).status_code == 403
    for name in ("localhost", "127.0.0.1"):
        ok = client.get("/api/state", headers={"Host": f"{name}:{client.port}"})
        assert ok.status_code == 200
    wrong_port = client.get("/api/state", headers={"Host": f"localhost:{client.port + 1}"})
    assert wrong_port.status_code == 403


def test_cross_origin_posts_are_refused(client: WebClient) -> None:
    refused = client.post("/api/stop", headers={"Origin": "http://evil.example"})
    assert refused.status_code == 403
    same = client.post("/api/stop", headers={"Origin": f"http://127.0.0.1:{client.port}"})
    assert same.status_code == 200


def test_posts_must_be_json(client: WebClient) -> None:
    response = client.http.post(
        "/api/stop",
        content=b"{}",
        headers={"X-AI-Neo-Token": TOKEN, "Content-Type": "text/plain"},
    )
    assert response.status_code == 415


def test_bad_bodies_are_rejected(client: WebClient, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = client.http.post(
        "/api/settings",
        content=b"not json",
        headers={"X-AI-Neo-Token": TOKEN, "Content-Type": "application/json"},
    )
    assert bad.status_code == 400
    assert client.post("/api/settings", [1, 2]).status_code == 400  # must be an object
    monkeypatch.setattr(server_module, "MAX_BODY_BYTES", 50)
    big = client.post("/api/settings", {"system": "x" * 500})
    assert big.status_code == 413


def test_options_requests_get_no_cors_headers(client: WebClient) -> None:
    response = client.http.options("/api/chat", headers={"Origin": "http://evil.example"})
    assert response.status_code == 405
    assert not any(k.lower().startswith("access-control") for k in response.headers)


def test_unknown_paths_and_methods(client: WebClient) -> None:
    assert client.get("/api/nothing").status_code == 404
    assert client.http.get("/nothing").status_code == 404
    assert client.http.get("/favicon.ico").status_code == 204
    assert client.get("/api/chat").status_code == 404  # chat is POST only


def test_errors_do_not_leak_internals(client: WebClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dict:
        raise RuntimeError("secret internal detail /home/someone")

    monkeypatch.setattr(client.web, "state", boom)
    response = client.get("/api/state")
    assert response.status_code == 500
    assert "secret" not in response.text and "/home" not in response.text


def test_only_localhost_hosts_are_accepted(
    fake_ollama: FakeOllama, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    web = make_web(fake_ollama, tmp_path)
    with pytest.raises(ValueError):
        make_server(web, TOKEN, "0.0.0.0", 0)
    web.close()
    argv = ["web", "--host", "0.0.0.0", "--no-browser", "--plain"]
    assert main([*argv, "--base-url", fake_ollama.url]) == EXIT_USAGE
    assert "only accepts connections from this computer" in capsys.readouterr().out


def test_the_page_has_no_external_resources_or_inline_styles() -> None:
    page = load_template()
    assert "__TOKEN__" in page and "__NONCE__" in page
    lowered = page.lower()
    assert "<script src" not in lowered and "<link" not in lowered and "@import" not in lowered
    assert 'style="' not in lowered  # inline style attributes would be blocked by the policy
    assert "innerhtml" in lowered  # sanity: the markdown renderer is present


# -- state and settings --------------------------------------------------------------------------


def test_state_describes_modes_models_and_the_governor(client: WebClient) -> None:
    state = client.state()
    assert [m["name"] for m in state["modes"]] == ["reason", "compute", "code", "write", "build"]
    assert state["ollama"] == {"ok": True, "model_installed": True}
    assert state["model"] == "qwen3:30b" and state["models"] == ["qwen3:30b"]
    assert state["settings"]["mode"] == "reason" and state["settings"]["think"] == "auto"
    assert state["settings"]["effective_think"] is True
    assert state["settings"]["tools"] is True and state["settings"]["auto_approve"] is False
    assert state["governor"]["reason"] and state["governor"]["enabled"] is False
    assert state["session"]["num_ctx"] == 8192 and state["busy"] is False
    assert state["messages"] == []


def test_state_reports_an_unreachable_ollama(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.stop()
    ollama = client.state()["ollama"]
    assert ollama["ok"] is False and "Cannot reach Ollama" in ollama["error"]
    fake_ollama.start = lambda: (
        None
    )  # the fixture's stop() runs again at teardown; keep it harmless


def test_settings_change_mode_thinking_tools_and_verify(client: WebClient) -> None:
    response = client.settings(mode="code", think="off", tools=False, verify=False)
    assert response.status_code == 200 and response.json() == {"warnings": []}
    settings = client.state()["settings"]
    assert settings["mode"] == "code" and settings["think"] == "off"
    assert settings["effective_think"] is False
    assert settings["tools"] is False and settings["verify"] is False
    client.settings(think="auto")
    assert client.state()["settings"]["effective_think"] is True  # code mode thinks by default


@pytest.mark.parametrize(
    "patch",
    [
        {"mode": "nope"},
        {"think": "maybe"},
        {"tools": "yes"},
        {"verify": 1},
        {"auto_approve": None},
        {"system": 5},
        {"system": "x" * 20_001},
        {"model": ""},
    ],
)
def test_bad_settings_are_rejected_and_change_nothing(client: WebClient, patch: dict) -> None:
    before = client.state()["settings"]
    response = client.post("/api/settings", {"mode": "write", **patch})
    assert response.status_code == 400 and response.json()["error"]
    assert client.state()["settings"] == before  # not even the valid mode was applied


def test_switching_to_an_uninstalled_model_is_refused(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    response = client.settings(model="llama3:8b")
    assert response.status_code == 400 and "ollama pull llama3:8b" in response.json()["error"]
    assert client.state()["model"] == "qwen3:30b"
    fake_ollama.models.append("llama3:8b")
    assert client.settings(model="llama3:8b").status_code == 200
    assert client.state()["model"] == "llama3:8b"
    fake_ollama.queue(text("hi"))
    client.chat("hello")
    assert fake_ollama.chat_payloads()[-1]["model"] == "llama3:8b"


def test_the_governor_stays_off_when_it_cannot_run(client: WebClient) -> None:
    response = client.settings(govern=True)
    assert response.status_code == 200
    warning = response.json()["warnings"][0]
    assert "cannot be turned on" in warning and "claude_model" in warning
    assert client.state()["settings"]["govern"] is False
    assert client.settings(govern=False).json() == {"warnings": []}


def test_yes_flag_starts_with_auto_approve_but_never_for_the_governor(
    make_client: Callable[..., WebClient],
) -> None:
    client = make_client(extra=["--yes"])
    assert client.state()["settings"]["auto_approve"] is True
    assert client.web.stack.governor.assume_yes is False  # consent to send text is always asked


def test_system_prompt_is_applied_and_reset(client: WebClient, fake_ollama: FakeOllama) -> None:
    assert client.settings(system="  Always answer in French.  ").status_code == 200
    assert client.state()["settings"]["system"] == "Always answer in French."
    fake_ollama.queue(text("Bonjour"), text("Hello"))
    client.chat("hi")
    first_system = fake_ollama.chat_payloads()[-1]["messages"][0]
    assert (
        first_system["role"] == "system" and "Always answer in French." in first_system["content"]
    )
    client.settings(system="")
    client.post("/api/new")
    client.chat("hi")
    assert "French" not in fake_ollama.chat_payloads()[-1]["messages"][0]["content"]


# -- chatting ------------------------------------------------------------------------------------


def test_a_plain_chat_streams_events_and_finishes(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    fake_ollama.queue(text("Hello ", "there.", thinking=["Let me ", "think."]))
    events = client.chat("Say hello")
    kinds = types(events)
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "segment" in kinds and "segment_end" in kinds
    assert "".join(e["text"] for e in of(events, "thinking")) == "Let me think."
    assert "".join(e["text"] for e in of(events, "answer")) == "Hello there."
    last = done(events)
    assert last["answer"] == "Hello there." and last["interrupted"] is False
    assert last["error"] is None and last["mode"] == "reason"
    assert "tok/s" in last["stats"]["text"] and last["stats"]["eval_count"] == 20
    assert last["context"]["num_ctx"] == 8192 and 0 <= last["context"]["percent"] <= 100
    assert client.state()["messages"] == [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello there."},
    ]


def test_the_conversation_continues_across_requests(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    fake_ollama.queue(text("Nice to meet you."), text("You said hi."))
    client.chat("hi")
    client.chat("what did I say?")
    roles = [m["role"] for m in fake_ollama.chat_payloads()[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_empty_or_oversized_prompts_are_refused(
    client: WebClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post("/api/chat", {"prompt": "   "}).status_code == 400
    assert client.post("/api/chat", {"prompt": 7}).status_code == 400
    assert client.post("/api/chat", {}).status_code == 400
    monkeypatch.setattr(server_module, "MAX_PROMPT_CHARS", 10)
    assert client.post("/api/chat", {"prompt": "x" * 11}).status_code == 400
    assert client.web.busy is False  # a refused prompt never leaves the server stuck


@pytest.mark.parametrize(
    ("think", "mode", "expected"),
    [
        ("off", "reason", False),
        ("on", "write", True),
        ("auto", "reason", True),
        ("auto", "write", False),
    ],
)
def test_the_thinking_setting_reaches_ollama(
    client: WebClient, fake_ollama: FakeOllama, think: str, mode: str, expected: bool
) -> None:
    client.settings(think=think, mode=mode)
    fake_ollama.queue(text("ok"))
    client.chat("hello")
    assert fake_ollama.chat_payloads()[-1]["think"] is expected


def test_tools_can_be_switched_off(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("a"), text("b"))
    client.chat("one")
    assert "calculator" in json.dumps(fake_ollama.chat_payloads()[-1]["tools"])
    client.settings(tools=False)
    client.chat("two")
    assert "tools" not in fake_ollama.chat_payloads()[-1]


def test_a_tool_call_shows_up_as_events(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(tool_call("calculator", {"expression": "12*12"}), text("It is 144."))
    events = client.chat("What is 12*12?")
    call, result = of(events, "tool_call")[0], of(events, "tool_result")[0]
    assert call["name"] == "calculator" and "12*12" in call["args"]
    assert result["ok"] is True and "144" in result["output"]
    assert done(events)["answer"] == "It is 144."
    assert types(events).count("segment") == 2  # the tool call, then the final answer


def test_running_code_asks_first_and_runs_when_approved(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    client.settings(mode="compute")
    fake_ollama.queue(tool_call("run_python", {"code": "print(6*7)"}), text("The result is 42."))
    events = client.chat("Compute 6*7 with code", answers={"confirm": "y"})
    prompt = of(events, "prompt")[0]
    assert prompt["kind"] == "confirm" and prompt["text"] == "Allow this?"
    assert [o["label"] for o in prompt["options"]] == ["Approve", "Deny"]
    preview = of(events, "preview")[0]
    assert "run_python" in preview["title"] and "print(6*7)" in preview["body"]
    assert types(events).index("preview") < types(events).index("prompt")
    result = of(events, "tool_result")[0]
    assert result["ok"] is True and "42" in result["output"]
    assert done(events)["answer"] == "The result is 42."


def test_denying_code_means_it_never_runs(
    client: WebClient, fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    client.settings(mode="compute")
    marker = tmp_path / "ran.txt"
    code = f"open({str(marker)!r}, 'w').write('x')"
    fake_ollama.queue(tool_call("run_python", {"code": code}), text("Understood, not running it."))
    events = client.chat("Write a file", answers={"confirm": "n"})
    result = of(events, "tool_result")[0]
    assert result["ok"] is False and "declined" in result["output"]
    assert not marker.exists()


def test_auto_approve_skips_the_question(client: WebClient, fake_ollama: FakeOllama) -> None:
    client.settings(mode="compute", auto_approve=True)
    fake_ollama.queue(tool_call("run_python", {"code": "print(1+1)"}), text("Two."))
    events = client.chat("1+1 with code")
    assert of(events, "prompt") == []
    assert of(events, "tool_result")[0]["ok"] is True
    client.settings(auto_approve=False)
    fake_ollama.queue(tool_call("run_python", {"code": "print(2+2)"}), text("Four."))
    asked = client.chat("2+2 with code", answers={"confirm": "n"})
    assert len(of(asked, "prompt")) == 1


def test_code_mode_runs_and_repairs_in_segments(client: WebClient, fake_ollama: FakeOllama) -> None:
    client.settings(mode="code")
    fake_ollama.queue(
        code_reply("def add(a, b):\n    return a - b\n\nassert add(2, 3) == 5\n"),
        code_reply("def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n"),
    )
    events = client.chat("Write add(a, b)")
    last = done(events)
    assert last["verify"]["passed"] is True
    assert "after 1 repair attempt" in last["verify"]["summary"]
    assert types(events).count("segment") == 2  # the first draft, then the repair
    assert "return a + b" in last["answer"]


def test_run_and_repair_can_be_turned_off(client: WebClient, fake_ollama: FakeOllama) -> None:
    client.settings(mode="code", verify=False)
    fake_ollama.queue(code_reply("print('x')"))
    last = done(client.chat("Print x"))
    assert last["verify"] is None


def test_build_mode_asks_you_to_approve_the_plan(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    client.settings(mode="build")
    plan = "Here is the plan:\n```json\n" + json.dumps(PLAN) + "\n```"
    fake_ollama.queue(text(plan))
    events = client.chat("Build a calculator", answers={"choose": "n"})
    prompt = of(events, "prompt")[0]
    assert prompt["kind"] == "choose" and prompt["default"] == "y"
    assert [(o["value"], o["label"]) for o in prompt["options"]] == [
        ("y", "Yes"),
        ("r", "Revise"),
        ("n", "No"),
    ]
    assert any(e["type"] == "preview" and e["title"] == "Build plan" for e in events)
    last = done(events)
    assert "cancelled" in last["answer"].lower()
    assert last["build"]["files"] == 0 and last["build"]["passed"] is False


def test_the_governor_asks_before_sending_anything(
    client: WebClient, fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    web = client.web
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        reply = json.dumps({"score": 9, "issues": [], "suggested_corrections": []})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )

    gov = web.stack.governor
    gov.cfg = type(gov.cfg)(**{**gov.cfg.__dict__, "claude_model": "some-claude-model"})
    gov.env = {"ANTHROPIC_API_KEY": "sk-not-a-real-key"}
    gov._client_factory = lambda: SimpleNamespace(messages=SimpleNamespace(create=create))
    assert client.settings(govern=True).json() == {"warnings": []}

    fake_ollama.queue(text("143"), text("144"))
    declined = client.chat("What is 12*12?", answers={"confirm": "n"})
    prompt = of(declined, "prompt")[0]
    assert prompt["text"] == "Send it?" and prompt["default"] == "n"
    assert "exactly this to Anthropic" in of(declined, "preview")[-1]["title"]
    assert sent == []  # nothing left the machine
    assert client.state()["settings"]["govern"] is False  # declining turns the governor off

    client.settings(govern=True)
    approved = client.chat("What is 12*12?", answers={"confirm": "y"})
    assert len(sent) == 1 and "What is 12*12?" in sent[0]["messages"][0]["content"]
    assert "last score 9/10" in done(approved)["governor"]


def test_ollama_being_down_is_reported_not_hung(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.stop()
    last = done(client.chat("hello"))
    assert "Cannot reach Ollama" in last["error"]
    wait_idle(client.web)
    fake_ollama.start = lambda: None  # the fixture's stop() runs again at teardown


def test_a_missing_model_tells_you_how_to_install_it(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    fake_ollama.models.clear()
    last = done(client.chat("hello"))
    assert "ollama pull qwen3:30b" in last["error"]


def test_a_mid_stream_error_keeps_the_partial_text(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    fake_ollama.queue(ChatReply(content=["Partial answer "], stream_error="model crashed"))
    last = done(client.chat("hello"))
    assert "model crashed" in last["error"]
    assert "Partial answer" in last["answer"]


# -- stopping and busy ----------------------------------------------------------------------------


def slow_reply(words: int = 60, delay: float = 0.05) -> ChatReply:
    return ChatReply(content=[f"w{i} " for i in range(words)], delay=delay)


def stream_in_background(client: WebClient, prompt: str) -> tuple[list, threading.Thread]:
    events: list[dict[str, Any]] = []

    def run() -> None:
        client.chat(prompt, into=events)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return events, thread


def wait_for(events: list, kind: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not any(e["type"] == kind for e in list(events)):
        assert time.monotonic() < deadline, f"never saw a {kind!r} event"
        time.sleep(0.02)


def test_stop_ends_the_reply_and_keeps_the_partial_text(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    fake_ollama.queue(slow_reply(), text("A fresh reply."))
    events, thread = stream_in_background(client, "count slowly")
    wait_for(events, "answer")
    assert client.post("/api/stop").json() == {"ok": True}
    thread.join(timeout=15)
    assert not thread.is_alive()
    last = done(events)
    assert last["interrupted"] is True
    assert last["answer"].startswith("w0 ") and len(last["answer"]) < 60 * 4
    wait_idle(client.web)
    # the server is usable again and the partial answer is part of the conversation
    again = done(client.chat("next"))
    assert again["answer"] == "A fresh reply." and again["interrupted"] is False
    roles = [m["role"] for m in client.state()["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_stop_while_a_question_is_waiting_cancels_it(
    client: WebClient, fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    client.settings(mode="compute")
    marker = tmp_path / "must-not-exist.txt"
    code = f"open({str(marker)!r}, 'w').write('x')"
    fake_ollama.queue(tool_call("run_python", {"code": code}), text("never reached"))
    events, thread = stream_in_background(client, "write a file")
    wait_for(events, "prompt")
    client.post("/api/stop")
    thread.join(timeout=15)
    assert done(events)["interrupted"] is True
    assert not marker.exists()
    wait_idle(client.web)


def test_a_second_chat_while_busy_is_refused(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(slow_reply())
    events, thread = stream_in_background(client, "slow one")
    wait_for(events, "answer")
    assert client.state()["busy"] is True
    refused = client.post("/api/chat", {"prompt": "second"})
    assert refused.status_code == 409 and "already running" in refused.json()["error"]
    assert client.settings(mode="write").status_code == 409
    assert client.post("/api/new").status_code == 409
    assert client.post("/api/load", {"name": "x"}).status_code == 409
    client.post("/api/stop")
    thread.join(timeout=15)
    wait_idle(client.web)
    assert client.settings(mode="write").status_code == 200


def test_closing_the_tab_stops_the_model(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(slow_reply(words=400, delay=0.03))
    with client.http.stream(
        "POST", "/api/chat", json={"prompt": "long"}, headers=client.headers()
    ) as response:
        for line in response.iter_lines():
            if '"answer"' in line:
                break
        # leaving the with-block closes the connection, like closing the browser tab
    wait_idle(client.web, timeout=15)
    assert client.web.busy is False


def test_respond_to_an_unknown_question_is_harmless(client: WebClient) -> None:
    assert client.post("/api/respond", {"id": "p999", "value": "y"}).json() == {"ok": False}
    assert client.post("/api/respond", {"id": 1, "value": "y"}).status_code == 400


# -- sessions ------------------------------------------------------------------------------------


def test_save_new_load_and_export(client: WebClient, fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("Paris."))
    client.chat("Capital of France?")
    saved = client.post("/api/save", {"name": "France trip!"}).json()
    assert saved == {"name": "France_trip"}  # same name rules as the terminal chat
    assert client.get("/api/sessions").json()["sessions"] == [
        {"name": "France_trip", "mode": "reason", "messages": 2}
    ]
    assert client.post("/api/new").json() == {"ok": True}
    assert client.state()["messages"] == []
    assert client.post("/api/load", {"name": "France_trip"}).json() == {"ok": True}
    assert [m["content"] for m in client.state()["messages"]] == ["Capital of France?", "Paris."]
    export = client.get("/api/export")
    assert export.status_code == 200 and export.headers["content-type"].startswith("text/markdown")
    assert 'filename="France_trip.md"' in export.headers["content-disposition"]
    assert (
        "## You" in export.text and "Capital of France?" in export.text and "Paris." in export.text
    )


def test_loading_a_missing_chat_or_a_bad_name(client: WebClient) -> None:
    assert client.post("/api/load", {"name": "ghost"}).status_code == 404
    assert client.post("/api/load", {"name": ""}).status_code == 400
    assert client.post("/api/save", {"name": 5}).status_code == 400


def test_a_chat_saved_by_the_terminal_loads_in_the_browser(
    client: WebClient, tmp_path: Path
) -> None:
    from ai_neo.sessions import Session, SessionStore
    from ai_neo.types import Message

    store = SessionStore(tmp_path / "home" / "sessions")
    session = Session(mode="write", model="qwen3:30b")
    session.messages += [Message("user", "Draft a line"), Message("assistant", "Here is a line.")]
    store.save(session, "from-terminal")
    assert client.post("/api/load", {"name": "from-terminal"}).status_code == 200
    state = client.state()
    assert state["settings"]["mode"] == "write"
    assert state["messages"][-1] == {"role": "assistant", "content": "Here is a line."}


def test_tool_results_appear_in_the_history_view(
    client: WebClient, fake_ollama: FakeOllama
) -> None:
    fake_ollama.queue(tool_call("calculator", {"expression": "2+2"}), text("Four."))
    client.chat("2+2?")
    roles = [m["role"] for m in client.state()["messages"]]
    assert roles == ["user", "tool", "assistant"]


# -- the UI adapter ------------------------------------------------------------------------------


def collect(ui: WebUI) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ui.attach(events.append)
    return events


def test_questions_block_until_the_browser_answers() -> None:
    ui = WebUI()
    events = collect(ui)
    results: dict[str, Any] = {}

    def ask() -> None:
        results["confirm"] = ui.confirm("Allow this?", default=False)
        results["choose"] = ui.choose("Approve? [Y]es / [R]evise / [N]o:", "yrn", default="y")
        results["ask"] = ui.ask("What should change?")

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    for expected, answer in (("confirm", "y"), ("choose", "r"), ("ask", "  make it shorter ")):
        deadline = time.monotonic() + 5
        while not any(e["type"] == "prompt" and e["kind"] == expected for e in list(events)):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        prompt = [e for e in events if e["type"] == "prompt" and e["kind"] == expected][0]
        assert ui.respond(prompt["id"], answer) is True
        assert ui.respond(prompt["id"], answer) is False  # a question is answered once
    thread.join(timeout=5)
    assert results == {"confirm": True, "choose": "r", "ask": "make it shorter"}
    assert ui.pending_ids() == []


def test_answers_are_validated() -> None:
    ui = WebUI()
    events = collect(ui)

    def reply(value: str) -> None:
        deadline = time.monotonic() + 5
        while not events:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        ui.respond(events.pop(0)["id"], value)

    for value, expected in (("zzz", "y"), ("N", "n"), ("", "y"), ("nope", "n")):
        threading.Thread(target=reply, args=(value,), daemon=True).start()
        assert ui.choose("Approve? [Y]es / [N]o:", "yn", default="y") == expected
    threading.Thread(target=reply, args=("maybe",), daemon=True).start()
    assert ui.confirm("Allow this?", default=True) is False  # anything but yes is a no
    threading.Thread(target=reply, args=("   ",), daemon=True).start()
    assert ui.ask("Name?", default="neo") == "neo"


def test_stop_releases_a_waiting_question_with_stoprequested() -> None:
    ui = WebUI()
    events = collect(ui)
    outcome: list[BaseException] = []

    def ask() -> None:
        try:
            ui.confirm("Allow this?", default=False)
        except KeyboardInterrupt as exc:  # StopRequested is a KeyboardInterrupt
            outcome.append(exc)

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    while not events:
        time.sleep(0.01)
    ui.request_stop()
    thread.join(timeout=5)
    assert len(outcome) == 1 and isinstance(outcome[0], StopRequested)
    assert ui.pending_ids() == []


def test_stop_interrupts_streams_and_tool_calls_but_attach_resets_it() -> None:
    ui = WebUI()
    events = collect(ui)
    stream = ui.stream()
    stream.event(StreamEvent("answer", "hello"))
    ui.request_stop()
    with pytest.raises(StopRequested):
        stream.event(StreamEvent("answer", "more"))
    with pytest.raises(StopRequested):
        ui.tool_call("calculator", {"expression": "1"})
    assert [e["type"] for e in events] == ["segment", "answer"]
    ui.detach()
    ui.attach(events.append)
    ui.stream().event(StreamEvent("thinking", "fine again"))  # a new task starts clean


def test_nothing_is_sent_without_a_listener_and_questions_use_defaults() -> None:
    ui = WebUI()
    ui.say("hello")
    ui.tag("OK", "nobody listens")
    assert ui.confirm("Allow this?", default=True) is True
    assert ui.choose("Pick [A]ll / [N]one:", "an", default="n") == "n"
    assert ui.ask("Name?", default="neo") == "neo"
    with pytest.raises(EOFError):
        ui.read_line("> ")


def test_output_events_are_plain_json_with_bounded_size() -> None:
    ui = WebUI()
    events = collect(ui)
    ui.say("note", dim=True)
    ui.heading("Title")
    ui.tag("WARN", "careful")
    ui.table(["a", "b"], [["1", "2"]], title="T")
    ui.tool_call("run_python", {"code": "x" * 5000})
    ui.tool_result(False, "y" * 50_000)
    ui.preview("Big", "z" * 50_000)
    ui.stats_line(GenStats(eval_count=10, eval_duration_ns=1_000_000_000), "context 5%")
    progress = ui.progress("Download")
    progress.update(50, 100)
    progress.update(50, 100)  # unchanged: not repeated
    progress.update(100, 100)
    progress.close()
    assert [e["type"] for e in events] == [
        "say", "heading", "tag", "table", "tool_call", "tool_result", "preview", "stats",
        "progress", "progress",
    ]  # fmt: skip
    json.dumps(events)  # everything must be serialisable
    assert len(events[4]["args"]) <= 400
    assert len(events[5]["output"]) < 6200 and "more characters" in events[5]["output"]
    assert len(events[6]["body"]) < 12_200
    assert "10.0 tok/s" in events[7]["text"] and "context 5%" in events[7]["text"]
    assert events[8]["percent"] == 50 and events[9]["percent"] == 100


def test_option_labels_come_from_the_prompt_text() -> None:
    labels = options_from_prompt("Approve this plan? [Y]es / [R]evise / [N]o:", "yrn")
    assert labels == [
        {"value": "y", "label": "Yes"},
        {"value": "r", "label": "Revise"},
        {"value": "n", "label": "No"},
    ]
    assert options_from_prompt("Pick one:", "ab") == [
        {"value": "a", "label": "A"},
        {"value": "b", "label": "B"},
    ]


def test_clip_marks_what_it_cut() -> None:
    assert clip("short", 10) == "short"
    cut = clip("x" * 30, 10)
    assert cut.startswith("x" * 10) and "20 more characters" in cut


# -- command line --------------------------------------------------------------------------------


def test_the_web_command_and_its_flags_parse() -> None:
    args = build_parser().parse_args(["web", "--port", "9000", "--no-browser", "--yes"])
    assert args.command == "web" and args.port == 9000 and args.no_browser is True
    assert args.host == "127.0.0.1" and args.yes is True
    defaults = build_parser().parse_args(["web"])
    assert defaults.port == 8765 and defaults.no_browser is False


def test_make_server_falls_back_to_a_free_port(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    web = make_web(fake_ollama, tmp_path)
    first = make_server(web, TOKEN, "127.0.0.1", 0)
    taken = first.server_address[1]
    try:
        second = make_server(web, TOKEN, "127.0.0.1", taken)
        try:
            assert second.server_address[1] != taken  # tried the next port instead of failing
        finally:
            second.server_close()
    finally:
        first.server_close()
        web.close()


def test_serve_prints_the_link_and_stops_on_ctrl_c(
    fake_ollama: FakeOllama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "web",
        "--base-url",
        fake_ollama.url,
        "--workspace",
        str(tmp_path / "ws"),
        "--no-browser",
    ]
    args = build_parser().parse_args(argv)
    ui = ScriptedUI()
    app = load_app(args, {"AI_NEO_HOME": str(tmp_path / "home")}, ui)

    def interrupt(self: Any, poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module.ThreadingHTTPServer, "serve_forever", interrupt)
    assert serve(app, host="127.0.0.1", port=0, open_browser=False) == 0
    output = ui.text
    assert "http://127.0.0.1:" in output and "?token=" in output
    assert "Only this computer can connect" in output
