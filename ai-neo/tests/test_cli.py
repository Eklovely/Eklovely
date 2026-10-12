from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_neo.cli import (
    HELP_ROWS,
    Repl,
    build_parser,
    build_stack,
    load_app,
    main,
    offer_first_run_setup,
    overrides_from,
)
from ai_neo.types import StreamEvent
from tests.fake_ollama import ChatReply, FakeOllama, code_reply, text, tool_call
from tests.helpers import FakeProbe, ScriptedUI

PLAIN = ["--plain"]


def make_repl(
    server: FakeOllama, tmp_path: Path, answers: list | None = None, extra: list[str] | None = None
):
    argv = ["--base-url", server.url, "--workspace", str(tmp_path / "ws"), *(extra or [])]
    args = build_parser().parse_args(argv)
    ui = ScriptedUI(answers or [])
    env = {"AI_NEO_HOME": str(tmp_path / "home")}
    app = load_app(args, env, ui)
    stack = build_stack(app)
    return Repl(app, stack), ui, app


# -- argument parsing ----------------------------------------------------------------------------


def test_flags_work_before_and_after_the_subcommand() -> None:
    parser = build_parser()
    before = parser.parse_args(["--model", "a:1", "ask", "hi"])
    after = parser.parse_args(["ask", "hi", "--model", "a:1", "--mode", "code", "--no-think"])
    assert overrides_from(before) == {"ollama.model": "a:1"}
    assert overrides_from(after) == {
        "ollama.model": "a:1",
        "chat.mode": "code",
        "chat.think": "off",
    }
    assert overrides_from(parser.parse_args([])) == {}  # untouched flags never override the config
    assert parser.parse_args(["-y"]).yes is True
    assert parser.parse_args(["ask", "x", "--git"]).git is True


def test_flags_beat_env_and_config(tmp_path: Path, fake_ollama: FakeOllama) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[ollama]\nmodel = "from-file"\nnum_ctx = 4096\n', encoding="utf-8"
    )
    args = build_parser().parse_args(["--model", "from-flag"])
    app = load_app(args, {"AI_NEO_HOME": str(home), "AI_NEO_MODEL": "from-env"}, ScriptedUI())
    assert app.cfg.ollama.model == "from-flag" and app.cfg.ollama.num_ctx == 4096


# -- REPL commands -------------------------------------------------------------------------------


def test_help_lists_every_command_from_the_spec(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    repl.handle_command("/help")
    for name in (
        "/help",
        "/new",
        "/think",
        "/mode",
        "/model",
        "/system",
        "/tools",
        "/govern",
        "/save",
        "/load",
        "/export",
        "/stats",
        "/config",
        "/setup",
        "/doctor",
        "/quit",
    ):
        assert name in ui.text, name
    assert len(HELP_ROWS) == 16
    repl.handle_command("/nonsense")
    assert "Unknown command /nonsense" in ui.text


def test_think_and_mode_commands(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    assert repl.effective_think() is True and repl.prompt_text() == "[reason+think] > "
    repl.handle_command("/think off")
    assert repl.effective_think() is False and repl.prompt_text() == "[reason] > "
    repl.handle_command("/think on")
    assert repl.think is True
    repl.handle_command("/think auto")
    assert repl.think is None
    repl.handle_command("/think maybe")
    assert "Usage: /think on|off|auto" in ui.text
    repl.handle_command("/mode write")
    assert repl.session.mode == "write" and repl.effective_think() is False
    repl.handle_command("/mode poetry")
    assert (
        "Usage: /mode reason|compute|code|write|build" in ui.text and repl.session.mode == "write"
    )
    for mode in ("reason", "compute", "code", "write", "build"):
        repl.handle_command(f"/mode {mode}")
        assert repl.session.mode == mode


def test_a_chat_turn_streams_and_reports_speed(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        ChatReply(
            thinking=["Let me think."],
            content=["Paris."],
            eval_count=40,
            eval_duration=4_000_000_000,
        )
    )
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    outcome = repl.ask_model("Capital of France?")
    assert outcome is not None and outcome.answer == "Paris."
    assert "(thinking) Let me think." in ui.text and "Paris." in ui.text
    assert "10.0 tok/s" in ui.text and "40 tokens" in ui.text and "context" in ui.text
    payload = fake_ollama.chat_payloads()[0]
    assert payload["options"]["num_ctx"] == 8192 and payload["keep_alive"] == "30m"
    assert payload["stream"] is True


def test_mode_changes_the_system_prompt_and_tools(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        text("a"), text("b"), text("c")
    )  # compute mode nudges once if no tool is used
    repl, _, _ = make_repl(fake_ollama, tmp_path)
    repl.handle_command("/mode write")
    repl.ask_model("write me something")
    repl.handle_command("/mode compute")
    repl.ask_model("compute")
    first, second = fake_ollama.chat_payloads()[:2]
    assert "Mode: writing" in first["messages"][0]["content"] and "tools" not in first
    assert "Mode: computation" in second["messages"][0]["content"]


def test_tools_toggle_and_system_prompt(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("a"), text("b"), text("c"))
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    repl.ask_model("one")
    repl.handle_command("/tools off")
    repl.ask_model("two")
    repl.handle_command("/system Answer in French.")
    repl.ask_model("three")
    payloads = fake_ollama.chat_payloads()
    assert "tools" in payloads[0] and "tools" not in payloads[1]
    assert payloads[2]["messages"][0]["content"] == "Answer in French."
    repl.handle_command("/system")
    repl.handle_command("/system reset")
    assert repl.session.system_override is None and "Answer in French." in ui.text
    repl.handle_command("/tools sideways")
    assert "Usage: /tools on|off" in ui.text


def test_new_save_load_export_use_the_user_data_folder(
    fake_ollama: FakeOllama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "somewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    fake_ollama.queue(text("first answer"), text("second answer"))
    repl, ui, app = make_repl(fake_ollama, tmp_path)
    repl.ask_model("first question")
    repl.handle_command("/save mychat")
    saved = app.paths.sessions / "mychat.json"
    assert saved.exists()
    repl.handle_command("/new")
    assert repl.session.messages == []
    repl.ask_model("second question")
    repl.handle_command("/load mychat")
    assert [m.content for m in repl.session.messages] == ["first question", "first answer"]
    repl.handle_command("/export")
    exported = app.paths.exports / "mychat.md"
    assert exported.exists()
    markdown = exported.read_text(encoding="utf-8")
    assert "## You" in markdown and "first question" in markdown and "first answer" in markdown
    repl.handle_command("/load")  # no name: newest
    repl.handle_command("/load ghost")
    assert "No saved session called 'ghost'" in ui.text
    assert list(cwd.iterdir()) == []  # nothing is ever written to the current directory


def test_model_command_checks_installation(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.models.append("qwen3:8b")
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    repl.handle_command("/model qwen3:8b")
    assert repl.stack.client.model == "qwen3:8b" and repl.session.model == "qwen3:8b"
    repl.handle_command("/model nonexistent:1b")
    assert "ollama pull nonexistent:1b" in ui.text and repl.stack.client.model == "qwen3:8b"
    repl.handle_command("/model")
    assert "Current model: qwen3:8b" in ui.text


def test_stats_and_config_commands(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("hello", eval_count=20, eval_duration=2_000_000_000))
    repl, ui, app = make_repl(fake_ollama, tmp_path)
    repl.ask_model("hi")
    repl.handle_command("/stats")
    for fragment in (
        "Session statistics",
        "tokens generated",
        "10.0 tok/s",
        "of 8192 tokens",
        "governor",
        "off (0 calls, 0 tokens)",
    ):
        assert fragment in ui.text, fragment
    repl.handle_command("/config")
    assert "[ollama]" in ui.text and f"Config file: {app.paths.config}" in ui.text


def test_govern_command_is_off_by_default_and_explains_refusals(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    assert not repl.stack.governor.enabled
    repl.handle_command("/govern on")
    assert "cannot be turned on" in ui.text and "AI_NEO_CLAUDE_MODEL" in ui.text
    assert not repl.stack.governor.enabled
    repl.handle_command("/govern")
    assert "Governor is off" in ui.text


def test_govern_flag_without_key_is_ignored_with_a_reason(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    repl, ui, app = make_repl(fake_ollama, tmp_path, extra=["--govern"])
    from ai_neo.cli import maybe_enable_governor

    maybe_enable_governor(app, repl.stack)
    assert "--govern was ignored" in ui.text and not repl.stack.governor.enabled


def test_main_loop_quit_eof_and_ctrl_c_at_the_prompt(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    repl, ui, _ = make_repl(fake_ollama, tmp_path, ["", KeyboardInterrupt(), "/quit"])
    assert repl.run() == 0
    assert "Ctrl+C stops a reply" in ui.text
    repl, ui, _ = make_repl(fake_ollama, tmp_path, [])  # EOF (Ctrl+D)
    assert repl.run() == 0


def test_multiline_input_with_triple_quotes(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("ok"))
    repl, ui, _ = make_repl(fake_ollama, tmp_path, ['"""def f():', "    return 1", '"""', "/quit"])
    repl.run()
    sent = fake_ollama.chat_payloads()[0]["messages"][-1]["content"]
    assert sent == "def f():\n    return 1"


def test_ctrl_c_during_a_reply_keeps_the_session_alive(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(text("one ", "two ", "three"), text("recovered"))
    repl, ui, _ = make_repl(fake_ollama, tmp_path, ["hello", "again", "/quit"])
    original = ui.stream
    state = {"first": True}

    def stream():  # type: ignore[no-untyped-def]
        renderer = original()
        interrupt = state["first"]
        state["first"] = False
        seen = []

        class Wrapper:
            def event(self, event: StreamEvent) -> None:
                renderer.event(event)
                seen.append(event)
                if interrupt and len(seen) == 2:
                    raise KeyboardInterrupt

            def close(self) -> None:
                renderer.close()

        return Wrapper()

    ui.stream = stream  # type: ignore[method-assign]
    assert repl.run() == 0  # did not exit
    assert "stopped by you" in ui.text and "recovered" in ui.text
    assert repl.session.messages[1].content == "one two"  # the partial text was kept


def test_code_mode_repairs_in_the_repl(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    bad = "def add(a, b):\n    return a - b\n\nif __name__ == '__main__':\n    assert add(1, 2) == 3\n"
    good = bad.replace("a - b", "a + b")
    fake_ollama.queue(code_reply(bad), code_reply(good))
    repl, ui, _ = make_repl(fake_ollama, tmp_path)
    repl.handle_command("/mode code")
    repl.handle_command("/tools off")
    repl.ask_model("write add")
    assert "PASSED after 1 repair attempt(s)" in ui.text


def test_first_run_offer(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    repl, ui, app = make_repl(fake_ollama, tmp_path, ["n"])
    offer_first_run_setup(app)
    assert "Start the guided setup now?" in ui.prompts[0] and "ai-neo setup" in ui.text
    # an existing config means no offer
    app.paths.root.mkdir(parents=True, exist_ok=True)
    app.paths.config.write_text("", encoding="utf-8")
    ui2 = ScriptedUI([])
    app.ui = ui2
    offer_first_run_setup(app)
    assert ui2.prompts == []


# -- main() --------------------------------------------------------------------------------------


@pytest.fixture
def cli_env(fake_ollama: FakeOllama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    monkeypatch.setenv("AI_NEO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AI_NEO_BASE_URL", fake_ollama.url)
    monkeypatch.setenv("AI_NEO_WORKSPACE", str(tmp_path / "ws"))
    return fake_ollama


def test_ask_prints_answer_and_stats(
    cli_env: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_env.queue(text("Four.", eval_count=10, eval_duration=1_000_000_000))
    assert main(["ask", "What is 2+2?", "--no-tools", "--plain"]) == 0
    out = capsys.readouterr().out
    assert "Four." in out and "10.0 tok/s" in out


def test_ask_mode_flag_and_stdin(cli_env: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    cli_env.queue(text("ok"))
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("piped question\n"))
    assert main(["ask", "--mode", "write", "--plain"]) == 0
    payload = cli_env.chat_payloads()[0]
    assert "Mode: writing" in payload["messages"][0]["content"]
    assert payload["messages"][-1]["content"] == "piped question"


def test_ask_code_with_tests_exit_codes(
    cli_env: FakeOllama, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tests = tmp_path / "t.py"
    tests.write_text(
        "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8"
    )
    cli_env.queue(code_reply("def add(a, b):\n    return a + b\n"))
    assert main(["ask", "write add", "--mode", "code", "--tests", str(tests), "--plain"]) == 0
    cli_env.queue(*[code_reply("def add(a, b):\n    return 0\n")] * 4)
    assert main(["ask", "write add", "--mode", "code", "--tests", str(tests), "--plain"]) == 1
    assert "FAILED after 3 repair attempt(s)" in capsys.readouterr().out


def test_ask_without_a_question_is_a_usage_error(
    cli_env: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tty:
        def isatty(self) -> bool:
            return True

        def read(self) -> str:
            return ""

    monkeypatch.setattr(sys, "stdin", Tty())
    assert main(["ask", "--plain"]) == 2


def test_ollama_down_gives_friendly_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AI_NEO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AI_NEO_BASE_URL", "http://127.0.0.1:9")
    assert main(["ask", "hi", "--plain"]) == 1
    out = capsys.readouterr().out
    assert "Cannot reach Ollama" in out and "ollama serve" in out and "ai-neo doctor" in out
    assert "Traceback" not in out


def test_missing_model_prints_the_pull_command(
    cli_env: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_env.models.clear()
    assert main(["ask", "hi", "--plain"]) == 1
    assert "ollama pull qwen3:30b" in capsys.readouterr().out


def test_bad_config_values_are_listed_and_exit_2(
    cli_env: FakeOllama, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[ollama]\nnum_ctx = "huge"\n[tools]\nmax_steps = 0\n', encoding="utf-8"
    )
    assert main(["ask", "hi", "--plain"]) == 2
    out = capsys.readouterr().out
    assert "ollama.num_ctx" in out and "tools.max_steps" in out and "ai-neo setup" in out


def test_config_command_prints_settings(
    cli_env: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "--plain"]) == 0
    out = capsys.readouterr().out
    assert "[ollama]" in out and f'base_url = "{cli_env.url}"' in out


def test_eval_command_end_to_end(
    cli_env: FakeOllama, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "reasoning.json").write_text(
        json.dumps(
            {
                "category": "reasoning",
                "tasks": [
                    {
                        "id": "r1",
                        "prompt": "2+2?",
                        "check": {"type": "exact", "answer": "4", "numeric": True},
                    },
                    {
                        "id": "r2",
                        "prompt": "3+3?",
                        "check": {"type": "exact", "answer": "6", "numeric": True},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (suite / "python.json").write_text(
        json.dumps(
            {
                "category": "python",
                "tasks": [
                    {
                        "id": "p1",
                        "prompt": "add",
                        "check": {
                            "type": "python_tests",
                            "tests": "from solution import add\n\ndef test_a():\n    assert add(1, 1) == 2\n",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cli_env.queue(  # task files load alphabetically: python, then reasoning
        code_reply("def add(a, b):\n    return a + b\n"),
        text("Final answer: 4"),
        text("Final answer: 7"),  # r1 passes, r2 fails
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "overall": {"pass_rate": 1.0},
                "categories": {"reasoning": {"pass_rate": 1.0}, "python": {"pass_rate": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "out.json"
    code = main(
        [
            "eval",
            "--suite",
            str(suite),
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--repair",
            "0",
            "--plain",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Evaluation results" in out and "reasoning" in out and "50%" in out and "100%" in out
    assert "ratio vs baseline" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["categories"]["reasoning"]["pass_rate"] == 0.5
    assert data["categories"]["python"]["pass_rate"] == 1.0
    assert data["baseline_comparison"]["categories"]["reasoning"]["ratio"] == 0.5
    assert data["overall"]["total"] == 3 and data["overall"]["mean_tok_s"] > 0
    # logging + dataset export: only the verified, local-only outputs
    out_file = tmp_path / "dataset.jsonl"
    assert main(["export-dataset", "--out", str(out_file), "--plain"]) == 0
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["meta"]["verified_by"] for r in rows) == ["exact_match", "tests"]
    assert "Claude's critique text is never" in capsys.readouterr().out


def test_eval_with_bad_suite_is_a_usage_error(cli_env: FakeOllama, tmp_path: Path) -> None:
    assert main(["eval", "--suite", str(tmp_path / "missing"), "--plain"]) == 2


def test_doctor_through_main(
    cli_env: FakeOllama, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ai_neo.setup.doctor.RealProbe", FakeProbe)
    code = main(["doctor", "--plain"])
    out = capsys.readouterr().out
    assert (
        "AI Neo doctor" in out and "[PASS] Ollama server" in out and "[PASS] Model qwen3:30b" in out
    )
    assert code == 0


def test_doctor_exit_code_is_nonzero_when_ollama_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AI_NEO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AI_NEO_BASE_URL", "http://127.0.0.1:9")
    assert main(["doctor", "--plain"]) == 1
    assert "[FAIL] Ollama server" in capsys.readouterr().out


def test_setup_non_interactive_through_main(
    cli_env: FakeOllama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ai_neo.setup.wizard.RealProbe", FakeProbe)  # a pretend laptop with Ollama
    cli_env.queue(ChatReply(thinking=["hm"], content=["ready"]))
    assert main(["setup", "--non-interactive", "--plain"]) == 0
    assert (tmp_path / "home" / "config.toml").exists()
    assert (tmp_path / "ws").is_dir() and (tmp_path / "home" / "setup_state.json").exists()


def test_version_flag_and_module_entry_point() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "ai_neo", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and done.stdout.strip() == "ai-neo 0.1.0"
    script = Path(sys.executable).parent / "ai-neo"
    if script.exists():  # the console script created by `pip install -e .`
        done = subprocess.run(
            [str(script), "--version"], capture_output=True, text=True, timeout=60, check=False
        )
        assert done.returncode == 0 and "ai-neo 0.1.0" in done.stdout


def test_unexpected_errors_do_not_show_tracebacks(
    cli_env: FakeOllama, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ai_neo.cli as cli

    def explode(app):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.COMMANDS, "config", explode)
    assert main(["config", "--plain"]) == 1
    out = capsys.readouterr().out
    assert "Unexpected problem: RuntimeError: boom" in out and "Traceback" not in out
    with pytest.raises(RuntimeError):
        main(["config", "--plain", "--debug"])


def test_tool_call_confirmation_in_chat_denies_without_a_terminal(
    cli_env: FakeOllama, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cli_env.queue(
        tool_call("write_file", {"path": "x.txt", "content": "hi"}), text("could not write")
    )
    assert main(["ask", "write x.txt", "--mode", "code", "--no-verify", "--plain"]) == 0
    assert not (tmp_path / "ws" / "x.txt").exists()
    assert "use --yes" in capsys.readouterr().out
    cli_env.queue(tool_call("write_file", {"path": "x.txt", "content": "hi"}), text("written"))
    assert main(["ask", "write x.txt", "--mode", "code", "--no-verify", "--yes", "--plain"]) == 0
    assert (tmp_path / "ws" / "x.txt").read_text(encoding="utf-8") == "hi"
