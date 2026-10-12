from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ai_neo.config import ConfigError, load_config
from ai_neo.ollama_client import OllamaClient
from ai_neo.paths import Paths
from ai_neo.setup.checks import Status
from ai_neo.setup.doctor import run_doctor
from ai_neo.setup.state import SetupState
from ai_neo.setup.steps import STEPS
from ai_neo.setup.wizard import Step, StepResult, run_setup
from tests.fake_ollama import ChatReply, FakeOllama, PullScript
from tests.helpers import FakeProbe, ScriptedUI

DEAD = "http://127.0.0.1:9"


class Env:
    """Everything one wizard run needs, built on the fake Ollama server."""

    def __init__(self, server: FakeOllama, tmp_path: Path, **flags: object) -> None:
        self.server = server
        self.paths = Paths(tmp_path / "home")
        self.workspace = tmp_path / "ws"
        self.probe = FakeProbe()
        self.environ: dict[str, str] = {}
        base = {"ollama.base_url": server.url, "workspace.path": str(self.workspace)}
        base.update(flags)  # type: ignore[arg-type]
        self.cfg = load_config(None, {}, base)

    def factory(self, dead_calls: int = 0):  # type: ignore[no-untyped-def]
        calls = {"n": 0}

        def make(cfg):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            url = DEAD if calls["n"] <= dead_calls else cfg.ollama.base_url
            return OllamaClient(url, cfg.ollama.model, keep_alive=cfg.ollama.keep_alive, timeout=15)

        return make

    def run(self, ui: ScriptedUI, **kw: object) -> int:
        kw.setdefault("client_factory", self.factory())
        return run_setup(self.paths, self.environ, ui, self.cfg, probe=self.probe, **kw)  # type: ignore[arg-type]

    def state(self) -> SetupState:
        return SetupState.load(self.paths.state)


def full_answers(ws: Path, *, ctx: str = "", think: str = "", keep: str = "") -> list[str]:
    return [
        "",  # step 1: may I run nvidia-smi?
        "",  # step 2: may I run ollama --version?
        str(ws),  # step 6: workspace
        ctx,  # context length
        think,  # thinking default
        keep,  # keep_alive
        "",  # write the settings?
        "",  # step 7: set up the governor? (default No)
        "",  # step 8: run the self-test?
    ]


def queue_first_test(server: FakeOllama) -> None:
    server.queue(
        ChatReply(
            thinking=["The user wants one word."],
            content=["ready"],
            eval_count=30,
            eval_duration=3_000_000_000,
            load_duration=4_000_000_000,
        )
    )


# -- the happy path --------------------------------------------------------------------------


def test_full_walkthrough_with_all_defaults(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    queue_first_test(fake_ollama)
    ui = ScriptedUI(full_answers(env.workspace))
    assert env.run(ui) == 0
    assert not ui.answers  # every scripted answer was asked for, in this order
    text = ui.text
    for number, step in enumerate(STEPS, 1):
        assert f"Step {number} of 9: {step.title}" in text
    for fragment in (
        "What will happen:",  # the exact command/action is shown before anything runs
        "nvidia-smi --query-gpu=name,memory.total",
        "[OK]",
        "RTX 500 Ada",
        "most of the model will run on the CPU",  # tells the laptop user what to expect
        "(thinking) The user wants one word.",  # thinking separated from the answer
        "--- answer ---",
        "10.0 tok/s",
        "keep_alive=30m",
        "Loading the model took 4.0s",
        "2+2*3 = 8",
        "an endless loop was stopped by the timeout",
        "a path containing ../ was rejected",
        "ai-neo ask",
        "--mode code",
        "--mode build",
        "ai-neo doctor",
        "Setup is complete",
    ):
        assert fragment in text, fragment
    # config written, workspace created, state recorded
    assert env.paths.config.exists() and env.workspace.is_dir()
    saved = load_config(env.paths.config, {})
    assert saved.workspace == env.workspace and saved.ollama.num_ctx == 8192
    assert not saved.governor.claude_model
    state = env.state()
    assert all(state.finished(s.id) for s in STEPS[:-1])
    assert "tokens/second" in text and str(env.paths.config) in text


def test_setup_asks_before_each_command(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI(["n", "n"])  # decline nvidia-smi and ollama --version
    assert env.run(ui, steps=STEPS[:2]) == 0
    assert env.probe.ran == []  # nothing was executed
    assert "GPU check skipped at your request" in ui.text


def test_low_resources_get_warnings_with_advice(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.probe.ram_gb, env.probe.disk_gb = 16.0, 12.0
    env.probe.executables.pop("nvidia-smi")
    ui = ScriptedUI(["y"])
    assert env.run(ui, steps=STEPS[:1]) == 0  # warnings do not fail the step
    assert "[WARN] Memory (RAM): 16 GB" in ui.text and "[WARN] Free disk space" in ui.text
    assert "none detected (the model will run on the CPU)" in ui.text


def test_old_python_fails_with_fix(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.probe.py = (3, 9, 7)
    ui = ScriptedUI(["", "q"])
    assert env.run(ui, steps=STEPS[:1]) == 1
    assert (
        "[FAIL]" in ui.text
        and "python.org" in ui.text
        and "Most likely causes and fixes" in ui.text
    )


# -- Ollama installed ----------------------------------------------------------------------------


def test_ollama_missing_then_installed_after_enter(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    del env.probe.executables["ollama"]

    def user_installs_ollama() -> None:
        env.probe.executables["ollama"] = "/usr/local/bin/ollama"

    ui = ScriptedUI(
        [user_installs_ollama, ""]
    )  # press Enter (after installing), then allow --version
    assert env.run(ui, steps=STEPS[1:2]) == 0
    assert "https://ollama.com/download" in ui.text
    assert "Press Enter once Ollama is installed" in ui.prompts[0]
    assert "Found Ollama at /usr/local/bin/ollama - ollama version is 0.12.0" in ui.text
    assert env.probe.ran == [("/usr/local/bin/ollama", "--version")]  # only the read-only command


def test_ollama_missing_retry_then_skip_and_nothing_is_auto_run(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    del env.probe.executables["ollama"]
    ui = ScriptedUI(["", "r", "", "s"])  # Enter, Retry, Enter again, Skip
    assert env.run(ui, steps=STEPS[1:2]) == 1  # skipped a failing step = unresolved
    assert ui.text.count("https://ollama.com/download") >= 2
    assert env.probe.ran == []  # never ran an installer
    assert env.state().steps["ollama-installed"].status == "skipped"
    assert "Installed but not found" in ui.text


def test_windows_default_install_location_is_searched() -> None:
    from ai_neo.setup.probe import known_locations

    paths = known_locations("ollama", "Windows", {"LOCALAPPDATA": r"C:\Users\me\AppData\Local"})
    assert paths[0].name == "ollama.exe" and "Programs" in str(paths[0])


# -- Ollama running ------------------------------------------------------------------------------


def test_server_down_then_retry_succeeds(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI(["r"])
    code = env.run(ui, steps=STEPS[2:3], client_factory=env.factory(dead_calls=1))
    assert code == 0
    text = ui.text
    assert "[FAIL] Nothing answered at" in text and "ollama serve" in text and "Start menu" in text
    assert "port 11434".lower() in text.lower()
    assert "[OK] Ollama is running" in text
    assert env.state().steps["ollama-running"].status == "done"


def test_server_down_skip_and_quit(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    factory = env.factory(dead_calls=99)
    assert env.run(ScriptedUI(["s"]), steps=STEPS[2:3], client_factory=factory) == 1
    assert env.state().steps["ollama-running"].status == "skipped"
    ui = ScriptedUI(["q"])
    assert env.run(ui, steps=STEPS[2:3], client_factory=factory) == 1
    assert "run `ai-neo setup` to continue" in ui.text


# -- model installed -----------------------------------------------------------------------------


def test_missing_model_is_pulled_with_permission(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.models.clear()
    fake_ollama.pull_scripts.append(
        PullScript(
            [
                {"status": "pulling manifest"},
                {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 500},
                {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 1000},
                {"status": "success"},
            ]
        )
    )
    ui = ScriptedUI(["y"])
    assert env.run(ui, steps=STEPS[3:4]) == 0
    assert "about 19 GB" in ui.text and "Download qwen3:30b now?" in ui.prompts[0]
    assert "100%" in ui.text and "downloaded and installed" in ui.text
    assert fake_ollama.models == ["qwen3:30b"]
    assert any(p == "/api/pull" for _, p, _ in fake_ollama.requests)


def test_declining_the_download_downloads_nothing(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.models.clear()
    ui = ScriptedUI(["n", "s"])
    assert env.run(ui, steps=STEPS[3:4]) == 1
    assert not any(p == "/api/pull" for _, p, _ in fake_ollama.requests)
    assert "ollama pull qwen3:30b" in ui.text


def test_interrupted_pull_can_be_retried_and_resumes(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.models.clear()
    fake_ollama.pull_scripts.extend(
        [
            PullScript(
                [{"status": "pulling abc", "total": 1000, "completed": 300}], add_model=False
            ),
            PullScript(
                [{"status": "pulling abc", "total": 1000, "completed": 1000}, {"status": "success"}]
            ),
        ]
    )
    ui = ScriptedUI(["y", "r", "y"])  # download? / the connection dropped: Retry / download again?
    assert env.run(ui, steps=STEPS[3:4]) == 0
    assert "The download failed" in ui.text and "connection closed" in ui.text
    assert "resumes where it left off" in ui.text or "Retry" in ui.text
    assert "disk space" in ui.text and "proxy" in ui.text.lower()
    pulls = [b for _, p, b in fake_ollama.requests if p == "/api/pull"]
    assert len(pulls) == 2


def test_ctrl_c_during_pull_is_a_pause_not_a_crash(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.models.clear()
    fake_ollama.pull_scripts.extend(
        [
            PullScript(
                [{"status": "pulling abc", "total": 1000, "completed": 100}] * 3, add_model=False
            ),
            PullScript([{"status": "success"}]),
        ]
    )
    ui = ScriptedUI(["y", "r", "y"])

    class Bar:
        calls = 0

        def update(self, completed: int, total: int, label: str = "") -> None:
            Bar.calls += 1
            if Bar.calls == 1:
                raise KeyboardInterrupt

        def close(self) -> None:
            pass

    ui.progress = lambda description: Bar()  # type: ignore[method-assign]
    assert env.run(ui, steps=STEPS[3:4]) == 0
    assert "Download cancelled" in ui.text and "Traceback" not in ui.text


# -- first test ----------------------------------------------------------------------------------


def test_first_test_reports_speed_and_slow_warning(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.queue(ChatReply(content=["ready"], eval_count=10, eval_duration=10_000_000_000))
    ui = ScriptedUI()
    assert env.run(ui, steps=STEPS[4:5]) == 0
    assert "[WARN] It works, but at 1.0 tokens/second it is very slow" in ui.text


def test_first_test_failure_offers_retry(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.queue(ChatReply(status=500, error="out of memory"), ChatReply(content=["ready"]))
    ui = ScriptedUI(["r"])
    assert env.run(ui, steps=STEPS[4:5]) == 0
    assert "[FAIL]" in ui.text and "out of memory" in ui.text.lower()
    assert "Out of memory: close other programs" in ui.text


# -- settings ------------------------------------------------------------------------------------


def test_settings_validate_answers_and_write_the_file(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    answers = [str(env.workspace), "lots", "4096", "on", "1h", ""]
    ui = ScriptedUI(answers)
    assert env.run(ui, steps=STEPS[5:6]) == 0
    assert "ollama.num_ctx" in ui.text and "must be an integer" in ui.text  # "lots" was rejected
    saved = load_config(env.paths.config, {})
    assert (saved.ollama.num_ctx, saved.chat.think, saved.ollama.keep_alive) == (4096, "on", "1h")
    assert f"Settings saved to {env.paths.config}" in ui.text
    assert "roughly another 1 GB" in ui.text


def test_low_ram_recommends_a_smaller_context(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI([str(env.workspace), "", "", "", ""])
    ctx_facts = {}

    from ai_neo.setup import steps as steps_module

    original = steps_module.run_settings

    def run(ctx):  # type: ignore[no-untyped-def]
        ctx.facts["ram_gb"] = 16.0
        ctx_facts["ctx"] = ctx
        return original(ctx)

    step = dataclasses.replace(STEPS[5], run=run)
    assert env.run(ui, steps=(step,)) == 0
    assert load_config(env.paths.config, {}).ollama.num_ctx == 4096


def test_settings_preserve_other_values_and_back_up(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    env.paths.root.mkdir(parents=True)
    env.paths.config.write_text('[ollama]\nbase_url = "http://gpu-box:11434"\n', encoding="utf-8")
    ui = ScriptedUI([str(env.workspace), "", "", "", ""])
    assert env.run(ui, steps=STEPS[5:6]) == 0
    assert load_config(env.paths.config, {}).ollama.base_url == "http://gpu-box:11434"
    assert env.paths.config.with_name("config.toml.bak").exists()


# -- governor ------------------------------------------------------------------------------------


def test_governor_default_is_no(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI([""])
    assert env.run(ui, steps=STEPS[6:7]) == 0
    assert "OPTIONAL and OFF by default" in ui.text
    assert "costs money" in ui.text and "Anthropic's servers" in ui.text
    assert "Not setting up the governor" in ui.text and "setx" not in ui.text


def test_governor_yes_shows_key_commands_and_never_prints_the_key(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI(["y", "s"])
    assert env.run(ui, steps=STEPS[6:7]) == 1
    assert 'setx ANTHROPIC_API_KEY "your-key"' in ui.text and "NEW one" in ui.text
    assert "not set in this window yet" in ui.text
    env.probe.system = "Linux"
    ui = ScriptedUI(["y", "s"])
    env.run(ui, steps=STEPS[6:7])
    assert "export ANTHROPIC_API_KEY" in ui.text and "~/.bashrc" in ui.text


def test_governor_with_key_saves_model_name_and_keeps_key_private(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    env.environ["ANTHROPIC_API_KEY"] = "sk-ant-TOPSECRET"
    ui = ScriptedUI(["y", "my-claude-model"])
    code = env.run(ui, steps=STEPS[6:7])
    assert code == 0  # a warning (missing SDK in the test env) does not fail the step
    assert "TOPSECRET" not in ui.text
    assert "ANTHROPIC_API_KEY is set (its value is never shown or stored)" in ui.text
    assert load_config(env.paths.config, {}).governor.claude_model == "my-claude-model"
    assert "TOPSECRET" not in env.paths.config.read_text(encoding="utf-8")


# -- self-test and done --------------------------------------------------------------------------


def test_selftest_runs_each_check(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI([""])
    assert env.run(ui, steps=STEPS[7:8]) == 0
    for fragment in (
        "Calculator: 2+2*3 = 8",
        "run_python: a one-line script printed 42",
        "run_python timeout",
        "File tools: ../",
        "File tools: absolute path",
        "Tiny reasoning task",
        "Tiny computation task",
        "Tiny python task",
        "Tiny writing task",
        "Tiny build task",
        "self-test checks passed",
    ):
        assert fragment in ui.text, fragment
    assert "[FAIL]" not in ui.text


def test_selftest_can_be_declined(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI(["n", "s"])
    assert env.run(ui, steps=STEPS[7:8]) == 1


# -- non-interactive -----------------------------------------------------------------------------


def test_non_interactive_never_prompts_never_downloads(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.models.clear()  # model missing
    fake_ollama.pull_scripts.append(PullScript([{"status": "success"}]))
    ui = ScriptedUI([])  # any prompt would raise EOFError
    code = env.run(ui, non_interactive=True)
    assert code == 1  # the model is missing, so setup is not complete
    assert not any(p == "/api/pull" for _, p, _ in fake_ollama.requests)
    assert ui.prompts == []
    assert env.paths.config.exists()  # defaults were written
    assert "Unfinished steps: model-installed" in ui.text or "model-installed" in ui.text
    assert "Setup finished with unresolved steps" in ui.text


def test_non_interactive_success_when_everything_is_ready(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    queue_first_test(fake_ollama)
    ui = ScriptedUI([])
    assert env.run(ui, non_interactive=True) == 0
    assert ui.prompts == [] and "Setup is complete" in ui.text
    assert env.workspace.is_dir()


# -- resume and idempotence ----------------------------------------------------------------------


def test_resume_after_quit_skips_finished_steps(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    first = ScriptedUI(["", "", "q"])  # nvidia-smi, --version, then Quit at the failing step 3
    assert env.run(first, steps=STEPS[:3], client_factory=env.factory(dead_calls=99)) == 1
    state = env.state()
    assert state.finished("system") and state.finished("ollama-installed")
    assert state.finished("ollama-running") is None
    ran_before = list(env.probe.ran)
    second = ScriptedUI(["", ""])  # confirm skipping steps 1 and 2
    assert env.run(second, steps=STEPS[:3]) == 0
    assert [("already completed on" in p) for p in second.prompts] == [True, True]
    assert len([p for p in second.prompts if "Skip it?" in p]) == 2
    assert env.probe.ran == ran_before  # finished steps were not run again
    assert env.state().finished("ollama-running")


def test_declining_the_skip_reruns_the_step(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.run(ScriptedUI(["", ""]), steps=STEPS[:2])
    before = len(env.probe.ran)
    ui = ScriptedUI(
        ["n", "", "n", ""]
    )  # don't skip step 1 (then allow nvidia-smi), same for step 2
    assert env.run(ui, steps=STEPS[:2]) == 0
    assert len(env.probe.ran) > before


def test_rerunning_everything_is_safe_and_changes_nothing(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    queue_first_test(fake_ollama)
    assert env.run(ScriptedUI(full_answers(env.workspace))) == 0
    config_before = env.paths.config.read_text(encoding="utf-8")
    again = ScriptedUI(["y"] * 8)  # skip all eight finished steps
    assert env.run(again) == 0
    assert env.paths.config.read_text(encoding="utf-8") == config_before
    assert not env.paths.config.with_name("config.toml.bak").exists()
    assert len([p for p in again.prompts if "Skip it?" in p]) == 8
    assert "Setup is complete" in again.text
    # --reset forgets progress
    queue_first_test(fake_ollama)
    assert env.run(ScriptedUI(full_answers(env.workspace)), reset=True) == 0


def test_corrupt_state_file_is_ignored(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.paths.root.mkdir(parents=True)
    env.paths.state.write_text("{not json", encoding="utf-8")
    ui = ScriptedUI(["", ""])
    assert env.run(ui, steps=STEPS[:2]) == 0
    assert "could not be read" in ui.text


def test_eof_mid_setup_saves_progress(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    ui = ScriptedUI([""])  # input ends right after step 1's question
    assert env.run(ui, steps=STEPS[:2]) == 1
    assert "Progress is saved" in ui.text


def test_a_crashing_step_shows_no_traceback(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)

    def boom(ctx):  # type: ignore[no-untyped-def]
        raise RuntimeError("kaboom")

    step = Step("boom", "Exploding step", "what", "why", ("try again",), lambda c: [], boom)
    ui = ScriptedUI(["s"])
    assert env.run(ui, steps=(step,)) == 1
    assert "Something unexpected went wrong: RuntimeError: kaboom" in ui.text
    assert "Traceback" not in ui.text
    assert any("[R]etry" in prompt for prompt in ui.prompts)


def test_ctrl_c_inside_a_step_offers_the_menu(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    calls = []

    def interrupted(ctx):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return StepResult(Status.OK, "fine now")

    step = Step("ctrlc", "Interruptible", "what", "why", (), lambda c: [], interrupted)
    ui = ScriptedUI(["r"])
    assert env.run(ui, steps=(step,)) == 0
    assert "Interrupted with Ctrl+C" in ui.text


def test_remote_ollama_skips_local_checks(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path, **{"ollama.base_url": "http://gpu-box.example.com:11434"})
    del env.probe.executables["ollama"]
    ui = ScriptedUI([])
    assert env.run(ui, steps=STEPS[:2], non_interactive=True) == 0
    assert "Memory, disk and GPU checks skipped" in ui.text and "[SKIP]" in ui.text
    assert env.probe.ran == []


# -- doctor --------------------------------------------------------------------------------------


def doctor(env: Env, **kw: object) -> tuple[int, ScriptedUI]:
    ui = ScriptedUI()
    env.workspace.mkdir(parents=True, exist_ok=True)
    code = run_doctor(
        env.paths, env.environ, ui, env.cfg, probe=env.probe, client_factory=env.factory(**kw), **{}
    )  # type: ignore[arg-type]
    return code, ui


def test_doctor_all_good_exits_zero(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    code, ui = doctor(env)
    assert code == 0
    assert "[PASS] Python version" in ui.text and "[PASS] Model qwen3:30b: installed" in ui.text
    assert "0 failed" in ui.text
    assert "[WARN] Configuration file: not created yet" in ui.text  # a warning, not a failure
    assert "fix: Run `ai-neo setup` to create it." in ui.text


def test_doctor_server_down_fails_with_one_line_fixes(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    code, ui = doctor(env, dead_calls=99)
    assert code == 1
    assert "[FAIL] Ollama server" in ui.text and "ollama serve" in ui.text
    assert "[SKIP] Model qwen3:30b" in ui.text


def test_doctor_missing_model_fails(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    fake_ollama.models.clear()
    code, ui = doctor(env)
    assert code == 1 and "[FAIL] Model qwen3:30b: not installed" in ui.text
    assert "fix: Run: ollama pull qwen3:30b" in ui.text


def test_doctor_missing_binary_is_fail_only_if_server_is_down(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    env = Env(fake_ollama, tmp_path)
    del env.probe.executables["ollama"]
    code, ui = doctor(env)
    assert code == 0 and "[WARN] Ollama program" in ui.text
    code, ui = doctor(env, dead_calls=99)
    assert code == 1 and "[FAIL] Ollama program" in ui.text


def test_doctor_warnings_alone_exit_zero(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.probe.ram_gb = 16.0
    code, ui = doctor(env)
    assert code == 0 and "[WARN] Memory (RAM): 16 GB" in ui.text and "0 failed" in ui.text


def test_doctor_bad_config_fails(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.paths.root.mkdir(parents=True)
    env.paths.config.write_text("[ollama]\nnum_ctx = 5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(env.paths.config, {})
    ui = ScriptedUI()
    code = run_doctor(
        env.paths, {}, ui, env.cfg, info.value, probe=env.probe, client_factory=env.factory()
    )
    assert code == 1 and "[FAIL] Configuration file" in ui.text and "ollama.num_ctx" in ui.text


def test_doctor_old_python_fails(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.probe.py = (3, 8, 0)
    code, ui = doctor(env)
    assert code == 1 and "[FAIL] Python version" in ui.text and "python.org" in ui.text


def test_doctor_remote_server_skips_local_hardware(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.cfg = dataclasses.replace(
        env.cfg, ollama=dataclasses.replace(env.cfg.ollama, base_url="http://127.0.0.2:9")
    )
    env.cfg = load_config(
        None,
        {},
        {"ollama.base_url": "http://gpu-box.example.com:9", "workspace.path": str(env.workspace)},
    )
    ui = ScriptedUI()
    code = run_doctor(
        env.paths, {}, ui, env.cfg, probe=env.probe, client_factory=env.factory(dead_calls=9)
    )
    assert "[SKIP] Local hardware and install" in ui.text
    assert code == 1  # the remote server is unreachable


def test_doctor_governor_states(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    env = Env(fake_ollama, tmp_path)
    env.environ["ANTHROPIC_API_KEY"] = "sk-ant-TOPSECRET"
    code, ui = doctor(env)
    assert code == 0 and "[WARN] Claude governor (optional)" in ui.text
    assert "no Claude model name" in ui.text and "TOPSECRET" not in ui.text
