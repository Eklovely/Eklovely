from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ai_neo.config import default_config
from ai_neo.governor import Governor
from ai_neo.logging_store import LogStore
from ai_neo.pipeline import TaskOptions, TaskPipeline
from ai_neo.sessions import Session
from ai_neo.tools import Workspace
from tests.fake_ollama import FakeOllama, code_reply, text, tool_call
from tests.helpers import ScriptedUI, make_agent
from tests.test_governor_logging import CRITIQUE_HIGH, CRITIQUE_LOW, SECRET, FakeAnthropic

GOOD_WITH_CHECKS = (
    "def add(a, b):\n    return a + b\n\n"
    "if __name__ == '__main__':\n    assert add(2, 3) == 5\n    print('ok')\n"
)
BAD_WITH_CHECKS = GOOD_WITH_CHECKS.replace("a + b", "a - b")
TESTS = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def make_pipeline(server: FakeOllama, tmp_path: Path, *, governor_replies=None, answers=None, **kw):
    ui = ScriptedUI(answers or [])
    agent, ui, cfg = make_agent(server, tmp_path, ui=ui, **kw)
    store = LogStore(tmp_path / "logs")
    governor = None
    fake = None
    if governor_replies is not None:
        gcfg = dataclasses.replace(cfg.governor, claude_model="claude-test")
        fake = FakeAnthropic(governor_replies)
        governor = Governor(
            gcfg, {"ANTHROPIC_API_KEY": SECRET}, ui, assume_yes=True, client_factory=lambda: fake
        )
        assert governor.enable() is None
    pipeline = TaskPipeline(
        cfg, agent, Workspace(cfg.workspace), ui, store=store, governor=governor
    )
    return pipeline, ui, store, fake


def logged(store: LogStore) -> list[dict]:
    return list(store.records())


def test_plain_turn_is_logged_with_all_fields(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("Paris.", eval_count=10, eval_duration=1_000_000_000))
    pipeline, ui, store, _ = make_pipeline(fake_ollama, tmp_path)
    session = Session(mode="reason", model="qwen3:30b")
    outcome = pipeline.run(session, "Capital of France?")
    assert outcome.answer == "Paris."
    (record,) = logged(store)
    assert record["prompt"] == "Capital of France?" and record["final_answer"] == "Paris."
    assert record["mode"] == "reason" and record["model"] == "qwen3:30b"
    assert record["settings"]["think"] is True and record["settings"]["num_ctx"] == 8192
    assert record["stats"]["tok_per_s"] == 10.0 and record["governor"] is None
    assert record["verified"] == {"method": None, "passed": None}
    assert "AI Neo" in record["system"]


def test_log_false_skips_logging(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("ok"))
    pipeline, _, store, _ = make_pipeline(fake_ollama, tmp_path)
    pipeline.run(Session("write", "m"), "hi", TaskOptions(log=False))
    assert logged(store) == []


def test_code_mode_repairs_automatically_and_logs_the_test_result(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(code_reply(BAD_WITH_CHECKS), code_reply(GOOD_WITH_CHECKS))
    pipeline, ui, store, _ = make_pipeline(fake_ollama, tmp_path)
    session = Session("code", "qwen3:30b")
    outcome = pipeline.run(session, "write add(a, b)", TaskOptions(use_tools=False))
    assert outcome.verify is not None and outcome.verify.passed and outcome.verify.repairs == 1
    (record,) = logged(store)
    assert record["verified"] == {"method": "self-checks", "passed": True}
    assert record["test_results"]["attempts"] == 2 and "def add" in record["final_answer"]
    assert outcome.stats.eval_count == 40  # two generations, counted once each


def test_code_mode_with_known_tests_is_verified_by_tests(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(code_reply("def add(a, b):\n    return a + b\n"))
    pipeline, _, store, _ = make_pipeline(fake_ollama, tmp_path)
    outcome = pipeline.run(Session("code", "m"), "add", TaskOptions(tests=TESTS))
    assert outcome.verify and outcome.verify.passed
    assert logged(store)[0]["verified"] == {"method": "tests", "passed": True}


def test_code_failure_is_logged_as_failed(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(*[code_reply(BAD_WITH_CHECKS)] * 2)
    pipeline, _, store, _ = make_pipeline(fake_ollama, tmp_path, **{"verify.repair_attempts": 1})
    outcome = pipeline.run(Session("code", "m"), "add", TaskOptions(use_tools=False))
    assert outcome.verify and not outcome.verify.passed
    assert logged(store)[0]["verified"]["passed"] is False


def test_compute_mode_shows_evidence(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        tool_call("calculator", {"expression": "12345*6789"}), text("Final answer: 83810205")
    )
    pipeline, ui, store, _ = make_pipeline(fake_ollama, tmp_path)
    outcome = pipeline.run(Session("compute", "m"), "12345*6789?")
    assert outcome.compute and outcome.compute.consistent is True
    assert "Computation (code and result)" in ui.text and "= 83810205" in ui.text
    assert logged(store)[0]["tool_calls"][0]["name"] == "calculator"


def test_build_mode_end_to_end(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    plan = {
        "name": "adder",
        "summary": "adds",
        "files": [
            {"path": "adder.py", "purpose": "add"},
            {"path": "tests/test_adder.py", "purpose": "tests"},
        ],
    }
    tests = (
        "import unittest\nfrom adder import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n"
    )
    fake_ollama.queue(
        text("```json\n" + json.dumps(plan) + "\n```"),
        text("```python\ndef add(a, b):\n    return a + b\n```"),
        text(f"```python\n{tests}```"),
    )
    pipeline, ui, store, _ = make_pipeline(fake_ollama, tmp_path)
    session = Session("build", "m")
    outcome = pipeline.run(session, "an adder", TaskOptions(auto_approve_plan=True))
    assert outcome.build and outcome.build.passed
    assert "Tests: PASSED" in outcome.answer
    assert logged(store)[0]["verified"] == {"method": "build_tests", "passed": True}
    assert [m.role for m in session.messages] == ["user", "assistant"]


def test_governor_revises_and_critique_stays_in_its_own_field(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(text("The answer is 143."), text("The answer is 144."))
    pipeline, ui, store, fake = make_pipeline(
        fake_ollama, tmp_path, governor_replies=[CRITIQUE_LOW, CRITIQUE_HIGH]
    )
    session = Session("reason", "m")
    outcome = pipeline.run(session, "What is 12*12?", TaskOptions(use_tools=False))
    assert outcome.answer == "The answer is 144." and outcome.draft == "The answer is 143."
    # history holds the final answer only, never the critique or the revision request
    assert [(m.role, m.content) for m in session.messages] == [
        ("user", "What is 12*12?"),
        ("assistant", "The answer is 144."),
    ]
    (record,) = logged(store)
    assert (
        record["local_draft"] == "The answer is 143."
        and record["final_answer"] == "The answer is 144."
    )
    assert record["governor"]["used"] and "wrong total" in json.dumps(record["governor"])
    assert "wrong total" not in record["final_answer"] and "wrong total" not in record["prompt"]
    assert len(fake.calls) == 2
    out = tmp_path / "ds.jsonl"
    assert store.export_dataset(out).exported == 0  # unverified + governed: never exported


def test_governor_untouched_when_off(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("fine"))
    pipeline, _, _, _ = make_pipeline(fake_ollama, tmp_path)
    assert pipeline.run(Session("reason", "m"), "q", TaskOptions(use_tools=False)).governor is None


def test_governor_revision_that_breaks_the_checks_is_rejected(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(code_reply(GOOD_WITH_CHECKS), code_reply(BAD_WITH_CHECKS))
    pipeline, ui, store, _ = make_pipeline(
        fake_ollama,
        tmp_path,
        governor_replies=[CRITIQUE_LOW, CRITIQUE_LOW],
        **{"governor.max_rounds": 1},
    )
    session = Session("code", "m")
    outcome = pipeline.run(session, "add", TaskOptions(use_tools=False))
    assert "a + b" in outcome.answer and "keeping the draft" in ui.text
    assert "a + b" in session.messages[-1].content


def test_ctrl_c_skips_the_governor(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("partial one ", "two ", "three"))
    pipeline, ui, store, fake = make_pipeline(
        fake_ollama, tmp_path, governor_replies=[CRITIQUE_HIGH]
    )
    original = ui.stream
    count = []

    def stream():  # type: ignore[no-untyped-def]
        renderer = original()

        class Stopper:
            def event(self, event) -> None:  # type: ignore[no-untyped-def]
                renderer.event(event)
                count.append(1)
                if len(count) == 2:
                    raise KeyboardInterrupt

            def close(self) -> None:
                renderer.close()

        return Stopper()

    ui.stream = stream  # type: ignore[method-assign]
    outcome = pipeline.run(Session("reason", "m"), "q", TaskOptions(use_tools=False))
    assert outcome.interrupted and fake.calls == []


def test_server_error_is_reported_not_raised(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    pipeline, ui, _, _ = make_pipeline(fake_ollama, tmp_path)  # no scripted reply -> HTTP 500
    outcome = pipeline.run(Session("write", "m"), "q")
    assert outcome.error and "[FAIL]" in ui.text


def test_default_config_unchanged() -> None:
    assert default_config().verify.repair_attempts == 3
