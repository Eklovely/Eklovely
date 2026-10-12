from __future__ import annotations

from pathlib import Path

import pytest

from ai_neo.sessions import Session
from ai_neo.textutil import clip_middle, extract_json_object, extract_python, extract_single_block
from ai_neo.verify import (
    answer_matches,
    last_numeric_result,
    run_compute_task,
    run_python_task,
)
from tests.fake_ollama import FakeOllama, code_reply, text, tool_call
from tests.helpers import make_agent

TESTS = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
BAD = "def add(a, b):\n    return a - b\n"
GOOD = "def add(a, b):\n    return a + b\n"


def session(mode: str = "code") -> Session:
    return Session(mode=mode, model="qwen3:30b")


@pytest.mark.parametrize("runner", ["builtin", "pytest"])
def test_failing_first_attempt_is_repaired_automatically(
    fake_ollama: FakeOllama, tmp_path: Path, runner: str
) -> None:
    fake_ollama.queue(code_reply(BAD), code_reply(GOOD, "Fixed: I used the wrong operator."))
    agent, ui, _ = make_agent(fake_ollama, tmp_path, **{"tools.test_runner": runner})
    report = run_python_task(
        agent, session(), "write add(a, b)", tests=TESTS, attempts=3, think=False, ui=ui
    )
    assert report.passed and report.repairs == 1 and report.method == "tests"
    assert [a.passed for a in report.attempts] == [False, True]
    assert report.final_code == GOOD
    feedback = fake_ollama.chat_payloads()[1]["messages"][-1]["content"]
    assert "Your code failed" in feedback and "assert" in feedback
    first_prompt = fake_ollama.chat_payloads()[0]["messages"][-1]["content"]
    assert "solution.py" in first_prompt and "test_add" in first_prompt
    assert "PASSED after 1 repair attempt(s)" in report.summary_line()
    assert "tools" not in fake_ollama.chat_payloads()[0]  # generation is tool-free


def test_exhausted_repairs_are_reported_as_failure(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(*[code_reply(BAD) for _ in range(3)])
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_python_task(agent, session(), "add", tests=TESTS, attempts=2, think=False, ui=ui)
    assert not report.passed and len(report.attempts) == 3 and report.repairs == 2
    assert report.summary_line().startswith("FAILED after 2 repair attempt(s)")
    assert len(fake_ollama.chat_payloads()) == 3
    assert "[FAIL]" in ui.text


def test_zero_attempts_means_no_repair_request(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(code_reply(BAD))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_python_task(agent, session(), "add", tests=TESTS, attempts=0, think=False, ui=ui)
    assert not report.passed and len(fake_ollama.chat_payloads()) == 1


def test_reply_without_code_block_is_asked_again(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("I would write an add function."), code_reply(GOOD))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_python_task(agent, session(), "add", tests=TESTS, attempts=2, think=False, ui=ui)
    assert report.passed
    assert "could not find a ```python" in fake_ollama.chat_payloads()[1]["messages"][-1]["content"]


def test_self_check_mode_without_tests(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    failing = BAD + "\nif __name__ == '__main__':\n    assert add(2, 3) == 5\n"
    passing = GOOD + "\nif __name__ == '__main__':\n    assert add(2, 3) == 5\n    print('ok')\n"
    fake_ollama.queue(code_reply(failing), code_reply(passing))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_python_task(agent, session(), "add", tests=None, attempts=3, think=False, ui=ui)
    assert report.passed and report.repairs == 1 and report.method == "self-checks"
    assert "AssertionError" in fake_ollama.chat_payloads()[1]["messages"][-1]["content"]


def test_code_without_any_checks_is_flagged_as_unverified(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(code_reply(GOOD))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_python_task(agent, session(), "add", tests=None, attempts=1, think=False, ui=ui)
    assert report.passed and report.method == "ran-without-checks"
    assert "NOT verified" in report.summary_line()


def test_infinite_loop_is_a_failure_with_timeout_feedback(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(code_reply("while True:\n    pass\n"), code_reply("print('done')\n"))
    agent, ui, _ = make_agent(fake_ollama, tmp_path, **{"tools.run_python_timeout": 1})
    report = run_python_task(agent, session(), "x", tests=None, attempts=1, think=False, ui=ui)
    assert report.attempts[0].timed_out and report.passed
    assert "timeout" in fake_ollama.chat_payloads()[1]["messages"][-1]["content"]


# -- compute -------------------------------------------------------------------------------------


def test_compute_uses_tools_and_shows_evidence(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        tool_call("calculator", {"expression": "12345*6789"}),
        tool_call("run_python", {"code": "print(sum(range(101)))"}),
        text("Final answer: 5050"),
    )
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_compute_task(agent, session("compute"), "sum 1..100", think=False, ui=ui)
    assert not report.nudged and report.consistent is True
    assert "calculator: 12345*6789 = 83810205" in report.evidence
    assert "run_python code:\nprint(sum(range(101)))" in report.evidence
    assert "5050" in report.evidence


def test_compute_nudges_once_when_no_tool_was_used(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        text("It's about 80 million."),
        tool_call("calculator", {"expression": "12345*6789"}),
        text("Final answer: 83,810,205"),
    )
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_compute_task(agent, session("compute"), "12345*6789?", think=False, ui=ui)
    assert report.nudged and report.consistent is True
    assert "without using a tool" in fake_ollama.chat_payloads()[1]["messages"][-1]["content"]


def test_compute_flags_answer_that_ignores_tool_result(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(tool_call("calculator", {"expression": "6*7"}), text("Final answer: 43"))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    report = run_compute_task(agent, session("compute"), "6*7?", think=False, ui=ui)
    assert report.consistent is False


def test_numeric_helpers() -> None:
    assert answer_matches("It is 83,810,205.", "83810205")
    assert not answer_matches("about 80 million", "83810205")
    assert last_numeric_result([]) is None


# -- text helpers --------------------------------------------------------------------------------


def test_extract_python_prefers_last_python_block() -> None:
    reply = "```python\nx = 1\n```\ntext\n```python\ny = 2\n```\n```bash\nls\n```"
    assert extract_python(reply) == "y = 2\n"
    assert extract_python("```\nz = 3\n```") == "z = 3\n"
    assert extract_python("no code") is None
    assert extract_python("```python\nunclosed = 1") is None


def test_extract_single_block_and_json() -> None:
    assert (
        extract_single_block("a\n```js\nlong block here\n```\n```\nx\n```") == "long block here\n"
    )
    assert extract_json_object('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert extract_json_object('```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    assert extract_json_object("nothing") is None
    assert extract_json_object("[1, 2]") is None


def test_clip_middle_keeps_the_tail() -> None:
    text_in = "start " + "m" * 5000 + " Traceback: ValueError"
    clipped = clip_middle(text_in, 500)
    assert len(clipped) < 600 and clipped.endswith("ValueError") and "omitted" in clipped
