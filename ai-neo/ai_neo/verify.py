"""Verify-and-repair loops for Python tasks, and evidence display for computation tasks."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ai_neo.agent import Agent, ToolEvent, TurnResult
from ai_neo.config import Config
from ai_neo.prompts import (
    CODE_NO_BLOCK,
    CODE_REPAIR,
    CODE_SELF_TEST_NOTE,
    CODE_TASK,
    CODE_TEST_NOTE,
    COMPUTE_NUDGE,
    fill,
)
from ai_neo.sandbox import run_python_process
from ai_neo.sessions import Session
from ai_neo.testrun import run_test_suite
from ai_neo.textutil import clip_middle, extract_python
from ai_neo.types import GenStats
from ai_neo.ui import UI

FEEDBACK_CHARS = 3000


@dataclass
class Attempt:
    number: int
    code: str
    passed: bool
    runner: str
    returncode: int | None
    output: str
    timed_out: bool = False


@dataclass
class VerifyReport:
    passed: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    final_code: str | None = None
    answer: str = ""
    method: str = "none"  # tests | self-checks | ran-without-checks | none
    stats: GenStats = field(default_factory=GenStats)
    interrupted: bool = False
    error: str | None = None

    @property
    def repairs(self) -> int:
        return max(0, len(self.attempts) - 1)

    def summary_line(self) -> str:
        if self.passed:
            how = (
                "on the first try"
                if self.repairs == 0
                else f"after {self.repairs} repair attempt(s)"
            )
            note = {
                "tests": "the provided tests passed",
                "self-checks": "the program's own assert checks passed",
                "ran-without-checks": "it ran without errors, but it contains no checks, so "
                "correctness is NOT verified",
            }.get(self.method, "")
            return f"PASSED {how}: {note}"
        last = self.attempts[-1] if self.attempts else None
        reason = ""
        if last is not None:
            reason = (
                "timed out"
                if last.timed_out
                else (
                    "no code block in the reply"
                    if not last.code
                    else f"exit code {last.returncode}"
                )
            )
        return (
            f"FAILED after {self.repairs} repair attempt(s) ({reason}). "
            "The code below is unverified."
        )


def run_solution(code: str, tests: str | None, cfg: Config) -> Attempt:
    """Run `code` (as solution.py) with optional tests in a throwaway folder."""
    with tempfile.TemporaryDirectory(prefix="ai_neo_verify_", ignore_cleanup_errors=True) as tmp:
        directory = Path(tmp)
        (directory / "solution.py").write_text(code, encoding="utf-8")
        if tests is not None:
            (directory / "test_solution.py").write_text(tests, encoding="utf-8")
            run = run_test_suite(
                directory,
                ["test_solution.py"],
                runner=cfg.tools.test_runner,
                timeout=cfg.tools.test_timeout,
                max_bytes=cfg.tools.max_output_bytes,
                memory_limit_mb=cfg.tools.memory_limit_mb,
            )
            return Attempt(
                0, code, run.passed, run.runner, run.returncode, run.output, run.timed_out
            )
        proc = run_python_process(
            ["solution.py"],
            cwd=directory,
            timeout=cfg.tools.run_python_timeout,
            max_bytes=cfg.tools.max_output_bytes,
            memory_limit_mb=cfg.tools.memory_limit_mb,
        )
        passed = proc.returncode == 0 and not proc.timed_out
        output = proc.output + ("\n[killed: timeout]" if proc.timed_out else "")
        return Attempt(0, code, passed, "python", proc.returncode, output, proc.timed_out)


def verification_method(code: str, tests: str | None) -> str:
    if tests is not None:
        return "tests"
    return "self-checks" if re.search(r"\bassert\b|unittest", code) else "ran-without-checks"


def verify_and_repair(
    agent: Agent,
    session: Session,
    turn: TurnResult,
    *,
    tests: str | None,
    attempts: int,
    think: bool,
    ui: UI,
) -> VerifyReport:
    """Run the code in `turn.answer`; on failure send the output back, up to `attempts` times."""
    report = VerifyReport(answer=turn.answer)
    report.stats += turn.stats
    current = turn
    for number in range(attempts + 1):
        code = extract_python(current.answer)
        if code is None:
            attempt = Attempt(number, "", False, "-", None, "no python code block found", False)
            feedback = CODE_NO_BLOCK
        else:
            ui.say(
                f"Running the code ({'with tests' if tests is not None else 'as a script'})...",
                dim=True,
            )
            attempt = run_solution(code, tests, agent.cfg)
            attempt.number = number
            feedback = fill(
                CODE_REPAIR,
                runner=attempt.runner,
                returncode=str(attempt.returncode),
                output=clip_middle(attempt.output.strip() or "(no output)", FEEDBACK_CHARS),
            )
        report.attempts.append(attempt)
        report.answer = current.answer
        if attempt.passed:
            report.passed = True
            report.final_code = code
            report.method = verification_method(code or "", tests)
            ui.tag("OK", report.summary_line())
            return report
        report.final_code = code
        ui.tag("FAIL", f"attempt {number + 1}: {_failure_reason(attempt)}")
        if number == attempts:
            break
        ui.say(f"Sending the failure back to the model (repair {number + 1} of {attempts})...")
        current = agent.run_turn(session, feedback, think=think, use_tools=False)
        report.stats += current.stats
        if current.interrupted or current.error:
            report.interrupted = current.interrupted
            report.error = current.error
            break
    report.method = "none"
    ui.tag("FAIL", report.summary_line())
    return report


def _failure_reason(attempt: Attempt) -> str:
    if attempt.timed_out:
        return "timed out"
    if not attempt.code:
        return "no code block in the reply"
    lines = [ln for ln in attempt.output.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else f"exit code {attempt.returncode}")[:160]


def run_python_task(
    agent: Agent,
    session: Session,
    request: str,
    *,
    tests: str | None,
    attempts: int,
    think: bool,
    ui: UI,
) -> VerifyReport:
    """Generate code for `request`, run it (and `tests`), and repair failures."""
    note = fill(CODE_TEST_NOTE, tests=tests) if tests is not None else CODE_SELF_TEST_NOTE
    prompt = fill(CODE_TASK, request=request, test_note=note)
    turn = agent.run_turn(session, prompt, think=think, use_tools=False)
    if turn.interrupted or turn.error:
        report = VerifyReport(answer=turn.answer, interrupted=turn.interrupted, error=turn.error)
        report.stats += turn.stats
        return report
    return verify_and_repair(
        agent, session, turn, tests=tests, attempts=attempts, think=think, ui=ui
    )


# -- computation -------------------------------------------------------------------------------


@dataclass
class ComputeReport:
    turn: TurnResult
    evidence: str
    nudged: bool = False
    consistent: bool | None = None  # None = could not tell


def format_evidence(events: list[ToolEvent]) -> str:
    """The code and result of each computation, shown next to the answer."""
    blocks = []
    for event in events:
        if event.name == "calculator" and isinstance(event.arguments, dict):
            blocks.append(f"calculator: {event.output}")
        elif event.name == "run_python" and isinstance(event.arguments, dict):
            code = str(event.arguments.get("code", "")).rstrip()
            blocks.append(f"run_python code:\n{code}\nresult:\n{event.output}")
        else:
            blocks.append(f"{event.name}: {event.output}")
    return "\n\n".join(blocks)


def last_numeric_result(events: list[ToolEvent]) -> str | None:
    for event in reversed(events):
        if not event.ok:
            continue
        if event.name == "calculator":
            value = event.output.rsplit(" = ", 1)[-1].strip()
        elif event.name == "run_python":
            body = event.output.split("--- output ---", 1)[-1].strip()
            value = body.splitlines()[-1].strip() if body else ""
        else:
            continue
        if re.fullmatch(r"-?\d+(\.\d{1,6})?", value):
            return value
    return None


def answer_matches(answer: str, value: str) -> bool:
    normalized = answer.replace(",", "").replace(" ", "").replace("_", "")
    return value in normalized


def run_compute_task(
    agent: Agent, session: Session, request: str, *, think: bool, ui: UI
) -> ComputeReport:
    """Ask for tool-based computation; nudge once if the model answered without any tool."""
    turn = agent.run_turn(session, request, think=think)
    nudged = False
    if not turn.tool_events and not turn.interrupted and not turn.error:
        ui.say("[no tool was used; asking the model to compute with a tool]", dim=True)
        nudged = True
        second = agent.run_turn(session, COMPUTE_NUDGE, think=think)
        second.stats += turn.stats
        second.thinking = "\n\n".join(part for part in (turn.thinking, second.thinking) if part)
        turn = second
    value = last_numeric_result(turn.tool_events)
    consistent = None if value is None or not turn.answer else answer_matches(turn.answer, value)
    return ComputeReport(turn, format_evidence(turn.tool_events), nudged, consistent)
