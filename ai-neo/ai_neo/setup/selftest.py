"""Offline self-test: AI Neo's tools, safety guards and answer checkers (no model, no network)."""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from ai_neo.calculator import CalcError, format_number, safe_eval
from ai_neo.config import Config
from ai_neo.evals import RunOutput, Task, evaluate
from ai_neo.setup.checks import Check, Status
from ai_neo.tools import ToolRegistry, Workspace
from ai_neo.types import ToolCall

PRIVATE_MARKER = "contents-that-must-stay-private"

TINY_TASKS: tuple[tuple[Task, str, str], ...] = (
    # (task, a correct answer, a wrong answer)
    (
        Task(
            "selftest-reasoning",
            "reasoning",
            "Is 7 greater than 3?",
            {"type": "exact", "answer": "yes"},
        ),
        "Final answer: yes",
        "Final answer: no",
    ),
    (
        Task(
            "selftest-computation",
            "computation",
            "What is 6 * 7?",
            {"type": "exact", "answer": "42", "numeric": True},
        ),
        "Final answer: 42",
        "Final answer: 41",
    ),
    (
        Task(
            "selftest-python",
            "python",
            "Write double(x).",
            {
                "type": "python_tests",
                "tests": (
                    "from solution import double\n\n\n"
                    "def test_double():\n    assert double(4) == 8\n"
                ),
            },
        ),
        "```python\ndef double(x):\n    return x * 2\n```",
        "```python\ndef double(x):\n    return x + 2\n```",
    ),
    (
        Task(
            "selftest-writing",
            "writing",
            "Greet the reader.",
            {
                "type": "rubric",
                "min_words": 3,
                "max_words": 12,
                "must_include": ["hello"],
                "banned": ["goodbye"],
            },
        ),
        "Hello there, welcome to AI Neo today.",
        "Goodbye.",
    ),
    (
        Task(
            "selftest-build",
            "build",
            "Build greeter.py.",
            {
                "type": "build",
                "tests": {
                    "acceptance/test_greeter.py": (
                        "from greeter import greet\n\n\n"
                        "def test_greet():\n    assert greet('Ann') == 'Hello, Ann!'\n"
                    )
                },
            },
        ),
        "def greet(name):\n    return f'Hello, {name}!'\n",
        "def greet(name):\n    return name\n",
    ),
)


def _check(check_id: str, label: str, ok: bool, detail: str, fix: str = "") -> Check:
    return Check(check_id, label, Status.OK if ok else Status.FAIL, detail, "" if ok else fix)


def _calculator() -> Check:
    try:
        shown = format_number(safe_eval("2+2*3"))
    except CalcError as exc:
        return _check("calc", "Calculator", False, str(exc), "Please report this as a bug.")
    return _check("calc", "Calculator", shown == "8", f"2+2*3 = {shown}")


def _hostile_calculator() -> Check:
    try:
        safe_eval("__import__('os').system('echo hi')")
    except CalcError:
        return _check("calc-safe", "Calculator safety", True, "refuses to run code in expressions")
    return _check(
        "calc-safe", "Calculator safety", False, "it ran code!", "Do not use the calculator tool."
    )


def _run_python(registry: ToolRegistry) -> list[Check]:
    quick = registry.execute(ToolCall("run_python", {"code": "print(6 * 7)"}))
    loop = registry.execute(ToolCall("run_python", {"code": "while True: pass"}))
    return [
        _check(
            "run-python",
            "run_python",
            quick.ok and "42" in quick.output,
            "a one-line script printed 42",
            "Python subprocesses may be blocked by antivirus.",
        ),
        _check(
            "timeout",
            "run_python timeout",
            bool(loop.meta.get("timed_out")),
            "an endless loop was stopped by the timeout",
            "Please report this as a bug; code runs are not being time-limited.",
        ),
    ]


def _path_guard(registry: ToolRegistry, root: Path) -> list[Check]:
    traversal = registry.execute(ToolCall("read_file", {"path": "../secret.txt"}))
    absolute = registry.execute(ToolCall("read_file", {"path": str(root.parent / "secret.txt")}))
    checks = [
        _check(
            "path-traversal",
            "File tools: ../",
            bool(traversal.meta.get("blocked")),
            "a path containing ../ was rejected",
            "Do not use the file tools; report a bug.",
        ),
        _check(
            "path-absolute",
            "File tools: absolute path",
            bool(absolute.meta.get("blocked")),
            "an absolute path outside the workspace was rejected",
            "Do not use the file tools.",
        ),
    ]
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(PRIVATE_MARKER, encoding="utf-8")
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        checks.append(
            Check(
                "path-symlink",
                "File tools: symlink escape",
                Status.WARN,
                "could not create a symlink here, so this check was skipped",
                "Windows needs Developer Mode to create symlinks; unit tests cover this.",
            )
        )
        return checks
    via_link = registry.execute(ToolCall("read_file", {"path": "link/secret.txt"}))
    checks.append(
        _check(
            "path-symlink",
            "File tools: symlink escape",
            bool(via_link.meta.get("blocked")) and PRIVATE_MARKER not in via_link.output,
            "a symlink pointing outside the workspace was rejected",
            "Do not use the file tools; report a bug.",
        )
    )
    return checks


def _tiny_tasks(cfg: Config) -> list[Check]:
    checks = []
    for task, good, bad in TINY_TASKS:
        passed, failed = _grade(task, good, bad, cfg)
        ok = passed and not failed
        detail = (
            "accepts a correct answer and rejects a wrong one"
            if ok
            else f"accepts correct: {passed}; rejects wrong: {not failed}"
        )
        checks.append(
            _check(
                task.id,
                f"Tiny {task.category} task",
                ok,
                detail,
                "Please report this as a bug; evaluation results would be unreliable.",
            )
        )
    return checks


def _grade(task: Task, good: str, bad: str, cfg: Config) -> tuple[bool, bool]:
    """(correct answer accepted, wrong answer accepted)."""
    if task.category != "build":
        return evaluate(task, RunOutput(good), cfg).passed, evaluate(
            task, RunOutput(bad), cfg
        ).passed
    results = []
    for code in (good, bad):
        with tempfile.TemporaryDirectory(
            prefix="ai_neo_tinybuild_", ignore_cleanup_errors=True
        ) as tmp:
            project = Path(tmp)
            (project / "greeter.py").write_text(code, encoding="utf-8")
            results.append(evaluate(task, RunOutput("", project_dir=project), cfg).passed)
    return results[0], results[1]


def run_selftest(cfg: Config) -> list[Check]:
    with tempfile.TemporaryDirectory(prefix="ai_neo_selftest_", ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        root = base / "workspace"
        root.mkdir()
        tools = dataclasses.replace(cfg.tools, run_python_timeout=2.0)
        registry = ToolRegistry(Workspace(root), tools, auto_yes=True)
        steps: list[Callable[[], list[Check]]] = [
            lambda: [_calculator(), _hostile_calculator()],
            lambda: _run_python(registry),
            lambda: _path_guard(registry, root.resolve()),
            lambda: _tiny_tasks(cfg),
        ]
        checks: list[Check] = []
        for step in steps:
            checks.extend(step())
    return checks
