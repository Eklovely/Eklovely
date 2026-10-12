"""Run a project's tests with pytest when available, else with the built-in mini runner."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

from ai_neo import minirunner
from ai_neo.sandbox import run_python_process

_RESULT_LINE = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)", re.MULTILINE)


@dataclass(frozen=True)
class SuiteRun:
    passed: bool
    returncode: int | None
    runner: str
    output: str
    timed_out: bool
    no_tests: bool
    passed_names: tuple[str, ...]
    failed_names: tuple[str, ...]

    def headline(self) -> str:
        if self.timed_out:
            return "tests timed out"
        if self.no_tests:
            return "no tests were found"
        if self.passed:
            return f"{len(self.passed_names)} test(s) passed"
        return f"{max(len(self.failed_names), 1)} test(s) failed"


def pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def resolve_runner(preference: str) -> str:
    if preference == "pytest":
        return "pytest"
    if preference == "builtin":
        return "builtin"
    return "pytest" if pytest_available() else "builtin"


def run_test_suite(
    cwd: Path,
    targets: list[str],
    *,
    runner: str = "auto",
    timeout: float = 120.0,
    max_bytes: int = 20000,
    memory_limit_mb: int = 0,
    home: Path | None = None,
) -> SuiteRun:
    chosen = resolve_runner(runner)
    if chosen == "pytest":
        args = [
            "-m",
            "pytest",
            "-q",
            "-x",
            "-rA",
            "--no-header",
            "-p",
            "no:cacheprovider",
            *targets,
        ]
    else:
        args = ["-P", str(Path(minirunner.__file__).resolve()), *targets]
    proc = run_python_process(
        args,
        cwd=cwd,
        timeout=timeout,
        max_bytes=max_bytes,
        home=home,
        memory_limit_mb=memory_limit_mb,
    )
    names: dict[str, list[str]] = {"PASSED": [], "FAILED": [], "ERROR": []}
    for status, name in _RESULT_LINE.findall(proc.output):
        names[status].append(name)
    no_tests = proc.returncode == 5 or "NO TESTS FOUND" in proc.output
    passed = proc.returncode == 0 and not proc.timed_out and not no_tests
    output = proc.output
    if proc.timed_out:
        output += f"\n[tests killed after {timeout:g}s timeout]"
    return SuiteRun(
        passed=passed,
        returncode=proc.returncode,
        runner=chosen,
        output=output,
        timed_out=proc.timed_out,
        no_tests=no_tests,
        passed_names=tuple(names["PASSED"]),
        failed_names=tuple(names["FAILED"] + names["ERROR"]),
    )
