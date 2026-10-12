"""A tiny dependency-free test runner, used when pytest is not installed.

Run as a script: `python minirunner.py [file-or-dir ...]`. It understands `unittest.TestCase`
classes, plain `test_*` functions and `Test*` classes without fixtures. Output lines use the same
`PASSED/FAILED/ERROR <id>` shape as `pytest -rA`, and the exit code is 0 only if at least one
test ran and none failed (5 means no tests were found, like pytest).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
import unittest
from pathlib import Path
from types import ModuleType

SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "venv", "env"}


def discover(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets or ["."]:
        path = Path(target)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.py")):
                parts = candidate.relative_to(path).parts[:-1]
                if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                    continue
                if candidate.name.startswith("test_") or candidate.name.endswith("_test.py"):
                    files.append(candidate)
    return files


def load_module(path: Path, index: int) -> ModuleType:
    for directory in (Path.cwd(), path.resolve().parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(f"{path.stem}__{index}", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def format_failure(exc: BaseException) -> str:
    frames = traceback.format_exception(
        type(exc), exc, exc.__traceback__.tb_next if exc.__traceback__ else None
    )
    return "".join(frames)


def call_plain(name: str, func: object) -> tuple[str, str]:
    """Return (status, detail) for one plain test callable."""
    signature = inspect.signature(func)  # type: ignore[arg-type]
    required = [
        p
        for p in signature.parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        return (
            "ERROR",
            f"{name} needs fixtures ({', '.join(p.name for p in required)}); install pytest",
        )
    try:
        func()  # type: ignore[operator]
    except AssertionError as exc:
        return "FAILED", format_failure(exc)
    except Exception as exc:
        return "ERROR", format_failure(exc)
    return "PASSED", ""


def run_unittest_case(test: unittest.TestCase) -> tuple[str, str]:
    result = unittest.TestResult()
    test.run(result)
    if result.skipped and result.wasSuccessful():
        return "SKIPPED", ""
    problems = result.failures + result.errors
    if problems:
        status = "FAILED" if result.failures else "ERROR"
        return status, "\n".join(text for _, text in problems)
    return "PASSED", ""


def flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(flatten(item))
        else:
            tests.append(item)
    return tests


def collect(module: ModuleType, label: str) -> list[tuple[str, tuple[str, str]]]:
    outcomes: list[tuple[str, tuple[str, str]]] = []
    loader = unittest.TestLoader()
    for name, obj in vars(module).items():
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj) and issubclass(obj, unittest.TestCase):
            for test in flatten(loader.loadTestsFromTestCase(obj)):
                outcomes.append(
                    (f"{label}::{name}::{test._testMethodName}", run_unittest_case(test))
                )
        elif inspect.isclass(obj) and name.startswith("Test"):
            for method_name, _ in inspect.getmembers(obj, inspect.isfunction):
                if method_name.startswith("test_"):
                    instance = obj()
                    outcomes.append(
                        (
                            f"{label}::{name}::{method_name}",
                            call_plain(method_name, getattr(instance, method_name)),
                        )
                    )
        elif inspect.isfunction(obj) and name.startswith("test_"):
            outcomes.append((f"{label}::{name}", call_plain(name, obj)))
    return outcomes


def main(argv: list[str]) -> int:
    started = time.monotonic()
    files = discover(argv)
    counts = {"PASSED": 0, "FAILED": 0, "ERROR": 0, "SKIPPED": 0}
    details: list[tuple[str, str]] = []
    for index, path in enumerate(files):
        label = path.as_posix()
        try:
            outcomes = collect(load_module(path, index), label)
        except Exception as exc:
            counts["ERROR"] += 1
            print(f"ERROR {label} (while importing)")
            details.append((label, format_failure(exc)))
            continue
        for test_id, (status, detail) in outcomes:
            counts[status] += 1
            print(f"{status} {test_id}")
            if detail:
                details.append((test_id, detail))
    for test_id, detail in details:
        print(f"\n===== {test_id} =====\n{detail.rstrip()}")
    elapsed = time.monotonic() - started
    total = sum(counts.values())
    print(
        f"\n{counts['PASSED']} passed, {counts['FAILED']} failed, {counts['ERROR']} errors, "
        f"{counts['SKIPPED']} skipped in {elapsed:.2f}s"
    )
    if total == 0:
        print("NO TESTS FOUND")
        return 5
    return 0 if counts["FAILED"] == 0 and counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
