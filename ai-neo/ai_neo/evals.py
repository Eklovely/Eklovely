"""Evaluation harness: task files, objective checkers, a runner and a pass-rate report.

Task files are JSON: {"category": "...", "tasks": [{"id", "prompt", "check": {...}}]}.
Check types: exact (string or numeric), regex, python_tests, build, rubric (for writing).
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from ai_neo.config import Config
from ai_neo.logging_store import LogStore, TaskRecord
from ai_neo.ollama_client import ModelNotFound, OllamaError, OllamaUnavailable
from ai_neo.pipeline import TaskOptions, TaskPipeline
from ai_neo.sessions import Session
from ai_neo.testrun import run_test_suite
from ai_neo.textutil import extract_python
from ai_neo.tools import PathEscape, Workspace
from ai_neo.types import GenStats
from ai_neo.ui import UI
from ai_neo.verify import run_solution

CATEGORIES = ("reasoning", "computation", "python", "writing", "build")
CHECK_TYPES = ("exact", "regex", "python_tests", "build", "rubric")
CATEGORY_MODE = {
    "reasoning": "reason",
    "computation": "compute",
    "python": "code",
    "writing": "write",
    "build": "build",
}
ANSWER_SUFFIX = "\n\nEnd your reply with a line of the form `Final answer: <answer>`."
DEFAULT_TARGET = 0.9


class SuiteError(Exception):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    check: Mapping[str, Any]


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str
    method: str  # exact_match | regex | tests | build_tests | rubric


# -- loading -------------------------------------------------------------------------------------


def _check_problems(task_id: str, category: str, check: Any) -> list[str]:
    if not isinstance(check, dict):
        return [f"{task_id}: 'check' must be an object"]
    kind = check.get("type")
    if kind not in CHECK_TYPES:
        return [f"{task_id}: check type must be one of {', '.join(CHECK_TYPES)}, got {kind!r}"]
    problems = []
    if category == "python" and kind != "python_tests":
        problems.append(f"{task_id}: python tasks need a 'python_tests' check")
    if category == "build" and kind != "build":
        problems.append(f"{task_id}: build tasks need a 'build' check")
    if category not in ("python", "build") and kind in ("python_tests", "build"):
        problems.append(f"{task_id}: a '{kind}' check does not fit category '{category}'")
    if kind == "exact" and not (check.get("answer") is not None or check.get("answers")):
        problems.append(f"{task_id}: exact checks need 'answer' or 'answers'")
    if kind == "regex":
        try:
            re.compile(str(check.get("pattern", "")))
        except re.error as exc:
            problems.append(f"{task_id}: bad regex ({exc})")
        if not check.get("pattern"):
            problems.append(f"{task_id}: regex checks need 'pattern'")
    if kind == "python_tests" and not isinstance(check.get("tests"), str):
        problems.append(f"{task_id}: python_tests checks need a 'tests' string")
    if kind == "build" and not (isinstance(check.get("tests"), dict) and check["tests"]):
        problems.append(f"{task_id}: build checks need a non-empty 'tests' object (path -> code)")
    if kind == "rubric" and not RUBRIC_KEYS & set(check):
        problems.append(
            f"{task_id}: rubric checks need at least one rule ({', '.join(RUBRIC_KEYS)})"
        )
    return problems


def load_suite(path: Path, categories: Sequence[str] | None = None) -> list[Task]:
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    if not files or not all(f.exists() for f in files):
        raise SuiteError(f"No task files found at {path}. Expected *.json files.")
    tasks: list[Task] = []
    problems: list[str] = []
    seen: set[str] = set()
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            problems.append(f"{file.name}: cannot read as JSON ({exc})")
            continue
        category = data.get("category") if isinstance(data, dict) else None
        if category not in CATEGORIES:
            problems.append(f"{file.name}: 'category' must be one of {', '.join(CATEGORIES)}")
            continue
        for item in data.get("tasks", []):
            task_id = str(item.get("id", "")).strip()
            if not task_id or not item.get("prompt"):
                problems.append(f"{file.name}: every task needs an 'id' and a 'prompt'")
                continue
            if task_id in seen:
                problems.append(f"{file.name}: duplicate task id {task_id!r}")
                continue
            seen.add(task_id)
            found = _check_problems(task_id, category, item.get("check"))
            problems.extend(found)
            if not found:
                tasks.append(Task(task_id, category, str(item["prompt"]), item["check"]))
    if problems:
        raise SuiteError("Problems in the task files:\n  " + "\n  ".join(problems))
    if categories:
        tasks = [t for t in tasks if t.category in categories]
    return tasks


# -- checkers ------------------------------------------------------------------------------------

_FINAL = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
RUBRIC_KEYS = frozenset(
    {"min_words", "max_words", "must_include", "banned", "must_match", "paragraphs", "bullets"}
)


def final_answer(text: str) -> str:
    """The text after the last `Final answer:`; otherwise the last non-empty line."""
    matches = _FINAL.findall(text)
    if matches:
        return matches[-1].strip().strip("*`_ ").strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip("*`_ ") if lines else ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("*`_\"' ").rstrip(".!").strip()).casefold()


def check_exact(answer: str, spec: Mapping[str, Any]) -> CheckResult:
    final = final_answer(answer)
    if spec.get("numeric"):
        wanted = Fraction(str(spec["answer"]).replace(",", ""))
        tolerance = Fraction(str(spec.get("tolerance", 0)))
        match = _NUMBER.search(final)
        if not match:
            return CheckResult(False, f"no number in the final answer {final!r}", "exact_match")
        got = Fraction(match.group().replace(",", ""))
        ok = abs(got - wanted) <= tolerance
        return CheckResult(ok, f"expected {wanted}, got {got}", "exact_match")
    accepted = [str(spec["answer"])] if "answer" in spec else []
    accepted += [str(a) for a in spec.get("answers", [])]
    normalized = _normalize(final)
    for candidate in accepted:
        wanted_text = _normalize(candidate)
        first_word = normalized.split(" ")[0].strip(",.;:!") if normalized else ""
        if normalized == wanted_text or (" " not in wanted_text and first_word == wanted_text):
            return CheckResult(True, f"matched {candidate!r}", "exact_match")
    return CheckResult(False, f"expected one of {accepted}, got {final!r}", "exact_match")


def check_regex(answer: str, spec: Mapping[str, Any]) -> CheckResult:
    flags = re.IGNORECASE if "i" in str(spec.get("flags", "")) else 0
    target = final_answer(answer) if spec.get("on") == "final" else answer
    ok = re.search(str(spec["pattern"]), target, flags | re.MULTILINE) is not None
    return CheckResult(ok, f"pattern {spec['pattern']!r} {'found' if ok else 'not found'}", "regex")


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)


def check_rubric(answer: str, spec: Mapping[str, Any]) -> CheckResult:
    failures: list[str] = []
    total = 0
    lowered = answer.casefold()
    words = len(_words(answer))

    def rule(ok: bool, message: str) -> None:
        nonlocal total
        total += 1
        if not ok:
            failures.append(message)

    if "min_words" in spec:
        rule(words >= spec["min_words"], f"too short ({words} words, need >= {spec['min_words']})")
    if "max_words" in spec:
        rule(words <= spec["max_words"], f"too long ({words} words, need <= {spec['max_words']})")
    for item in spec.get("must_include", []):
        options = [item] if isinstance(item, str) else list(item)
        rule(any(o.casefold() in lowered for o in options), f"missing required point {options}")
    for phrase in spec.get("banned", []):
        rule(phrase.casefold() not in lowered, f"contains banned phrase {phrase!r}")
    for pattern in spec.get("must_match", []):
        rule(
            re.search(pattern, answer.strip(), re.MULTILINE) is not None,
            f"does not match required pattern {pattern!r}",
        )
    if "paragraphs" in spec:
        count = len([p for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()])
        rule(count == spec["paragraphs"], f"{count} paragraphs, need {spec['paragraphs']}")
    if "bullets" in spec:
        rules = spec["bullets"]
        bullets = [ln for ln in answer.splitlines() if re.match(r"\s*[-*•]\s+\S", ln)]
        if "exact" in rules:
            rule(len(bullets) == rules["exact"], f"{len(bullets)} bullets, need {rules['exact']}")
        if "max_words_each" in rules:
            longest = max((len(_words(b)) for b in bullets), default=0)
            rule(longest <= rules["max_words_each"], f"a bullet has {longest} words")
    passed = not failures
    detail = f"{total - len(failures)}/{total} rubric checks passed"
    return CheckResult(passed, detail if passed else detail + ": " + "; ".join(failures), "rubric")


def check_python(code: str | None, spec: Mapping[str, Any], cfg: Config) -> CheckResult:
    if not code:
        return CheckResult(False, "no Python code block in the answer", "tests")
    attempt = run_solution(code, str(spec["tests"]), cfg)
    detail = "tests passed" if attempt.passed else _last_line(attempt.output)
    return CheckResult(attempt.passed, detail, "tests")


def check_build(project: Path | None, spec: Mapping[str, Any], cfg: Config) -> CheckResult:
    """Copy the hidden acceptance tests into the built project and run them."""
    if project is None or not project.is_dir():
        return CheckResult(False, "no project was built", "build_tests")
    workspace = Workspace(project)
    targets: list[str] = []
    for relative, code in spec["tests"].items():
        try:
            path = workspace.resolve(relative)
        except PathEscape as exc:
            return CheckResult(False, f"bad acceptance test path: {exc}", "build_tests")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(code), encoding="utf-8")
        targets.append(relative)
    with tempfile.TemporaryDirectory(prefix="ai_neo_accept_", ignore_cleanup_errors=True) as home:
        run = run_test_suite(
            project,
            targets,
            runner=cfg.tools.test_runner,
            timeout=cfg.tools.test_timeout,
            max_bytes=cfg.tools.max_output_bytes,
            memory_limit_mb=cfg.tools.memory_limit_mb,
            home=Path(home),
        )
    detail = run.headline() if run.passed else run.headline() + ": " + _last_line(run.output)
    return CheckResult(run.passed, detail, "build_tests")


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1][:200] if lines else "no output"


# -- running -------------------------------------------------------------------------------------


@dataclass
class RunOutput:
    answer: str
    stats: GenStats = field(default_factory=GenStats)
    code: str | None = None
    project_dir: Path | None = None
    error: str | None = None
    record: TaskRecord | None = None


class TaskRunner(Protocol):
    def run(self, task: Task) -> RunOutput: ...


def evaluate(task: Task, output: RunOutput, cfg: Config) -> CheckResult:
    spec = task.check
    kind = spec["type"]
    if output.error and not output.answer and kind != "build":
        return CheckResult(False, f"model error: {output.error}", "exact_match")
    if kind == "exact":
        return check_exact(output.answer, spec)
    if kind == "regex":
        return check_regex(output.answer, spec)
    if kind == "rubric":
        return check_rubric(output.answer, spec)
    if kind == "python_tests":
        return check_python(output.code or extract_python(output.answer), spec, cfg)
    return check_build(output.project_dir, spec, cfg)


class PipelineRunner:
    """Runs each task through the real pipeline in a fresh session."""

    def __init__(self, pipeline: TaskPipeline, repair_attempts: int | None = None) -> None:
        self.pipeline = pipeline
        self.repair_attempts = repair_attempts

    def run(self, task: Task) -> RunOutput:
        session = Session(mode=CATEGORY_MODE[task.category], model=self.pipeline.cfg.ollama.model)
        prompt = task.prompt
        opts = TaskOptions(log=False, repair_attempts=self.repair_attempts, auto_approve_plan=True)
        if task.check["type"] in ("exact", "regex") and not task.check.get("raw"):
            prompt += ANSWER_SUFFIX
        if task.category == "python":
            opts.tests = str(task.check["tests"])
        outcome = self.pipeline.run(session, prompt, opts)
        code = outcome.verify.final_code if outcome.verify else None
        project = outcome.build.project_dir if outcome.build else None
        return RunOutput(
            outcome.answer, outcome.stats, code, project, outcome.error, outcome.record
        )


# -- reporting -----------------------------------------------------------------------------------


@dataclass
class TaskResult:
    id: str
    category: str
    passed: bool
    method: str
    detail: str
    tok_per_s: float
    seconds: float
    answer_excerpt: str


@dataclass
class Report:
    suite: str
    model: str
    timestamp: str
    results: list[TaskResult] = field(default_factory=list)

    def category_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for category in CATEGORIES:
            rows = [r for r in self.results if r.category == category]
            if rows:
                stats[category] = _summarize(rows)
        return stats

    def overall(self) -> dict[str, Any]:
        return _summarize(self.results)

    def to_dict(
        self, baseline: Mapping[str, Any] | None = None, target: float = DEFAULT_TARGET
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "suite": self.suite,
            "model": self.model,
            "timestamp": self.timestamp,
            "categories": self.category_stats(),
            "overall": self.overall(),
            "tasks": [vars(r) for r in self.results],
        }
        if baseline is not None:
            data["baseline_comparison"] = compare_to_baseline(data, baseline, target)
        return data


def _summarize(rows: list[TaskResult]) -> dict[str, Any]:
    speeds = [r.tok_per_s for r in rows if r.tok_per_s > 0]
    passed = sum(r.passed for r in rows)
    return {
        "total": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "mean_tok_s": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
    }


def compare_to_baseline(
    current: Mapping[str, Any], baseline: Mapping[str, Any], target: float = DEFAULT_TARGET
) -> dict[str, Any]:
    """Ratio of our pass rate to the baseline's (for example Claude's), per category and overall."""

    def ratio(mine: Mapping[str, Any], base: Mapping[str, Any] | None) -> dict[str, Any]:
        base_rate = float(base["pass_rate"]) if base else 0.0
        value = round(float(mine["pass_rate"]) / base_rate, 4) if base_rate > 0 else None
        return {
            "pass_rate": mine["pass_rate"],
            "baseline_pass_rate": base_rate if base else None,
            "ratio": value,
            "meets_target": value is not None and value >= target,
        }

    base_categories = baseline.get("categories", {})
    comparison = {
        "target_ratio": target,
        "overall": ratio(current["overall"], baseline.get("overall")),
        "categories": {
            name: ratio(stats, base_categories.get(name))
            for name, stats in current["categories"].items()
        },
    }
    return comparison


def run_suite(
    tasks: Sequence[Task],
    runner: TaskRunner,
    cfg: Config,
    *,
    ui: UI,
    store: LogStore | None = None,
    suite_name: str = "",
) -> Report:
    report = Report(
        suite=suite_name,
        model=cfg.ollama.model,
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    for index, task in enumerate(tasks, 1):
        ui.heading(f"[{index}/{len(tasks)}] {task.id} ({task.category})")
        try:
            output = runner.run(task)
        except (OllamaUnavailable, ModelNotFound):
            raise
        except OllamaError as exc:
            output = RunOutput("", error=str(exc))
        result = evaluate(task, output, cfg)
        ui.tag("PASS" if result.passed else "FAIL", f"{task.id}: {result.detail}")
        report.results.append(
            TaskResult(
                id=task.id,
                category=task.category,
                passed=result.passed,
                method=result.method,
                detail=result.detail,
                tok_per_s=round(output.stats.tok_per_s, 2),
                seconds=round(output.stats.seconds, 2),
                answer_excerpt=output.answer.strip()[:300],
            )
        )
        if store is not None and output.record is not None:
            output.record.verified = {
                "method": result.method
                if result.method in ("exact_match", "tests", "build_tests")
                else None,
                "passed": result.passed,
            }
            store.append(output.record)
    return report


def render_report(data: Mapping[str, Any], ui: UI) -> None:
    rows = [
        [
            name,
            f"{s['passed']}/{s['total']}",
            f"{s['pass_rate'] * 100:.0f}%",
            f"{s['mean_tok_s']:.1f}",
        ]
        for name, s in data["categories"].items()
    ]
    overall = data["overall"]
    rows.append(
        [
            "OVERALL",
            f"{overall['passed']}/{overall['total']}",
            f"{overall['pass_rate'] * 100:.0f}%",
            f"{overall['mean_tok_s']:.1f}",
        ]
    )
    ui.table(["category", "passed", "pass rate", "mean tok/s"], rows, title="Evaluation results")
    comparison = data.get("baseline_comparison")
    if comparison:
        base_rows = []
        for name, entry in {**comparison["categories"], "OVERALL": comparison["overall"]}.items():
            ratio = "n/a" if entry["ratio"] is None else f"{entry['ratio'] * 100:.0f}%"
            ok = "yes" if entry["meets_target"] else "no"
            base_rows.append([name, ratio, ok])
        ui.table(
            ["category", "ratio vs baseline", f">= {comparison['target_ratio'] * 100:.0f}% target"],
            base_rows,
            title="Progress against the baseline",
        )


def write_report(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
