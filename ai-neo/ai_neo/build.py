"""Build mode: plan -> approve -> generate files one at a time -> test -> fix -> honest summary."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_neo.agent import Agent
from ai_neo.context import CHARS_PER_TOKEN
from ai_neo.prompts import BUILD_FILE, BUILD_FIX, BUILD_PLAN, MODES, fill
from ai_neo.sandbox import minimal_env
from ai_neo.testrun import SuiteRun, run_test_suite
from ai_neo.textutil import clip_middle, extract_json_object, extract_single_block
from ai_neo.tools import PathEscape, Workspace
from ai_neo.types import GenStats, Message
from ai_neo.ui import UI

MAX_FILES = 8
PLAN_ATTEMPTS = 3
FILE_ATTEMPTS = 2
FIX_PROMPT_CHARS = 6000


class PlanError(ValueError):
    pass


@dataclass
class PlannedFile:
    path: str
    purpose: str


@dataclass
class BuildPlan:
    name: str
    summary: str
    files: list[PlannedFile]

    def render(self) -> str:
        lines = [f"{self.name}: {self.summary}", ""]
        lines += [f"  {f.path}  -  {f.purpose}" for f in self.files]
        return "\n".join(lines)


def is_test_path(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith("_test.py")


def _clean_relative(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise PlanError("every file needs a non-empty 'path'")
    path = raw.strip().replace("\\", "/").removeprefix("./")
    parts = path.split("/")
    if (
        path.startswith("/")
        or ":" in path
        or ".." in parts
        or any(p.startswith(".") for p in parts)
    ):
        raise PlanError(f"unsafe file path in plan: {raw!r}")
    if len(path) > 100:
        raise PlanError(f"file path too long: {raw!r}")
    return path


def parse_plan(text: str, max_files: int = MAX_FILES) -> BuildPlan:
    obj = extract_json_object(text)
    if obj is None:
        raise PlanError("the reply contained no JSON object")
    raw_files = obj.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PlanError("the JSON needs a non-empty 'files' list")
    files: list[PlannedFile] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            raise PlanError("each entry of 'files' must be an object with 'path' and 'purpose'")
        path = _clean_relative(item.get("path"))
        if path in seen:
            continue
        seen.add(path)
        files.append(PlannedFile(path, str(item.get("purpose", "")).strip() or "(no description)"))
    if len(files) > max_files:
        raise PlanError(f"too many files ({len(files)}); keep it to at most {max_files}")
    if not any(is_test_path(f.path) for f in files):
        raise PlanError("the plan must include a unittest file such as tests/test_<thing>.py")
    files.sort(key=lambda f: is_test_path(f.path))  # implementation first, tests last
    name = re.sub(r"[^a-z0-9_-]+", "-", str(obj.get("name", "")).lower()).strip("-")[:40]
    return BuildPlan(name or "project", str(obj.get("summary", "")).strip(), files)


def signature_digest(path: str, code: str) -> str:
    """Top-level signatures of a Python file (so later files can use it without its full text)."""
    if path.endswith(".py"):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        if tree is not None:
            lines = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    lines.append(f"def {node.name}({ast.unparse(node.args)})")
                elif isinstance(node, ast.ClassDef):
                    lines.append(f"class {node.name}:")
                    lines += [
                        f"    def {n.name}({ast.unparse(n.args)})"
                        for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    ]
            if lines:
                return "\n".join(lines)
    return "\n".join(code.splitlines()[:10])


@dataclass
class BuildReport:
    plan: BuildPlan | None = None
    project_dir: Path | None = None
    files_written: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    tests: SuiteRun | None = None
    fix_rounds: int = 0
    passed: bool = False
    cancelled: bool = False
    interrupted: bool = False
    notes: list[str] = field(default_factory=list)
    stats: GenStats = field(default_factory=GenStats)

    def render(self) -> str:
        """Summary built from facts (files on disk, real test results), not model claims."""
        if self.cancelled:
            return "Build cancelled before any files were written."
        lines = []
        if self.plan:
            lines.append(f"Project: {self.plan.name} - {self.plan.summary}")
        if self.project_dir:
            lines.append(f"Location: {self.project_dir}")
        lines.append(
            f"Files written ({len(self.files_written)}): {', '.join(self.files_written) or 'none'}"
        )
        if self.files_failed:
            lines.append(f"Files that could not be generated: {', '.join(self.files_failed)}")
        lines.append("")
        if self.tests is None:
            lines.append("Tests: not run.")
        elif self.passed:
            lines.append(f"Tests: PASSED ({self.tests.headline()}, runner {self.tests.runner}).")
            lines.append(
                "What works: " + (", ".join(self.tests.passed_names) or "all collected tests")
            )
        else:
            lines.append(
                f"Tests: FAILED ({self.tests.headline()}) after {self.fix_rounds} fix round(s)."
            )
            if self.tests.passed_names:
                lines.append("What works: " + ", ".join(self.tests.passed_names))
            failing = ", ".join(self.tests.failed_names) or "see the output below"
            lines.append("What does not work: " + failing)
            lines.append("Last test output:\n" + clip_middle(self.tests.output.strip(), 1500))
        lines += [f"Note: {note}" for note in self.notes]
        return "\n".join(lines)


class GitCheckpoints:
    """Opt-in `git init` plus a commit at each checkpoint. Does nothing unless enabled."""

    def __init__(self, directory: Path, enabled: bool, ui: UI) -> None:
        self.directory = directory
        self.enabled = enabled
        self.ui = ui
        self.available = enabled and shutil.which("git") is not None
        if enabled and not self.available:
            ui.tag("WARN", "git was not found on PATH, so checkpoints are skipped.")

    def commit(self, message: str) -> str | None:
        if not self.available:
            return None
        with tempfile.TemporaryDirectory(prefix="ai_neo_git_", ignore_cleanup_errors=True) as home:
            env = minimal_env(Path(home))
            env["GIT_TERMINAL_PROMPT"] = "0"
            identity = ["-c", "user.name=AI Neo", "-c", "user.email=ai-neo@localhost"]
            steps = [
                ["git", "add", "-A"],
                ["git", *identity, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message],
            ]
            if not (self.directory / ".git").exists():
                steps.insert(0, ["git", "init", "-q"])
            for argv in steps:
                self.ui.say("$ " + " ".join(argv), dim=True)
                done = subprocess.run(  # noqa: S603
                    argv,
                    cwd=self.directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if done.returncode != 0 and argv[1:2] != ["commit"]:
                    return f"git {argv[1]} failed: {done.stderr.strip()[:200]}"
        return None


def _unique_project_dir(workspace: Workspace, name: str) -> tuple[str, Path]:
    for suffix in ("", *[f"-{i}" for i in range(2, 100)]):
        candidate = f"{name}{suffix}"
        path = workspace.resolve(candidate)
        if not path.exists():
            return candidate, path
    raise PlanError("could not find an unused project folder name")


class Builder:
    def __init__(self, agent: Agent, workspace: Workspace, ui: UI, *, git: bool = False) -> None:
        self.agent = agent
        self.workspace = workspace
        self.ui = ui
        self.git = git
        self.cfg = agent.cfg
        self.system = MODES["build"].system
        self.stats = GenStats()

    def _ask_model(self, prompt: str, think: bool) -> tuple[str, bool]:
        result = self.agent.generate(
            [Message("system", self.system), Message("user", prompt)], "build", think
        )
        self.stats += result.stats
        if result.interrupted:
            raise KeyboardInterrupt
        return result.content, result.done_reason == "length"

    # -- stage 1 and 2: plan and approval ------------------------------------------------------

    def make_plan(self, request: str, think: bool, feedback: str = "") -> BuildPlan:
        note = f"\n\nAdjust the plan using this feedback: {feedback}" if feedback else ""
        problem = ""
        for _ in range(PLAN_ATTEMPTS):
            prompt = fill(BUILD_PLAN, request=request, max_files=str(MAX_FILES), feedback=note)
            if problem:
                prompt += (
                    f"\n\nYour previous reply was rejected: {problem}. Reply with ONLY the JSON."
                )
            text, _ = self._ask_model(prompt, think)
            try:
                return parse_plan(text)
            except PlanError as exc:
                problem = str(exc)
                self.ui.tag("WARN", f"plan rejected: {problem}")
        raise PlanError(
            f"The model did not produce a usable plan after {PLAN_ATTEMPTS} tries ({problem})."
        )

    def approve_plan(
        self, request: str, plan: BuildPlan, think: bool, auto: bool
    ) -> BuildPlan | None:
        while True:
            self.ui.preview("Build plan", plan.render())
            if auto:
                return plan
            choice = self.ui.choose(
                "Approve this plan? [Y]es / [R]evise / [N]o:", "yrn", default="y"
            )
            if choice == "y":
                return plan
            if choice == "n":
                return None
            feedback = self.ui.ask("What should change?")
            if feedback:
                plan = self.make_plan(request, think, feedback)

    # -- stage 3: generate files ---------------------------------------------------------------

    def generate_file(
        self, plan: BuildPlan, planned: PlannedFile, written: dict[str, str], think: bool
    ) -> str | None:
        file_list = "\n".join(f"- {f.path}: {f.purpose}" for f in plan.files)
        digest = (
            "\n\n".join(
                f"# {path}\n{signature_digest(path, code)}" for path, code in written.items()
            )
            or "(none yet)"
        )
        prompt = fill(
            BUILD_FILE,
            name=plan.name,
            summary=plan.summary,
            file_list=file_list,
            digest=digest,
            path=planned.path,
            purpose=planned.purpose,
        )
        for attempt in range(FILE_ATTEMPTS):
            text, truncated = self._ask_model(prompt, think)
            code = extract_single_block(text)
            if code is not None and not truncated:
                return code
            reason = "was cut off (too long)" if truncated else "had no fenced code block"
            self.ui.tag("WARN", f"{planned.path}: the reply {reason}")
            prompt += "\n\nKeep it short and reply with the complete file in ONE fenced code block."
            if attempt + 1 == FILE_ATTEMPTS:
                break
        return None

    # -- stage 4 and 5: test and fix -----------------------------------------------------------

    def run_tests(self, project: Path) -> SuiteRun:
        with tempfile.TemporaryDirectory(
            prefix="ai_neo_build_", ignore_cleanup_errors=True
        ) as home:
            return run_test_suite(
                project,
                [],
                runner=self.cfg.tools.test_runner,
                timeout=self.cfg.tools.test_timeout,
                max_bytes=self.cfg.tools.max_output_bytes,
                memory_limit_mb=self.cfg.tools.memory_limit_mb,
                home=Path(home),
            )

    def _fix_context(self, project: Path, rel_paths: list[str], output: str) -> str:
        mentioned = [p for p in rel_paths if p in output or Path(p).name in output]
        ordered = mentioned + [p for p in rel_paths if is_test_path(p) and p not in mentioned]
        ordered += [p for p in rel_paths if p not in ordered]
        budget = min(FIX_PROMPT_CHARS, int(self.cfg.ollama.num_ctx * CHARS_PER_TOKEN * 0.35))
        parts, used = [], 0
        for rel in ordered:
            body = (project / rel).read_text(encoding="utf-8", errors="replace")
            chunk = f"FILE: {rel}\n```\n{body}\n```\n"
            if used + len(chunk) > budget and parts:
                parts.append(f"(file {rel} omitted to save space)\n")
                continue
            parts.append(chunk)
            used += len(chunk)
        return "\n".join(parts)

    def apply_fix(
        self, project: Path, plan: BuildPlan, reply: str, rel_paths: list[str]
    ) -> list[str]:
        allowed = set(rel_paths) | {f.path for f in plan.files}
        changed = []
        pattern = re.compile(r"FILE:\s*`?([^\s`]+)`?\s*\r?\n```[\w+-]*\r?\n(.*?)```", re.DOTALL)
        for rel, body in pattern.findall(reply):
            try:
                rel_clean = _clean_relative(rel)
            except PlanError:
                continue
            if rel_clean not in allowed:
                continue
            target = project / rel_clean
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body.strip("\n") + "\n", encoding="utf-8", newline="\n")
            changed.append(rel_clean)
        return changed

    # -- the whole flow ------------------------------------------------------------------------

    def build(
        self, request: str, *, think: bool, auto_approve: bool, fix_rounds: int
    ) -> BuildReport:
        report = BuildReport()
        try:
            self._build(report, request, think, auto_approve, fix_rounds)
        except KeyboardInterrupt:
            report.interrupted = True
            report.notes.append("Interrupted by the user; the project may be incomplete.")
        except PlanError as exc:
            report.notes.append(str(exc))
        report.stats += self.stats
        return report

    def _build(
        self, report: BuildReport, request: str, think: bool, auto: bool, fix_rounds: int
    ) -> None:
        self.ui.say("Step 1 of 6: asking the model for a short plan...")
        plan = self.make_plan(request, think)
        self.ui.say("Step 2 of 6: waiting for your approval of the plan.")
        approved = self.approve_plan(request, plan, think, auto)
        if approved is None:
            report.cancelled = True
            return
        plan = report.plan = approved
        folder, project = _unique_project_dir(self.workspace, plan.name)
        project.mkdir(parents=True)
        report.project_dir = project
        git = GitCheckpoints(project, self.git, self.ui)

        self.ui.say(f"Step 3 of 6: writing {len(plan.files)} file(s) into {project}")
        written: dict[str, str] = {}
        for index, planned in enumerate(plan.files, 1):
            self.ui.say(f"  [{index}/{len(plan.files)}] {planned.path}")
            code = self.generate_file(plan, planned, written, think)
            if code is None:
                report.files_failed.append(planned.path)
                continue
            try:
                target = self.workspace.resolve(f"{folder}/{planned.path}")
            except PathEscape as exc:
                report.files_failed.append(planned.path)
                report.notes.append(f"blocked path {planned.path}: {exc}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8", newline="\n")
            written[planned.path] = code
            report.files_written.append(planned.path)
        self._note(git.commit("AI Neo: initial files"), report)

        self.ui.say("Step 4 of 6: running the tests...")
        run = report.tests = self.run_tests(project)
        self._show_tests(run)
        rounds = 0
        while not run.passed and rounds < fix_rounds and report.files_written:
            rounds += 1
            self.ui.say(f"Step 5 of 6: fix round {rounds} of {fix_rounds}...")
            rel_paths = sorted(report.files_written)
            prompt = fill(
                BUILD_FIX,
                name=plan.name,
                output=clip_middle(run.output.strip(), 3000),
                files=self._fix_context(project, rel_paths, run.output),
            )
            reply, _ = self._ask_model(prompt, think)
            changed = self.apply_fix(project, plan, reply, rel_paths)
            if not changed:
                self.ui.tag("WARN", "the model gave no usable file changes this round")
                continue
            self.ui.say(f"  rewrote: {', '.join(changed)}")
            run = report.tests = self.run_tests(project)
            self._show_tests(run)
            self._note(git.commit(f"AI Neo: fix round {rounds}"), report)
        report.fix_rounds = rounds
        report.passed = run.passed and not report.files_failed
        if run.passed and report.files_failed:
            report.notes.append("Tests pass, but some planned files were never generated.")
        self._note(git.commit("AI Neo: final"), report)
        self.ui.say("Step 6 of 6: summary below.")

    def _show_tests(self, run: SuiteRun) -> None:
        self.ui.tag("OK" if run.passed else "FAIL", f"tests: {run.headline()} ({run.runner})")

    @staticmethod
    def _note(message: str | None, report: BuildReport) -> None:
        if message:
            report.notes.append(message)
