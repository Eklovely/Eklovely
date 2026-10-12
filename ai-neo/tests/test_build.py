from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_neo.build import Builder, PlanError, parse_plan, signature_digest
from ai_neo.tools import Workspace
from tests.fake_ollama import FakeOllama, text
from tests.helpers import ScriptedUI, make_agent

PLAN = {
    "name": "Calc Tool",
    "summary": "A tiny calculator module.",
    "files": [
        {"path": "tests/test_calc.py", "purpose": "unit tests"},
        {"path": "calc.py", "purpose": "add function"},
    ],
}
BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TESTS = (
    "import unittest\nfrom calc import add\n\n\n"
    "class AddTests(unittest.TestCase):\n"
    "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n\n"
    "if __name__ == '__main__':\n    unittest.main()\n"
)


def plan_reply(plan: dict | None = None) -> str:
    return "Here is the plan:\n```json\n" + json.dumps(plan or PLAN) + "\n```"


def block(code: str) -> str:
    return f"```python\n{code}```"


# -- plan parsing --------------------------------------------------------------------------------


def test_parse_plan_sorts_tests_last_and_cleans_name() -> None:
    plan = parse_plan(plan_reply())
    assert plan.name == "calc-tool"
    assert [f.path for f in plan.files] == ["calc.py", "tests/test_calc.py"]


@pytest.mark.parametrize(
    "files",
    [
        [{"path": "calc.py", "purpose": "x"}],  # no test file
        [{"path": "../evil.py", "purpose": "x"}, {"path": "test_a.py", "purpose": "t"}],
        [{"path": "/etc/passwd", "purpose": "x"}, {"path": "test_a.py", "purpose": "t"}],
        [{"path": "C:\\x.py", "purpose": "x"}, {"path": "test_a.py", "purpose": "t"}],
        [{"path": ".git/config", "purpose": "x"}, {"path": "test_a.py", "purpose": "t"}],
        [{"path": f"f{i}.py", "purpose": "x"} for i in range(9)] + [{"path": "test_a.py"}],
        [],
        ["calc.py"],
    ],
)
def test_parse_plan_rejects_bad_plans(files: list) -> None:
    with pytest.raises(PlanError):
        parse_plan(json.dumps({"name": "p", "summary": "s", "files": files}))


def test_parse_plan_rejects_non_json() -> None:
    with pytest.raises(PlanError, match="no JSON"):
        parse_plan("I think we should build a calculator.")


def test_signature_digest() -> None:
    code = (
        "import os\n\ndef add(a: int, b: int = 2) -> int:\n    return a + b\n\n"
        "class C:\n    def m(self, x): ...\n"
    )
    digest = signature_digest("m.py", code)
    assert (
        "def add(a: int, b: int=2)" in digest
        and "class C:" in digest
        and "def m(self, x)" in digest
    )
    assert signature_digest("notes.txt", "a\nb") == "a\nb"
    assert signature_digest("broken.py", "def (:\n") == "def (:"


# -- full flow -----------------------------------------------------------------------------------


def make_builder(server: FakeOllama, tmp_path: Path, ui: ScriptedUI | None = None, **kw: object):
    agent, ui, cfg = make_agent(server, tmp_path, ui=ui, **kw)
    workspace = Workspace(cfg.workspace)
    return Builder(agent, workspace, ui), ui, cfg, workspace


@pytest.mark.parametrize("runner", ["builtin", "pytest"])
def test_build_generates_fixes_and_passes(
    fake_ollama: FakeOllama, tmp_path: Path, runner: str
) -> None:
    fake_ollama.queue(
        text(plan_reply()),
        text(block(BUGGY)),
        text(block(TESTS)),
        text("The operator is wrong.\nFILE: calc.py\n" + block(FIXED)),
    )
    builder, ui, cfg, _ = make_builder(fake_ollama, tmp_path, **{"tools.test_runner": runner})
    report = builder.build("a calculator", think=False, auto_approve=True, fix_rounds=3)
    assert report.passed and report.fix_rounds == 1
    assert report.files_written == ["calc.py", "tests/test_calc.py"]
    project = cfg.workspace / "calc-tool"
    assert report.project_dir == project
    assert (project / "calc.py").read_text(encoding="utf-8") == FIXED
    assert "Tests: PASSED" in report.render() and "What works:" in report.render()
    # the test file prompt only saw the *signature* of calc.py, not its body
    test_prompt = fake_ollama.chat_payloads()[2]["messages"][-1]["content"]
    assert "def add(a, b)" in test_prompt and "return a - b" not in test_prompt
    fix_prompt = fake_ollama.chat_payloads()[3]["messages"][-1]["content"]
    assert "FILE: calc.py" in fix_prompt and "return a - b" in fix_prompt


def test_build_reports_failure_honestly(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        text(plan_reply()),
        text(block(BUGGY)),
        text(block(TESTS)),
        text("no idea"),
        text("still no idea"),
    )
    builder, ui, _, _ = make_builder(fake_ollama, tmp_path)
    report = builder.build("calc", think=False, auto_approve=True, fix_rounds=2)
    assert not report.passed and report.fix_rounds == 2
    rendered = report.render()
    assert "Tests: FAILED" in rendered and "What does not work" in rendered
    assert "no usable file changes" in ui.text


def test_plan_can_be_revised_then_approved(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    second = dict(PLAN, name="calc-v2")
    fake_ollama.queue(
        text(plan_reply()), text(plan_reply(second)), text(block(FIXED)), text(block(TESTS))
    )
    ui = ScriptedUI(["r", "please use a different name", "y"])
    builder, ui, cfg, _ = make_builder(fake_ollama, tmp_path, ui=ui)
    report = builder.build("calc", think=False, auto_approve=False, fix_rounds=1)
    assert report.passed and report.plan is not None and report.plan.name == "calc-v2"
    assert (
        "please use a different name" in fake_ollama.chat_payloads()[1]["messages"][-1]["content"]
    )


def test_declined_plan_writes_nothing(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text(plan_reply()))
    ui = ScriptedUI(["n"])
    builder, ui, cfg, workspace = make_builder(fake_ollama, tmp_path, ui=ui)
    report = builder.build("calc", think=False, auto_approve=False, fix_rounds=1)
    assert report.cancelled and "cancelled" in report.render()
    assert list(workspace.root.iterdir()) == []


def test_bad_plans_are_retried_then_give_up(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    evil = dict(PLAN, files=[{"path": "../evil.py", "purpose": "x"}, {"path": "test_x.py"}])
    fake_ollama.queue(text("no json"), text(plan_reply(evil)), text(plan_reply(evil)))
    builder, ui, _, workspace = make_builder(fake_ollama, tmp_path)
    report = builder.build("calc", think=False, auto_approve=True, fix_rounds=1)
    assert report.plan is None and "did not produce a usable plan" in report.render()
    assert not (workspace.root.parent / "evil.py").exists()
    assert "plan rejected" in ui.text


def test_file_reply_without_fence_is_retried(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        text(plan_reply()),
        text("def add(a, b): return a + b"),
        text(block(FIXED)),
        text(block(TESTS)),
    )
    builder, ui, _, _ = make_builder(fake_ollama, tmp_path)
    report = builder.build("calc", think=False, auto_approve=True, fix_rounds=0)
    assert report.passed
    assert "Keep it short" in fake_ollama.chat_payloads()[2]["messages"][-1]["content"]


def test_ungenerated_file_is_listed_and_blocks_success(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(text(plan_reply()), text(block(FIXED)), text("nope"), text("nope again"))
    builder, ui, _, _ = make_builder(fake_ollama, tmp_path)
    report = builder.build("calc", think=False, auto_approve=True, fix_rounds=0)
    assert report.files_failed == ["tests/test_calc.py"] and not report.passed
    assert "could not be generated" in report.render()


def test_fix_cannot_write_outside_the_plan(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        text(plan_reply()),
        text(block(BUGGY)),
        text(block(TESTS)),
        text(
            "FILE: ../escape.py\n" + block("x = 1\n") + "\nFILE: surprise.py\n" + block("y = 2\n")
        ),
    )
    builder, ui, cfg, workspace = make_builder(fake_ollama, tmp_path)
    report = builder.build("calc", think=False, auto_approve=True, fix_rounds=1)
    assert not report.passed
    assert not (workspace.root / "escape.py").exists()
    assert not (workspace.root / "calc-tool" / "surprise.py").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_checkpoints_are_opt_in(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text(plan_reply()), text(block(FIXED)), text(block(TESTS)))
    agent, ui, cfg = make_agent(fake_ollama, tmp_path)
    builder = Builder(agent, Workspace(cfg.workspace), ui, git=True)
    report = builder.build("calc", think=False, auto_approve=True, fix_rounds=0)
    assert report.passed
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=report.project_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "AI Neo: initial files" in log
    # and nothing is created without the flag
    fake_ollama.queue(text(plan_reply()), text(block(FIXED)), text(block(TESTS)))
    plain = Builder(agent, Workspace(cfg.workspace), ui, git=False)
    again = plain.build("calc", think=False, auto_approve=True, fix_rounds=0)
    assert again.project_dir is not None and not (again.project_dir / ".git").exists()
    assert again.project_dir.name == "calc-tool-2"  # never overwrites an existing project
