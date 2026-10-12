from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from ai_neo.config import ToolCfg, default_config
from ai_neo.tools import PathEscape, ToolRegistry, Workspace
from ai_neo.types import ToolCall


def cfg(**kw: object) -> ToolCfg:
    return dataclasses.replace(default_config().tools, **kw)  # type: ignore[arg-type]


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    root.mkdir()
    return Workspace(root)


def registry(
    ws: Workspace, *, yes: bool = True, confirm: object = None, **kw: object
) -> ToolRegistry:
    return ToolRegistry(ws, cfg(**kw), confirm=confirm, auto_yes=yes)  # type: ignore[arg-type]


def call(reg: ToolRegistry, name: str, **args: object):
    return reg.execute(ToolCall(name, args))


# -- path guard --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "../secret.txt",
        "a/../../secret.txt",
        "..\\secret.txt",
        "sub\\..\\..\\x",
        "/etc/passwd",
        "\\Windows\\System32",
        "C:\\Windows\\win.ini",
        "C:win.ini",
        "//server/share/x",
        "",
        "   ",
        "a\x00b",
        "..",
    ],
)
def test_workspace_rejects_traversal_and_absolute(ws: Workspace, bad: str) -> None:
    with pytest.raises(PathEscape):
        ws.resolve(bad)


def test_workspace_accepts_normal_paths(ws: Workspace) -> None:
    assert ws.resolve("a/b/c.py") == ws.root / "a" / "b" / "c.py"
    assert ws.resolve("./x.txt") == ws.root / "x.txt"
    assert ws.resolve(".") == ws.root


def make_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks on this system")


def test_symlinked_directory_escape_is_blocked(ws: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")
    make_symlink(ws.root / "link", outside)
    reg = registry(ws)
    result = call(reg, "read_file", path="link/secret.txt")
    assert not result.ok and result.meta.get("blocked")
    assert "top secret" not in result.output
    assert not call(reg, "write_file", path="link/new.txt", content="x").ok
    assert not (outside / "new.txt").exists()
    assert not call(reg, "list_dir", path="link").ok


def test_symlinked_file_and_dangling_symlink_escape_blocked(ws: Workspace, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    make_symlink(ws.root / "peek.txt", secret)
    make_symlink(ws.root / "plant.txt", tmp_path / "not_yet_there.txt")
    reg = registry(ws)
    assert not call(reg, "read_file", path="peek.txt").ok
    assert not call(reg, "write_file", path="plant.txt", content="pwn").ok
    assert not (tmp_path / "not_yet_there.txt").exists()


def test_symlink_inside_workspace_is_allowed(ws: Workspace) -> None:
    (ws.root / "real").mkdir()
    (ws.root / "real" / "f.txt").write_text("fine", encoding="utf-8")
    make_symlink(ws.root / "alias", ws.root / "real")
    assert call(registry(ws), "read_file", path="alias/f.txt").output == "fine"


# -- file tools --------------------------------------------------------------------------------


def test_write_read_list_roundtrip(ws: Workspace) -> None:
    reg = registry(ws)
    assert call(reg, "write_file", path="pkg/mod.py", content="x = 1\n").ok
    assert call(reg, "read_file", path="pkg/mod.py").output == "x = 1\n"
    listing = call(reg, "list_dir", path=".").output
    assert "pkg/" in listing
    assert "mod.py" in call(reg, "list_dir", path="pkg").output
    assert call(reg, "list_dir").ok  # default path


def test_read_errors_are_results_not_exceptions(ws: Workspace) -> None:
    reg = registry(ws)
    assert not call(reg, "read_file", path="missing.txt").ok
    (ws.root / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert "binary" in call(reg, "read_file", path="bin.dat").output
    (ws.root / "d").mkdir()
    assert not call(reg, "write_file", path="d", content="x").ok


def test_read_is_truncated(ws: Workspace) -> None:
    (ws.root / "big.txt").write_text("x" * 5000, encoding="utf-8")
    out = call(registry(ws, max_output_bytes=1000), "read_file", path="big.txt").output
    assert "truncated" in out and len(out) < 1200


def test_write_requires_confirmation(ws: Workspace) -> None:
    seen: list[tuple[str, str]] = []

    def decline(title: str, preview: str) -> bool:
        seen.append((title, preview))
        return False

    reg = registry(ws, yes=False, confirm=decline)
    result = call(reg, "write_file", path="a.txt", content="hello")
    assert not result.ok and result.meta["declined"]
    assert not (ws.root / "a.txt").exists()
    assert seen[0][0] == "write_file" and "a.txt" in seen[0][1] and "hello" in seen[0][1]
    accept = registry(ws, yes=False, confirm=lambda t, p: True)
    assert call(accept, "write_file", path="a.txt", content="hello").ok


def test_no_confirmer_and_no_yes_means_denied(ws: Workspace) -> None:
    reg = registry(ws, yes=False)
    assert not call(reg, "write_file", path="a.txt", content="x").ok
    assert not call(reg, "run_python", code="print(1)").ok
    assert call(reg, "calculator", expression="1+1").ok  # no confirmation needed


def test_overwrite_preview_says_so(ws: Workspace) -> None:
    (ws.root / "a.txt").write_text("old", encoding="utf-8")
    previews: list[str] = []
    reg = registry(ws, yes=False, confirm=lambda t, p: previews.append(p) or True)
    call(reg, "write_file", path="a.txt", content="new")
    assert "OVERWRITE" in previews[0]


# -- tool-call hygiene ---------------------------------------------------------------------------


def test_malformed_calls_return_errors(ws: Workspace) -> None:
    reg = registry(ws)
    cases = [
        ToolCall("nope", {}),
        ToolCall("calculator", "{not json"),
        ToolCall("calculator", "[1, 2]"),
        ToolCall("calculator", {}),
        ToolCall("calculator", {"expression": 5}),
        ToolCall("calculator", {"expression": "1+1", "extra": 1}),
        ToolCall("run_python", {"code": ["print(1)"]}),
        ToolCall("", None),
        ToolCall("read_file", None),
    ]
    for item in cases:
        result = reg.execute(item)
        assert not result.ok and result.output.startswith("ERROR"), item


def test_json_string_arguments_and_numeric_strings_are_accepted(ws: Workspace) -> None:
    reg = registry(ws)
    assert reg.execute(ToolCall("calculator", json.dumps({"expression": "6*7"}))).output.endswith(
        "42"
    )
    assert call(reg, "run_python", code="print(1)", timeout="5").ok


def test_specs_are_ollama_shaped(ws: Workspace) -> None:
    specs = registry(ws).specs(["calculator", "write_file", "ghost"])
    assert [s["function"]["name"] for s in specs] == ["calculator", "write_file"]
    assert specs[0]["function"]["parameters"]["required"] == ["expression"]
    assert specs[0]["type"] == "function"


# -- calculator tool ---------------------------------------------------------------------------


def test_calculator_tool(ws: Workspace) -> None:
    reg = registry(ws)
    assert call(reg, "calculator", expression="2+2*3").output == "2+2*3 = 8"
    bad = call(reg, "calculator", expression="__import__('os')")
    assert not bad.ok and "ERROR" in bad.output


# -- run_python tool ---------------------------------------------------------------------------


def test_run_python_success_and_failure(ws: Workspace) -> None:
    reg = registry(ws)
    ok = call(reg, "run_python", code="print(6*7)")
    assert ok.ok and "42" in ok.output and ok.meta["returncode"] == 0
    bad = call(reg, "run_python", code="raise RuntimeError('nope')")
    assert not bad.ok and "RuntimeError: nope" in bad.output


def test_run_python_times_out_on_infinite_loop(ws: Workspace) -> None:
    result = call(registry(ws, run_python_timeout=1.0), "run_python", code="while True: pass")
    assert not result.ok and result.meta["timed_out"] and "TIMED OUT" in result.output


def test_model_cannot_raise_the_timeout_ceiling(ws: Workspace) -> None:
    reg = registry(ws, run_python_timeout=1.0)
    result = call(reg, "run_python", code="while True: pass", timeout=3600)
    assert result.meta["timed_out"] and result.meta["seconds"] < 5


def test_run_python_caps_huge_output(ws: Workspace) -> None:
    result = call(registry(ws, max_output_bytes=1000), "run_python", code="print('y' * 3_000_000)")
    assert result.meta["truncated"] and len(result.output) < 1400


def test_run_python_runs_outside_the_workspace(ws: Workspace) -> None:
    result = call(registry(ws), "run_python", code="import os; print(os.getcwd())")
    assert str(ws.root) not in result.output


@pytest.mark.skipif(os.name == "nt", reason="rlimits are POSIX only")
def test_run_python_memory_limit(ws: Workspace) -> None:
    result = call(
        registry(ws, memory_limit_mb=256), "run_python", code="x = bytearray(2 * 1024**3)"
    )
    assert not result.ok and "MemoryError" in result.output


# -- run_tests tool ----------------------------------------------------------------------------


@pytest.mark.parametrize("runner", ["builtin", "pytest"])
def test_run_tests_pass_and_fail(ws: Workspace, runner: str) -> None:
    project = ws.root / "proj"
    project.mkdir()
    (project / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (project / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
        "def test_plain():\n    assert add(1, 1) == 2\n",
        encoding="utf-8",
    )
    reg = registry(ws, test_runner=runner)
    good = call(reg, "run_tests", path="proj")
    assert good.ok and good.meta["runner"] == runner, good.output
    assert "PASSED" in good.output
    (project / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    bad = call(reg, "run_tests", path="proj")
    assert not bad.ok and "FAILED" in bad.output


@pytest.mark.parametrize("runner", ["builtin", "pytest"])
def test_run_tests_reports_no_tests(ws: Workspace, runner: str) -> None:
    (ws.root / "empty").mkdir()
    result = call(registry(ws, test_runner=runner), "run_tests", path="empty")
    assert not result.ok and result.meta["no_tests"]


def test_run_tests_cannot_leave_workspace(ws: Workspace) -> None:
    result = call(registry(ws), "run_tests", path="../")
    assert not result.ok and result.meta.get("blocked")
