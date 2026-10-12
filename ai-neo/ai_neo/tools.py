"""Tools the model may call: calculator, run_python, file tools and run_tests.

`ToolRegistry.execute` never raises: every problem becomes a `ToolResult(ok=False, ...)` that is
sent back to the model so it can retry.
"""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ai_neo.calculator import CalcError, format_number, safe_eval
from ai_neo.config import ToolCfg
from ai_neo.sandbox import ProcResult, run_python_process
from ai_neo.testrun import run_test_suite
from ai_neo.types import ToolCall

Confirmer = Callable[[str, str], bool]

MAX_WRITE_BYTES = 1_000_000
MAX_ARG_CHARS = 200_000
MAX_LIST_ENTRIES = 200


class PathEscape(ValueError):
    """A path would leave the workspace (absolute path, `..`, or a symlink pointing outside)."""


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)


def error(message: str, **meta: Any) -> ToolResult:
    return ToolResult(False, f"ERROR: {message}", meta)


class Workspace:
    """The only directory tree the file tools may touch."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def resolve(self, relative: str) -> Path:
        """Resolve `relative` inside the root, following symlinks, or raise PathEscape.

        Every component that exists is resolved, so a symlink that points outside the workspace
        is caught even when the final file does not exist yet. (A symlink swapped in after this
        check can still win a race; see the README safety notes.)
        """
        if not isinstance(relative, str) or not relative.strip():
            raise PathEscape("path must be a non-empty string")
        if "\x00" in relative:
            raise PathEscape("path contains a NUL byte")
        for flavour in (PurePosixPath(relative), PureWindowsPath(relative)):
            if flavour.is_absolute() or flavour.anchor or getattr(flavour, "drive", ""):
                raise PathEscape(f"absolute paths are not allowed: {relative!r}")
        if ".." in re.split(r"[\\/]+", relative):
            raise PathEscape(f"'..' is not allowed in paths: {relative!r}")
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise PathEscape(f"{relative!r} resolves outside the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, dict[str, str]]
    required: tuple[str, ...]
    run: Callable[..., ToolResult]
    needs_confirm: bool = False
    preview: Callable[[dict[str, Any]], str] | None = None

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                },
            },
        }


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return True


def _coerce_numbers(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(args)
    for key, value in args.items():
        kind = tool.properties.get(key, {}).get("type")
        if kind in ("number", "integer") and isinstance(value, str):
            with contextlib.suppress(ValueError):
                fixed[key] = int(value) if kind == "integer" else float(value)
    return fixed


def _validate(tool: Tool, args: dict[str, Any]) -> list[str]:
    problems = [f"missing required argument '{k}'" for k in tool.required if k not in args]
    for key, value in args.items():
        spec = tool.properties.get(key)
        if spec is None:
            problems.append(f"unknown argument '{key}'")
        elif not _type_ok(spec["type"], value):
            problems.append(f"argument '{key}' must be of type {spec['type']}")
        elif isinstance(value, str) and len(value) > MAX_ARG_CHARS:
            problems.append(f"argument '{key}' is too long")
    return problems


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"arguments are not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise ValueError(f"arguments must be a JSON object, got {type(raw).__name__}")
    return raw


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


def format_process(proc: ProcResult) -> str:
    if proc.timed_out:
        header = f"TIMED OUT after {proc.seconds:.0f}s (process killed)"
    else:
        header = f"exit code {proc.returncode}"
    body = proc.output.rstrip()
    return f"{header}\n--- output ---\n{body}" if body else f"{header}\n(no output)"


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        cfg: ToolCfg,
        *,
        confirm: Confirmer | None = None,
        auto_yes: bool = False,
    ) -> None:
        self.workspace = workspace
        self.cfg = cfg
        self._confirm = confirm
        self._auto_yes = auto_yes
        tools = [
            Tool(
                "calculator",
                "Evaluate an exact arithmetic expression (+ - * / // % **, parentheses, sqrt, "
                "log, sin, factorial, ...). Use this instead of mental math.",
                {"expression": {"type": "string", "description": "e.g. 12345*6789 + 2**10"}},
                ("expression",),
                self._calculator,
            ),
            Tool(
                "run_python",
                "Run a Python 3 script in a temporary folder and return its output. Use print(). "
                "Standard library only; no input(); it is killed after the timeout.",
                {
                    "code": {"type": "string", "description": "complete Python source"},
                    "timeout": {"type": "number", "description": "seconds (optional, capped)"},
                },
                ("code",),
                self._run_python,
                needs_confirm=True,
                preview=self._preview_run_python,
            ),
            Tool(
                "read_file",
                "Read a text file inside the workspace.",
                {"path": {"type": "string", "description": "path relative to the workspace"}},
                ("path",),
                self._read_file,
            ),
            Tool(
                "write_file",
                "Create or overwrite a text file inside the workspace.",
                {
                    "path": {"type": "string", "description": "path relative to the workspace"},
                    "content": {"type": "string", "description": "complete file contents"},
                },
                ("path", "content"),
                self._write_file,
                needs_confirm=True,
                preview=self._preview_write_file,
            ),
            Tool(
                "list_dir",
                "List files and folders inside the workspace.",
                {"path": {"type": "string", "description": "folder, default '.'"}},
                (),
                self._list_dir,
            ),
            Tool(
                "run_tests",
                "Run the tests of a project folder inside the workspace and return the results.",
                {"path": {"type": "string", "description": "project folder, default '.'"}},
                (),
                self._run_tests,
                needs_confirm=True,
                preview=lambda a: (
                    f"Run the tests in '{a.get('path', '.')}' (this executes the "
                    "project's code on your computer)."
                ),
            ),
        ]
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = list(names) if names is not None else list(self._tools)
        return [self._tools[n].spec() for n in wanted if n in self._tools]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises; bad calls come back as error results."""
        if not call.name:
            return error(
                "malformed tool call: no tool name. Call one of "
                f"{', '.join(self._tools)} with JSON arguments."
            )
        tool = self._tools.get(call.name)
        if tool is None:
            return error(f"unknown tool {call.name!r}. Available tools: {', '.join(self._tools)}.")
        try:
            args = _coerce_numbers(tool, _parse_arguments(call.arguments))
            problems = _validate(tool, args)
            if problems:
                usage = ", ".join(f"{k}: {v['type']}" for k, v in tool.properties.items())
                return error(
                    f"invalid arguments for {tool.name}: {'; '.join(problems)}. "
                    f"Expected {{{usage}}}."
                )
            if tool.needs_confirm and not self._approved(tool, args):
                return error("the user declined this action.", declined=True)
            return tool.run(**args)
        except PathEscape as exc:
            return error(f"blocked: {exc}", blocked=True)
        except ValueError as exc:
            return error(str(exc))
        except Exception as exc:  # deliberate: a tool bug must come back as an error, not crash
            return error(f"{type(exc).__name__}: {exc}")

    def _approved(self, tool: Tool, args: dict[str, Any]) -> bool:
        if self._auto_yes:
            return True
        if self._confirm is None:
            return False
        preview = tool.preview(args) if tool.preview else json.dumps(args)[:500]
        return self._confirm(tool.name, preview)

    # -- tool implementations ----------------------------------------------------------------

    def _calculator(self, expression: str) -> ToolResult:
        try:
            return ToolResult(
                True, f"{expression.strip()} = {format_number(safe_eval(expression))}"
            )
        except CalcError as exc:
            return error(str(exc))

    def _effective_timeout(self, requested: float | None) -> float:
        ceiling = self.cfg.run_python_timeout
        if requested is None or requested <= 0:
            return ceiling
        return min(float(requested), ceiling)

    def _run_python(self, code: str, timeout: float | None = None) -> ToolResult:
        limit = self._effective_timeout(timeout)
        with tempfile.TemporaryDirectory(prefix="ai_neo_run_", ignore_cleanup_errors=True) as tmp:
            directory = Path(tmp)
            (directory / "main.py").write_text(code, encoding="utf-8")
            proc = run_python_process(
                ["main.py"],
                cwd=directory,
                timeout=limit,
                max_bytes=self.cfg.max_output_bytes,
                memory_limit_mb=self.cfg.memory_limit_mb,
            )
        ok = proc.returncode == 0 and not proc.timed_out
        meta = {
            "returncode": proc.returncode,
            "timed_out": proc.timed_out,
            "truncated": proc.truncated,
            "seconds": round(proc.seconds, 2),
        }
        return ToolResult(ok, format_process(proc), meta)

    def _preview_run_python(self, args: dict[str, Any]) -> str:
        limit = self._effective_timeout(args.get("timeout"))
        code = _clip_lines(str(args.get("code", "")), 40)
        return f"Run this Python code (timeout {limit:g}s) in a temporary folder:\n{code}"

    def _read_file(self, path: str) -> ToolResult:
        target = self.workspace.resolve(path)
        if not target.is_file():
            return error(f"{path!r} is not a file in the workspace.")
        limit = self.cfg.max_output_bytes
        data = target.read_bytes()[: limit + 1]
        if b"\x00" in data[:8000]:
            return error(f"{path!r} looks like a binary file.")
        text = data.decode("utf-8", errors="replace")
        return ToolResult(True, _clip(text, limit), {"path": self.workspace.relative(target)})

    def _write_file(self, path: str, content: str) -> ToolResult:
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            return error(f"content is larger than {MAX_WRITE_BYTES} bytes.")
        target = self.workspace.resolve(path)
        if target == self.workspace.root or target.is_dir():
            return error(f"{path!r} is a directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        relative = self.workspace.relative(target)
        return ToolResult(True, f"wrote {len(data)} bytes to {relative}", {"path": relative})

    def _preview_write_file(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        try:
            existing = self.workspace.resolve(path).exists()
        except PathEscape as exc:
            return f"BLOCKED: {exc}"
        action = "OVERWRITE existing file" if existing else "Create new file"
        return f"{action} {path}\n{_clip_lines(str(args.get('content', '')), 40)}"

    def _list_dir(self, path: str = ".") -> ToolResult:
        target = self.workspace.resolve(path)
        if not target.is_dir():
            return error(f"{path!r} is not a folder in the workspace.")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:MAX_LIST_ENTRIES]:
            if entry.is_symlink():
                lines.append(f"{entry.name}  (symlink)")
            elif entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"... and {len(entries) - MAX_LIST_ENTRIES} more")
        return ToolResult(True, "\n".join(lines) or "(empty folder)")

    def _run_tests(self, path: str = ".") -> ToolResult:
        target = self.workspace.resolve(path)
        if not target.is_dir():
            return error(f"{path!r} is not a folder in the workspace.")
        with tempfile.TemporaryDirectory(prefix="ai_neo_test_", ignore_cleanup_errors=True) as home:
            run = run_test_suite(
                target,
                [],
                runner=self.cfg.test_runner,
                timeout=self.cfg.test_timeout,
                max_bytes=self.cfg.max_output_bytes,
                memory_limit_mb=self.cfg.memory_limit_mb,
                home=Path(home),
            )
        text = (
            f"{'PASSED' if run.passed else 'FAILED'}: {run.headline()} ({run.runner})\n{run.output}"
        )
        meta = {"passed": run.passed, "runner": run.runner, "no_tests": run.no_tests}
        return ToolResult(run.passed, text, meta)


def _clip_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    shown = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        shown += f"\n... ({len(lines) - max_lines} more lines)"
    return _clip(shown, 4000)
