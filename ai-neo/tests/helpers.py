from __future__ import annotations

import io
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai_neo.agent import Agent
from ai_neo.config import Config, load_config
from ai_neo.ollama_client import OllamaClient
from ai_neo.tools import ToolRegistry, Workspace
from ai_neo.ui import PlainUI
from tests.fake_ollama import FakeOllama


class ScriptedUI(PlainUI):
    """A PlainUI whose input comes from a script. Exhausting it raises EOFError (no hangs)."""

    def __init__(self, answers: Iterable[Any] = ()) -> None:
        self.answers = deque(answers)
        self.prompts: list[str] = []
        super().__init__(read=self._next, out=io.StringIO(), color=False, interactive=True)

    def _next(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError(f"scripted input exhausted at: {prompt!r}")
        item = self.answers.popleft()
        if isinstance(item, BaseException):
            raise item
        return str(item)

    @property
    def text(self) -> str:
        return self.out.getvalue()  # type: ignore[attr-defined]


def make_config(server: FakeOllama, tmp_path: Path, **overrides: Any) -> Config:
    flags = {
        "ollama.base_url": server.url,
        "workspace.path": str(tmp_path / "workspace"),
        "tools.run_python_timeout": 5,
    }
    flags.update(overrides)
    return load_config(None, {}, flags)


def make_agent(
    server: FakeOllama,
    tmp_path: Path,
    *,
    ui: ScriptedUI | None = None,
    yes: bool = True,
    **overrides: Any,
) -> tuple[Agent, ScriptedUI, Config]:
    cfg = make_config(server, tmp_path, **overrides)
    ui = ui or ScriptedUI()
    client = OllamaClient(cfg.ollama.base_url, cfg.ollama.model, keep_alive="30m", timeout=15)
    workspace = Workspace(cfg.workspace)
    workspace.root.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(
        workspace,
        cfg.tools,
        confirm=lambda title, preview: ui.confirm(f"{title}?"),
        auto_yes=yes,
    )
    return Agent(client, registry, cfg, ui), ui, cfg
