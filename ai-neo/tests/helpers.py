from __future__ import annotations

import io
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_neo.agent import Agent
from ai_neo.config import Config, load_config
from ai_neo.ollama_client import OllamaClient
from ai_neo.setup.probe import CommandResult
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
        if callable(item):  # run a side effect (for example "the user installed Ollama")
            return str(item() or "")
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


@dataclass
class FakeProbe:
    """A pretend computer. Defaults match the target laptop: 32 GB RAM, 4 GB GPU, Windows 11."""

    system: str = "Windows"
    py: tuple[int, int, int] = (3, 12, 1)
    ram_gb: float | None = 32.0
    disk_gb: float | None = 200.0
    executables: dict[str, str] = field(
        default_factory=lambda: {
            "ollama": r"C:\Users\me\AppData\Local\Programs\Ollama\ollama.exe",
            "nvidia-smi": "nvidia-smi",
        }
    )
    environment: dict[str, str] = field(default_factory=dict)
    gpu_output: str = "NVIDIA RTX 500 Ada Generation Laptop GPU, 4094\n"
    ran: list[tuple[str, ...]] = field(default_factory=list)

    def os_name(self) -> str:
        return self.system

    def python_version(self) -> tuple[int, int, int]:
        return self.py

    def total_ram_gb(self) -> float | None:
        return self.ram_gb

    def free_disk_gb(self, path: object) -> float | None:
        return self.disk_gb

    def find_executable(self, name: str) -> str | None:
        return self.executables.get(name)

    def env(self, name: str) -> str | None:
        return self.environment.get(name)

    def run(self, argv, timeout: float = 15.0) -> CommandResult:  # type: ignore[no-untyped-def]
        self.ran.append(tuple(argv))
        program = str(argv[0]).replace("\\", "/").rsplit("/", 1)[-1]
        if program.startswith("nvidia-smi"):
            return CommandResult(0, self.gpu_output)
        if program.startswith("ollama"):
            return CommandResult(0, "ollama version is 0.12.0\n")
        return CommandResult(127, "", "not found")
