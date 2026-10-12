"""Locations of AI Neo's files inside the platform user-data directory."""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

HOME_ENV = "AI_NEO_HOME"


def default_user_data_dir(env: Mapping[str, str] | None = None, system: str | None = None) -> Path:
    """`AI_NEO_HOME`, else %APPDATA%\\AINeo on Windows, else ~/.ai_neo."""
    env = os.environ if env is None else env
    override = env.get(HOME_ENV)
    if override:
        return Path(override).expanduser()
    if (system or platform.system()) == "Windows":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "AINeo"
    return Path.home() / ".ai_neo"


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def default(cls, env: Mapping[str, str] | None = None) -> Paths:
        return cls(default_user_data_dir(env))

    @property
    def config(self) -> Path:
        return self.root / "config.toml"

    @property
    def state(self) -> Path:
        return self.root / "setup_state.json"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    def ensure(self) -> None:
        for directory in (self.root, self.sessions, self.logs, self.datasets, self.exports):
            directory.mkdir(parents=True, exist_ok=True)
