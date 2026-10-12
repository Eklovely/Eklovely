from __future__ import annotations

from pathlib import Path

import pytest

from tests.fake_ollama import FakeOllama


@pytest.fixture(autouse=True)
def isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """No test may read or write the real home directory or see real credentials."""
    home = tmp_path_factory.mktemp("fake_home")
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    for name in (
        "AI_NEO_HOME",
        "ANTHROPIC_API_KEY",
        "AI_NEO_CLAUDE_MODEL",
        "AI_NEO_MODEL",
        "AI_NEO_BASE_URL",
        "AI_NEO_THINK",
        "AI_NEO_MODE",
        "AI_NEO_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    server.start()
    try:
        yield server
    finally:
        server.stop()
