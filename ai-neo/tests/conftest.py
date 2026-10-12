from __future__ import annotations

import pytest

from tests.fake_ollama import FakeOllama


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    server.start()
    try:
        yield server
    finally:
        server.stop()
