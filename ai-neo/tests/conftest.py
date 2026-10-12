from __future__ import annotations

import ipaddress
import socket
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


def _is_local(host: object) -> bool:
    if not isinstance(host, str):
        return True  # unix sockets and other non-IP addresses
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite must not need the internet: only loopback (the fake Ollama) is reachable."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guard(address: object) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_local(host):
            raise RuntimeError(f"test tried to reach the network: {address!r}")

    def connect(self: socket.socket, address: object) -> None:
        guard(address)
        return real_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        guard(address)
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    def getaddrinfo(host: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        guard((host,))
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
