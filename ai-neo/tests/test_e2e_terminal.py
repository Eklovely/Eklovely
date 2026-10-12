"""Drives the real program in a real pseudo-terminal (Rich UI, real Ctrl+C), POSIX only."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fake_ollama import ChatReply, FakeOllama

pty = pytest.importorskip("pty") if os.name != "nt" else None
pytestmark = pytest.mark.skipif(os.name == "nt", reason="needs a POSIX pseudo-terminal")


BOOT = (
    "import fcntl, os, sys, termios\n"
    "os.setsid()\n"
    "fcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"
    "os.execv(sys.executable, [sys.executable, '-m', 'ai_neo', *sys.argv[1:]])\n"
)


class Terminal:
    def __init__(self, server: FakeOllama, tmp_path: Path, args: list[str]) -> None:
        master, slave = pty.openpty()  # type: ignore[union-attr]
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "AI_NEO_HOME": str(tmp_path / "home"),
            "AI_NEO_BASE_URL": server.url,
            "AI_NEO_WORKSPACE": str(tmp_path / "ws"),
            "TERM": "xterm",
            "COLUMNS": "100",
            "LINES": "40",
        }
        self.master = master
        (tmp_path / "home").mkdir(exist_ok=True)
        if "--first-run" in args:
            args = [a for a in args if a != "--first-run"]
        else:  # an existing config means no first-run offer
            (tmp_path / "home" / "config.toml").write_text("", encoding="utf-8")
        # The child must have the pseudo-terminal as its *controlling* terminal, otherwise Ctrl+C
        # typed into it never becomes a SIGINT. So it starts its own session first.
        self.proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", BOOT, *args],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            close_fds=True,
        )
        os.close(slave)
        self.output = ""

    def read_until(self, needle: str, timeout: float = 30.0) -> str:
        deadline = time.monotonic() + timeout
        while needle not in self.output:
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"timed out waiting for {needle!r}; saw:\n{self.output[-1500:]}"
                )
            ready, _, _ = select.select([self.master], [], [], 0.2)
            if ready:
                try:
                    self.output += os.read(self.master, 4096).decode("utf-8", errors="replace")
                except OSError:
                    break
        return self.output

    def send(self, data: str) -> None:
        os.write(self.master, data.encode())

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        os.close(self.master)


def test_streaming_thinking_and_ctrl_c_in_a_real_terminal(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(
        ChatReply(
            thinking=["Considering the request. "] * 3,
            content=[f"word{i} " for i in range(40)],
            delay=0.25,
        ),
        ChatReply(content=["second reply works"]),
    )
    term = Terminal(fake_ollama, tmp_path, ["--mode", "write", "--think"])
    try:
        term.read_until("[write+think] > ")
        term.send("tell me a story\n")
        term.read_until("word3")  # the answer is streaming
        assert "thought for" in term.output  # thinking was collapsed once the answer began
        term.send("\x03")  # real Ctrl+C
        term.read_until("stopped by you")
        term.read_until("[write+think] > ")  # the program is still alive and asking for input
        assert term.proc.poll() is None
        deadline = time.monotonic() + 10
        while fake_ollama.disconnects < 1 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert (
            fake_ollama.disconnects >= 1
        )  # Ollama was told to stop generating (connection closed)
        term.send("again\n")
        term.read_until("second reply works")
        term.send("/quit\n")
        assert term.proc.wait(timeout=15) == 0
    finally:
        term.close()


def test_ctrl_c_at_the_prompt_does_not_exit_and_ctrl_d_does(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    term = Terminal(fake_ollama, tmp_path, ["--mode", "write"])
    try:
        term.read_until("[write")
        term.send("\x03")
        term.read_until("Use /quit or Ctrl+D to leave")  # (the banner text differs)
        term.read_until("> ", timeout=10)
        assert term.proc.poll() is None
        time.sleep(0.3)
        term.send("\x04")  # Ctrl+D
        assert term.proc.wait(timeout=15) == 0
    finally:
        term.close()


def test_first_run_offers_setup_and_can_be_declined(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    term = Terminal(fake_ollama, tmp_path, ["--mode", "write", "--first-run"])
    try:
        term.read_until("Start the guided setup now? (recommended the first time) [Y/n]")
        term.send("n\n")
        term.read_until("Run `ai-neo setup` whenever you like")
        term.read_until("[write")
        term.send("/quit\n")
        assert term.proc.wait(timeout=15) == 0
    finally:
        term.close()
