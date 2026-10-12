from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from ai_neo.sandbox import minimal_env, run_python_process

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="uses POSIX process groups/rlimits")


def run(tmp_path: Path, code: str, **kw: object) -> object:
    (tmp_path / "main.py").write_text(code, encoding="utf-8")
    params = {"timeout": 20.0, "max_bytes": 20_000, "memory_limit_mb": 0}
    params.update(kw)
    return run_python_process(["main.py"], cwd=tmp_path, **params)  # type: ignore[arg-type]


def test_normal_run(tmp_path: Path) -> None:
    result = run(tmp_path, "print('hello')")
    assert result.returncode == 0 and result.output.strip() == "hello"  # type: ignore[attr-defined]
    assert not result.timed_out  # type: ignore[attr-defined]


def test_infinite_loop_times_out(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run(tmp_path, "while True: pass", timeout=1)
    assert result.timed_out  # type: ignore[attr-defined]
    assert time.monotonic() - started < 6


def test_huge_output_is_capped_and_traceback_survives(tmp_path: Path) -> None:
    code = "print('a' * 5_000_000)\nraise ValueError('boom')"
    result = run(tmp_path, code, max_bytes=2000)
    assert result.truncated  # type: ignore[attr-defined]
    assert len(result.output) < 2400  # type: ignore[attr-defined]
    assert "ValueError: boom" in result.output  # type: ignore[attr-defined]
    assert result.total_bytes > 5_000_000  # type: ignore[attr-defined]


def test_endless_printing_does_not_exhaust_memory(tmp_path: Path) -> None:
    result = run(tmp_path, "while True: print('spam' * 100)", timeout=1, max_bytes=1000)
    assert result.timed_out and len(result.output) < 1300  # type: ignore[attr-defined]


def test_environment_is_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    monkeypatch.setenv("MY_PRIVATE_VAR", "private")
    code = "import os; print(sorted(k for k in os.environ if 'KEY' in k or 'PRIVATE' in k))"
    result = run(tmp_path, code)
    assert result.output.strip() == "[]"  # type: ignore[attr-defined]
    env = minimal_env(tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and env["HOME"] == str(tmp_path)


def test_stdin_is_closed(tmp_path: Path) -> None:
    result = run(tmp_path, "input()")
    assert result.returncode != 0 and "EOFError" in result.output  # type: ignore[attr-defined]


@POSIX_ONLY
def test_timeout_kills_child_processes(tmp_path: Path) -> None:
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    result = run(tmp_path, code, timeout=1)
    pid = int(result.output.split()[0])  # type: ignore[attr-defined]
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("child process survived the timeout")


@POSIX_ONLY
def test_memory_limit(tmp_path: Path) -> None:
    result = run(tmp_path, "x = bytearray(2 * 1024**3)\nprint('allocated')", memory_limit_mb=256)
    assert "allocated" not in result.output  # type: ignore[attr-defined]
    assert "MemoryError" in result.output  # type: ignore[attr-defined]


def test_python_args_support_modules(tmp_path: Path) -> None:
    result = run_python_process(
        ["-c", "import sys; print(sys.version_info[0])"],
        cwd=tmp_path,
        timeout=10,
        max_bytes=1000,
    )
    assert result.output.strip() == "3" and sys.version_info[0] == 3


def test_kill_running_stops_a_script_that_is_in_progress(tmp_path) -> None:
    import threading
    import time

    from ai_neo.sandbox import kill_running, run_python_process

    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    box: list = []
    thread = threading.Thread(
        target=lambda: box.append(
            run_python_process([str(script)], cwd=tmp_path, timeout=30, max_bytes=1000)
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(1.0)
    started = time.monotonic()
    assert kill_running() == 1
    thread.join(timeout=10)
    assert not thread.is_alive() and time.monotonic() - started < 5
    assert box[0].returncode != 0 and not box[0].timed_out
    assert kill_running() == 0  # the registry is empty again
