"""Best-effort subprocess runner for model-written code.

This limits accidents (runaway loops, huge output, memory hogs, environment leaks). It is NOT a
security sandbox: the child process runs as you and can read and write anything you can.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PYTHON_FLAGS = (
    "-E",
    "-s",
    "-B",
    "-u",
)  # ignore PYTHON* env vars and user site, no .pyc, unbuffered

# POSIX only: set resource limits, then exec the real interpreter (limits survive exec).
_POSIX_BOOTSTRAP = (
    "import os, resource, sys\n"
    "cpu, mem, fsize = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])\n"
    "def lim(which, soft):\n"
    "    try:\n"
    "        resource.setrlimit(which, (soft, soft))\n"
    "    except (ValueError, OSError):\n"
    "        pass\n"
    "lim(resource.RLIMIT_CPU, cpu)\n"
    "if mem > 0:\n"
    "    lim(resource.RLIMIT_AS, mem)\n"
    "lim(resource.RLIMIT_FSIZE, fsize)\n"
    "os.execv(sys.executable, [sys.executable, '-E', '-s', '-B', '-u', *sys.argv[4:]])\n"
)
_FILE_SIZE_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True)
class ProcResult:
    returncode: int | None
    output: str
    timed_out: bool
    truncated: bool
    total_bytes: int
    seconds: float


def minimal_env(home: Path) -> dict[str, str]:
    """Environment for child processes: no API keys, no user variables, HOME/TMP inside `home`."""
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(home),
        "USERPROFILE": str(home),
        "TMPDIR": str(home),
        "TEMP": str(home),
        "TMP": str(home),
        "LANG": "C.UTF-8",
    }
    for key in ("PATH", "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


class _Capture:
    """Keeps the first and last bytes of a stream and counts the rest (so tracebacks survive)."""

    def __init__(self, cap: int) -> None:
        self.head_cap = cap // 2
        self.tail_cap = cap - self.head_cap
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        room = self.head_cap - len(self.head)
        if room > 0:
            self.head += data[:room]
            data = data[room:]
        if data:
            self.tail += data
            if len(self.tail) > self.tail_cap:
                del self.tail[: len(self.tail) - self.tail_cap]

    def render(self) -> tuple[str, bool]:
        kept = len(self.head) + len(self.tail)
        head = self.head.decode("utf-8", errors="replace")
        tail = self.tail.decode("utf-8", errors="replace")
        if self.total <= kept:
            return head + tail, False
        omitted = self.total - kept
        return f"{head}\n...[{omitted} bytes of output omitted]...\n{tail}", True


def _reader(stream: object, capture: _Capture) -> None:
    read = stream.read1 if hasattr(stream, "read1") else stream.read  # type: ignore[attr-defined]
    while True:
        try:
            data = read(65536)
        except (OSError, ValueError):
            return
        if not data:
            return
        capture.feed(data)


def _windows_job(pid: int, memory_bytes: int) -> int | None:  # pragma: no cover - Windows only
    """Put the process in a Job Object that kills the whole tree on close and caps memory."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("a", "b", "c", "d", "e", "f")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        limits = ExtendedLimits()
        flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_bytes > 0:
            flags |= 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            limits.ProcessMemoryLimit = memory_bytes
        limits.BasicLimitInformation.LimitFlags = flags
        ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
        process = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not ok or not process:
            kernel32.CloseHandle(job)
            return None
        assigned = kernel32.AssignProcessToJobObject(job, process)
        kernel32.CloseHandle(process)
        if not assigned:
            kernel32.CloseHandle(job)
            return None
        return int(job)
    except Exception:
        return None


def _close_windows_job(job: int | None) -> None:  # pragma: no cover - Windows only
    if job is None:
        return
    try:
        import ctypes

        ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(job))
    except Exception:
        return


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":  # pragma: no cover - Windows only; the job is closed by the caller
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
            timeout=10,
        )
        proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


_LIVE: set[subprocess.Popen[bytes]] = set()
_LIVE_LOCK = threading.Lock()


def kill_running() -> int:
    """Kill every sandboxed process that is running right now; returns how many were running.

    Used by the browser's Stop button and when the web server shuts down, so a script that was
    abandoned does not keep running in the background.
    """
    with _LIVE_LOCK:
        running = list(_LIVE)
    killed = 0
    for proc in running:
        if proc.poll() is None:
            _kill_tree(proc)
            killed += 1
    return killed


def run_python_process(
    python_args: list[str],
    *,
    cwd: Path,
    timeout: float,
    max_bytes: int,
    home: Path | None = None,
    memory_limit_mb: int = 0,
) -> ProcResult:
    """Run `python <python_args>` in `cwd` with a timeout, output cap and minimal environment."""
    home = home or cwd
    memory_bytes = memory_limit_mb * 1024 * 1024
    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":  # pragma: no cover - Windows only
        command = [sys.executable, *PYTHON_FLAGS, *python_args]
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        )
    else:
        cpu_seconds = int(timeout) + 2
        command = [
            sys.executable,
            "-E",
            "-s",
            "-c",
            _POSIX_BOOTSTRAP,
            str(cpu_seconds),
            str(memory_bytes),
            str(_FILE_SIZE_LIMIT),
            *python_args,
        ]
        popen_kwargs["start_new_session"] = True
    started = time.monotonic()
    proc = subprocess.Popen(  # noqa: S603 - arguments are built here, never via a shell
        command,
        cwd=cwd,
        env=minimal_env(home),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **popen_kwargs,  # type: ignore[arg-type]
    )
    with _LIVE_LOCK:
        _LIVE.add(proc)
    job = _windows_job(proc.pid, memory_bytes) if os.name == "nt" else None
    capture = _Capture(max_bytes)
    thread = threading.Thread(target=_reader, args=(proc.stdout, capture), daemon=True)
    thread.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        proc.wait()
    except BaseException:
        _kill_tree(proc)
        proc.wait()
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.discard(proc)
        thread.join(timeout=2)
        _close_windows_job(job)
        if proc.stdout:
            proc.stdout.close()
    output, truncated = capture.render()
    return ProcResult(
        returncode=proc.returncode,
        output=output,
        timed_out=timed_out,
        truncated=truncated,
        total_bytes=capture.total,
        seconds=time.monotonic() - started,
    )
