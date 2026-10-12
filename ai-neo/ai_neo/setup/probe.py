"""Looking at the machine: OS, memory, disk, GPU and installed programs.

Everything goes through the `SystemProbe` interface so tests can fake a machine.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_mb: int | None


class SystemProbe(Protocol):
    def os_name(self) -> str: ...

    def python_version(self) -> tuple[int, int, int]: ...

    def total_ram_gb(self) -> float | None: ...

    def free_disk_gb(self, path: Path) -> float | None: ...

    def find_executable(self, name: str) -> str | None: ...

    def run(self, argv: Sequence[str], timeout: float = 15.0) -> CommandResult: ...

    def env(self, name: str) -> str | None: ...


def _ram_windows() -> float | None:  # pragma: no cover - Windows only
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return status.ullTotalPhys / 1024**3


class RealProbe:
    def os_name(self) -> str:
        return platform.system()

    def python_version(self) -> tuple[int, int, int]:
        info = sys.version_info
        return info.major, info.minor, info.micro

    def total_ram_gb(self) -> float | None:
        system = platform.system()
        try:
            if system == "Windows":
                return _ram_windows()
            if system == "Linux":
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024**2
            if system == "Darwin":
                done = self.run(["sysctl", "-n", "hw.memsize"])
                if done.returncode == 0:
                    return int(done.stdout.strip()) / 1024**3
        except (OSError, ValueError):
            return None
        return None

    def free_disk_gb(self, path: Path) -> float | None:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free / 1024**3
        except OSError:
            return None

    def find_executable(self, name: str) -> str | None:
        found = shutil.which(name)
        if found:
            return found
        for candidate in known_locations(name, self.os_name(), os.environ):
            if candidate.is_file():
                return str(candidate)
        return None

    def run(self, argv: Sequence[str], timeout: float = 15.0) -> CommandResult:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argument lists, no shell
                list(argv), capture_output=True, text=True, timeout=timeout, check=False
            )
        except FileNotFoundError:
            return CommandResult(127, "", "not found")
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "timed out")
        except OSError as exc:
            return CommandResult(126, "", str(exc))
        return CommandResult(done.returncode, done.stdout, done.stderr)

    def env(self, name: str) -> str | None:
        return os.environ.get(name)


def known_locations(name: str, system: str, env: dict[str, str] | os._Environ[str]) -> list[Path]:
    """Usual install places for a program that may not be on PATH."""
    if name != "ollama":
        return []
    if system == "Windows":
        local = env.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return [base / "Programs" / "Ollama" / "ollama.exe"]
    if system == "Darwin":
        return [
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/usr/local/bin/ollama"),
        ]
    return [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]


def detect_gpu(probe: SystemProbe) -> GpuInfo | None:
    """First NVIDIA GPU from `nvidia-smi`, or None if there is none (or it is not installed)."""
    if probe.find_executable("nvidia-smi") is None:
        return None
    done = probe.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if done.returncode != 0 or not done.stdout.strip():
        return None
    first = done.stdout.strip().splitlines()[0]
    name, _, memory = first.rpartition(",")
    try:
        vram: int | None = int(float(memory.strip()))
    except ValueError:
        return GpuInfo(first.strip(), None)
    return GpuInfo(name.strip(), vram)


def ollama_models_dir(probe: SystemProbe) -> Path:
    """Where Ollama keeps models (OLLAMA_MODELS, else ~/.ollama/models)."""
    override = probe.env("OLLAMA_MODELS")
    return Path(override) if override else Path.home() / ".ollama" / "models"
