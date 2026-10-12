"""Individual health checks, shared by the setup wizard and `ai-neo doctor`."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ai_neo.config import Config, ConfigError
from ai_neo.governor import key_present
from ai_neo.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    model_installed,
)
from ai_neo.paths import Paths
from ai_neo.setup.probe import GpuInfo, SystemProbe, detect_gpu

MIN_PYTHON = (3, 11)
MIN_RAM_GB = 24.0
MIN_FREE_DISK_GB = 25.0
MODEL_DISK_GB = 19  # qwen3:30b download size, approximately


class Status(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Check:
    id: str
    label: str
    status: Status
    detail: str
    fix: str = ""


def worst(statuses: list[Status]) -> Status:
    for level in (Status.FAIL, Status.WARN):
        if level in statuses:
            return level
    return Status.OK


def check_python(probe: SystemProbe) -> Check:
    major, minor, micro = probe.python_version()
    text = f"Python {major}.{minor}.{micro}"
    if (major, minor) >= MIN_PYTHON:
        return Check("python", "Python version", Status.OK, text)
    return Check(
        "python",
        "Python version",
        Status.FAIL,
        f"{text} is too old (need 3.11 or newer)",
        "Install Python 3.11+ from https://www.python.org/downloads/ and tick "
        "'Add python.exe to PATH'.",
    )


def check_dependencies() -> Check:
    if importlib.util.find_spec("rich") is None:
        return Check(
            "deps",
            "Optional packages",
            Status.WARN,
            "rich is not installed (plain text mode)",
            "Run: pip install rich   (optional; makes the screen prettier)",
        )
    return Check("deps", "Optional packages", Status.OK, "httpx and rich are installed")


def check_ram(probe: SystemProbe) -> Check:
    ram = probe.total_ram_gb()
    if ram is None:
        return Check(
            "ram",
            "Memory (RAM)",
            Status.WARN,
            "could not detect the amount of RAM",
            "qwen3:30b needs about 19 GB for the model plus room for context; "
            "32 GB is comfortable.",
        )
    if ram < MIN_RAM_GB:
        return Check(
            "ram",
            "Memory (RAM)",
            Status.WARN,
            f"{ram:.0f} GB (24 GB or more recommended)",
            "Expect heavy swapping and very slow answers. Use a smaller model "
            "(for example qwen3:8b via --model) or a remote Ollama server.",
        )
    return Check("ram", "Memory (RAM)", Status.OK, f"{ram:.0f} GB")


def check_disk(probe: SystemProbe, models_dir: Path, needed_gb: float = MIN_FREE_DISK_GB) -> Check:
    free = probe.free_disk_gb(models_dir)
    if free is None:
        return Check(
            "disk", "Free disk space", Status.WARN, f"could not read free space near {models_dir}"
        )
    text = f"{free:.0f} GB free where Ollama stores models ({models_dir})"
    if free < needed_gb:
        return Check(
            "disk",
            "Free disk space",
            Status.WARN,
            text + f"; about {needed_gb:.0f} GB is recommended",
            f"The model download is about {MODEL_DISK_GB} GB. Free some space, or set "
            "OLLAMA_MODELS "
            "to a bigger drive and restart Ollama.",
        )
    return Check("disk", "Free disk space", Status.OK, text)


def gpu_check(gpu: GpuInfo | None) -> Check:
    if gpu is None:
        return Check("gpu", "GPU", Status.OK, "none detected (the model will run on the CPU)")
    vram = f"{gpu.vram_mb / 1024:.0f} GB VRAM" if gpu.vram_mb else "unknown VRAM"
    note = ""
    if gpu.vram_mb is not None and gpu.vram_mb < 12000:
        note = "; the model is much bigger than that, so most layers run on the CPU"
    return Check("gpu", "GPU", Status.OK, f"{gpu.name} ({vram}){note}")


def check_gpu(probe: SystemProbe) -> Check:
    return gpu_check(detect_gpu(probe))


def check_ollama_binary(probe: SystemProbe, server_ok: bool) -> Check:
    found = probe.find_executable("ollama")
    if found:
        return Check("ollama-bin", "Ollama program", Status.OK, f"found at {found}")
    fix = (
        "Download and run the installer from https://ollama.com/download, then open a "
        "NEW terminal. "
        "On Windows it installs to %LOCALAPPDATA%\\Programs\\Ollama."
    )
    if server_ok:
        return Check(
            "ollama-bin",
            "Ollama program",
            Status.WARN,
            "not found here, but a server answers (it may run elsewhere)",
            fix,
        )
    return Check(
        "ollama-bin", "Ollama program", Status.FAIL, "not found on PATH or in the usual place", fix
    )


def check_server(client: OllamaClient) -> tuple[Check, list[str] | None]:
    try:
        models = client.list_models()
    except OllamaUnavailable:
        return (
            Check(
                "server",
                "Ollama server",
                Status.FAIL,
                f"no answer at {client.base_url}",
                "Start the Ollama app (Start menu or tray icon), or run `ollama serve` in another "
                "terminal. If it runs elsewhere, set ollama.base_url.",
            ),
            None,
        )
    except OllamaError as exc:
        return Check(
            "server", "Ollama server", Status.FAIL, str(exc), "Restart Ollama and try again."
        ), None
    return Check("server", "Ollama server", Status.OK, f"answers at {client.base_url}"), models


def check_model(models: list[str] | None, model: str) -> Check:
    if models is None:
        return Check(
            "model", f"Model {model}", Status.SKIP, "not checked because Ollama is not running"
        )
    if model_installed(models, model):
        return Check("model", f"Model {model}", Status.OK, "installed")
    return Check(
        "model", f"Model {model}", Status.FAIL, "not installed", f"Run: ollama pull {model}"
    )


def check_config(path: Path, error: ConfigError | None) -> Check:
    if error is not None:
        return Check(
            "config",
            "Configuration file",
            Status.FAIL,
            "; ".join(error.problems[:3]),
            f"Fix the values in {path}, or delete the file and run `ai-neo setup`.",
        )
    if not path.exists():
        return Check(
            "config",
            "Configuration file",
            Status.WARN,
            "not created yet (defaults are used)",
            "Run `ai-neo setup` to create it.",
        )
    return Check("config", "Configuration file", Status.OK, f"valid ({path})")


def _writable(directory: Path) -> str | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory):
            pass
    except OSError as exc:
        return str(exc)
    return None


def check_workspace(cfg: Config) -> Check:
    path = cfg.workspace
    if not path.exists():
        return Check(
            "workspace",
            "Workspace folder",
            Status.WARN,
            f"{path} does not exist yet",
            "It is created by `ai-neo setup` (or on first use).",
        )
    problem = _writable(path)
    if problem:
        return Check(
            "workspace",
            "Workspace folder",
            Status.FAIL,
            f"{path} is not writable ({problem})",
            "Pick another folder with `ai-neo setup` or set workspace.path.",
        )
    return Check("workspace", "Workspace folder", Status.OK, f"{path} is writable")


def check_userdata(paths: Paths) -> Check:
    root = paths.root
    if not root.exists():
        ancestor = next((p for p in root.parents if p.exists()), root)
        if os.access(ancestor, os.W_OK):
            return Check(
                "userdata", "AI Neo data folder", Status.OK, f"{root} will be created when needed"
            )
        return Check(
            "userdata",
            "AI Neo data folder",
            Status.FAIL,
            f"{root} cannot be created",
            "Set AI_NEO_HOME to a folder you can write to.",
        )
    problem = _writable(root)
    if problem:
        return Check(
            "userdata",
            "AI Neo data folder",
            Status.FAIL,
            f"{root} is not writable ({problem})",
            "Set AI_NEO_HOME to a folder you can write to.",
        )
    return Check("userdata", "AI Neo data folder", Status.OK, str(root))


def check_governor(cfg: Config, env: Mapping[str, str]) -> Check:
    has_key = key_present(env)
    has_model = bool(cfg.governor.claude_model)
    has_sdk = importlib.util.find_spec("anthropic") is not None
    if not has_key and not has_model:
        return Check(
            "governor", "Claude governor (optional)", Status.OK, "not configured (it is optional)"
        )
    problems = []
    if not has_key:
        problems.append("ANTHROPIC_API_KEY is not set")
    if not has_model:
        problems.append("no Claude model name (governor.claude_model / AI_NEO_CLAUDE_MODEL)")
    if not has_sdk:
        problems.append("the 'anthropic' package is missing")
    if problems:
        return Check(
            "governor",
            "Claude governor (optional)",
            Status.WARN,
            "; ".join(problems),
            "Finish the governor step of `ai-neo setup`, or ignore it: the governor is optional.",
        )
    return Check(
        "governor",
        "Claude governor (optional)",
        Status.OK,
        "ready (stays off until you pass --govern or type /govern on)",
    )
