"""Configuration: defaults < config.toml < AI_NEO_* environment variables < CLI flags."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_neo.prompts import MODES

THINK_SETTINGS = ("auto", "on", "off")
TEST_RUNNERS = ("auto", "pytest", "builtin")
DEFAULT_WORKSPACE = "~/AI-Neo-Workspace"


class ConfigError(Exception):
    """Raised with every problem found, one `section.key (source): message` per line."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class OllamaCfg:
    base_url: str
    model: str
    num_ctx: int
    keep_alive: str
    num_predict: int
    request_timeout: float


@dataclass(frozen=True)
class ChatCfg:
    mode: str
    think: str  # "auto" = the mode's preset, "on", "off"


@dataclass(frozen=True)
class ToolCfg:
    max_steps: int
    run_python_timeout: float
    max_output_bytes: int
    test_timeout: float
    test_runner: str
    memory_limit_mb: int


@dataclass(frozen=True)
class VerifyCfg:
    repair_attempts: int


@dataclass(frozen=True)
class GovernorCfg:
    claude_model: str
    max_rounds: int
    score_threshold: int
    max_calls_per_session: int
    max_tokens_per_session: int
    max_tokens_per_call: int


@dataclass(frozen=True)
class ContextCfg:
    summarize_at: float
    keep_last_turns: int


@dataclass(frozen=True)
class Config:
    ollama: OllamaCfg
    chat: ChatCfg
    workspace: Path
    tools: ToolCfg
    verify: VerifyCfg
    governor: GovernorCfg
    context: ContextCfg


@dataclass(frozen=True)
class _Field:
    section: str
    key: str
    kind: str  # str | int | float | bool | path | url | think | keepalive | mode | runner
    default: Any
    env: str | None = None
    lo: float | None = None
    hi: float | None = None
    required: bool = False  # string must be non-empty

    @property
    def label(self) -> str:
        return f"{self.section}.{self.key}"


FIELDS: tuple[_Field, ...] = (
    _Field("ollama", "base_url", "url", "http://localhost:11434", "AI_NEO_BASE_URL"),
    _Field("ollama", "model", "str", "qwen3:30b", "AI_NEO_MODEL", required=True),
    _Field("ollama", "num_ctx", "int", 8192, "AI_NEO_NUM_CTX", lo=1024, hi=262144),
    _Field("ollama", "keep_alive", "keepalive", "30m", "AI_NEO_KEEP_ALIVE"),
    _Field("ollama", "num_predict", "int", 4096, "AI_NEO_NUM_PREDICT", lo=64, hi=131072),
    _Field("ollama", "request_timeout", "float", 600.0, "AI_NEO_REQUEST_TIMEOUT", lo=5, hi=7200),
    _Field("chat", "mode", "mode", "reason", "AI_NEO_MODE"),
    _Field("chat", "think", "think", "auto", "AI_NEO_THINK"),
    _Field("workspace", "path", "path", DEFAULT_WORKSPACE, "AI_NEO_WORKSPACE"),
    _Field("tools", "max_steps", "int", 8, "AI_NEO_MAX_STEPS", lo=1, hi=50),
    _Field("tools", "run_python_timeout", "float", 20.0, "AI_NEO_RUN_TIMEOUT", lo=1, hi=600),
    _Field("tools", "max_output_bytes", "int", 20000, None, lo=1000, hi=5_000_000),
    _Field("tools", "test_timeout", "float", 120.0, None, lo=5, hi=3600),
    _Field("tools", "test_runner", "runner", "auto", None),
    _Field("tools", "memory_limit_mb", "int", 2048, None, lo=0, hi=1_000_000),
    _Field("verify", "repair_attempts", "int", 3, "AI_NEO_REPAIR_ATTEMPTS", lo=0, hi=10),
    _Field("governor", "claude_model", "str", "", "AI_NEO_CLAUDE_MODEL"),
    _Field("governor", "max_rounds", "int", 2, None, lo=0, hi=5),
    _Field("governor", "score_threshold", "int", 8, None, lo=0, hi=10),
    _Field("governor", "max_calls_per_session", "int", 6, None, lo=0, hi=100),
    _Field("governor", "max_tokens_per_session", "int", 30000, None, lo=0, hi=2_000_000),
    _Field("governor", "max_tokens_per_call", "int", 1500, None, lo=100, hi=16000),
    _Field("context", "summarize_at", "float", 0.75, None, lo=0.3, hi=0.95),
    _Field("context", "keep_last_turns", "int", 4, None, lo=1, hi=20),
)

_KEEP_ALIVE = re.compile(r"^(-1|\d+|\d+(\.\d+)?(ns|us|ms|s|m|h))$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(field: _Field, value: Any) -> Any:
    """Return the validated value or raise ValueError with a human-readable reason."""
    kind = field.kind
    if kind == "bool":
        return _to_bool(value)
    if kind in ("int", "float"):
        return _to_number(field, value)
    if not isinstance(value, str):
        if kind == "think" and isinstance(value, bool):
            return "on" if value else "off"
        raise ValueError(f"must be a string, got {value!r}")
    text = value.strip()
    if kind == "think":
        lowered = text.lower()
        if lowered in _TRUE:
            return "on"
        if lowered in _FALSE:
            return "off"
        if lowered == "auto":
            return "auto"
        raise ValueError(f"must be one of auto, on, off; got {value!r}")
    if kind == "mode":
        if text not in MODES:
            raise ValueError(f"must be one of {', '.join(MODES)}; got {value!r}")
        return text
    if kind == "runner":
        if text not in TEST_RUNNERS:
            raise ValueError(f"must be one of {', '.join(TEST_RUNNERS)}; got {value!r}")
        return text
    if kind == "url":
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"must look like http://localhost:11434, got {value!r}")
        return text.rstrip("/")
    if kind == "keepalive":
        if not _KEEP_ALIVE.match(text):
            raise ValueError(f'must be a duration like "30m", "1h", "300" or "-1"; got {value!r}')
        return text
    if kind == "path":
        if not text:
            raise ValueError("must not be empty")
        return text
    if field.required and not text:
        raise ValueError("must not be empty")
    return text


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise ValueError(f"must be true or false, got {value!r}")


def _to_number(field: _Field, value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"must be a number, got {value!r}")
    try:
        number = int(value) if field.kind == "int" else float(value)
        if field.kind == "int" and isinstance(value, float) and value != number:
            raise ValueError
    except (TypeError, ValueError):
        word = "an integer" if field.kind == "int" else "a number"
        raise ValueError(f"must be {word}, got {value!r}") from None
    too_low = field.lo is not None and number < field.lo
    too_high = field.hi is not None and number > field.hi
    if too_low or too_high:
        raise ValueError(f"must be between {field.lo:g} and {field.hi:g}, got {number}")
    return number


def _read_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError([f"{path}: not valid TOML ({exc})"]) from exc
    except OSError as exc:
        raise ConfigError([f"{path}: cannot be read ({exc})"]) from exc


def _unknown_keys(raw: Mapping[str, Any], source: str) -> list[str]:
    known: dict[str, set[str]] = {}
    for f in FIELDS:
        known.setdefault(f.section, set()).add(f.key)
    problems = []
    for section, body in raw.items():
        if section not in known:
            problems.append(f"[{section}] ({source}): unknown section")
        elif not isinstance(body, dict):
            problems.append(f"[{section}] ({source}): must be a table")
        else:
            problems.extend(
                f"{section}.{key} ({source}): unknown setting"
                for key in body
                if key not in known[section]
            )
    return problems


def load_config(
    path: Path | None,
    env: Mapping[str, str],
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Build a validated Config, or raise ConfigError listing every problem at once."""
    raw = _read_toml(path)
    source = str(path) if path else "config.toml"
    problems = _unknown_keys(raw, source)
    values: dict[str, Any] = {}
    for f in FIELDS:
        chosen = f.default
        layers: list[tuple[str, Any]] = []
        section = raw.get(f.section)
        if isinstance(section, dict) and f.key in section:
            layers.append((source, section[f.key]))
        if f.env and f.env in env and env[f.env] != "":
            layers.append((f"env {f.env}", env[f.env]))
        if overrides and overrides.get(f.label) is not None:
            layers.append(("command-line flag", overrides[f.label]))
        for layer_name, candidate in layers:
            try:
                chosen = _coerce(f, candidate)
            except ValueError as exc:
                problems.append(f"{f.label} ({layer_name}): {exc}")
        values[f.label] = chosen
    if problems:
        raise ConfigError(problems)
    return _assemble(values)


def _assemble(v: Mapping[str, Any]) -> Config:
    def group(section: str) -> dict[str, Any]:
        return {f.key: v[f.label] for f in FIELDS if f.section == section}

    return Config(
        ollama=OllamaCfg(**group("ollama")),
        chat=ChatCfg(**group("chat")),
        workspace=Path(v["workspace.path"]).expanduser(),
        tools=ToolCfg(**group("tools")),
        verify=VerifyCfg(**group("verify")),
        governor=GovernorCfg(**group("governor")),
        context=ContextCfg(**group("context")),
    )


def default_config() -> Config:
    return load_config(None, {})


def config_values(cfg: Config) -> dict[str, Any]:
    """Flat `section.key -> value` mapping (the inverse of `_assemble`)."""
    sections: dict[str, Any] = {
        "ollama": cfg.ollama,
        "chat": cfg.chat,
        "tools": cfg.tools,
        "verify": cfg.verify,
        "governor": cfg.governor,
        "context": cfg.context,
    }
    out: dict[str, Any] = {"workspace.path": str(cfg.workspace)}
    for name, obj in sections.items():
        for key, value in vars(obj).items():
            out[f"{name}.{key}"] = value
    return out


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


_COMMENTS = {
    "ollama": "Connection to the Ollama server. Point base_url at a remote server if you like.",
    "chat": "think = auto uses each mode's own preset; on/off forces it.",
    "workspace": "The only folder the file tools may read or write.",
    "tools": "Limits for the tool loop and sandboxed code runs (best-effort isolation).",
    "verify": "How many times a failing Python task is sent back to the model for repair.",
    "governor": "Optional Claude critique. Always off at start; use --govern or /govern on.",
    "context": "History is summarized when it passes summarize_at of num_ctx.",
}


def render_toml(cfg: Config) -> str:
    values = config_values(cfg)
    lines = ["# AI Neo configuration. Environment variables and command-line flags override this."]
    sections = list(dict.fromkeys(f.section for f in FIELDS))
    for section in sections:
        lines += ["", f"# {_COMMENTS[section]}", f"[{section}]"]
        for f in FIELDS:
            if f.section == section:
                lines.append(f"{f.key} = {_toml_value(values[f.label])}")
    return "\n".join(lines) + "\n"


def write_config(cfg: Config, path: Path) -> None:
    """Write `cfg`, keeping the previous file as `<name>.bak` if its content differs."""
    text = render_toml(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous == text:
            return
        path.with_name(path.name + ".bak").write_text(previous, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def is_local_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
