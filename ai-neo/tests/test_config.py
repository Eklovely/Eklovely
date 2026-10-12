from __future__ import annotations

from pathlib import Path

import pytest

from ai_neo.config import (
    ConfigError,
    default_config,
    is_local_url,
    load_config,
    render_toml,
    write_config,
)
from ai_neo.paths import Paths, default_user_data_dir


def test_defaults_match_spec() -> None:
    cfg = default_config()
    assert cfg.ollama.base_url == "http://localhost:11434"
    assert cfg.ollama.model == "qwen3:30b"
    assert cfg.ollama.num_ctx == 8192
    assert cfg.ollama.keep_alive == "30m"
    assert cfg.tools.max_steps == 8
    assert cfg.tools.run_python_timeout == 20
    assert cfg.verify.repair_attempts == 3
    assert not hasattr(cfg.governor, "enabled")  # only --govern / /govern on can enable it
    assert cfg.governor.claude_model == ""  # no hard-coded Claude model
    assert cfg.governor.max_rounds == 2
    assert cfg.governor.score_threshold == 8
    assert cfg.context.summarize_at == 0.75


def test_precedence_toml_then_env_then_flags(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[ollama]\nmodel = "from-toml"\nnum_ctx = 4096\n', encoding="utf-8")
    env = {"AI_NEO_MODEL": "from-env"}
    cfg = load_config(path, env)
    assert cfg.ollama.model == "from-env"
    assert cfg.ollama.num_ctx == 4096
    cfg = load_config(path, env, {"ollama.model": "from-flag"})
    assert cfg.ollama.model == "from-flag"


def test_claude_model_comes_from_env() -> None:
    cfg = load_config(None, {"AI_NEO_CLAUDE_MODEL": "some-model"})
    assert cfg.governor.claude_model == "some-model"


def test_all_problems_reported_at_once(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[ollama]\nnum_ctx = "lots"\nbase_url = "ftp://x"\nbogus = 1\n[nothing]\nx = 1\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_config(path, {"AI_NEO_MAX_STEPS": "0"})
    text = str(info.value)
    assert "ollama.num_ctx" in text and "ollama.base_url" in text
    assert "ollama.bogus" in text and "[nothing]" in text
    assert "tools.max_steps (env AI_NEO_MAX_STEPS)" in text
    assert len(info.value.problems) == 5


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[ollama\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(path, {})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ollama.num_ctx", 100),
        ("ollama.keep_alive", "soon"),
        ("chat.mode", "poetry"),
        ("chat.think", "maybe"),
        ("context.summarize_at", 1.5),
        ("governor.score_threshold", 11),
        ("tools.test_runner", "nose"),
    ],
)
def test_bad_values_rejected(key: str, value: object) -> None:
    with pytest.raises(ConfigError):
        load_config(None, {}, {key: value})


def test_think_accepts_booleans_and_words(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[chat]\nthink = true\n", encoding="utf-8")
    assert load_config(path, {}).chat.think == "on"
    assert load_config(None, {"AI_NEO_THINK": "false"}).chat.think == "off"


def test_render_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(None, {}, {"workspace.path": r"C:\Users\me\AI Neo", "ollama.num_ctx": 4096})
    path = tmp_path / "config.toml"
    write_config(cfg, path)
    again = load_config(path, {})
    assert again == cfg
    assert "[governor]" in render_toml(cfg)


def test_write_config_keeps_backup(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_config(default_config(), path)
    changed = load_config(None, {}, {"ollama.num_ctx": 4096})
    write_config(changed, path)
    assert path.with_name("config.toml.bak").exists()
    write_config(changed, path)  # idempotent: identical content leaves the backup alone
    assert load_config(path, {}).ollama.num_ctx == 4096


def test_is_local_url() -> None:
    assert is_local_url("http://localhost:11434")
    assert is_local_url("http://127.0.0.1:11434")
    assert not is_local_url("http://gpu-box.example.com:11434")


def test_user_data_dir_rules(tmp_path: Path) -> None:
    assert default_user_data_dir({"AI_NEO_HOME": str(tmp_path)}) == tmp_path
    win = default_user_data_dir({"APPDATA": r"C:\Users\x\AppData\Roaming"}, system="Windows")
    assert win.name == "AINeo"
    assert default_user_data_dir({}, system="Linux").name == ".ai_neo"


def test_paths_layout(tmp_path: Path) -> None:
    paths = Paths(tmp_path / "home")
    paths.ensure()
    assert paths.sessions.is_dir() and paths.logs.is_dir()
    assert paths.config == tmp_path / "home" / "config.toml"
