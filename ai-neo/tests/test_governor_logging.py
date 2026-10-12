from __future__ import annotations

import dataclasses
import importlib.machinery
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_neo.config import default_config
from ai_neo.governor import (
    Governor,
    build_payload,
    govern,
    key_present,
    parse_critique,
)
from ai_neo.logging_store import DatasetStats, LogStore, TaskRecord
from tests.helpers import ScriptedUI

SECRET = "sk-ant-SECRET123456"
CRITIQUE_LOW = json.dumps(
    {"score": 4, "issues": ["wrong total"], "suggested_corrections": ["recompute the sum"]}
)
CRITIQUE_HIGH = json.dumps({"score": 9, "issues": [], "suggested_corrections": []})


class FakeAnthropic:
    def __init__(self, replies: list[Any], usage: tuple[int, int] = (100, 50)) -> None:
        self.replies = list(replies)
        self.usage = usage
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        stop = "refusal" if reply == "REFUSAL" else "end_turn"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]),
            stop_reason=stop,
        )


def make_governor(
    replies: list[Any],
    *,
    env: dict[str, str] | None = None,
    model: str = "some-claude-model",
    answers: list[str] | None = None,
    assume_yes: bool = False,
    **cfg_changes: Any,
) -> tuple[Governor, FakeAnthropic, ScriptedUI]:
    cfg = dataclasses.replace(default_config().governor, claude_model=model, **cfg_changes)
    fake = FakeAnthropic(replies)
    ui = ScriptedUI(answers if answers is not None else ["y"])
    environment = {"ANTHROPIC_API_KEY": SECRET} if env is None else env
    gov = Governor(cfg, environment, ui, assume_yes=assume_yes, client_factory=lambda: fake)
    return gov, fake, ui


def run_govern(gov: Governor, revise_with: list[str | None] | None = None):
    revise_calls: list[str] = []
    replies = list(revise_with or ["better answer"])

    def revise(prompt: str) -> str | None:
        revise_calls.append(prompt)
        return replies.pop(0) if replies else None

    result = govern(
        gov,
        task_type="computation",
        prompt="What is 12*12?",
        draft="143",
        tool_results=["12*12 = 144"],
        revise=revise,
    )
    return result, revise_calls


# -- disabled by default ---------------------------------------------------------------------


def test_off_without_the_flag_even_with_key_and_model() -> None:
    gov, fake, ui = make_governor([CRITIQUE_LOW])
    assert not gov.enabled
    result, revise_calls = run_govern(gov)
    assert result.final == "143" and not result.used and fake.calls == [] and revise_calls == []


def test_cannot_be_enabled_without_the_key() -> None:
    gov, fake, _ = make_governor([CRITIQUE_LOW], env={})
    reason = gov.enable()
    assert reason is not None and "ANTHROPIC_API_KEY" in reason
    assert not gov.enabled
    assert run_govern(gov)[0].final == "143" and fake.calls == []
    assert not key_present({"ANTHROPIC_API_KEY": ""})


def test_cannot_be_enabled_without_a_model_name() -> None:
    gov, fake, _ = make_governor([CRITIQUE_LOW], model="")
    assert "AI_NEO_CLAUDE_MODEL" in (gov.enable() or "")
    assert fake.calls == []


def test_no_hardcoded_default_model() -> None:
    assert default_config().governor.claude_model == ""


def test_missing_sdk_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    gov = Governor(
        dataclasses.replace(default_config().governor, claude_model="m"),
        {"ANTHROPIC_API_KEY": SECRET},
        ScriptedUI(),
    )
    assert "pip install" in (gov.unavailable_reason() or "")


# -- consent and flow --------------------------------------------------------------------------


def test_exact_payload_is_shown_before_the_first_call_and_must_be_approved() -> None:
    gov, fake, ui = make_governor([CRITIQUE_HIGH], answers=["y"])
    assert gov.enable() is None
    result, _ = run_govern(gov)
    payload = fake.calls[0]["messages"][0]["content"]
    assert payload in ui.text  # the user saw exactly what was sent
    assert "Send it? [y/N]" in ui.prompts[0]
    assert "some-claude-model" in ui.text
    assert result.used and result.final == "143"


def test_declining_turns_the_governor_off_and_sends_nothing() -> None:
    gov, fake, ui = make_governor([CRITIQUE_HIGH], answers=["n"])
    gov.enable()
    result, _ = run_govern(gov)
    assert fake.calls == [] and not gov.enabled and not result.used


def test_consent_is_asked_only_once_per_session() -> None:
    gov, fake, ui = make_governor([CRITIQUE_LOW, CRITIQUE_HIGH], answers=["y"])
    gov.enable()
    result, _ = run_govern(gov)
    assert len(fake.calls) == 2 and len(ui.prompts) == 1
    assert result.final == "better answer"


def test_revision_loop_stops_at_threshold() -> None:
    gov, fake, _ = make_governor([CRITIQUE_LOW, CRITIQUE_HIGH])
    gov.enable()
    result, revise_calls = run_govern(gov)
    assert [r.critique.score for r in result.rounds] == [4, 9]
    assert (
        result.final == "better answer" and result.stopped_reason == "score reached the threshold"
    )
    assert "recompute the sum" in revise_calls[0] and "wrong total" in revise_calls[0]
    assert (
        fake.calls[1]["messages"][0]["content"].count("better answer") == 1
    )  # re-critiques revision
    assert "wrong total" in json.dumps(result.to_log("m"))  # critique kept in its own log field


def test_round_cap_is_respected() -> None:
    gov, fake, _ = make_governor([CRITIQUE_LOW] * 5, max_rounds=2)
    gov.enable()
    result, revise_calls = run_govern(gov, ["v2", "v3", "v4"])
    assert len(fake.calls) == 2 and len(revise_calls) == 2
    assert result.stopped_reason == "round limit reached" and result.final == "v3"


def test_call_cap_stops_the_loop() -> None:
    gov, fake, ui = make_governor([CRITIQUE_LOW] * 5, max_rounds=5, max_calls_per_session=1)
    gov.enable()
    result, _ = run_govern(gov, ["v2", "v3"])
    assert len(fake.calls) == 1 and "cap" in ui.text
    assert gov.unavailable_reason() is not None


def test_token_cap_stops_the_loop() -> None:
    # Each call reserves max_tokens_per_call as its worst case, so the cap can never be exceeded.
    gov, fake, ui = make_governor(
        [CRITIQUE_LOW] * 5, max_rounds=5, max_tokens_per_session=2000, max_tokens_per_call=1500
    )
    gov.enable()
    run_govern(gov, ["v2", "v3", "v4", "v5", "v6"])
    assert 1 <= len(fake.calls) < 5 and "token cap" in ui.text
    assert gov.tokens <= 2000


def test_usage_is_counted_from_the_api_response() -> None:
    gov, _, _ = make_governor([CRITIQUE_HIGH])
    gov.enable()
    run_govern(gov)
    assert gov.tokens == 150 and gov.calls == 1


# -- failure handling ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply", [RuntimeError("network down"), "not json at all", '{"issues": []}', "REFUSAL"]
)
def test_any_failure_falls_back_to_the_local_answer(reply: Any) -> None:
    gov, fake, ui = make_governor([reply])
    gov.enable()
    result, revise_calls = run_govern(gov)
    assert result.final == "143" and not result.used and revise_calls == []
    assert "keeping the local answer" in ui.text
    assert gov.calls == 1  # the failed attempt still counts against the cap


def test_key_is_never_printed_logged_or_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    gov, fake, ui = make_governor([CRITIQUE_LOW, CRITIQUE_HIGH, RuntimeError(f"bad key {'x'}")])
    gov.enable()
    result, _ = run_govern(gov)
    blob = ui.text + json.dumps(result.to_log("m")) + json.dumps(fake.calls, default=str)
    assert SECRET not in blob and "SECRET123" not in blob
    # the real client factory is constructed with no arguments (the SDK reads the env itself)
    created: list[dict[str, Any]] = []
    module = types.ModuleType("anthropic")
    module.__spec__ = importlib.machinery.ModuleSpec("anthropic", None)

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    real = Governor(
        dataclasses.replace(default_config().governor, claude_model="m"),
        {"ANTHROPIC_API_KEY": SECRET},
        ScriptedUI(),
    )
    real._client()
    assert created == [{}]


# -- payload and parsing ---------------------------------------------------------------------


def test_payload_contains_only_prompt_draft_and_tool_results() -> None:
    payload = build_payload("python", "write add()", "def add(): ...", ["tests passed"])
    assert "write add()" in payload and "def add(): ..." in payload and "tests passed" in payload
    assert "Task type: python" in payload
    long_payload = build_payload("python", "p" * 50_000, "d" * 50_000, ["t" * 50_000] * 5)
    assert len(long_payload) < 6000 + 8000 + 6200 + 1000


def test_parse_critique_variants() -> None:
    fenced = parse_critique(
        'Here you go:\n```json\n{"score": 7.5, "issues": "one", "corrections": ["a"]}\n```'
    )
    assert (
        fenced.score == 7.5 and fenced.issues == ["one"] and fenced.suggested_corrections == ["a"]
    )
    assert parse_critique('{"score": 99}').score == 10
    assert parse_critique('{"score": -3}').score == 0
    with pytest.raises(ValueError):
        parse_critique('{"score": "high"}')
    with pytest.raises(ValueError):
        parse_critique("no json")


# -- logging and dataset -----------------------------------------------------------------------


def record(**kw: Any) -> TaskRecord:
    base: dict[str, Any] = {
        "mode": "code",
        "model": "qwen3:30b",
        "prompt": "write add",
        "final_answer": "def add(a, b): return a + b",
        "system": "You are AI Neo.",
        "verified": {"method": "tests", "passed": True},
    }
    base.update(kw)
    return TaskRecord(**base)


def test_log_file_naming_and_content(tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    rec = record(timestamp="2026-03-04T10:00:00+00:00", tool_calls=[{"name": "calculator"}])
    path = store.append(rec)
    store.append(record(timestamp="2026-03-04T11:00:00+00:00"))
    store.append(record(timestamp="2026-03-05T11:00:00+00:00"))
    assert path.name == "ai_neo-20260304.jsonl"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "logs" / "ai_neo-20260305.jsonl").exists()
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    for key in (
        "id",
        "timestamp",
        "mode",
        "model",
        "settings",
        "prompt",
        "local_draft",
        "tool_calls",
        "test_results",
        "governor",
        "final_answer",
        "stats",
    ):
        assert key in first
    assert datetime.fromisoformat(first["timestamp"]).year == 2026


def test_dataset_contains_only_verified_local_output(tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    claude_text = "CLAUDE SAYS: the draft is wrong because of UNIQUE_CRITIQUE_MARKER"
    governed = {"used": True, "rounds": [{"critique_text": claude_text, "score": 4}]}
    store.append(record(prompt="p-tests"))
    store.append(record(prompt="p-exact", verified={"method": "exact_match", "passed": True}))
    store.append(record(prompt="p-build", verified={"method": "build_tests", "passed": True}))
    store.append(record(prompt="p-self", verified={"method": "self-checks", "passed": True}))
    store.append(record(prompt="p-failed", verified={"method": "tests", "passed": False}))
    store.append(record(prompt="p-unverified", verified={"method": None, "passed": None}))
    store.append(record(prompt="p-governed", governor=governed))
    store.append(record(prompt="p-empty", final_answer="  "))
    out = tmp_path / "datasets" / "out.jsonl"
    stats = store.export_dataset(out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats == DatasetStats(total=8, exported=3, skipped_unverified=4, skipped_governed=1)
    assert sorted(r["messages"][1]["content"] for r in rows) == ["p-build", "p-exact", "p-tests"]
    text = out.read_text(encoding="utf-8")
    assert "UNIQUE_CRITIQUE_MARKER" not in text and "governor" not in text
    assert rows[0]["messages"][0] == {"role": "system", "content": "You are AI Neo."}
    assert set(rows[0]) == {"messages", "meta"} and rows[0]["meta"]["verified_by"] in {
        "tests",
        "exact_match",
        "build_tests",
    }


def test_governed_tasks_are_opt_in_and_still_exclude_critique_text(tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    claude_text = "UNIQUE_CRITIQUE_MARKER"
    store.append(record(governor={"used": True, "rounds": [{"critique_text": claude_text}]}))
    out = tmp_path / "out.jsonl"
    stats = store.export_dataset(out, include_governed=True)
    assert stats.exported == 1
    assert claude_text not in out.read_text(encoding="utf-8")


def test_corrupt_log_lines_are_skipped(tmp_path: Path) -> None:
    store = LogStore(tmp_path / "logs")
    path = store.append(record())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{half a line\n")
    store.append(record(prompt="second"))
    assert len(list(store.records())) == 2
    assert store.export_dataset(tmp_path / "o.jsonl").exported == 2
