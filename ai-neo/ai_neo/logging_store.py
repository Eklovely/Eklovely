"""Local task log (one JSON line per task) and dataset export.

Everything stays on this machine. The log keeps Claude's critique text in its own `governor` field.

`export_dataset` writes only the local model's own outputs that passed an automated check
(provided tests, project tests, or an exact-match answer) and NEVER includes any governor field or
critique text. By default it also skips tasks where the governor was used, because a revision shaped
by Claude's feedback is no longer purely the local model's output.

NOTE: before using any Claude-generated text for training, check Anthropic's current terms of use.
This tool does not do that for you; it simply keeps Claude's text out of the dataset.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Verification methods strong enough for a training dataset. "self-checks" (the model's own
# asserts) and "ran-without-checks" are deliberately excluded.
DATASET_METHODS = frozenset({"tests", "exact_match", "build_tests"})


def new_id() -> str:
    return uuid.uuid4().hex


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class TaskRecord:
    mode: str
    model: str
    prompt: str
    final_answer: str
    system: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    local_draft: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    test_results: dict[str, Any] | None = None
    governor: dict[str, Any] | None = None
    verified: dict[str, Any] = field(default_factory=lambda: {"method": None, "passed": None})
    stats: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    timestamp: str = field(default_factory=timestamp)


@dataclass
class DatasetStats:
    total: int = 0
    exported: int = 0
    skipped_unverified: int = 0
    skipped_governed: int = 0


class LogStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, when: str) -> Path:
        day = datetime.fromisoformat(when).strftime("%Y%m%d")
        return self.directory / f"ai_neo-{day}.jsonl"

    def append(self, record: TaskRecord) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(record.timestamp)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return path

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("ai_neo-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue  # a half-written line must not break export
                    if isinstance(value, dict):
                        yield value

    def export_dataset(self, out: Path, *, include_governed: bool = False) -> DatasetStats:
        stats = DatasetStats()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for record in self.records():
                stats.total += 1
                verified = record.get("verified") or {}
                if (
                    verified.get("passed") is not True
                    or verified.get("method") not in DATASET_METHODS
                ):
                    stats.skipped_unverified += 1
                    continue
                governor = record.get("governor") or {}
                if governor.get("used") and not include_governed:
                    stats.skipped_governed += 1
                    continue
                row = dataset_row(record)
                if row is None:
                    stats.skipped_unverified += 1
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats.exported += 1
        return stats


def dataset_row(record: dict[str, Any]) -> dict[str, Any] | None:
    """Whitelist of fields only: no governor data, thinking, tool output or critique text."""
    prompt, answer = record.get("prompt"), record.get("final_answer")
    if not isinstance(prompt, str) or not isinstance(answer, str) or not answer.strip():
        return None
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return {
        "messages": messages,
        "meta": {
            "id": record.get("id"),
            "mode": record.get("mode"),
            "model": record.get("model"),
            "verified_by": (record.get("verified") or {}).get("method"),
        },
    }
