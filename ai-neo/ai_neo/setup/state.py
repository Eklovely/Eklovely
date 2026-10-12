"""Remembers which setup steps are finished, so the wizard can resume and be re-run safely."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_VERSION = 1


@dataclass
class StepRecord:
    status: str  # done | skipped | failed
    at: str
    result: str = ""
    message: str = ""


@dataclass
class SetupState:
    steps: dict[str, StepRecord] = field(default_factory=dict)
    was_corrupt: bool = False

    @classmethod
    def load(cls, path: Path) -> SetupState:
        if not path.exists():
            return cls()
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            steps = {
                key: StepRecord(
                    **{k: v for k, v in value.items() if k in StepRecord.__annotations__}
                )
                for key, value in data.get("steps", {}).items()
            }
            return cls(steps=steps)
        except (ValueError, TypeError, OSError, AttributeError):
            return cls(was_corrupt=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": STATE_VERSION,
            "steps": {key: vars(record) for key, record in self.steps.items()},
        }
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(body, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def finished(self, step_id: str) -> StepRecord | None:
        record = self.steps.get(step_id)
        return record if record is not None and record.status == "done" else None

    def record(self, step_id: str, status: str, result: str, message: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self.steps[step_id] = StepRecord(status, stamp, result, message[:300])
