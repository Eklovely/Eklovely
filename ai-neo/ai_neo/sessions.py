"""Chat sessions: the data model, JSON persistence and markdown export."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_neo.prompts import MODES
from ai_neo.types import Message

SESSION_VERSION = 1


class SessionNotFound(Exception):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Session:
    mode: str
    model: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str | None = None
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    system_override: str | None = None
    created: str = field(default_factory=now_iso)
    turns: int = 0
    generated_tokens: int = 0
    generation_seconds: float = 0.0

    def system_prompt(self) -> str:
        base = self.system_override or MODES[self.mode].system
        if self.summary:
            note = "Summary of the earlier conversation (older turns were condensed):"
            return f"{base}\n\n{note}\n{self.summary}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "model": self.model,
            "created": self.created,
            "summary": self.summary,
            "system_override": self.system_override,
            "turns": self.turns,
            "generated_tokens": self.generated_tokens,
            "generation_seconds": round(self.generation_seconds, 2),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        mode = data.get("mode", "reason")
        return cls(
            mode=mode if mode in MODES else "reason",
            model=data.get("model", ""),
            id=data.get("id") or uuid.uuid4().hex[:12],
            name=data.get("name"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            summary=data.get("summary", ""),
            system_override=data.get("system_override"),
            created=data.get("created", now_iso()),
            turns=int(data.get("turns", 0)),
            generated_tokens=int(data.get("generated_tokens", 0)),
            generation_seconds=float(data.get("generation_seconds", 0.0)),
        )


_WINDOWS_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def safe_name(name: str) -> str:
    """A file-name-safe version of `name`. Letters and digits of any language are kept (so two
    different non-English names stay different); separators and control characters become `_`,
    and names Windows reserves for devices (CON, NUL, ...) get a leading underscore."""
    cleaned = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")[:60] or "session"
    if cleaned.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES:
        cleaned = "_" + cleaned
    return cleaned


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, name: str) -> Path:
        return self.directory / f"{safe_name(name)}.json"

    def save(self, session: Session, name: str | None = None) -> Path:
        label = name or session.name or f"{session.mode}-{session.created[:19].replace(':', '')}"
        session.name = safe_name(label)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session.name)
        temp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.json.tmp")  # unique per save
        temp.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temp, path)
        return path

    def load(self, name: str | None = None) -> Session:
        if name:
            path = self.path_for(name)
        else:
            candidates = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise SessionNotFound("There are no saved sessions yet. Use /save first.")
            path = candidates[-1]
        if not path.exists():
            raise SessionNotFound(
                f"No saved session called '{name}'. Use /load with no name to see the latest."
            )
        try:
            return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise SessionNotFound(f"{path.name} is not a readable session file ({exc}).") from exc

    def list(self) -> list[tuple[str, str, int]]:
        """(name, mode, message count) for each saved session, newest first."""
        if not self.directory.exists():
            return []
        out = []
        for path in sorted(
            self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append((path.stem, str(data.get("mode", "?")), len(data.get("messages", []))))
            except (ValueError, OSError):
                continue
        return out


def export_markdown(session: Session) -> str:
    lines = [
        f"# AI Neo session {session.name or session.id}",
        "",
        f"- Mode: {session.mode}",
        f"- Model: {session.model}",
        f"- Started: {session.created}",
        "",
    ]
    if session.summary:
        lines += ["## Summary of earlier conversation", "", session.summary, ""]
    for message in session.messages:
        if message.role == "user":
            lines += ["## You", "", message.content, ""]
        elif message.role == "assistant":
            lines += ["## AI Neo", "", message.content or "(tool call)", ""]
            for call in message.tool_calls:
                lines += [
                    "```json",
                    json.dumps({call.name: call.arguments}, indent=2, default=str),
                    "```",
                    "",
                ]
        elif message.role == "tool":
            lines += [
                f"### Tool result: {message.tool_name}",
                "",
                "```",
                message.content,
                "```",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"
