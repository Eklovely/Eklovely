"""Plain data types shared by the client, agent and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A tool call requested by the model. `arguments` is untrusted and may be malformed."""

    name: str
    arguments: Any = field(default_factory=dict)


@dataclass
class Message:
    """One chat message. There is deliberately no field for thinking text."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None

    def to_ollama(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls
            ]
        if self.role == "tool" and self.tool_name:
            out["tool_name"] = self.tool_name
        return out

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {"name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_name:
            out["tool_name"] = self.tool_name
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        calls = [ToolCall(c["name"], c.get("arguments", {})) for c in data.get("tool_calls", [])]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=calls,
            tool_name=data.get("tool_name"),
        )


@dataclass(frozen=True)
class Sampling:
    """Sampler settings sent as Ollama `options`. Greedy decoding is never used."""

    temperature: float
    top_p: float
    top_k: int
    num_ctx: int
    num_predict: int

    def options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


@dataclass
class GenStats:
    """Timing and token counts for one or more generations (durations in nanoseconds)."""

    eval_count: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    wall_seconds: float = 0.0

    @property
    def tok_per_s(self) -> float:
        if self.eval_duration_ns > 0:
            return self.eval_count / (self.eval_duration_ns / 1e9)
        if self.wall_seconds > 0:
            return self.eval_count / self.wall_seconds
        return 0.0

    @property
    def seconds(self) -> float:
        if self.wall_seconds > 0:
            return self.wall_seconds
        return self.total_duration_ns / 1e9

    def __iadd__(self, other: GenStats) -> GenStats:
        self.eval_count += other.eval_count
        self.eval_duration_ns += other.eval_duration_ns
        self.total_duration_ns += other.total_duration_ns
        self.load_duration_ns += other.load_duration_ns
        self.prompt_eval_count += other.prompt_eval_count
        self.wall_seconds += other.wall_seconds
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_count": self.eval_count,
            "tok_per_s": round(self.tok_per_s, 2),
            "seconds": round(self.seconds, 2),
            "load_seconds": round(self.load_duration_ns / 1e9, 2),
            "prompt_tokens": self.prompt_eval_count,
        }


@dataclass
class ChatResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stats: GenStats = field(default_factory=GenStats)
    interrupted: bool = False
    done_reason: str | None = None


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["thinking", "answer"]
    text: str
