"""Context-window management.

Token counts are estimated as characters / 4. That is a rough rule of thumb for English text and
code (Qwen3's tokenizer is a little less efficient on other languages and symbols), so the 75%
trigger leaves headroom. Ollama's `prompt_eval_count` is NOT used to correct the estimate because
it only counts tokens that were not already in the KV cache.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from ai_neo.prompts import SUMMARIZE, fill
from ai_neo.sessions import Session
from ai_neo.types import Message

CHARS_PER_TOKEN = 4
TRANSCRIPT_MSG_CHARS = 1500
TOOL_OUTPUT_KEEP = 800


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def estimate_message(message: Message) -> int:
    total = estimate_tokens(message.content) + 4
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(
            json.dumps(call.arguments, default=str)
        )
    return total


def estimate_messages(messages: list[Message]) -> int:
    return sum(estimate_message(m) for m in messages)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ...[cut]"


def render_transcript(messages: list[Message], max_chars: int) -> str:
    """One line per message; each is clipped so the whole transcript fits in `max_chars`."""
    share = max(120, max_chars // max(1, len(messages)) - 40)
    lines = []
    for m in messages:
        default = TOOL_OUTPUT_KEEP // 2 if m.role == "tool" else TRANSCRIPT_MSG_CHARS
        label = f"tool({m.tool_name})" if m.role == "tool" else m.role
        text = _clip(m.content, min(default, share))
        for call in m.tool_calls:
            text += f" [called {call.name}]"
        lines.append(f"{label}: {text}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = "...[earlier part omitted]\n" + transcript[-max_chars:]
    return transcript


class ContextManager:
    def __init__(self, num_ctx: int, summarize_at: float, keep_last_turns: int) -> None:
        self.num_ctx = num_ctx
        self.summarize_at = summarize_at
        self.keep_last_turns = keep_last_turns

    @property
    def budget(self) -> int:
        return int(self.num_ctx * self.summarize_at)

    def used(self, session: Session) -> int:
        return estimate_tokens(session.system_prompt()) + estimate_messages(session.messages)

    def percent(self, session: Session) -> int:
        return round(100 * self.used(session) / self.num_ctx)

    def needs_compaction(self, session: Session) -> bool:
        return self.used(session) > self.budget

    def split(self, messages: list[Message]) -> tuple[list[Message], list[Message]]:
        """Split into (older, recent); `recent` starts at the Nth-last user message."""
        user_positions = [i for i, m in enumerate(messages) if m.role == "user"]
        if len(user_positions) <= self.keep_last_turns:
            return [], messages
        cut = user_positions[-self.keep_last_turns]
        return messages[:cut], messages[cut:]

    def compact(
        self,
        session: Session,
        summarizer: Callable[[str], str],
        notify: Callable[[str], None],
    ) -> bool:
        """Summarize old turns if the history is over budget. Returns True if anything changed."""
        if not self.needs_compaction(session):
            return False
        before = self.percent(session)
        changed = False
        older, recent = self.split(session.messages)
        if older:
            previous = f"Earlier summary:\n{session.summary}\n\n" if session.summary else ""
            max_chars = max(2000, int(self.num_ctx * CHARS_PER_TOKEN * 0.5))
            prompt = fill(
                SUMMARIZE, previous=previous, transcript=render_transcript(older, max_chars)
            )
            summary = summarizer(prompt).strip()
            if not summary:
                summary = (session.summary + "\n" if session.summary else "") + (
                    "(some earlier messages were dropped to save space)"
                )
            session.summary = summary
            session.messages = recent
            changed = True
            notify(
                f"Context was {before}% full: summarized {len(older)} older messages "
                f"(kept the system prompt and the last {self.keep_last_turns} turns)."
            )
        if self.needs_compaction(session) and self._shrink_tool_outputs(session):
            changed = True
            notify("Context is still large: shortened old tool outputs.")
        return changed

    @staticmethod
    def _shrink_tool_outputs(session: Session) -> bool:
        changed = False
        for index, message in enumerate(session.messages[:-1]):
            if message.role == "tool" and len(message.content) > TOOL_OUTPUT_KEEP:
                session.messages[index] = Message(
                    "tool",
                    _clip(message.content, TOOL_OUTPUT_KEEP),
                    tool_name=message.tool_name,
                )
                changed = True
        return changed
