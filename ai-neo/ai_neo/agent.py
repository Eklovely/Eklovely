"""The bounded tool-calling loop: one user turn in, one answer (plus tool events) out."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_neo.config import Config
from ai_neo.context import ContextManager
from ai_neo.ollama_client import OllamaClient, OllamaError, ToolCallParseError
from ai_neo.prompts import (
    MODES,
    NON_THINKING,
    STEP_LIMIT_NOTE,
    TOOL_CALL_RETRY,
    ModePreset,
    fill,
)
from ai_neo.sessions import Session
from ai_neo.tools import ToolRegistry, ToolResult
from ai_neo.types import GenStats, Message, Sampling, ToolCall
from ai_neo.ui import UI

_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


@dataclass
class ToolEvent:
    name: str
    arguments: Any
    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, limit: int = 4000) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "output": self.output[:limit],
            "meta": self.meta,
        }


@dataclass
class TurnResult:
    answer: str = ""
    thinking: str = ""
    tool_events: list[ToolEvent] = field(default_factory=list)
    stats: GenStats = field(default_factory=GenStats)
    steps: int = 0
    interrupted: bool = False
    step_limit_hit: bool = False
    truncated: bool = False
    error: str | None = None


def resolve_think(setting: str, override: bool | None, preset: ModePreset) -> bool:
    """/think override > config (`on`/`off`) > the mode's own default."""
    if override is not None:
        return override
    if setting in ("on", "off"):
        return setting == "on"
    return preset.think_default


def sampling_for(cfg: Config, preset: ModePreset, think: bool) -> Sampling:
    chosen = preset.sampling(think)
    return Sampling(
        temperature=chosen.temperature,
        top_p=chosen.top_p,
        top_k=chosen.top_k,
        num_ctx=cfg.ollama.num_ctx,
        num_predict=cfg.ollama.num_predict,
    )


def extract_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Fallback for models that print `<tool_call>{json}</tool_call>` as plain text."""
    calls: list[ToolCall] = []
    for match in _TEXT_TOOL_CALL.finditer(content):
        raw = match.group(1)
        try:
            obj = json.loads(raw)
            calls.append(ToolCall(str(obj["name"]), obj.get("arguments", {})))
        except (ValueError, KeyError, TypeError, AttributeError):
            calls.append(ToolCall("", raw))
    cleaned = _TEXT_TOOL_CALL.sub("", content).strip() if calls else content
    return calls, cleaned


class Agent:
    def __init__(self, client: OllamaClient, registry: ToolRegistry, cfg: Config, ui: UI) -> None:
        self.client = client
        self.registry = registry
        self.cfg = cfg
        self.ui = ui
        self.context = ContextManager(
            cfg.ollama.num_ctx, cfg.context.summarize_at, cfg.context.keep_last_turns
        )

    # -- helpers -----------------------------------------------------------------------------

    def summarize(self, prompt: str) -> str:
        sampling = Sampling(
            NON_THINKING.temperature,
            NON_THINKING.top_p,
            NON_THINKING.top_k,
            self.cfg.ollama.num_ctx,
            700,
        )
        try:
            result = self.client.chat([Message("user", prompt)], sampling=sampling, think=False)
        except OllamaError:
            return ""
        return result.content

    def _compact(self, session: Session) -> None:
        self.context.compact(
            session, self.summarize, lambda note: self.ui.say(f"[{note}]", dim=True)
        )

    def generate(
        self,
        messages: Sequence[Message],
        mode: str,
        think: bool,
        *,
        tools: list[dict[str, Any]] | None = None,
        show: bool = True,
    ) -> Any:
        """One streamed model call outside any session (used by verify, build and evals)."""
        sampling = sampling_for(self.cfg, MODES[mode], think)
        renderer = self.ui.stream()
        try:
            return self.client.chat(
                messages,
                sampling=sampling,
                think=think,
                tools=tools,
                on_event=renderer.event if show else None,
            )
        finally:
            renderer.close()

    # -- the loop ----------------------------------------------------------------------------

    def run_turn(
        self,
        session: Session,
        user_text: str,
        *,
        think: bool,
        tool_names: Sequence[str] | None = None,
        use_tools: bool = True,
    ) -> TurnResult:
        preset = MODES[session.mode]
        names = tuple(tool_names) if tool_names is not None else preset.tools
        specs = self.registry.specs(names) if use_tools and names else None
        sampling = sampling_for(self.cfg, preset, think)
        result = TurnResult()
        session.messages.append(Message("user", user_text))
        thinking_parts: list[str] = []
        max_steps = self.cfg.tools.max_steps
        try:
            for step in range(1, max_steps + 1):
                result.steps = step
                is_last = step == max_steps
                self._compact(session)
                outgoing = [Message("system", session.system_prompt()), *session.messages]
                if is_last and specs and step > 1:
                    outgoing.append(Message("user", STEP_LIMIT_NOTE))
                renderer = self.ui.stream()
                try:
                    chat = self.client.chat(
                        outgoing,
                        sampling=sampling,
                        think=think,
                        tools=None if is_last else specs,
                        on_event=renderer.event,
                    )
                except ToolCallParseError as exc:
                    renderer.close()
                    self.ui.say(
                        "[the model wrote a malformed tool call; asking it to retry]", dim=True
                    )
                    problem = str(exc)[:300]
                    session.messages.append(Message("user", fill(TOOL_CALL_RETRY, problem=problem)))
                    continue
                except OllamaError as exc:
                    renderer.close()
                    result.error = str(exc)
                    self._keep_partial(session, exc.partial.content if exc.partial else "", result)
                    break
                renderer.close()
                result.stats += chat.stats
                if chat.thinking:
                    thinking_parts.append(chat.thinking)
                calls = list(chat.tool_calls)
                content = chat.content
                if not calls and specs:
                    calls, content = extract_text_tool_calls(content)
                if chat.interrupted:
                    result.interrupted = True
                    self._keep_partial(session, content, result)
                    break
                if is_last and calls:
                    result.step_limit_hit = True
                    calls = []
                session.messages.append(Message("assistant", content, tool_calls=calls))
                if not calls:
                    result.answer = content
                    result.truncated = chat.done_reason == "length"
                    result.step_limit_hit = result.step_limit_hit or (is_last and not content)
                    break
                self._run_tools(session, calls, result)
                if result.interrupted:
                    break
        except KeyboardInterrupt:
            result.interrupted = True
        result.thinking = "\n\n".join(thinking_parts)
        session.turns += 1
        session.generated_tokens += result.stats.eval_count
        session.generation_seconds += result.stats.seconds
        return result

    def _keep_partial(self, session: Session, text: str, result: TurnResult) -> None:
        if text.strip():
            session.messages.append(Message("assistant", text))
            result.answer = text

    def _run_tools(self, session: Session, calls: list[ToolCall], result: TurnResult) -> None:
        for index, call in enumerate(calls):
            name = call.name or "unknown"
            try:
                self.ui.tool_call(name, call.arguments)
                outcome: ToolResult = self.registry.execute(call)
            except KeyboardInterrupt:
                result.interrupted = True
                for rest in calls[index:]:
                    note = "ERROR: interrupted by the user before this tool ran."
                    session.messages.append(Message("tool", note, tool_name=rest.name or "unknown"))
                return
            self.ui.tool_result(outcome.ok, outcome.output)
            result.tool_events.append(
                ToolEvent(name, call.arguments, outcome.ok, outcome.output, dict(outcome.meta))
            )
            session.messages.append(Message("tool", outcome.output, tool_name=name))
