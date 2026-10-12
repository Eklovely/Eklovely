"""Optional Claude "governor": critiques the local model's draft so the local model can revise it.

Off by default. It runs only after `--govern` or `/govern on` AND when ANTHROPIC_API_KEY is set
AND a Claude model name is configured (governor.claude_model or AI_NEO_CLAUDE_MODEL; there is no
built-in default). Only the prompt, the draft and tool results are ever sent. The key is never
read into a variable or logged: we only check that it exists, and the Anthropic SDK reads it from
the environment itself. Every API failure falls back to the local answer.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_neo.config import GovernorCfg
from ai_neo.context import estimate_tokens
from ai_neo.prompts import GOVERNOR_REVISE, GOVERNOR_SYSTEM, RUBRICS, fill
from ai_neo.textutil import extract_json_object
from ai_neo.ui import UI

KEY_ENV = "ANTHROPIC_API_KEY"
MAX_PROMPT_CHARS = 6000
MAX_DRAFT_CHARS = 8000
MAX_TOOL_CHARS = 2000
MAX_TOOLS_TOTAL = 6000


def key_present(env: Mapping[str, str]) -> bool:
    """True if the key variable exists and is non-empty. The value is never kept."""
    return bool(env.get(KEY_ENV))


@dataclass
class Critique:
    score: float
    issues: list[str]
    suggested_corrections: list[str]
    raw_text: str = ""  # Claude's own words; kept apart so datasets can exclude them


@dataclass
class GovernorRound:
    number: int
    critique: Critique
    revised: bool = False


@dataclass
class GovernResult:
    final: str
    used: bool = False
    rounds: list[GovernorRound] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0
    stopped_reason: str = ""

    def to_log(self, model: str) -> dict[str, Any]:
        return {
            "used": self.used,
            "claude_model": model,
            "calls": self.calls,
            "tokens": self.tokens,
            "stopped_reason": self.stopped_reason,
            "rounds": [
                {
                    "round": r.number,
                    "score": r.critique.score,
                    "issues": r.critique.issues,
                    "suggested_corrections": r.critique.suggested_corrections,
                    "critique_text": r.critique.raw_text,
                    "revised": r.revised,
                }
                for r in self.rounds
            ],
        }


def parse_critique(raw: str) -> Critique:
    """Tolerant parser for Claude's JSON critique. Raises ValueError if unusable."""
    obj = extract_json_object(raw)
    if obj is None or "score" not in obj:
        raise ValueError("the critique was not a JSON object with a score")
    try:
        score = max(0.0, min(10.0, float(obj["score"])))
    except (TypeError, ValueError):
        raise ValueError("the critique score was not a number") from None

    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value] if isinstance(value, list) else []

    corrections = obj.get("suggested_corrections", obj.get("corrections"))
    return Critique(score, strings(obj.get("issues")), strings(corrections), raw.strip())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def build_payload(task_type: str, prompt: str, draft: str, tool_results: Sequence[str]) -> str:
    """The exact user message sent to Claude: prompt, draft and tool results only."""
    parts = [
        f"Task type: {task_type}",
        f"Rubric: {RUBRICS.get(task_type, RUBRICS['reasoning'])}",
        "",
        "=== User prompt ===",
        _clip(prompt, MAX_PROMPT_CHARS),
        "",
        "=== Local model's draft answer ===",
        _clip(draft, MAX_DRAFT_CHARS),
    ]
    if tool_results:
        parts += ["", "=== Tool results ==="]
        budget = MAX_TOOLS_TOTAL
        for result in tool_results:
            chunk = _clip(result, min(MAX_TOOL_CHARS, budget))
            parts.append(chunk)
            budget -= len(chunk)
            if budget <= 0:
                break
    return "\n".join(parts)


class Governor:
    def __init__(
        self,
        cfg: GovernorCfg,
        env: Mapping[str, str],
        ui: UI,
        *,
        assume_yes: bool = False,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.ui = ui
        self.assume_yes = assume_yes
        self._client_factory = client_factory
        self.enabled = False
        self.calls = 0
        self.tokens = 0
        self._consented = False

    # -- availability --------------------------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        if not self.cfg.claude_model:
            return (
                "no Claude model is configured (set governor.claude_model in config.toml "
                "or the AI_NEO_CLAUDE_MODEL environment variable)"
            )
        if not key_present(self.env):
            return f"the {KEY_ENV} environment variable is not set"
        if self._client_factory is None and importlib.util.find_spec("anthropic") is None:
            return "the 'anthropic' package is not installed (run: pip install -e .[claude])"
        if self.calls >= self.cfg.max_calls_per_session:
            return (
                f"the per-session cap of {self.cfg.max_calls_per_session} Claude calls was reached"
            )
        if self.tokens >= self.cfg.max_tokens_per_session:
            return f"the per-session cap of {self.cfg.max_tokens_per_session} tokens was reached"
        return None

    def enable(self) -> str | None:
        """Turn the governor on for this session. Returns the reason if that is not possible."""
        reason = self.unavailable_reason()
        if reason is None:
            self.enabled = True
        return reason

    def disable(self) -> None:
        self.enabled = False

    def reset_consent(self) -> None:
        """Forget the approval: the next call shows what it will send and asks again."""
        self._consented = False

    # -- one critique --------------------------------------------------------------------------

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        import anthropic

        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY itself; we never touch the value

    def _consent(self, payload: str) -> bool:
        if self._consented:
            self.ui.say(
                f"[governor: sending {len(payload)} characters to Claude "
                f"(call {self.calls + 1} of {self.cfg.max_calls_per_session})]",
                dim=True,
            )
            return True
        body = (
            f"Destination: Anthropic API, model '{self.cfg.claude_model}'\n"
            f"Limits: {self.cfg.max_calls_per_session} calls and "
            f"{self.cfg.max_tokens_per_session} tokens per session\n"
            "Nothing else (no files, no history, no environment) is sent.\n\n"
            f"--- system message ---\n{GOVERNOR_SYSTEM}\n\n--- user message ---\n{payload}"
        )
        self.ui.preview("The governor will send exactly this to Anthropic (costs money)", body)
        if self.assume_yes or self.ui.confirm("Send it?", default=False):
            self._consented = True
            return True
        self.disable()
        self.ui.tag("WARN", "Governor turned off for this session (you declined).")
        return False

    def review(
        self, task_type: str, prompt: str, draft: str, tool_results: Sequence[str] = ()
    ) -> Critique | None:
        """Ask Claude to critique `draft`. Returns None (and explains why) on any problem."""
        if getattr(self.ui, "stop_requested", False):
            return None  # the user pressed Stop: nothing leaves this computer
        reason = self.unavailable_reason()
        if reason:
            self.ui.tag("WARN", f"Governor skipped: {reason}.")
            return None
        payload = build_payload(task_type, prompt, draft, tool_results)
        estimate = estimate_tokens(payload) + estimate_tokens(GOVERNOR_SYSTEM)
        if self.tokens + estimate + self.cfg.max_tokens_per_call > self.cfg.max_tokens_per_session:
            self.ui.tag("WARN", "Governor skipped: it would exceed the per-session token cap.")
            return None
        if not self._consent(payload):
            return None
        self.calls += 1
        try:
            response = self._client().messages.create(
                model=self.cfg.claude_model,
                max_tokens=self.cfg.max_tokens_per_call,
                system=GOVERNOR_SYSTEM,
                messages=[{"role": "user", "content": payload}],
            )
            self.tokens += self._usage(response, estimate)
            if getattr(response, "stop_reason", None) == "refusal":
                raise RuntimeError("Claude declined to review this content")
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            return parse_critique(text)
        except Exception as exc:  # deliberate: any API/parse failure must fall back to local
            self.ui.tag(
                "WARN",
                f"Governor unavailable ({type(exc).__name__}: {str(exc)[:160]}); "
                "keeping the local answer.",
            )
            return None

    @staticmethod
    def _usage(response: Any, fallback: int) -> int:
        usage = getattr(response, "usage", None)
        used = int(getattr(usage, "input_tokens", 0) or 0) + int(
            getattr(usage, "output_tokens", 0) or 0
        )
        return used or fallback


def format_critique(critique: Critique) -> tuple[str, str]:
    issues = "\n".join(f"- {i}" for i in critique.issues) or "- (none listed)"
    fixes = "\n".join(f"- {c}" for c in critique.suggested_corrections) or "- (none listed)"
    return issues, fixes


def revision_prompt(critique: Critique) -> str:
    issues, fixes = format_critique(critique)
    return fill(GOVERNOR_REVISE, score=f"{critique.score:g}", issues=issues, corrections=fixes)


def govern(
    gov: Governor,
    *,
    task_type: str,
    prompt: str,
    draft: str,
    tool_results: Sequence[str],
    revise: Callable[[str], str | None],
) -> GovernResult:
    """Critique -> local revision, up to max_rounds or until the score reaches the threshold.

    `revise(revision_prompt)` runs the local model and returns the revised answer (or None).
    """
    result = GovernResult(final=draft)
    if not gov.enabled:
        return result
    current = draft
    for number in range(1, gov.cfg.max_rounds + 1):
        critique = gov.review(task_type, prompt, current, tool_results)
        if critique is None:
            result.stopped_reason = "governor unavailable or declined"
            break
        entry = GovernorRound(number, critique)
        result.rounds.append(entry)
        result.used = True
        gov.ui.say(f"[governor round {number}: Claude scored the draft {critique.score:g}/10]")
        if critique.score >= gov.cfg.score_threshold:
            result.stopped_reason = "score reached the threshold"
            break
        revised = revise(revision_prompt(critique))
        if not revised or not revised.strip() or revised.strip() == current.strip():
            result.stopped_reason = "the local model produced no revision"
            break
        entry.revised = True
        current = revised
    else:
        result.stopped_reason = "round limit reached"
    result.final = current
    result.calls = gov.calls
    result.tokens = gov.tokens
    return result
