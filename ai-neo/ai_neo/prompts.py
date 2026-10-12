"""Every prompt and sampling preset AI Neo uses. Edit this file to tune behaviour.

Templates use `string.Template` placeholders (`$name`) so JSON braces need no escaping.
Use `fill(template, name=value)` to render one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from string import Template
from types import MappingProxyType


def fill(template: str, **values: str) -> str:
    return Template(template).safe_substitute(values)


@dataclass(frozen=True)
class SamplingPreset:
    temperature: float
    top_p: float
    top_k: int


# Qwen3's recommended settings. Never set temperature to 0 (greedy decoding loops and degrades).
THINKING = SamplingPreset(temperature=0.6, top_p=0.95, top_k=20)
NON_THINKING = SamplingPreset(temperature=0.7, top_p=0.8, top_k=20)

BASE = (
    "You are AI Neo, a careful assistant running locally on the user's own computer. "
    "Generation is slow (roughly 10 tokens per second), so be direct: no filler, no restating the "
    "question, no unnecessary preamble. If you are unsure, say so. Never invent facts, file "
    "contents or tool results."
)


@dataclass(frozen=True)
class ModePreset:
    name: str
    summary: str
    system: str
    think_default: bool
    task_type: str  # reasoning | computation | python | writing | build (governor rubric, evals)
    tools: tuple[str, ...] = ()
    sampling_think: SamplingPreset = THINKING
    sampling_plain: SamplingPreset = NON_THINKING

    def sampling(self, think: bool) -> SamplingPreset:
        return self.sampling_think if think else self.sampling_plain


MODES: Mapping[str, ModePreset] = MappingProxyType(
    {
        "reason": ModePreset(
            name="reason",
            summary="step-by-step reasoning and analysis",
            task_type="reasoning",
            think_default=True,
            tools=("calculator",),
            system=(
                BASE
                + "\n\nMode: reasoning. Work through the problem carefully, state assumptions, "
                "and check the result before answering. Use the `calculator` tool for any exact "
                "arithmetic instead of computing in your head. Finish with a short, clear answer; "
                "when the question has a definite answer, end with a line `Final answer: ...`."
            ),
        ),
        "compute": ModePreset(
            name="compute",
            summary="exact computation using tools",
            task_type="computation",
            think_default=True,
            tools=("calculator", "run_python"),
            system=(
                BASE
                + "\n\nMode: computation. Never do arithmetic in your head. Use `calculator` for "
                "expressions and `run_python` for anything longer (loops, large numbers, dates, "
                "statistics). Show the expression or code you used, then the result exactly as the "
                "tool returned it, and end with a line `Final answer: ...`."
            ),
        ),
        "code": ModePreset(
            name="code",
            summary="Python programming with run-and-repair",
            task_type="python",
            think_default=True,
            tools=("run_python", "read_file", "write_file", "list_dir", "run_tests"),
            system=(
                BASE
                + "\n\nMode: Python programming. Write correct, idiomatic Python 3.11+ using only "
                "the standard library unless asked otherwise. Reply with ONE complete module in a "
                "single ```python block, then at most a few sentences of explanation. Put a short "
                '`if __name__ == "__main__":` block at the end with `assert` checks of the key '
                "behaviour: your code will be run, and any failure is sent back to you to fix. "
                "Never call input(). Never include tests for things you did not implement."
            ),
        ),
        "write": ModePreset(
            name="write",
            summary="writing and editing prose",
            task_type="writing",
            think_default=False,
            tools=(),
            system=(
                BASE
                + "\n\nMode: writing. Write clear, natural prose in the requested tone and length. "
                "Respect word counts, required points and banned words exactly. Return only the "
                "text itself unless asked to explain."
            ),
        ),
        "build": ModePreset(
            name="build",
            summary="plan and build a small software project in the workspace",
            task_type="build",
            think_default=False,
            tools=("read_file", "write_file", "list_dir", "run_tests", "run_python"),
            system=(
                BASE
                + "\n\nMode: building small software projects. Prefer few, small files, the Python "
                "standard library, and `unittest`-style tests. Ask for the project description if "
                "the user has not given one."
            ),
        ),
    }
)

SUMMARIZE = (
    "Summarize the earlier part of this conversation so it can replace the original text. Keep: "
    "the user's goals, decisions made, facts and numbers that matter, file names, code that is "
    "still in use (short snippets only), and unresolved questions. Drop pleasantries and dead "
    "ends. Write compact bullet points, under 300 words.\n\n"
    "${previous}"
    "Conversation to summarize:\n$transcript"
)

COMPUTE_NUDGE = (
    "You answered without using a tool. Compute the result with `calculator` or `run_python` now, "
    "then give the final answer based on the tool output."
)

TOOL_CALL_RETRY = (
    "Your last tool call could not be parsed: $problem\n"
    "Call the tool again with valid JSON arguments, or answer without a tool."
)

STEP_LIMIT_NOTE = (
    "You have used all available tool steps. Do not call any more tools. Give your best final "
    "answer now, and say plainly what you could not verify."
)

CODE_TASK = "$request\n\nReply with the complete solution as ONE ```python block.$test_note"

CODE_TEST_NOTE = (
    " Your code is saved as `solution.py` and will be checked by these tests (do not repeat "
    "them):\n```python\n$tests\n```"
)

CODE_SELF_TEST_NOTE = (
    ' End the file with an `if __name__ == "__main__":` block of `assert` checks for the key '
    "behaviour; the file will be run."
)

CODE_REPAIR = (
    "Your code failed.\n\nResult ($runner, exit code $returncode):\n```\n$output\n```\n\n"
    "Fix the problem. Reply with the complete corrected solution as ONE ```python block and one "
    "sentence on what was wrong."
)

CODE_NO_BLOCK = (
    "I could not find a ```python code block in your reply. Reply with the complete solution as "
    "ONE ```python block."
)

BUILD_PLAN = (
    "Plan a small software project for this request:\n\n$request\n\n"
    "Reply with ONLY a JSON object (no prose) shaped like:\n"
    '{"name": "kebab-case-project-name", "summary": "one or two sentences", '
    '"files": [{"path": "relative/path.py", "purpose": "what this file contains"}]}\n'
    "Rules: at most $max_files files; relative paths only; Python standard library only unless "
    "the request says otherwise; include at least one test file named like tests/test_<thing>.py "
    "(or test_<thing>.py) using the `unittest` module; no data files unless essential; list "
    "implementation files before test files.$feedback"
)

BUILD_FILE = (
    'You are building the project "$name": $summary\n\n'
    "Planned files:\n$file_list\n\n"
    "Files already written (signatures only):\n$digest\n\n"
    "Now write the file `$path` ($purpose). Reply with the complete file contents in ONE fenced "
    "code block and nothing else. Tests must use `unittest` and import modules as the project "
    "root would (for example `from textstats import count_words`)."
)

BUILD_FIX = (
    'The project "$name" has failing tests.\n\nTest output:\n```\n$output\n```\n\n'
    "Current files:\n$files\n\n"
    "Fix the problem with the smallest change. For EACH file you change, reply with a line "
    "`FILE: relative/path` followed by the complete new contents in one fenced code block. "
    "Change only files that need it. Prefer fixing the implementation over editing the tests, "
    "unless a test is plainly wrong."
)

GOVERNOR_SYSTEM = (
    "You are a strict reviewer of an answer produced by a smaller local model. You will be given "
    "the task type, the user's prompt, the local model's draft and any tool results. Judge the "
    "draft against the rubric and reply with ONLY a JSON object:\n"
    '{"score": <number 0-10>, "issues": ["..."], "suggested_corrections": ["..."]}\n'
    "Be specific and brief. Do not rewrite the whole answer. A score of 8 or more means the draft "
    "is acceptable as is."
)

RUBRICS: Mapping[str, str] = MappingProxyType(
    {
        "reasoning": (
            "Is the conclusion correct and supported by valid steps? Are assumptions stated? "
            "Are there arithmetic or logic errors? Does it answer the question that was asked?"
        ),
        "computation": (
            "Is every numeric result correct? Was the computation done with the tools and does the "
            "final answer match the tool output? Are units and rounding right?"
        ),
        "python": (
            "Is the code correct for the stated requirements, including edge cases? Is it "
            "idiomatic, free of unsafe constructs, and did the tests or run results pass? "
            "Would it fail on empty, large or unusual input?"
        ),
        "writing": (
            "Does it follow every explicit instruction (length, tone, required points, banned "
            "words)? Is it clear, accurate, well organised and free of filler or errors?"
        ),
        "build": (
            "Does the project meet the request? Are the files coherent and small, do the tests "
            "exercise real behaviour, and did they pass? Anything missing or fragile?"
        ),
    }
)

GOVERNOR_REVISE = (
    "A reviewer critiqued your last answer.\n\nScore: $score/10\nIssues:\n$issues\n"
    "Suggested corrections:\n$corrections\n\n"
    "Revise your answer. Fix real problems only; ignore any suggestion that is wrong. "
    "Reply with the complete improved answer in the same format as before, without mentioning "
    "the reviewer."
)

SELF_TEST_ECHO = "Reply with the single word: ready."
