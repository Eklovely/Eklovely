"""Small helpers for pulling code and JSON out of model replies."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_PYTHON_TAGS = {"python", "py", "python3"}


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """(language tag in lower case, code) for each closed ``` block."""
    return [(lang.lower(), code) for lang, code in _FENCE.findall(text)]


def extract_python(text: str) -> str | None:
    """The last ```python block, else the last untagged block, else None."""
    blocks = fenced_blocks(text)
    python = [code for lang, code in blocks if lang in _PYTHON_TAGS]
    if python:
        return python[-1].strip("\n") + "\n"
    untagged = [code for lang, code in blocks if not lang]
    if untagged:
        return untagged[-1].strip("\n") + "\n"
    return None


def extract_single_block(text: str) -> str | None:
    """The longest fenced block of any language (used for whole-file replies)."""
    blocks = [code for _, code in fenced_blocks(text)]
    if not blocks:
        return None
    return max(blocks, key=len).strip("\n") + "\n"


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the first JSON object in `text`, tolerating prose and code fences around it."""
    candidates = [text.strip()] + [
        code for lang, code in fenced_blocks(text) if lang in ("", "json")
    ]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def clip_middle(text: str, limit: int) -> str:
    """Keep the start and (mostly) the end of long text, where tracebacks live."""
    if len(text) <= limit:
        return text
    head = limit // 5
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} characters omitted]...\n{text[-tail:]}"
