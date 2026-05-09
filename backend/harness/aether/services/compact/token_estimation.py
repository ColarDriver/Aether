"""Lightweight token estimation for compaction decisions."""

from __future__ import annotations

import json
from typing import Any


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Return a deterministic rough token estimate for a message list.

    The estimate intentionally mirrors the rough shape used by
    open-claude-code: count message text, add small structural overhead,
    convert chars to tokens at roughly four chars per token, then pad by
    4/3 so preflight compaction triggers slightly early instead of late.
    """
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            total_chars += len(str(msg))
            continue
        total_chars += 4
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += sum(_estimate_block_chars(block) for block in content)
        elif content is not None:
            total_chars += len(json.dumps(content, ensure_ascii=False, default=str))
    return int((total_chars / 4) * (4 / 3))


def _estimate_block_chars(block: Any) -> int:
    if not isinstance(block, dict):
        return len(str(block))
    block_type = block.get("type")
    if block_type == "text":
        return len(str(block.get("text", "")))
    if block_type == "tool_use":
        return 20 + len(json.dumps(block.get("input") or {}, ensure_ascii=False, default=str))
    if block_type == "tool_result":
        inner = block.get("content")
        if isinstance(inner, str):
            return 20 + len(inner)
        if isinstance(inner, list):
            return 20 + sum(_estimate_block_chars(item) for item in inner)
        if inner is not None:
            return 20 + len(json.dumps(inner, ensure_ascii=False, default=str))
        return 20
    if block_type in {"image", "document"}:
        return 8000
    if block_type in {"thinking", "redacted_thinking"}:
        return 20 + len(str(block.get("thinking", "")))
    return len(json.dumps(block, ensure_ascii=False, default=str))
