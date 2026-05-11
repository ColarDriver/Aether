"""Helpers for extracting provider-emitted reasoning from message history."""

from __future__ import annotations

import json
from typing import Any


def extract_last_reasoning(
    messages: list[dict[str, Any]],
    turn_start_idx: int,
) -> str | None:
    """Return the latest non-empty reasoning block in the current turn."""

    start = max(0, int(turn_start_idx))
    for idx in range(len(messages) - 1, start - 1, -1):
        message = messages[idx]
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            break
        if message.get("role") != "assistant":
            continue
        for key in ("reasoning", "reasoning_content"):
            value = _normalise_reasoning_value(message.get(key))
            if value:
                return value
        metadata = message.get("metadata")
        if isinstance(metadata, dict):
            for key in ("reasoning_content", "reasoning_details"):
                value = _normalise_reasoning_value(metadata.get(key))
                if value:
                    return value
    return None


def _normalise_reasoning_value(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(value)
    return None
