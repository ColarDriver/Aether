"""JSON-schema sanitizer for local backend grammar recovery."""

from __future__ import annotations

import copy
from typing import Any

from aether.tools.base import ToolDescriptor


UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset({"pattern", "format"})


def strip_pattern_and_format(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Return a deep-copied tool payload with unsupported schema keys removed."""

    cleaned = copy.deepcopy(tools)
    removed = strip_pattern_and_format_in_place(cleaned)
    return cleaned, removed > 0


def strip_pattern_and_format_with_count(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, int]:
    """Count-preserving sibling used by observability metadata."""

    cleaned = copy.deepcopy(tools)
    removed = strip_pattern_and_format_in_place(cleaned)
    return cleaned, removed > 0, removed


def strip_pattern_and_format_in_place(value: Any) -> int:
    """Remove ``pattern`` / ``format`` keys recursively from mutable structures."""

    return _strip_unsupported_schema_keys(value, seen=set())


def sanitize_tool_descriptors(
    descriptors: list[Any],
) -> tuple[list[ToolDescriptor], int]:
    """Clone descriptors with local-backend-incompatible schema fields removed."""

    sanitized: list[ToolDescriptor] = []
    removed = 0
    for descriptor in descriptors:
        parameters = copy.deepcopy(getattr(descriptor, "parameters", {}) or {})
        removed += strip_pattern_and_format_in_place(parameters)
        sanitized.append(
            ToolDescriptor(
                name=str(getattr(descriptor, "name", "")),
                description=str(getattr(descriptor, "description", "") or ""),
                parameters=parameters,
                required=list(getattr(descriptor, "required", []) or []),
            )
        )
    return sanitized, removed


def _strip_unsupported_schema_keys(value: Any, *, seen: set[int]) -> int:
    if isinstance(value, (str, bytes, bytearray, memoryview)) or value is None:
        return 0

    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    removed = 0
    if isinstance(value, dict):
        for key in tuple(value.keys()):
            if key in UNSUPPORTED_SCHEMA_KEYS:
                value.pop(key, None)
                removed += 1
                continue
            removed += _strip_unsupported_schema_keys(value[key], seen=seen)
        return removed

    if isinstance(value, list):
        for item in value:
            removed += _strip_unsupported_schema_keys(item, seen=seen)
        return removed

    if isinstance(value, tuple):
        for item in value:
            removed += _strip_unsupported_schema_keys(item, seen=seen)
        return removed

    if isinstance(value, set):
        for item in tuple(value):
            removed += _strip_unsupported_schema_keys(item, seen=seen)
        return removed

    return 0
