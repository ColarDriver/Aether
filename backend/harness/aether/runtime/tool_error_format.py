"""Format tool-call errors into model-friendly corrective messages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FormattedToolError:
    text: str
    category: str
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class SchemaIssue:
    code: str
    path: str
    expected: str | None = None
    received: str | None = None


_MAX_HINT_CHARS = 120


def format_invalid_tool_args_error(
    tool_name: str,
    exc: Exception,
    raw_args: str,
) -> FormattedToolError:
    line, column, parser_msg = _extract_jsondecodeerror_metadata(exc)
    pieces: list[str] = []
    if line is not None and column is not None:
        pieces.append(
            f"Invalid JSON arguments for tool `{tool_name}` at line {line} column {column}: "
            f"`{parser_msg}`."
        )
    else:
        pieces.append(f"Invalid JSON arguments for tool `{tool_name}`: `{parser_msg}`.")

    snippet = _build_snippet(raw_args, line, column)
    if snippet:
        pieces.append(f"The parser stopped after: {snippet}")
    pieces.append(
        "Hint: emit a single JSON object as the `arguments` field. "
        "Every key/value pair must be comma-separated and the whole payload "
        "must be wrapped in `{}`. For tools with no required parameters, use `{}`."
    )
    return FormattedToolError(
        text="\n".join(pieces),
        category="json_syntax",
        line=line,
        column=column,
    )


def format_schema_error(
    tool_name: str,
    parameters_schema: Mapping[str, Any] | None,
    parsed_args: Any,
) -> FormattedToolError:
    issues = _collect_schema_issues(parameters_schema, parsed_args)
    return format_schema_error_from_issues(tool_name, issues)


def format_schema_error_from_issues(
    tool_name: str,
    issues: list[SchemaIssue],
) -> FormattedToolError:
    if not issues:
        return FormattedToolError(
            text=f"`{tool_name}` arguments validation failed (no specific issue extracted).",
            category="schema_unknown",
        )

    parts: list[str] = []
    for issue in issues:
        if issue.code == "missing":
            parts.append(f"The required parameter `{issue.path}` is missing")
        elif issue.code == "unexpected":
            parts.append(f"An unexpected parameter `{issue.path}` was provided")
        elif issue.code == "type_mismatch":
            parts.append(
                f"The parameter `{issue.path}` type is expected as `{issue.expected}` "
                f"but provided as `{issue.received}`"
            )
        else:
            parts.append(f"`{issue.path}`: {issue.code}")

    summary = (
        f"`{tool_name}` failed due to the following "
        f"{'issue' if len(parts) == 1 else f'{len(parts)} issues'}:"
    )
    category = "schema_unknown"
    if any(i.code == "missing" for i in issues):
        category = "schema_missing"
    elif any(i.code == "type_mismatch" for i in issues):
        category = "schema_type_mismatch"
    elif any(i.code == "unexpected" for i in issues):
        category = "schema_unexpected"
    return FormattedToolError(text="\n".join([summary, *parts]), category=category)


def format_unknown_tool_error(
    requested_name: str,
    available_names: list[str],
    repair_attempts: list[tuple[str, str]] | None = None,
) -> FormattedToolError:
    parts = [f"Unknown tool: `{requested_name}` does not exist."]
    if repair_attempts:
        attempts_txt = "; ".join(f"{stage}->`{candidate}`" for stage, candidate in repair_attempts)
        parts.append(f"Repair attempts: {attempts_txt}.")
    suggestions = _suggest_similar_tools(requested_name, available_names, top_k=3)
    if suggestions:
        parts.append(f"Did you mean: {', '.join('`' + s + '`' for s in suggestions)}?")
    if available_names:
        shown = available_names[:20]
        suffix = "" if len(available_names) <= len(shown) else f" (+{len(available_names) - len(shown)} more)"
        parts.append(f"Available tools: {', '.join('`' + name + '`' for name in shown)}{suffix}.")
    parts.append("Please pick one of the registered tools and retry.")
    return FormattedToolError(text="\n".join(parts), category="unknown_tool")


def _extract_jsondecodeerror_metadata(exc: Exception) -> tuple[int | None, int | None, str]:
    if isinstance(exc, json.JSONDecodeError):
        return (
            exc.lineno if exc.lineno > 0 else None,
            exc.colno if exc.colno > 0 else None,
            exc.msg or str(exc),
        )
    return None, None, str(exc) or exc.__class__.__name__


def _build_snippet(
    raw_args: str,
    line: int | None,
    column: int | None,
) -> str | None:
    if not raw_args:
        return None
    offset: int | None = None
    if line is not None and column is not None:
        try:
            offset = _line_col_to_offset(raw_args, line, column)
        except ValueError:
            offset = None
    if offset is None:
        return _truncate(raw_args, _MAX_HINT_CHARS)
    start = max(0, offset - _MAX_HINT_CHARS // 2)
    end = min(len(raw_args), offset + _MAX_HINT_CHARS // 2)
    snippet = raw_args[start:end]
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(raw_args):
        snippet = f"{snippet}..."
    return snippet


def _line_col_to_offset(text: str, line: int, column: int) -> int:
    lines = text.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        raise ValueError("line out of range")
    return sum(len(item) for item in lines[: line - 1]) + max(0, column - 1)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}...{text[-half:]}"


_PRIMITIVE_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _collect_schema_issues(
    schema: Mapping[str, Any] | None,
    parsed_args: Any,
) -> list[SchemaIssue]:
    if schema is None or not isinstance(parsed_args, dict):
        return []
    issues: list[SchemaIssue] = []
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        properties = {}
    required = list(schema.get("required") or [])
    additional_allowed = schema.get("additionalProperties", None) is not False

    for key in required:
        if key not in parsed_args:
            issues.append(SchemaIssue(code="missing", path=str(key)))

    if not additional_allowed:
        for key in parsed_args:
            if key not in properties:
                issues.append(SchemaIssue(code="unexpected", path=str(key)))

    for key, prop_schema in properties.items():
        if key not in parsed_args or not isinstance(prop_schema, Mapping):
            continue
        expected = prop_schema.get("type")
        if not expected:
            continue
        value = parsed_args[key]
        if isinstance(expected, list):
            if any(_value_matches_type(value, str(item)) for item in expected):
                continue
            issues.append(
                SchemaIssue(
                    code="type_mismatch",
                    path=str(key),
                    expected=" or ".join(str(item) for item in expected),
                    received=_python_type_name(value),
                )
            )
            continue
        if not _value_matches_type(value, str(expected)):
            issues.append(
                SchemaIssue(
                    code="type_mismatch",
                    path=str(key),
                    expected=str(expected),
                    received=_python_type_name(value),
                )
            )
    return issues


def _value_matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    types = _PRIMITIVE_TYPE_MAP.get(expected)
    if types is None:
        return True
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in {"number", "integer"}:
        return isinstance(value, types) and not isinstance(value, bool)
    return isinstance(value, types)


def _python_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _suggest_similar_tools(
    requested: str,
    available: list[str],
    *,
    top_k: int,
) -> list[str]:
    if not available or not requested:
        return []
    scored: list[tuple[int, str]] = []
    for name in available:
        distance = _levenshtein(requested.lower(), name.lower())
        if distance <= 2:
            scored.append((distance, name))
    scored.sort()
    return [name for _, name in scored[:top_k]]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


__all__ = [
    "FormattedToolError",
    "SchemaIssue",
    "format_invalid_tool_args_error",
    "format_schema_error",
    "format_schema_error_from_issues",
    "format_unknown_tool_error",
]
