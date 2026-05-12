"""Trajectory conversion and JSONL persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aether.runtime.observability.reasoning import extract_last_reasoning


def convert_to_trajectory_format(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert Aether message history into a simple training trajectory."""

    conversations: list[dict[str, str]] = []
    pending_tool_responses: list[str] = []

    def flush_tools() -> None:
        if pending_tool_responses:
            conversations.append(
                {"from": "tool", "value": "\n".join(pending_tool_responses)}
            )
            pending_tool_responses.clear()

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            pending_tool_responses.append(_format_tool_response(message))
            continue

        flush_tools()
        if role == "user":
            conversations.append(
                {"from": "human", "value": _content_to_text(message.get("content"))}
            )
        elif role == "assistant":
            value = _format_assistant_message(message)
            if value:
                conversations.append({"from": "gpt", "value": value})

    flush_tools()
    return conversations


def build_trajectory_record(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_id: str | None,
    task_id: str | None,
    model: str,
    provider: str,
    completed: bool,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
        "turn_id": turn_id,
        "task_id": task_id,
        "model": model,
        "provider": provider,
        "completed": completed,
        "conversations": convert_to_trajectory_format(messages),
    }


def save_trajectory_record(
    record: dict[str, Any],
    *,
    trajectory_dir: Path,
    session_id: str,
    completed: bool,
) -> Path:
    bucket = "sessions" if completed else "failed"
    target_dir = trajectory_dir / bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_safe_session_id(session_id)}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
    return path


def _format_assistant_message(message: dict[str, Any]) -> str:
    parts: list[str] = []
    reasoning = extract_last_reasoning([message], 0)
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")

    content = _content_to_text(message.get("content")).strip()
    if content:
        parts.append(content)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            formatted = _format_tool_call(tool_call)
            if formatted:
                parts.append(f"<tool_call>\n{formatted}\n</tool_call>")
    return "\n".join(parts).strip()


def _format_tool_call(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = tool_call.get("name") or function.get("name")
    if not name:
        return None
    raw_arguments = function.get("arguments", tool_call.get("arguments", {}))
    arguments = _parse_arguments(raw_arguments)
    payload = {
        "name": str(name),
        "arguments": arguments,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _format_tool_response(message: dict[str, Any]) -> str:
    payload = {
        "tool_call_id": str(message.get("tool_call_id") or ""),
        "name": str(message.get("name") or ""),
        "content": _maybe_json_content(message.get("content")),
    }
    return (
        "<tool_response>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_response>"
    )


def _parse_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value if value is not None else {}


def _maybe_json_content(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _safe_session_id(session_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in session_id)
    return safe[:120] or "session"
