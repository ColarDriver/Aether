"""Map transport-neutral run service events to gateway event payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from aether.gateway.protocol import (
    ApprovalQuestion,
    ApprovalRequest,
    Cancelled,
    Done,
    Error,
    IterationEnd,
    IterationStart,
    LoopStateChanged as GatewayLoopStateChanged,
    Reasoning,
    Status,
    StreamProgress,
    TextDelta,
    TokenUsage,
    ToolCall as ToolCallEvent,
    ToolResult as ToolResultEvent,
)
from aether.services.runs.events import (
    AssistantDelta,
    IterationFinished,
    IterationStarted,
    LoopStateChanged,
    PermissionRequested,
    ReasoningDelta,
    RunCancelled,
    RunEvent,
    RunFailed,
    RunFinished,
    RunStarted,
    RunStatusChanged,
    SilentProgress,
    TokenUsageUpdated,
    ToolFinished,
    ToolStarted,
)


def service_event_to_gateway_payload(event: RunEvent) -> dict[str, Any] | None:
    """Return the existing ``notify("event", payload)`` shape.

    ``RunStarted`` is service lifecycle state, not part of the current gateway
    agent event stream, so it intentionally maps to ``None``.
    """
    gateway_event = service_event_to_gateway_model(event)
    if gateway_event is None:
        return None
    return gateway_event.model_dump(mode="json", exclude_none=True)


def service_event_to_gateway_model(event: RunEvent):
    if isinstance(event, RunStarted):
        return None
    if isinstance(event, AssistantDelta):
        return TextDelta(
            session_id=event.session_id,
            run_id=event.run_id,
            text=event.text,
            sequence=event.sequence,
        )
    if isinstance(event, ReasoningDelta):
        return Reasoning(
            session_id=event.session_id,
            run_id=event.run_id,
            text=event.text,
            sequence=event.sequence,
        )
    if isinstance(event, SilentProgress):
        return StreamProgress(
            session_id=event.session_id,
            run_id=event.run_id,
            chars=event.chars,
            sequence=event.sequence,
        )
    if isinstance(event, RunStatusChanged):
        return Status(
            session_id=event.session_id,
            run_id=event.run_id,
            kind=event.kind,
            detail=event.detail,
        )
    if isinstance(event, LoopStateChanged):
        return GatewayLoopStateChanged(
            session_id=event.session_id,
            run_id=event.run_id,
            state=event.state,
        )
    if isinstance(event, IterationStarted):
        return IterationStart(
            session_id=event.session_id,
            run_id=event.run_id,
            iteration=event.iteration,
        )
    if isinstance(event, IterationFinished):
        return IterationEnd(
            session_id=event.session_id,
            run_id=event.run_id,
            iteration=event.iteration,
        )
    if isinstance(event, ToolStarted):
        return ToolCallEvent(
            session_id=event.session_id,
            run_id=event.run_id,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            arguments=dict(event.arguments or {}),
            iteration=event.iteration,
        )
    if isinstance(event, ToolFinished):
        return ToolResultEvent(
            session_id=event.session_id,
            run_id=event.run_id,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            content=event.content,
            is_error=event.is_error,
            iteration=event.iteration,
            metadata=safe_metadata(event.metadata),
        )
    if isinstance(event, TokenUsageUpdated):
        return TokenUsage(
            session_id=event.session_id,
            run_id=event.run_id,
            input_tokens=max(0, int(event.input_tokens or 0)),
            output_tokens=max(0, int(event.output_tokens or 0)),
            cache_read_tokens=max(0, int(event.cache_read_tokens or 0)),
            cache_write_tokens=max(0, int(event.cache_write_tokens or 0)),
        )
    if isinstance(event, RunFinished):
        return Done(
            session_id=event.session_id,
            run_id=event.run_id,
            final_text=event.final_text,
            exit_reason=event.exit_reason,
        )
    if isinstance(event, RunCancelled):
        return Cancelled(
            session_id=event.session_id,
            run_id=event.run_id,
            reason=event.reason,
            partial_text=event.partial_text,
        )
    if isinstance(event, RunFailed):
        return Error(
            session_id=event.session_id,
            run_id=event.run_id,
            message=event.message,
        )
    if isinstance(event, PermissionRequested):
        return None
    raise TypeError(f"unsupported run event: {type(event).__name__}")


def permission_request_to_gateway_payload(event: PermissionRequested) -> dict[str, Any]:
    return ApprovalRequest(
        kind="plan" if event.kind == "plan" else "questions",
        session_id=event.session_id,
        run_id=event.run_id,
        tool_call_id=event.tool_call_id,
        plan_text=event.plan_text,
        plan_path=event.plan_path,
        questions=_approval_questions(event.questions),
        deadline_ms=event.deadline_ms,
    ).model_dump(mode="json", exclude_none=True)


def safe_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        out[str(key)] = value
    return out


def _approval_questions(raw: list[dict[str, Any]]) -> list[ApprovalQuestion]:
    questions: list[ApprovalQuestion] = []
    for item in raw:
        questions.append(ApprovalQuestion.model_validate(item))
    return questions


__all__ = [
    "permission_request_to_gateway_payload",
    "safe_metadata",
    "service_event_to_gateway_model",
    "service_event_to_gateway_payload",
]
