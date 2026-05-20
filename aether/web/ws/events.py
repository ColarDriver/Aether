"""Run-service event mapping for browser WebSocket clients."""

from __future__ import annotations

from typing import Any

from aether.services.runs import (
    AssistantDelta,
    IterationFinished,
    IterationStarted,
    LoopStateChanged,
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
from aether.services.runs.events import event_to_public_dict

_EVENT_TYPES: tuple[tuple[type[object], str], ...] = (
    (RunStarted, "run.started"),
    (AssistantDelta, "assistant.delta"),
    (ReasoningDelta, "reasoning.delta"),
    (SilentProgress, "silent.progress"),
    (RunStatusChanged, "run.status"),
    (LoopStateChanged, "loop.state"),
    (IterationStarted, "iteration.started"),
    (IterationFinished, "iteration.finished"),
    (ToolStarted, "tool.started"),
    (ToolFinished, "tool.finished"),
    (TokenUsageUpdated, "token.usage"),
    (RunFinished, "run.finished"),
    (RunFailed, "run.failed"),
    (RunCancelled, "run.cancelled"),
)


def run_event_to_frame(event: RunEvent) -> dict[str, Any]:
    return {
        "type": _event_type(event),
        "payload": event_to_public_dict(event),
    }


def _event_type(event: RunEvent) -> str:
    for event_class, event_name in _EVENT_TYPES:
        if isinstance(event, event_class):
            return event_name
    return "run.event"


__all__ = ["run_event_to_frame"]
