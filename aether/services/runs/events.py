"""Agent run service events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from typing import Any, Literal, TypeAlias

RunStatusKind: TypeAlias = Literal["thinking", "responding", "tool_use", "idle"]
PermissionKind: TypeAlias = Literal["plan", "questions", "tool"]


@dataclass(frozen=True, slots=True)
class RunStarted:
    session_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class AssistantDelta:
    session_id: str
    run_id: str
    text: str
    sequence: int


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    session_id: str
    run_id: str
    text: str
    sequence: int


@dataclass(frozen=True, slots=True)
class SilentProgress:
    session_id: str
    run_id: str
    chars: int
    sequence: int


@dataclass(frozen=True, slots=True)
class RunStatusChanged:
    session_id: str
    run_id: str
    kind: RunStatusKind
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class LoopStateChanged:
    session_id: str
    run_id: str
    state: str


@dataclass(frozen=True, slots=True)
class IterationStarted:
    session_id: str
    run_id: str
    iteration: int


@dataclass(frozen=True, slots=True)
class IterationFinished:
    session_id: str
    run_id: str
    iteration: int


@dataclass(frozen=True, slots=True)
class ToolStarted:
    session_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0


@dataclass(frozen=True, slots=True)
class ToolFinished:
    session_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PermissionRequested:
    session_id: str
    run_id: str
    kind: PermissionKind
    tool_call_id: str | None = None
    plan_text: str | None = None
    plan_path: str | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)
    deadline_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsageUpdated:
    session_id: str
    run_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class RunFinished:
    session_id: str
    run_id: str
    final_text: str = ""
    exit_reason: str = "done"


@dataclass(frozen=True, slots=True)
class RunFailed:
    session_id: str
    run_id: str
    message: str


@dataclass(frozen=True, slots=True)
class RunCancelled:
    session_id: str
    run_id: str
    reason: str | None = None
    partial_text: str = ""


RunEvent: TypeAlias = (
    RunStarted
    | AssistantDelta
    | ReasoningDelta
    | SilentProgress
    | RunStatusChanged
    | LoopStateChanged
    | IterationStarted
    | IterationFinished
    | ToolStarted
    | ToolFinished
    | PermissionRequested
    | TokenUsageUpdated
    | RunFinished
    | RunFailed
    | RunCancelled
)


def event_to_public_dict(event: RunEvent) -> dict[str, Any]:
    """Return a JSON-compatible service event payload.

    Service events intentionally do not contain gateway wire ``type`` fields.
    Adapters add transport-specific type names at the edge.
    """
    if not is_dataclass(event):
        raise TypeError(f"unsupported run event: {type(event).__name__}")
    payload = asdict(event)
    payload["event"] = type(event).__name__
    return _json_safe(payload)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        out[str(key)] = value
    return out


__all__ = [
    "AssistantDelta",
    "IterationFinished",
    "IterationStarted",
    "LoopStateChanged",
    "PermissionRequested",
    "ReasoningDelta",
    "RunCancelled",
    "RunEvent",
    "RunFailed",
    "RunFinished",
    "RunStarted",
    "RunStatusChanged",
    "SilentProgress",
    "TokenUsageUpdated",
    "ToolFinished",
    "ToolStarted",
    "event_to_public_dict",
]
