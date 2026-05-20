"""Session service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TranscriptToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    role: Literal["user", "assistant", "system", "tool"]
    text: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[TranscriptToolCall] = field(default_factory=list)
    is_error: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    created_at: float
    updated_at: float
    provider: str
    model: str
    base_url: str | None = None
    system_prompt: str | None = None
    message_count: int = 0
    summary: str | None = None
    mode: str | None = None


@dataclass(frozen=True, slots=True)
class SessionCreateRequest:
    provider: str
    model: str
    base_url: str | None = None
    system_prompt: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class SessionUpdateRequest:
    session_id: str
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    update_base_url: bool = False
    update_system_prompt: bool = False


@dataclass(frozen=True, slots=True)
class SessionResumeRequest:
    session_id_or_prefix: str


@dataclass(frozen=True, slots=True)
class SessionDeleteRequest:
    session_id: str


@dataclass(frozen=True, slots=True)
class SessionRenameRequest:
    session_id: str
    new_session_id: str


@dataclass(frozen=True, slots=True)
class SessionExportRequest:
    session_id_or_prefix: str


@dataclass(frozen=True, slots=True)
class SessionExportResult:
    session_id: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SessionListResult:
    sessions: list[SessionInfo]


@dataclass(frozen=True, slots=True)
class SessionCurrentResult:
    session_id: str
    info: SessionInfo
    messages: list[TranscriptMessage] = field(default_factory=list)


__all__ = [
    "SessionCreateRequest",
    "SessionCurrentResult",
    "SessionDeleteRequest",
    "SessionExportRequest",
    "SessionExportResult",
    "SessionInfo",
    "SessionListResult",
    "SessionRenameRequest",
    "SessionResumeRequest",
    "SessionUpdateRequest",
    "TranscriptMessage",
    "TranscriptToolCall",
]
