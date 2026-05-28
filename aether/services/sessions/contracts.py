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
class TranscriptAttachment:
    type: Literal["file", "image", "text"]
    name: str | None = None
    path: str | None = None
    url: str | None = None
    mime_type: str | None = None
    data: str | None = None
    is_directory: bool = False
    line_start: int | None = None
    line_end: int | None = None
    note: str | None = None
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    role: Literal["user", "assistant", "system", "tool"]
    text: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[TranscriptToolCall] = field(default_factory=list)
    attachments: list[TranscriptAttachment] = field(default_factory=list)
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
    permission_mode: str | None = None
    cwd: str | None = None


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
class SessionForkRequest:
    session_id_or_prefix: str
    message_index: int | None = None
    target_user_message_id: str | None = None
    user_message_index: int | None = None
    expected_content: str | None = None
    new_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class SessionForkResult:
    source_session_id: str
    forked_from_index: int
    messages_copied: int
    info: SessionInfo
    messages: list[TranscriptMessage] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SessionRewindRequest:
    session_id_or_prefix: str
    message_index: int | None = None
    target_user_message_id: str | None = None
    user_message_index: int | None = None
    expected_content: str | None = None
    rewind_before_target: bool = False


@dataclass(frozen=True, slots=True)
class SessionRewindResult:
    session_id: str
    rewound_to_index: int
    messages_kept: int
    messages_removed: int
    info: SessionInfo
    messages: list[TranscriptMessage] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SessionTurnTarget:
    target_user_message_id: str
    user_message_index: int
    user_message_count: int
    message_index: int
    content: str | None = None


@dataclass(frozen=True, slots=True)
class SessionTurnCodeSnapshot:
    available: bool
    files_changed: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    checkpoint_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SessionTurnCheckpoint:
    target: SessionTurnTarget
    code: SessionTurnCodeSnapshot
    work_dir: str | None = None
    conversation: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SessionTurnCheckpointsResult:
    session_id: str
    checkpoints: list[SessionTurnCheckpoint] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SessionTurnCheckpointDiffResult:
    session_id: str
    state: Literal["ok", "missing", "error"]
    target: SessionTurnTarget
    path: str
    diff: str | None = None
    work_dir: str | None = None
    checkpoint_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SessionMessageAction:
    name: str
    supported: bool
    label: str
    reason: str | None = None
    checkpoint_id: str | None = None
    destructive: bool = False


@dataclass(frozen=True, slots=True)
class SessionMessageActionsResult:
    session_id: str
    message_index: int
    role: str
    target_user_message_id: str | None = None
    user_message_index: int | None = None
    actions: list[SessionMessageAction] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SessionExportRequest:
    session_id_or_prefix: str


@dataclass(frozen=True, slots=True)
class SessionExportResult:
    session_id: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SessionImportRequest:
    data: dict[str, Any]
    new_session_id: str | None = None
    overwrite: bool = False
    make_current: bool = True


@dataclass(frozen=True, slots=True)
class SessionImportResult:
    source_session_id: str | None
    overwritten: bool
    info: SessionInfo
    messages: list[TranscriptMessage] = field(default_factory=list)


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
    "SessionForkRequest",
    "SessionForkResult",
    "SessionImportRequest",
    "SessionImportResult",
    "SessionInfo",
    "SessionListResult",
    "SessionMessageAction",
    "SessionMessageActionsResult",
    "SessionRenameRequest",
    "SessionResumeRequest",
    "SessionRewindRequest",
    "SessionRewindResult",
    "SessionTurnCheckpoint",
    "SessionTurnCheckpointDiffResult",
    "SessionTurnCheckpointsResult",
    "SessionTurnCodeSnapshot",
    "SessionTurnTarget",
    "SessionUpdateRequest",
    "TranscriptAttachment",
    "TranscriptMessage",
    "TranscriptToolCall",
]
