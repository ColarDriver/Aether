"""Context compression service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextBreakdownItem:
    label: str
    tokens: int
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ContextStatusResult:
    session_id: str
    context_engine: str = "default"
    compression_count: int = 0
    last_compression: dict[str, Any] | None = None
    message_count: int = 0
    token_estimate: int = 0
    provider: str | None = None
    model: str | None = None
    context_window: int | None = None
    prompt_tokens: int = 0
    transcript_tokens: int = 0
    system_tokens: int = 0
    memory_tokens: int = 0
    attachment_tokens: int = 0
    tool_result_tokens: int = 0
    pressure_level: str = "unknown"
    next_action: str = "none"
    breakdown: list[ContextBreakdownItem] = field(default_factory=list)
    status: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ContextEstimateRequest:
    session_id: str
    draft: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ContextCompressRequest:
    session_id: str
    focus: str | None = None
    force: bool = True


@dataclass(frozen=True, slots=True)
class ContextCompressResult(ContextStatusResult):
    status: str = "skipped"


__all__ = [
    "ContextBreakdownItem",
    "ContextCompressRequest",
    "ContextCompressResult",
    "ContextEstimateRequest",
    "ContextStatusResult",
]
