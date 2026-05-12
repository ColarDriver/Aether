"""Stable contracts for Aether's memory subsystem.

The memory layer is intentionally separate from ``AgentEngine``.  Providers
return structured, budgetable blocks; the engine decides later whether any
block is safe to inject into the provider-bound message copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MemoryMode(str, Enum):
    """Configured memory operating mode."""

    OFF = "off"
    TASK = "task"
    PROJECT = "project"
    PERSONAL_ASSISTANT = "personal_assistant"


class MemoryScope(str, Enum):
    """Isolation boundary for a memory block."""

    SESSION = "session"
    TASK = "task"
    PROJECT = "project"
    USER = "user"


class MemoryKind(str, Enum):
    """Semantic category for a memory block."""

    TASK_STATE = "task_state"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    PROJECT_FACT = "project_fact"
    USER_PREFERENCE = "user_preference"
    REFERENCE = "reference"
    WARNING = "warning"


def normalize_memory_mode(value: Any, *, default: MemoryMode = MemoryMode.PROJECT) -> MemoryMode:
    """Return a valid :class:`MemoryMode` for untrusted config values."""

    if isinstance(value, MemoryMode):
        return value
    if value is None:
        return default
    try:
        return MemoryMode(str(value).strip().lower())
    except ValueError:
        return default


def scopes_for_mode(
    mode: MemoryMode | str,
    *,
    user_profile_enabled: bool = False,
) -> tuple[MemoryScope, ...]:
    """Return the scopes that may participate in retrieval for ``mode``."""

    normalized = normalize_memory_mode(mode, default=MemoryMode.OFF)
    if normalized is MemoryMode.OFF:
        return ()
    if normalized is MemoryMode.TASK:
        return (MemoryScope.SESSION, MemoryScope.TASK)
    if normalized is MemoryMode.PROJECT:
        return (MemoryScope.SESSION, MemoryScope.TASK, MemoryScope.PROJECT)
    if not user_profile_enabled:
        return (MemoryScope.SESSION, MemoryScope.TASK, MemoryScope.PROJECT)
    return (MemoryScope.SESSION, MemoryScope.TASK, MemoryScope.PROJECT, MemoryScope.USER)


@dataclass(slots=True, frozen=True)
class MemoryQuery:
    """Inputs used by a memory provider to retrieve relevant blocks."""

    session_id: str
    task_id: str | None
    user_message: str
    recent_messages: list[dict[str, Any]]
    working_directory: str | None = None
    active_files: tuple[str, ...] = ()
    mode: MemoryMode = MemoryMode.PROJECT
    token_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.token_budget < 0:
            raise ValueError("token_budget must be non-negative")


@dataclass(slots=True, frozen=True)
class MemoryBlock:
    """One retrievable memory unit with provenance and budget metadata."""

    id: str
    scope: MemoryScope
    kind: MemoryKind
    text: str
    source: str
    token_estimate: int
    relevance: float = 0.0
    confidence: str = "medium"
    created_at: str | None = None
    updated_at: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("memory block id is required")
        if not self.source.strip():
            raise ValueError("memory block source is required")
        if not self.text.strip():
            raise ValueError("memory block text is required")
        if self.token_estimate <= 0:
            raise ValueError("memory block token_estimate must be positive")


@dataclass(slots=True, frozen=True)
class MemoryBundle:
    """Result returned by a memory provider."""

    blocks: tuple[MemoryBlock, ...] = ()
    token_estimate: int = 0
    skipped_reason: str | None = None
    latency_ms: float = 0.0
    provider_errors: tuple[str, ...] = ()

    @classmethod
    def skipped(
        cls,
        reason: str,
        *,
        latency_ms: float = 0.0,
        provider_errors: tuple[str, ...] = (),
    ) -> "MemoryBundle":
        return cls(
            blocks=(),
            token_estimate=0,
            skipped_reason=reason,
            latency_ms=latency_ms,
            provider_errors=provider_errors,
        )

    @classmethod
    def from_blocks(
        cls,
        blocks: tuple[MemoryBlock, ...] | list[MemoryBlock],
        *,
        latency_ms: float = 0.0,
        skipped_reason: str | None = None,
        provider_errors: tuple[str, ...] = (),
    ) -> "MemoryBundle":
        normalized = tuple(blocks)
        return cls(
            blocks=normalized,
            token_estimate=sum(block.token_estimate for block in normalized),
            skipped_reason=skipped_reason,
            latency_ms=latency_ms,
            provider_errors=provider_errors,
        )


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol implemented by concrete memory backends."""

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        """Return relevant memory for a provider-bound request."""

    def observe_turn(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Best-effort turn-end observation hook."""

    def before_compaction(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Best-effort hook before compaction mutates/projection-builds context."""
