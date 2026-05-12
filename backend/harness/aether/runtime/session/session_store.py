"""Session store abstractions for AgentEngine persistence."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


class SessionStore(ABC):
    """Minimal session persistence contract used by AgentEngine."""

    @abstractmethod
    def get_session(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def update_system_prompt(self, session_id: str, prompt_snapshot: str) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory store suitable for tests/local runs."""

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.sessions.get(session_id)
            return copy.deepcopy(row) if row is not None else None

    def update_system_prompt(self, session_id: str, prompt_snapshot: str) -> None:
        with self._lock:
            row = self.sessions.setdefault(session_id, {})
            row["system_prompt"] = prompt_snapshot
