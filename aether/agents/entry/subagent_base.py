"""Subagent delegation abstraction placeholder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class SubagentDispatcher(ABC):
    """Dispatch delegated tasks to child agents (future module)."""

    @abstractmethod
    def dispatch(self, goal: str, context: Dict[str, Any] | None = None) -> str:
        raise NotImplementedError
