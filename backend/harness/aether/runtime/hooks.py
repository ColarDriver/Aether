"""Lifecycle hooks for AgentEngine session/turn events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EngineHooks:
    """Override methods to observe lifecycle events without middleware coupling."""

    def on_session_start(self, *, session_id: str, context_metadata: dict[str, Any]) -> None:
        return None

    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> None:
        return None

    def post_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        response_text: str,
        context_metadata: dict[str, Any],
    ) -> None:
        return None

    def on_session_end(
        self,
        *,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        return None
