"""Lifecycle hooks for AgentEngine session/turn events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.runtime.contracts import NormalizedResponse


@dataclass(slots=True)
class HookOutcome:
    """Structured result a hook can return to influence the next LLM call.

    Hooks remain optional lifecycle observers by default.  Returning this
    dataclass lets a hook inject transient context into the provider-bound
    message copy or short-circuit the provider call entirely.
    """

    inject_user_context: str | None = None
    inject_system_addendum: str | None = None
    short_circuit_response: NormalizedResponse | None = None


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
    ) -> HookOutcome | None:
        return None

    def pre_api_request(
        self,
        *,
        session_id: str,
        iteration: int,
        model: str,
        provider: str,
        api_mode: str,
        api_call_count: int,
        message_count: int,
        tool_count: int,
        approx_input_tokens: int,
        request_char_count: int,
        max_tokens: int | None,
        context_metadata: dict[str, Any],
    ) -> None:
        return None

    def post_api_request(
        self,
        *,
        session_id: str,
        iteration: int,
        model: str,
        provider: str,
        api_mode: str,
        api_call_count: int,
        elapsed_ms: float,
        response_finish_reason: str | None,
        error: Exception | None,
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

    def on_task_cleanup(
        self,
        *,
        task_id: str,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        return None
