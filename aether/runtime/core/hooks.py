"""Lifecycle hooks for AgentEngine session/turn events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.runtime.core.contracts import NormalizedResponse, ToolResult


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

    def post_tool_use(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        result: "ToolResult",
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        """Fire-and-forget hook after a tool call completes.

        Parity with open-claude-code ``PostToolUse``.  ``result`` is the
        ToolResult **after** the ``after_tool`` middleware chain ran, so
        the hook sees what downstream code will record / send.

        Implementations MUST NOT raise — failures are logged and
        swallowed by the engine.  Implementations MUST NOT mutate
        ``result`` either; use middleware ``after_tool`` for ToolResult
        transformation.  This hook is meant for side effects such as
        scheduling background diagnostic collection, telemetry, or
        marking metadata on ``context_metadata`` for the next turn.

        Mutually exclusive with :meth:`post_tool_use_failure` — exactly
        one of the two fires per tool call.  This one fires when the
        registry dispatch returned without raising; the failure variant
        fires when dispatch raised an exception the engine had to
        recover from.
        """
        return None

    def post_tool_use_failure(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        """Fire-and-forget hook after a tool call raised during dispatch.

        Parity with open-claude-code ``PostToolUseFailure``.  Fires
        when :meth:`ToolRegistry.dispatch` raised — whether the engine
        then recovered (``fail_on_tool_error=False``) or aborted the
        loop (``fail_on_tool_error=True``).  Middleware ``after_tool``
        failures do NOT fire this hook; those propagate via the
        middleware error path instead.

        Implementations MUST NOT raise.  Same contract as
        :meth:`post_tool_use`.
        """
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
