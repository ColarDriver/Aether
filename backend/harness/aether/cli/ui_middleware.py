"""Engine-side bridge that feeds UI events from middleware/hook callbacks.

Translates ``EngineMiddleware.before_tool``/``after_tool`` and
``EngineHooks.pre_llm_call``/``post_llm_call`` into ``CLIUI`` calls so
that the run loop visibly narrates what the agent is doing.
"""

from __future__ import annotations

from typing import Any

from aether.agents.middlewares.base import EngineMiddleware
from aether.cli.ui import CLIUI, _verb_for_tool
from aether.runtime.contracts import (
    LoopState,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.hooks import EngineHooks


class CLIUIMiddleware(EngineMiddleware):
    """Forward tool dispatch / completion to the UI."""

    def __init__(self, ui: CLIUI) -> None:
        self.ui = ui

    def before_llm(self, messages: list[dict], context: TurnContext) -> list[dict]:
        # Each iteration the engine asks the LLM what's next.  The first
        # iteration the REPL has already announced its verb — for
        # subsequent iterations (after tool execution) we re-rotate the
        # verb so the user sees the loop progress without it feeling
        # mechanical.
        if int(context.iteration or 0) > 1:
            self.ui.set_status(f"{self.ui._next_spinner_verb()}…")
        return messages

    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        # If the LLM streamed text we keep that on screen; otherwise the
        # spinner is still active and either tool-dispatch or the engine's
        # final-text fallback will take over.  We do not stop the spinner
        # here — that's done at tool-call / stream boundaries.
        return response

    def before_tool(self, call: ToolCall | ToolResult, context: TurnContext) -> ToolCall | ToolResult:
        # before_tool runs once per call, possibly receiving a short-circuit
        # ToolResult from earlier middleware (e.g. policy denials).  Render
        # accordingly so the user sees both real and short-circuited paths.
        if isinstance(call, ToolResult):
            self.ui.render_tool_call(call.name, {"_short_circuit": True})
            self.ui.render_tool_result(
                call.name,
                call.content,
                is_error=call.is_error,
                metadata=dict(call.metadata),
            )
            return call

        self.ui.render_tool_call(call.name, dict(call.arguments or {}), tool_call_id=call.id)
        # Reuse the same human-readable verb the call line just printed
        # ("Reading file…", "Running command…", …) so the spinner visibly
        # follows the action that's underway.
        verb = _verb_for_tool(call.name)
        self.ui.set_status(f"{verb}…", spinner="dots")
        return call

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        self.ui.render_tool_result(
            result.name,
            result.content,
            is_error=result.is_error,
            metadata=dict(result.metadata),
        )
        # Once the tool has returned we hand the spinner back to the next
        # LLM iteration which will overwrite it via ``before_llm``.
        self.ui.set_status(f"{self.ui._next_spinner_verb()}…")
        return result

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        # We don't want to spam — the REPL handles the final error display
        # — but the spinner shouldn't keep ticking after a fatal.
        self.ui.clear_status()


class CLIUIHooks(EngineHooks):
    """Engine lifecycle hooks (currently a no-op; reserved for future."""

    def __init__(self, ui: CLIUI) -> None:
        super().__init__()
        self.ui = ui

    def on_session_start(self, *, session_id: str, context_metadata: dict[str, Any]) -> None:
        return None

    def post_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        response_text: str,
        context_metadata: dict[str, Any],
    ) -> None:
        # The provider may have already streamed `response_text` via the
        # stream_callback. If it did, the UI's stream is open and we let
        # ``end_stream`` happen naturally when the engine finalises.
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
