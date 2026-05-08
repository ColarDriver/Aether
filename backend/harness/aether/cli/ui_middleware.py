"""Engine-side bridge that feeds UI events from middleware/hook callbacks.

Translates ``EngineMiddleware.before_tool``/``after_tool`` and
``EngineHooks.pre_llm_call``/``post_llm_call`` into ``CLIUI`` calls so
that the run loop visibly narrates what the agent is doing.

The bridge also drives :class:`aether.cli.activity.TurnState` mode
transitions so the bottom activity bar reflects the engine's current
phase (thinking / responding / tool-use), and feeds dispatched calls
into the :class:`aether.cli.tool_groups.ToolGroupTracker` so consecutive
read/search/list tools coalesce into a single rolling line instead of
one block per call.
"""

from __future__ import annotations

from typing import Any

from aether.agents.middlewares.base import EngineMiddleware
from aether.cli.activity import (
    MODE_RESPONDING,
    MODE_THINKING,
    MODE_TOOL_USE,
)
from aether.cli.tool_groups import hint_for_call
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
        # Each iteration the engine asks the LLM what's next.  Flip the
        # bar back to "thinking" so we get a fresh "thought for Ns" once
        # the first delta lands; rotate the verb so the loop visibly
        # progresses without feeling mechanical.  Also flush the prior
        # iteration's tool group so its past-tense summary lands in
        # scrollback before the new iteration's prelude/tool group.
        self.ui.tool_groups.begin_iteration()
        st = self.ui._turn_state
        st.mode = MODE_THINKING
        st.has_active_tools = False
        st.mark_thinking_start()
        if int(context.iteration or 0) > 1:
            self.ui.set_status(f"{self.ui._next_spinner_verb()}…")
        else:
            # First iteration — the REPL already set a verb in begin_turn,
            # but ``mark_thinking_start`` reset thinking state above so we
            # still get a clean elapsed measurement.
            pass
        return messages

    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        # ``responding`` was already set when the first stream delta
        # arrived (see ``CLIUI.stream_delta``).  For non-streaming
        # providers we set it here so the bar reflects "model produced
        # output" before any tool dispatch.
        st = self.ui._turn_state
        if response.content and st.mode == MODE_THINKING:
            st.mark_first_response_token()
            st.mode = MODE_RESPONDING
        return response

    def before_tool(self, call: ToolCall | ToolResult, context: TurnContext) -> ToolCall | ToolResult:
        # before_tool runs once per call, possibly receiving a short-circuit
        # ToolResult from earlier middleware (e.g. policy denials).  Render
        # accordingly so the user sees both real and short-circuited paths.
        st = self.ui._turn_state
        st.mode = MODE_TOOL_USE
        st.has_active_tools = True

        if isinstance(call, ToolResult):
            # Short-circuit: the call never runs.  We still feed the
            # tracker so the headline reflects what the model asked for,
            # then immediately mark it finished so the active-group line
            # doesn't linger.
            self.ui.tool_groups.start_call(call.name, {})
            self.ui.tool_groups.finish_call(is_error=call.is_error)
            self.ui.stats.tool_calls += 1
            if call.is_error:
                self.ui.stats.tool_errors += 1
            self.ui.render_tool_call(call.name, {"_short_circuit": True})
            self.ui.render_tool_result(
                call.name,
                call.content,
                is_error=call.is_error,
                metadata=dict(call.metadata),
            )
            st.has_active_tools = False
            st.tool_use_count += 1
            return call

        args = dict(call.arguments or {})
        # End any in-flight stream so the prelude lands in scrollback
        # before the tool group line / active-group preview shows up.
        # ``ToolGroupTracker`` itself doesn't care about ordering, but
        # the active-group rendering inside ``_TurnSurface`` relies on
        # the prelude being flushed so it doesn't get tail-cropped.
        if self.ui._stream.active:
            self.ui.end_stream()

        self.ui.tool_groups.start_call(call.name, args)
        self.ui.stats.tool_calls += 1

        # Reuse the same human-readable verb the tracker will display
        # ("Reading file…", "Running command…", …) so the activity bar
        # visibly follows the action that's underway.  Prefer a short
        # hint over the raw verb because it's more informative.
        verb = _verb_for_tool(call.name)
        hint = hint_for_call(call.name, args)
        bar_verb = f"{verb}…  {hint}" if hint else f"{verb}…"
        self.ui.set_status(bar_verb, spinner="dots")

        # Verbose: keep the per-call legacy block for debugging.  In
        # normal mode this is a no-op (see ``CLIUI.render_tool_call``).
        self.ui.render_tool_call(call.name, args, tool_call_id=call.id)
        st.tool_use_count += 1
        return call

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        self.ui.tool_groups.finish_call(is_error=result.is_error)
        if result.is_error:
            self.ui.stats.tool_errors += 1
        self.ui.render_tool_result(
            result.name,
            result.content,
            is_error=result.is_error,
            metadata=dict(result.metadata),
        )
        # Tool finished — bar goes back to a thinking verb for the gap
        # between tool result and the next LLM iteration.
        st = self.ui._turn_state
        st.mode = MODE_THINKING
        st.has_active_tools = False
        self.ui.set_status(f"{self.ui._next_spinner_verb()}…")
        return result

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        # We don't want to spam — the REPL handles the final error display
        # — but the bar shouldn't keep ticking after a fatal.
        self.ui.clear_status()
        # Drop any partially-resolved tool group so we don't print a
        # half-baked past-tense headline after the fatal.
        self.ui.tool_groups.discard_active()


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
