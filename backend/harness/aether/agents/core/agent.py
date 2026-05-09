"""Primary agent implementation for Aether harness."""

from __future__ import annotations

import copy
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List

from aether.agents.core.phantom_tool import (
    PhantomToolIntent,
    build_corrective_user_message,
    detect_phantom_tool_intent,
    synthesize_tool_calls_from_phantom,
)
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    LoopState,
    NormalizedResponse,
    ToolResult,
    TurnContext,
)
from aether.runtime.hooks import EngineHooks
from aether.runtime.interrupts import InterruptController
from aether.runtime.provider_errors import ProviderInvocationError, ResponseInvalidError
from aether.runtime.recovery import (
    AttemptState,
    GenericBackoffStrategy,
    RecoveryStrategy,
    wait_interruptible,
)
from aether.runtime.services import EngineServices
from aether.runtime.session_runtime import (
    TURN_KEY_EMPTY_RESPONSE_RETRIES,
    TURN_KEY_INVALID_JSON_RETRIES,
    TURN_KEY_PHANTOM_TOOL_RETRIES,
    TURN_KEY_PHANTOM_TOOL_SYNTHESIZED,
    TURN_KEY_PROVIDER_ERROR_RETRIES,
    TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES,
    SessionRuntimeRegistry,
    SessionRuntimeState,
)
from aether.runtime.session_store import InMemorySessionStore, SessionStore
from aether.runtime.state_machine import EngineStateMachine
from aether.runtime.usage import CanonicalUsage, normalize_usage
from aether.tools.base import UnknownToolError
from aether.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aether.subagents.contracts import SubagentResult, SubagentTask
    from aether.subagents.manager import SubagentManager


# Sprint 3 / PR 3.1: keys that live on context.metadata as runtime helpers
# but should NOT leak into the JSON-serialisable ``EngineResult.metadata['turn']``
# snapshot.  Their normalised, public-facing form lives at the metadata top
# level (e.g. ``usage_accumulator`` → ``metadata['usage']``).  Add new
# internal-only keys here whenever a future PR introduces one.
_METADATA_INTERNAL_KEYS: frozenset[str] = frozenset({
    "usage_accumulator",
})


class AgentEngine:
    """Main agent orchestrator with an explicit run loop state machine."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        tool_registry: ToolRegistry | None = None,
        middleware_pipeline: MiddlewarePipeline | None = None,
        config: EngineConfig | None = None,
        interrupt_controller: InterruptController | None = None,
        logger: logging.Logger | None = None,
        delegate_depth: int = 0,
        subagent_id: str | None = None,
        parent_subagent_id: str | None = None,
        subagent_manager: "SubagentManager | None" = None,
        session_store: SessionStore | None = None,
        hooks: EngineHooks | None = None,
        recovery_strategy: RecoveryStrategy | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        if tool_registry is None:
            if self.config.use_builtin_tools:
                # Local import keeps subprocess/pathlib import cost
                # off the critical path for callers that explicitly
                # disable builtins (use_builtin_tools=False) or pass
                # their own registry.
                from aether.tools.builtins import build_default_tool_registry

                tool_registry = build_default_tool_registry()
            else:
                tool_registry = ToolRegistry()
        self.services = EngineServices(
            provider=provider,
            tool_registry=tool_registry,
            middleware_pipeline=middleware_pipeline or MiddlewarePipeline(),
            interrupt_controller=interrupt_controller or InterruptController(),
            logger=logger or logging.getLogger(__name__),
            # Default recovery strategy mirrors the pre-Sprint-0 in-provider
            # retry behaviour (3 attempts, 2s/4s exponential backoff on the
            # well-known retriable status codes).  Only structured
            # ``ProviderInvocationError`` errors are routed through this
            # strategy — generic ``Exception`` instances bypass it and go
            # straight to the middleware on_error pipeline as before.
            recovery_strategy=recovery_strategy or GenericBackoffStrategy(),
        )
        if self.services.middleware_pipeline.logger is None:
            self.services.middleware_pipeline.logger = self.services.logger

        self._delegate_depth = max(0, int(delegate_depth))
        self._subagent_id = subagent_id
        self._parent_subagent_id = parent_subagent_id
        self._subagent_manager = subagent_manager

        self._current_session_id: str | None = None
        self._active_children: dict[str, AgentEngine] = {}
        self._active_children_lock = RLock()
        self._session_store = session_store or InMemorySessionStore()
        self._hooks = hooks or EngineHooks()

        # Per-session state registry (cross-turn nudge counters, future
        # cached system prompt for prefix-cache stability).  Storing these
        # on a session-keyed registry instead of plain ``self._*`` attributes
        # is what makes a single ``AgentEngine`` instance safe to share
        # across many concurrent sessions — see ``runtime/session_runtime.py``
        # for the rationale.
        #
        # Note: per-turn retry counters (empty_response_retries,
        # provider_error_retries) live on ``TurnContext.metadata`` instead
        # of here, because their lifetime is exactly one turn.
        self._session_runtime = SessionRuntimeRegistry()

    @property
    def delegate_depth(self) -> int:
        return self._delegate_depth

    @property
    def subagent_id(self) -> str | None:
        return self._subagent_id

    @property
    def parent_subagent_id(self) -> str | None:
        return self._parent_subagent_id

    @property
    def subagent_manager(self) -> "SubagentManager | None":
        return self._subagent_manager

    def set_subagent_manager(self, manager: "SubagentManager") -> None:
        self._subagent_manager = manager

    def _register_child(self, child: "AgentEngine") -> None:
        key = child.subagent_id or f"child-{id(child)}"
        with self._active_children_lock:
            self._active_children[key] = child

    def _unregister_child(self, child: "AgentEngine") -> None:
        with self._active_children_lock:
            to_remove = [k for k, v in self._active_children.items() if v is child]
            for key in to_remove:
                self._active_children.pop(key, None)

    def _interrupt_active_children(self, reason: str | None = None) -> None:
        with self._active_children_lock:
            children = list(self._active_children.values())
        for child in children:
            child.interrupt(reason=reason)

    def interrupt(self, session_id: str | None = None, reason: str | None = None) -> None:
        effective_session = session_id or self._current_session_id
        if effective_session:
            self.services.interrupt_controller.request(effective_session, reason)
        self._interrupt_active_children(reason=reason)

    def clear_interrupt(self, session_id: str | None = None) -> None:
        effective_session = session_id or self._current_session_id
        if effective_session:
            self.services.interrupt_controller.clear(effective_session)

    def run_subagents(
        self,
        tasks: List["SubagentTask"],
        *,
        max_concurrent_children: int | None = None,
    ) -> List["SubagentResult"]:
        if self._subagent_manager is None:
            raise RuntimeError("Subagent manager is not configured")
        return self._subagent_manager.run_tasks(
            parent=self,
            tasks=tasks,
            max_concurrent_children=max_concurrent_children,
        )

    def resume(self, request: EngineRequest) -> EngineResult:
        return self.run_loop(request)

    def run_turn(self, request: EngineRequest) -> EngineResult:
        return self.run_loop(request)

    def run_loop(self, request: EngineRequest) -> EngineResult:
        """Execute one full agent loop turn until completion/failure/interruption."""
        self._current_session_id = request.session_id
        context: TurnContext | None = None
        active_system_prompt: str | None = None
        stream_callback_wrapped = None

        final_response: str | None = None
        error_text: str | None = None
        exit_reason = ExitReason.EMPTY_RESPONSE
        iterations = 0

        try:
            state_machine, messages, context = self._prepare_turn_entry(request)
            messages, active_system_prompt = self._prepare_session_and_system_prompt(
                request,
                messages,
                context,
            )
            self._apply_turn_nudges(context)
            stream_callback_wrapped = self._build_stream_callback(request, context)
            stream_silent_callback_wrapped = self._build_stream_silent_callback(
                request, context
            )

            state_machine.transition(LoopState.PREPARE)
            if self._is_interrupted(request.session_id):
                state_machine.transition(LoopState.INTERRUPTED)
                exit_reason = ExitReason.INTERRUPTED
            else:
                state_machine.transition(LoopState.PRE_LLM)

                # Main iterative loop: each iteration can produce tool calls or a terminal text response.
                while iterations < self.config.max_iterations:
                    context.iteration = iterations + 1

                    if self._is_interrupted(request.session_id):
                        state_machine.transition(LoopState.INTERRUPTED)
                        exit_reason = ExitReason.INTERRUPTED
                        break

                    self._safe_call_hook(
                        "pre_llm_call",
                        session_id=request.session_id,
                        iteration=context.iteration,
                        messages=copy.deepcopy(messages),
                        context_metadata=context.metadata,
                    )

                    # PRE_LLM middleware stage: rewrite/enrich outbound message list.
                    # PR 1.2 continuation path can override the message list
                    # for exactly one next iteration (partial assistant +
                    # continuation user instruction).  Pop it here so retries
                    # do not accidentally keep reusing a stale override.
                    loop_messages = context.metadata.pop("_messages_override", None)
                    if isinstance(loop_messages, list):
                        messages = loop_messages

                    try:
                        prepared_messages = self.services.middleware_pipeline.run_before_llm(messages, context)
                    except Exception as exc:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break

                    state_machine.transition(LoopState.LLM_CALL)
                    # Allow middleware to short-circuit the provider call (e.g. circuit breaker).
                    response = self._pop_context_response(context, "llm_pre_response")
                    if response is None:
                        # ProviderInvocationError → engine-side recovery strategy
                        # decides whether to retry; any other Exception bypasses
                        # the strategy and goes straight to the middleware
                        # on_error path (preserving pre-Sprint-0 behaviour for
                        # bugs / programming errors / scripted-test providers).
                        invoke_outcome = self._invoke_provider_with_recovery(
                            request=request,
                            prepared_messages=prepared_messages,
                            stream_callback=stream_callback_wrapped,
                            stream_silent_callback=stream_silent_callback_wrapped,
                            context=context,
                        )
                        if invoke_outcome.interrupted:
                            state_machine.transition(LoopState.INTERRUPTED)
                            exit_reason = ExitReason.INTERRUPTED
                            break
                        if invoke_outcome.response is not None:
                            response = invoke_outcome.response
                        else:
                            # Recovery exhausted (or never applicable): fall
                            # through to the existing middleware on_error
                            # pipeline so it can build a user-facing message.
                            exc = invoke_outcome.error
                            assert exc is not None  # type-checker: mutually exclusive with response
                            self._handle_pipeline_error(exc, state_machine.state, context)
                            recovered_response = self._pop_context_response(context, "llm_error_response")
                            if recovered_response is None:
                                error_text = str(exc)
                                # Sprint 1 / PR 1.1: surface RESPONSE_INVALID
                                # as a distinct terminal so observers can
                                # tell "API kept handing us malformed bodies"
                                # apart from "API itself errored".
                                if isinstance(exc, ResponseInvalidError):
                                    exit_reason = ExitReason.RESPONSE_INVALID
                                else:
                                    exit_reason = ExitReason.PROVIDER_ERROR
                                state_machine.transition(LoopState.FAILED)
                                break
                            response = recovered_response

                    state_machine.transition(LoopState.POST_LLM)
                    # POST_LLM middleware stage: normalize/annotate response before control-flow split.
                    try:
                        response = self.services.middleware_pipeline.run_after_llm(response, context)
                    except Exception as exc:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break

                    # Sprint 3 / PR 3.1: accumulate token usage per LLM call.
                    # We do this AFTER the after_llm middleware pipeline so any
                    # response-level normalisation it performed is reflected in
                    # the raw ``metadata["usage"]`` dict we read here.  The
                    # accumulator lives on context.metadata so it is per-turn
                    # and per-session safe (no cross-session leak).
                    self._accumulate_usage(response, context)
                    iterations += 1

                    self._safe_call_hook(
                        "post_llm_call",
                        session_id=request.session_id,
                        iteration=context.iteration,
                        response_text=response.content or "",
                        context_metadata=context.metadata,
                    )

                    if response.finish_reason == "length":
                        # Sprint 1 / PR 1.3: ``finish_reason="length"`` +
                        # tool_calls is a structurally different failure
                        # from prose-truncation.  The continuation prompt
                        # used by PR 1.2 cannot finish a half-emitted JSON
                        # tool argument, so we route this case through a
                        # dedicated retry-once-then-refuse handler instead
                        # of the regular length-continuation path.
                        if (
                            response.tool_calls
                            and getattr(self.config, "truncated_tool_call_detection_enabled", True)
                        ):
                            handled = self._handle_length_with_tool_calls(
                                response=response,
                                messages=messages,
                                context=context,
                            )
                        else:
                            handled = self._handle_length_finish_reason(
                                response=response,
                                messages=messages,
                                request=request,
                                context=context,
                            )
                        if handled.action == "continue":
                            messages = handled.messages
                            state_machine.transition(LoopState.CHECK_EXIT)
                            if iterations >= self.config.max_iterations:
                                state_machine.transition(LoopState.FINALIZE)
                                exit_reason = ExitReason.MAX_ITERATIONS
                                break
                            state_machine.transition(LoopState.PRE_LLM)
                            continue
                        if handled.action == "finalize":
                            messages = handled.messages
                            final_response = handled.final_response
                            exit_reason = handled.exit_reason
                            state_machine.transition(LoopState.FINALIZE)
                            break

                    if response.tool_calls:
                        # Sprint 1 / PR 1.3 — gate the dispatcher on
                        # tool-call argument validation.  The validator
                        # normalises argument types in place, decides
                        # whether the response is truncated (and we
                        # should retry without poisoning history), and
                        # falls back to a tool-error injection so the
                        # model can self-correct on persistent JSON
                        # mistakes.  See ``_validate_tool_call_arguments``
                        # for the full state machine.
                        if getattr(self.config, "truncated_tool_call_detection_enabled", True):
                            validation = self._validate_tool_call_arguments(
                                response=response,
                                messages=messages,
                                context=context,
                            )
                            if validation.action == "retry":
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if iterations >= self.config.max_iterations:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue
                            if validation.action == "truncated":
                                rollback = self._get_messages_up_to_last_assistant(messages)
                                visible_text = self._extract_visible_text(response.content or "")
                                if visible_text:
                                    prefix_parts = context.metadata.setdefault(
                                        "truncated_response_prefix_parts", []
                                    )
                                    if isinstance(prefix_parts, list):
                                        prefix_parts.append(visible_text)
                                context.metadata["partial"] = True
                                context.metadata.setdefault(
                                    "length_exit_reason", "tool_call_truncated"
                                )
                                messages = rollback
                                final_response = visible_text or None
                                exit_reason = ExitReason.TOOL_CALL_TRUNCATED
                                state_machine.transition(LoopState.FINALIZE)
                                break
                            if validation.action == "inject_error":
                                # Append assistant tool_calls + per-call
                                # tool error stubs.  The model sees the
                                # JSON it produced and the parse error,
                                # then has the next iteration to retry.
                                messages.extend(validation.injection_messages)
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if iterations >= self.config.max_iterations:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue
                            # action == "ok" → fall through to dispatch.
                        self._register_skill_nudge(context)

                        # Tool path: append assistant tool-call message first, then execute each call.
                        state_machine.transition(LoopState.TOOL_DISPATCH)
                        self._append_assistant_tool_message(messages, response)

                        state_machine.transition(LoopState.TOOL_EXECUTE)
                        tool_failed = False
                        for call in response.tool_calls:
                            if self._is_interrupted(request.session_id):
                                state_machine.transition(LoopState.INTERRUPTED)
                                exit_reason = ExitReason.INTERRUPTED
                                break

                            try:
                                # before_tool can rewrite a ToolCall or short-circuit with ToolResult
                                # (for guardrails/policy blocks).
                                pre_tool = self.services.middleware_pipeline.run_before_tool(call, context)
                            except Exception as exc:
                                self._handle_pipeline_error(exc, state_machine.state, context)
                                error_text = str(exc)
                                exit_reason = ExitReason.MIDDLEWARE_ERROR
                                state_machine.transition(LoopState.FAILED)
                                tool_failed = True
                                break

                            if isinstance(pre_tool, ToolResult):
                                result = pre_tool
                            else:
                                tool_call = pre_tool
                                # Track active call so middleware on_error handlers can build
                                # a deterministic fallback ToolResult for this exact invocation.
                                context.metadata.pop("tool_error_result", None)
                                context.metadata["_active_tool_call"] = tool_call
                                try:
                                    result = self.services.tool_registry.dispatch(tool_call, context)
                                except UnknownToolError:
                                    if self.config.fail_on_unknown_tool:
                                        error_text = f"Unknown tool: {tool_call.name}"
                                        exit_reason = ExitReason.UNKNOWN_TOOL
                                        state_machine.transition(LoopState.FAILED)
                                        tool_failed = True
                                        break
                                    result = ToolResult(
                                        tool_call_id=tool_call.id,
                                        name=tool_call.name,
                                        content=f"Unknown tool: {tool_call.name}",
                                        is_error=True,
                                    )
                                except Exception as exc:
                                    if self.config.fail_on_tool_error:
                                        # If strict mode is enabled, middleware may still recover
                                        # by providing a synthetic ToolResult in metadata.
                                        self._handle_pipeline_error(exc, state_machine.state, context)
                                        recovered_tool_result = context.metadata.pop("tool_error_result", None)
                                        if not isinstance(recovered_tool_result, ToolResult):
                                            error_text = str(exc)
                                            exit_reason = ExitReason.TOOL_ERROR
                                            state_machine.transition(LoopState.FAILED)
                                            tool_failed = True
                                            break
                                        result = recovered_tool_result
                                    else:
                                        result = ToolResult(
                                            tool_call_id=tool_call.id,
                                            name=tool_call.name,
                                            content=f"Tool execution error: {exc}",
                                            is_error=True,
                                        )
                                finally:
                                    context.metadata.pop("_active_tool_call", None)

                            try:
                                # after_tool middleware stage for redaction, auditing, or shaping.
                                result = self.services.middleware_pipeline.run_after_tool(result, context)
                            except Exception as exc:
                                self._handle_pipeline_error(exc, state_machine.state, context)
                                error_text = str(exc)
                                exit_reason = ExitReason.MIDDLEWARE_ERROR
                                state_machine.transition(LoopState.FAILED)
                                tool_failed = True
                                break

                            self._append_tool_result_message(messages, result)

                        if state_machine.state in {LoopState.FAILED, LoopState.INTERRUPTED}:
                            break
                        if tool_failed:
                            break

                        # Continue iterative tool-use loop unless max iteration budget is exhausted.
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if iterations >= self.config.max_iterations:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break

                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    # No tool calls -> finalize response (or recover
                    # from phantom-tool intent first).
                    prefix_parts = context.metadata.pop("truncated_response_prefix_parts", None)
                    prefix = " ".join(part.strip() for part in prefix_parts if isinstance(part, str) and part.strip()) if isinstance(prefix_parts, list) else ""
                    suffix = (response.content or "").strip()
                    combined_content = ((prefix + " " + suffix).strip() if prefix and suffix else (prefix or suffix))
                    response_to_store = response
                    if combined_content != (response.content or ""):
                        response_to_store = NormalizedResponse(
                            content=combined_content,
                            tool_calls=list(response.tool_calls),
                            finish_reason=response.finish_reason,
                            metadata=dict(response.metadata),
                        )

                    # Phantom-tool recovery — when the assistant body
                    # carries clear evidence of attempted tool use
                    # (\u0060\u0060\u0060bash blocks, ``<function=NAME>`` inline
                    # tags, ``<invoke>`` XML) but ``tool_calls`` is
                    # empty, the model is "describing" rather than
                    # invoking.  Without this branch the loop would
                    # silently finalise as TEXT_RESPONSE, leaving the
                    # user with a green checkmark and no work done.
                    #
                    # Three outcomes:
                    #
                    # * ``"retry"`` — append the assistant's prose to
                    #   history, append a corrective ``role=user``
                    #   nudge, bump the retry counter, continue the
                    #   loop and let the model self-correct.
                    # * ``"exhausted"`` — phantom intent was present
                    #   but ``max_phantom_tool_retries`` already
                    #   burned; fall through to finalise but tag the
                    #   turn ``PHANTOM_TOOL_INTENT`` so the UI shows
                    #   "model never invoked anything" instead of a
                    #   misleading green checkmark.
                    # * ``"none"`` — body was prose, no recovery
                    #   needed; fall through unchanged.
                    phantom_outcome = self._maybe_recover_phantom_tool_intent(
                        response_to_store=response_to_store,
                        messages=messages,
                        context=context,
                    )
                    if phantom_outcome == "synthesized":
                        # The recovery method has populated
                        # ``response_to_store.tool_calls`` from the
                        # parsed prose.  Run the synthesized calls
                        # through the dispatch path inline so the
                        # next LLM iteration sees a clean tool/result
                        # pair, just as if the model had emitted
                        # structured tool_calls in the first place.
                        synth_outcome, synth_exit, synth_error = self._dispatch_synthesized_tool_calls(
                            response=response_to_store,
                            messages=messages,
                            context=context,
                            state_machine=state_machine,
                            request=request,
                        )
                        if synth_outcome == "interrupted":
                            exit_reason = ExitReason.INTERRUPTED
                            break
                        if synth_outcome == "failed":
                            assert synth_exit is not None
                            exit_reason = synth_exit
                            error_text = synth_error
                            break
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if iterations >= self.config.max_iterations:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    if phantom_outcome == "retry":
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if iterations >= self.config.max_iterations:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    self._append_assistant_text_message(messages, response_to_store)
                    final_response = response_to_store.content or ""
                    if stream_callback_wrapped and final_response and not context.metadata.get("streamed_output"):
                        stream_callback_wrapped(final_response)
                        context.metadata["stream_fallback_emitted"] = True

                    if final_response:
                        # Reset the per-turn empty-response counter on success.
                        # (Currently no consumer reads this counter — Sprint 4
                        # will introduce the 9-step empty-response degradation
                        # path that consumes it.)
                        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
                        if phantom_outcome == "exhausted":
                            # Body looked like attempted tool use but
                            # the recovery budget ran out — surface
                            # this as a distinct exit reason so the
                            # UI footer can flag it ("model described
                            # commands but never invoked them") rather
                            # than the misleading TEXT_RESPONSE green
                            # checkmark.
                            exit_reason = ExitReason.PHANTOM_TOOL_INTENT
                        else:
                            exit_reason = (
                                ExitReason.LENGTH_RECOVERED if prefix else ExitReason.TEXT_RESPONSE
                            )
                    else:
                        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = (
                            int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)) + 1
                        )
                        exit_reason = ExitReason.EMPTY_RESPONSE
                    state_machine.transition(LoopState.FINALIZE)
                    break

            # Force a deterministic terminal state if loop exits by condition rather than break.
            if state_machine.state not in {LoopState.FAILED, LoopState.INTERRUPTED, LoopState.FINALIZE}:
                state_machine.transition(LoopState.FINALIZE)
                if iterations >= self.config.max_iterations:
                    exit_reason = ExitReason.MAX_ITERATIONS

            # FINALIZE -> DONE transition for successful/terminal completion paths.
            if state_machine.state == LoopState.FINALIZE:
                state_machine.transition(LoopState.DONE)

            result = self._build_result(
                request,
                messages,
                iterations,
                final_response,
                error_text,
                exit_reason,
                context=context,
                active_system_prompt=active_system_prompt,
            )
            self.services.interrupt_controller.clear(request.session_id)

            self._safe_call_hook(
                "on_session_end",
                session_id=request.session_id,
                completed=result.status in {EngineStatus.COMPLETED, EngineStatus.MAX_ITERATIONS},
                interrupted=result.status == EngineStatus.INTERRUPTED,
                context_metadata=context.metadata,
            )
            return result
        finally:
            self._current_session_id = None

    def _prepare_turn_entry(
        self,
        request: EngineRequest,
    ) -> tuple[EngineStateMachine, List[Dict[str, Any]], TurnContext]:
        state_machine = EngineStateMachine()
        messages = self._sanitize_messages(copy.deepcopy(request.messages))

        if request.user_message is not None:
            messages.append({"role": "user", "content": self._sanitize_text(request.user_message)})

        task_id = self._sanitize_text(str(request.metadata.get("task_id", "")).strip()) if request.metadata else ""
        if not task_id:
            task_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())

        metadata = dict(request.metadata)
        # Sprint 3 / PR 3.1: stamp the active provider's identity onto the
        # turn so middleware (TokenUsageMiddleware) and downstream tools can
        # pick the right ``normalize_usage`` parser without dragging
        # ``EngineServices`` into the middleware contract.  Re-read from
        # ``self.services.provider`` each turn so PR 3.4's fallback chain
        # (which swaps the active provider mid-session) gets a fresh value.
        active_provider = self.services.provider
        metadata.update(
            {
                "entry_prepared": True,
                "task_id": task_id,
                "turn_id": turn_id,
                "request_has_stream_callback": bool(request.stream_callback),
                "active_provider_name": getattr(active_provider, "provider_name", "openai"),
                "active_provider_api_mode": getattr(active_provider, "api_mode", "chat"),
                # Per-turn retry counters live on TurnContext.metadata so they
                # cannot leak across concurrent sessions sharing this engine.
                # Initialised to 0 here; mutated by the run-loop's empty-
                # response / provider-error / truncated-tool-call branches.
                TURN_KEY_EMPTY_RESPONSE_RETRIES: 0,
                TURN_KEY_PROVIDER_ERROR_RETRIES: 0,
                TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES: 0,
                TURN_KEY_INVALID_JSON_RETRIES: 0,
                TURN_KEY_PHANTOM_TOOL_RETRIES: 0,
                TURN_KEY_PHANTOM_TOOL_SYNTHESIZED: 0,
            }
        )

        context = TurnContext(
            session_id=request.session_id,
            iteration=0,
            metadata=metadata,
            task_id=task_id,
            turn_id=turn_id,
        )
        return state_machine, messages, context

    def _prepare_session_and_system_prompt(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> tuple[List[Dict[str, Any]], str | None]:
        stored_prompt = None
        is_new_session = not bool(request.messages)
        if self._session_store:
            session_row = self._session_store.get_session(request.session_id)
            if session_row:
                maybe_prompt = session_row.get("system_prompt")
                if isinstance(maybe_prompt, str) and maybe_prompt.strip():
                    stored_prompt = self._sanitize_text(maybe_prompt)
                is_new_session = False
            else:
                is_new_session = True

        requested_prompt = self._sanitize_text(request.system_message) if request.system_message else None
        selected_prompt = requested_prompt or stored_prompt

        # Sprint 1.5 / P0-9: augment the prompt with a registry-derived
        # <tool_use_contract> block before injecting it into messages.
        # We deliberately do NOT mutate ``selected_prompt`` itself —
        # storage / metadata / EngineResult should reflect what the
        # caller passed in, not the engine's boilerplate.  The contract
        # is regenerated each turn from the live registry, so resuming
        # a session with a different toolset still produces an accurate
        # block.
        prompt_for_messages = selected_prompt
        if self.config.tool_use_contract_enabled:
            from aether.agents.core.system_prompt import (
                augment_system_with_tool_contract,
            )

            prompt_for_messages = augment_system_with_tool_contract(
                selected_prompt,
                self.services.tool_registry.list_descriptors(),
            )

        if prompt_for_messages:
            messages = self._inject_system_prompt(messages, prompt_for_messages)

        context.metadata["system_prompt_applied"] = bool(selected_prompt)
        context.metadata["system_prompt_source"] = (
            "request_field" if requested_prompt else ("session_store" if stored_prompt else "none")
        )
        context.metadata["tool_use_contract_applied"] = bool(
            self.config.tool_use_contract_enabled
            and prompt_for_messages is not None
            and prompt_for_messages != selected_prompt
        )

        if self.config.enable_todo_hydration:
            self._hydrate_todo_snapshot(messages, context)

        if selected_prompt and self._session_store and (is_new_session or requested_prompt is not None):
            self._session_store.update_system_prompt(request.session_id, selected_prompt)

        context.metadata["system_prompt_preview"] = selected_prompt[:120] if selected_prompt else ""

        if is_new_session:
            self._safe_call_hook(
                "on_session_start",
                session_id=request.session_id,
                context_metadata=context.metadata,
            )

        return messages, selected_prompt

    def _apply_turn_nudges(self, context: TurnContext) -> None:
        """Bump the memory-review counter for this session and surface the flag.

        Cadence is per-session (not per-engine), so the counter must come from
        ``SessionRuntimeRegistry`` rather than an instance attribute.  This is
        what guarantees that two concurrent sessions sharing the same
        ``AgentEngine`` cannot interfere with each other's nudge timing.
        """
        context.metadata.setdefault("should_review_memory", False)
        context.metadata.setdefault("should_review_skills", False)

        interval = max(0, int(getattr(self.config, "memory_nudge_interval", 0)))
        if interval <= 0:
            return

        state = self._session_runtime.get(context.session_id)
        state.memory_nudge_counter += 1
        if state.memory_nudge_counter >= interval:
            context.metadata["should_review_memory"] = True
            state.memory_nudge_counter = 0

    def _register_skill_nudge(self, context: TurnContext) -> None:
        """Bump the skill-review counter on every tool-call iteration.

        Like ``_apply_turn_nudges`` this lives on ``SessionRuntimeState`` so
        the cadence cannot bleed across concurrent sessions.
        """
        interval = max(0, int(getattr(self.config, "skill_nudge_interval", 0)))
        if interval <= 0:
            return

        state = self._session_runtime.get(context.session_id)
        state.skill_nudge_counter += 1
        if state.skill_nudge_counter >= interval:
            context.metadata["should_review_skills"] = True
            state.skill_nudge_counter = 0

    def _maybe_recover_phantom_tool_intent(
        self,
        *,
        response_to_store: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> str:
        """Detect phantom-tool intent and decide what to do about it.

        Returns one of four string values:

        * ``"synthesized"`` — phantom intent was detected **and** the
          parsed prose mapped cleanly to registered tools, so the
          engine turned them into structured ``ToolCall``s.  The
          caller's ``response_to_store`` was mutated in place
          (``response_to_store.tool_calls`` now carries the synth
          calls) so the caller can fall straight through the normal
          tool-dispatch branch as if the model had emitted them.  Also
          bumps ``TURN_KEY_PHANTOM_TOOL_SYNTHESIZED`` and stashes the
          per-call notes in ``context.metadata['phantom_synth_notes']``
          so the UI can render a soft "synthesized" hint instead of
          the loud ``inline tool tags`` warning.
        * ``"retry"``  — phantom intent detected, retry budget had
          room, ``messages`` was extended with the assistant's prose
          + a corrective ``role=user`` nudge, and the per-turn
          counter was bumped.  The caller should ``continue`` the
          loop (after honouring max_iterations).
        * ``"exhausted"`` — phantom intent was present but
          ``EngineConfig.max_phantom_tool_retries`` had already been
          burned this turn.  ``messages`` is left untouched; the
          caller falls through to the normal text-response finalise
          path but should set ``exit_reason = PHANTOM_TOOL_INTENT``.
          Also stashes ``"phantom_tool_attempts"`` in
          ``context.metadata`` so observers / diagnostics can show
          *what* the model tried to invoke without re-parsing.
        * ``"none"`` — body was prose; no recovery needed.  Caller
          falls through unchanged.

        Designed as a pure decision routine: callers control the loop
        transitions and ``exit_reason`` assignment.  This keeps the
        hot path (no phantom intent at all — the common case) at one
        ``re.search`` against the response body and zero list
        mutations.
        """
        if not getattr(self.config, "phantom_tool_recovery_enabled", True):
            return "none"

        intent = detect_phantom_tool_intent(response_to_store.content or "")
        if intent.is_empty():
            return "none"

        # Cache the parsed intent for downstream observers (footer,
        # diagnostic).  Stored as plain dicts so middleware can
        # serialise it without importing the dataclass.
        context.metadata["phantom_tool_attempts"] = {
            "shell_commands": list(intent.shell_commands),
            "invoke_calls": [
                {"name": name, "arguments": dict(args)}
                for name, args in intent.invoke_calls
            ],
            "raw_intents_count": intent.raw_intents_count,
        }

        # Synthesis path — try to turn the prose intents into real
        # ``ToolCall``s the registry can dispatch *this iteration*.
        # When successful the caller skips the corrective-message
        # retry entirely and the loop continues forward with one
        # fewer wasted round-trip.  Synthesis is conservative: it
        # only emits calls for tools the registry already knows about
        # (after case-fold + underscore-normalise), so unknown tool
        # names still fall through to the corrective-message path.
        if getattr(self.config, "phantom_tool_synthesis_enabled", True):
            synth = synthesize_tool_calls_from_phantom(
                intent,
                self.services.tool_registry,
                id_prefix=f"phantom_{context.iteration}",
            )
            if synth is not None and synth.tool_calls:
                response_to_store.tool_calls = list(synth.tool_calls)
                # Strip phantom-tool prose out of the visible content
                # — leaving raw ``<function=...>`` tags in the
                # assistant message would re-trigger detection on the
                # next turn (when the message is re-sent in history).
                # We keep a brief acknowledgement in the content slot
                # so the UI's "● Listing 1 directory…" headline still
                # has *something* to render.
                response_to_store.content = ""
                context.metadata.setdefault("phantom_synth_notes", []).extend(synth.notes)
                if synth.unresolved:
                    context.metadata.setdefault(
                        "phantom_synth_unresolved", []
                    ).extend(
                        [{"name": name, "reason": reason} for name, reason in synth.unresolved]
                    )
                context.metadata[TURN_KEY_PHANTOM_TOOL_SYNTHESIZED] = (
                    int(context.metadata.get(TURN_KEY_PHANTOM_TOOL_SYNTHESIZED, 0))
                    + len(synth.tool_calls)
                )
                try:
                    self.services.logger.debug(
                        "phantom_tool_synthesis: emitted=%d unresolved=%d session=%s",
                        len(synth.tool_calls),
                        len(synth.unresolved),
                        context.session_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return "synthesized"

        attempts = int(context.metadata.get(TURN_KEY_PHANTOM_TOOL_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "max_phantom_tool_retries", 2)))
        if attempts >= max_attempts:
            return "exhausted"

        # Append the assistant's prose so the next iteration's
        # context window includes the model's broken response, then
        # the corrective nudge.  Order matters: removing the
        # assistant message would let the model "forget" what it
        # just wrote and re-emit the same broken pattern.
        self._append_assistant_text_message(messages, response_to_store)
        messages.append(
            build_corrective_user_message(intent, attempt_index=attempts + 1)
        )
        context.metadata[TURN_KEY_PHANTOM_TOOL_RETRIES] = attempts + 1
        # Reset the empty-response counter — this iteration produced
        # *content*, just in the wrong shape.  Carrying the
        # empty-response budget over would cause spurious 9-step
        # degradation kicks (P0-8) when phantom recovery is doing
        # the right thing.
        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
        try:
            self.services.logger.debug(
                "phantom_tool_recovery: attempt=%d / max=%d shell=%d invoke=%d session=%s",
                attempts + 1,
                max_attempts,
                len(intent.shell_commands),
                len(intent.invoke_calls),
                context.session_id,
            )
        except Exception:  # noqa: BLE001
            pass
        return "retry"

    def _dispatch_synthesized_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        state_machine: EngineStateMachine,
        request: EngineRequest,
    ) -> tuple[str, ExitReason | None, str | None]:
        """Dispatch ``response.tool_calls`` synthesized from prose intent.

        Returns ``(outcome, exit_reason, error_text)`` where ``outcome``
        is one of:

        * ``"continue"`` — every call dispatched, results appended,
          loop should iterate to the next LLM call.
        * ``"interrupted"`` — user pressed Ctrl-C mid-dispatch; caller
          should ``break`` and surface ``ExitReason.INTERRUPTED``.
        * ``"failed"`` — middleware / strict-mode tool error; caller
          should ``break`` with the returned ``exit_reason`` and
          ``error_text``.

        This helper is intentionally a tighter version of the regular
        tool-dispatch branch: we skip the truncation /
        invalid-JSON validators because the synthesized arguments
        were built by us (not parsed from a model JSON blob), so the
        failure modes those gates protect against do not apply.
        """
        state_machine.transition(LoopState.TOOL_DISPATCH)
        self._append_assistant_tool_message(messages, response)

        state_machine.transition(LoopState.TOOL_EXECUTE)
        for call in response.tool_calls:
            if self._is_interrupted(request.session_id):
                state_machine.transition(LoopState.INTERRUPTED)
                return "interrupted", ExitReason.INTERRUPTED, None

            try:
                pre_tool = self.services.middleware_pipeline.run_before_tool(call, context)
            except Exception as exc:
                self._handle_pipeline_error(exc, state_machine.state, context)
                state_machine.transition(LoopState.FAILED)
                return "failed", ExitReason.MIDDLEWARE_ERROR, str(exc)

            if isinstance(pre_tool, ToolResult):
                result = pre_tool
            else:
                tool_call = pre_tool
                context.metadata.pop("tool_error_result", None)
                context.metadata["_active_tool_call"] = tool_call
                try:
                    result = self.services.tool_registry.dispatch(tool_call, context)
                except UnknownToolError:
                    if self.config.fail_on_unknown_tool:
                        state_machine.transition(LoopState.FAILED)
                        return (
                            "failed",
                            ExitReason.UNKNOWN_TOOL,
                            f"Unknown tool: {tool_call.name}",
                        )
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
                except Exception as exc:
                    if self.config.fail_on_tool_error:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        recovered_tool_result = context.metadata.pop("tool_error_result", None)
                        if not isinstance(recovered_tool_result, ToolResult):
                            state_machine.transition(LoopState.FAILED)
                            return "failed", ExitReason.TOOL_ERROR, str(exc)
                        result = recovered_tool_result
                    else:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=f"Tool execution error: {exc}",
                            is_error=True,
                        )
                finally:
                    context.metadata.pop("_active_tool_call", None)

            try:
                result = self.services.middleware_pipeline.run_after_tool(result, context)
            except Exception as exc:
                self._handle_pipeline_error(exc, state_machine.state, context)
                state_machine.transition(LoopState.FAILED)
                return "failed", ExitReason.MIDDLEWARE_ERROR, str(exc)

            self._append_tool_result_message(messages, result)

        return "continue", None, None

    def _build_stream_callback(self, request: EngineRequest, context: TurnContext):
        callback = request.stream_callback
        if callback is None:
            return None

        # Sprint 1 / PR 1.1 emergency rollback: if the operator has flipped
        # ``EngineConfig.streaming_enabled`` off (e.g. because a gateway is
        # serving broken SSE), pretend the request had no callback at all.
        # The provider then takes the non-streaming path and the user
        # gets a single final-response chunk instead of a stream.  No
        # exception is raised — graceful degradation is the whole point.
        if not getattr(self.config, "streaming_enabled", True):
            self.services.logger.debug(
                "streaming_enabled=False; suppressing stream_callback for session %s",
                request.session_id,
            )
            return None

        def _wrapped(delta: str) -> None:
            if not isinstance(delta, str) or not delta:
                return
            try:
                callback(delta)
                context.metadata["streamed_output"] = True
                context.metadata["stream_callback_calls"] = int(context.metadata.get("stream_callback_calls", 0)) + 1
            except Exception:
                self.services.logger.exception("stream callback failed")

        return _wrapped

    def _build_stream_silent_callback(self, request: EngineRequest, context: TurnContext):
        """Wrap :attr:`EngineRequest.stream_silent_callback` for provider use.

        Sprint 3 / PR 3.1 — the silent counterpart of
        :meth:`_build_stream_callback`.  Providers forward count-only
        chunks (tool-arg JSON, signatures) here; the wrapped callback
        bumps a separate ``stream_silent_callback_calls`` metadata
        counter so observability can tell how often the live token
        estimator was advanced via this path.

        Same rollback semantics: when ``streaming_enabled`` is off we
        return ``None`` so providers don't try to emit silent deltas
        either.  Same isolation guarantee: any exception from the
        downstream callback is logged and swallowed — a UI render
        failure must never poison the model call.
        """
        callback = request.stream_silent_callback
        if callback is None:
            return None

        if not getattr(self.config, "streaming_enabled", True):
            return None

        def _wrapped_silent(delta: str) -> None:
            if not isinstance(delta, str) or not delta:
                return
            try:
                callback(delta)
                context.metadata["stream_silent_callback_calls"] = (
                    int(context.metadata.get("stream_silent_callback_calls", 0)) + 1
                )
            except Exception:
                self.services.logger.exception("stream silent callback failed")

        return _wrapped_silent

    def _safe_call_hook(self, name: str, **kwargs: Any) -> None:
        hook = getattr(self._hooks, name, None)
        if hook is None:
            return
        try:
            hook(**kwargs)
        except Exception:
            self.services.logger.exception("Engine hook failed: %s", name)

    def _hydrate_todo_snapshot(self, messages: List[Dict[str, Any]], context: TurnContext) -> None:
        snapshot = self._extract_todo_snapshot(messages)
        if not snapshot:
            context.metadata["todo_restored_count"] = 0
            return
        context.metadata["todo_snapshot"] = snapshot
        context.metadata["todo_restored_count"] = len(snapshot)

    def _extract_todo_snapshot(self, messages: List[Dict[str, Any]]) -> list[dict[str, Any]]:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue

            metadata = message.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("todos"), list):
                todos = metadata.get("todos")
                if todos:
                    return [todo for todo in todos if isinstance(todo, dict)]

            content = message.get("content")
            if isinstance(content, str) and content.strip().startswith("["):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
                    return payload
        return []

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._sanitize_structure(msg) for msg in messages]

    def _sanitize_structure(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_text(value)
        if isinstance(value, list):
            return [self._sanitize_structure(item) for item in value]
        if isinstance(value, dict):
            return {k: self._sanitize_structure(v) for k, v in value.items()}
        return value

    @staticmethod
    def _sanitize_text(text: str | None) -> str:
        if not text:
            return ""
        return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))

    def _inject_system_prompt(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._sanitize_text(system_prompt).strip()
        if not prompt:
            return messages

        merged = list(messages)
        if merged and isinstance(merged[0], dict) and merged[0].get("role") == "system":
            existing = merged[0].get("content")
            if isinstance(existing, str) and existing.strip() == prompt:
                return merged

        merged.insert(0, {"role": "system", "content": prompt})
        return merged

    def _is_interrupted(self, session_id: str) -> bool:
        return self.services.interrupt_controller.is_interrupted(session_id)

    def _handle_pipeline_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        # Pass ``exc_info=error`` explicitly: this method is sometimes called
        # *outside* the original ``except`` block (e.g. after the recovery
        # loop in ``_invoke_provider_with_recovery``), where
        # ``sys.exc_info()`` would be cleared and ``logger.exception`` would
        # otherwise emit a misleading "NoneType: None" stack trace.
        self.services.logger.error(
            "AgentEngine error at %s: %s",
            state,
            error,
            exc_info=error,
        )
        self.services.middleware_pipeline.run_on_error(error, state, context)

    # ------------------------------------------------------------------
    # Provider invocation with engine-side recovery (Sprint 0 / PR 0.3)
    # ------------------------------------------------------------------

    @dataclass(slots=True)
    class _ProviderInvocationOutcome:
        """Tri-state result of ``_invoke_provider_with_recovery``.

        Exactly one of (``response``, ``error``) is non-None unless
        ``interrupted`` is True (in which case both are None — the engine
        treats that as an INTERRUPTED exit reason without consulting either).
        """

        response: NormalizedResponse | None = None
        error: Exception | None = None
        interrupted: bool = False

    def _invoke_provider_with_recovery(
        self,
        *,
        request: EngineRequest,
        prepared_messages: List[Dict[str, Any]],
        stream_callback,
        stream_silent_callback,
        context: TurnContext,
    ) -> "_ProviderInvocationOutcome":
        """Issue ``provider.generate`` and apply the recovery strategy on failure.

        Loop semantics:

        * On success → returns ``_ProviderInvocationOutcome(response=...)``.
        * On ``ProviderInvocationError`` → bumps the per-turn provider-error
          counter, asks the configured ``RecoveryStrategy``.  If the strategy
          says retry, we sleep interruptibly then loop.  If it says give up
          (or the wait gets interrupted), we hand the *most recent* error
          back to the caller so the existing middleware on_error path can
          run as before.
        * On any other ``Exception`` → bumps the counter once and returns
          immediately with that exception (no retry).  This preserves the
          old behaviour for non-provider bugs (e.g. the test suite's
          ``ScriptedProvider`` raising ``RuntimeError``).
        * If an interrupt arrives during a retry wait → returns
          ``interrupted=True``; the caller transitions to LoopState.INTERRUPTED.

        The retry trail is recorded in
        ``context.metadata['recovery_decisions']`` as a list of
        ``{"reason": ..., "wait_seconds": ..., "attempt": ...}`` dicts so
        callers and tests can inspect what the strategy did without parsing
        logs.
        """
        attempt_state = AttemptState()
        last_error: Exception | None = None
        decisions_log = context.metadata.setdefault("recovery_decisions", [])

        while True:
            try:
                call_config = request.model_config
                ephemeral_max_output_tokens = context.metadata.pop("_ephemeral_max_output_tokens", None)
                if isinstance(ephemeral_max_output_tokens, int) and ephemeral_max_output_tokens > 0:
                    # PR 1.2 length continuation temporarily raises the output
                    # budget without mutating EngineRequest.model_config in-place.
                    call_config = ModelCallConfig(
                        temperature=request.model_config.temperature,
                        max_tokens=ephemeral_max_output_tokens,
                        extra=dict(request.model_config.extra),
                    )

                response = self.services.provider.generate(
                    prepared_messages,
                    self.services.tool_registry.list_descriptors(),
                    call_config,
                    context,
                    stream_callback=stream_callback,
                    stream_silent_callback=stream_silent_callback,
                )
                # Sprint 1 / PR 1.1: post-LLM response-shape validation.
                # ``validate_response`` is non-mutating; if it returns False
                # we lift the structured failure into the recovery loop so
                # the existing retry / give-up machinery handles it.  The
                # default base-class implementation always returns valid,
                # so providers that don't care pay zero cost here.
                ok, reasons = self.services.provider.validate_response(response)
                if not ok:
                    raise ResponseInvalidError(
                        validation_errors=list(reasons),
                        body_summary="invalid response: " + "; ".join(reasons[:5]),
                        metadata={"phase": "validate_response"},
                    )
                return AgentEngine._ProviderInvocationOutcome(response=response)
            except ProviderInvocationError as exc:
                # Bump the observability counter once per failed attempt.
                context.metadata[TURN_KEY_PROVIDER_ERROR_RETRIES] = (
                    int(context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)) + 1
                )
                attempt_state.attempt += 1
                attempt_state.errors.append(exc)
                last_error = exc

                decision = self.services.recovery_strategy.decide(
                    error=exc,
                    attempt_state=attempt_state,
                    context=context,
                )
                decisions_log.append(
                    {
                        "attempt": attempt_state.attempt,
                        "retry": decision.retry,
                        "wait_seconds": decision.wait_seconds,
                        "reason": decision.reason,
                        "status_code": exc.status_code,
                        "is_network_error": exc.is_network_error,
                    }
                )

                if not decision.retry:
                    return AgentEngine._ProviderInvocationOutcome(error=exc)

                # Interruptible wait — if the user cancels mid-retry we abort
                # the whole turn rather than serve a stale recovery attempt.
                if decision.wait_seconds > 0:
                    completed = wait_interruptible(
                        decision.wait_seconds,
                        interrupt_controller=self.services.interrupt_controller,
                        session_id=request.session_id,
                    )
                    attempt_state.total_wait_seconds += decision.wait_seconds
                    if not completed:
                        return AgentEngine._ProviderInvocationOutcome(interrupted=True)

                # Loop and retry the provider call.
                continue
            except Exception as exc:
                # Non-structured errors (programming errors, ScriptedProvider
                # exhaustion, etc.) bypass the recovery strategy entirely —
                # one bump + immediate hand-off to middleware.
                context.metadata[TURN_KEY_PROVIDER_ERROR_RETRIES] = (
                    int(context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)) + 1
                )
                last_error = exc
                return AgentEngine._ProviderInvocationOutcome(error=last_error)

    @dataclass(slots=True)
    class _LengthHandlingOutcome:
        action: str
        messages: List[Dict[str, Any]]
        final_response: str | None = None
        exit_reason: ExitReason = ExitReason.TEXT_RESPONSE

    @dataclass(slots=True)
    class _ToolCallValidationOutcome:
        """Verdict from ``_validate_tool_call_arguments``.

        ``action`` is one of:

        * ``"ok"``               — every tool_call has parseable args; the
                                   engine proceeds with the existing
                                   dispatch path.
        * ``"retry"``            — at least one tool_call had truncated /
                                   invalid args but we still have retry
                                   budget; the engine continues without
                                   appending to history (re-issues the
                                   provider call).
        * ``"truncated"``        — args looked truncated and we are out of
                                   retry budget; the engine finalises the
                                   turn with TOOL_CALL_TRUNCATED.
        * ``"inject_error"``    — args were unparseable JSON (not
                                   truncated) and we exhausted the silent
                                   retry budget; the caller appends the
                                   assistant message + a tool-error stub
                                   per failing call so the model can
                                   self-correct on the next iteration.
        """

        action: str
        invalid_json_args: List[tuple[str, str]] = field(default_factory=list)
        # When action="inject_error", the messages the caller should append
        # before continuing the loop — assistant tool_calls message followed
        # by one ``role="tool"`` error stub per failing tool_call.
        injection_messages: List[Dict[str, Any]] = field(default_factory=list)

    def _handle_length_finish_reason(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        request: EngineRequest,
        context: TurnContext,
    ) -> "_LengthHandlingOutcome":
        """Handle ``finish_reason == "length"`` before normal tool/text split.

        Sprint 1 / PR 1.2 introduces the first half of Hermes' length-handling
        behaviour:

        1. **Thinking-budget exhaustion** — if the model used its output budget
           entirely on hidden reasoning-like text (``<think>`` / ``<reasoning>``)
           and produced no visible answer, stop politely instead of returning an
           empty response.
        2. **Continuation retry** — append the partial assistant message plus a
           user continuation instruction, raise an ephemeral max token budget,
           and re-enter PRE_LLM up to ``max_length_continue_retries`` times.
        3. **Rollback / partial return** — once retries are exhausted, drop the
           continuation scaffolding and return the best visible prefix we have,
           marking the turn as ``LENGTH_EXHAUSTED``.

        P0-4's truncated tool-call detection is intentionally NOT implemented
        here yet; tool-call + length currently falls into the same continuation
        path and will be tightened in PR 1.3.
        """
        if not getattr(self.config, "length_continuation_enabled", True):
            merged = list(messages)
            self._append_assistant_text_message(merged, response)
            return AgentEngine._LengthHandlingOutcome(
                action="finalize",
                messages=merged,
                final_response=response.content or "",
                exit_reason=ExitReason.TEXT_RESPONSE,
            )

        visible_text = self._extract_visible_text(response.content or "")
        attempts = int(context.metadata.get("length_continue_attempts", 0))

        if self._looks_like_thinking_only_length_response(response):
            merged = list(messages)
            friendly = (
                "The model ran out of output budget while reasoning and did not produce a visible answer. "
                "Please try again with a lower reasoning effort or a larger max token limit."
            )
            merged.append(
                {
                    "role": "assistant",
                    "content": friendly,
                    "finish_reason": response.finish_reason,
                    "metadata": {"partial": True, "length_reason": "thinking_budget"},
                }
            )
            context.metadata["partial"] = True
            context.metadata["length_exit_reason"] = "thinking_budget"
            return AgentEngine._LengthHandlingOutcome(
                action="finalize",
                messages=merged,
                final_response=friendly,
                exit_reason=ExitReason.LENGTH_EXHAUSTED,
            )

        max_retries = max(0, int(getattr(self.config, "max_length_continue_retries", 3)))
        if attempts < max_retries:
            continuation_messages = list(messages)
            continuation_messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "finish_reason": response.finish_reason,
                    "metadata": {
                        "partial": True,
                        "length_continue_attempt": attempts + 1,
                    },
                }
            )
            continuation_messages.append(
                {
                    "role": "user",
                    "content": (
                        "[System: Your previous response was truncated by the output limit. "
                        "Continue exactly where you left off without repeating earlier text.]"
                    ),
                    "metadata": {"_length_continue_prompt": True},
                }
            )
            context.metadata["length_continue_attempts"] = attempts + 1
            if visible_text:
                prefix_parts = context.metadata.setdefault("truncated_response_prefix_parts", [])
                if isinstance(prefix_parts, list):
                    prefix_parts.append(visible_text)
            base_max_tokens = request.model_config.max_tokens or 0
            if base_max_tokens > 0:
                context.metadata["_ephemeral_max_output_tokens"] = min(32000, base_max_tokens * (attempts + 2))
            context.metadata["_messages_override"] = continuation_messages
            return AgentEngine._LengthHandlingOutcome(action="continue", messages=continuation_messages)

        rollback_messages = self._get_messages_up_to_last_assistant(messages)
        prefix_parts = context.metadata.get("truncated_response_prefix_parts", [])
        prefix_items = [part.strip() for part in prefix_parts if isinstance(part, str) and part.strip()]
        if visible_text:
            prefix_items.append(visible_text)
        prefix = " ".join(prefix_items).strip()
        context.metadata["partial"] = True
        context.metadata["length_exit_reason"] = "retries_exhausted"
        context.metadata["truncated_response_prefix"] = prefix
        if prefix:
            rollback_messages.append(
                {
                    "role": "assistant",
                    "content": prefix,
                    "finish_reason": response.finish_reason,
                    "metadata": {"partial": True, "length_exhausted": True},
                }
            )
        return AgentEngine._LengthHandlingOutcome(
            action="finalize",
            messages=rollback_messages,
            final_response=prefix or (response.content or ""),
            exit_reason=ExitReason.LENGTH_EXHAUSTED,
        )

    @staticmethod
    def _strip_reasoning_markup(text: str) -> str:
        cleaned = text.replace("<think>", "").replace("</think>", "")
        cleaned = cleaned.replace("<reasoning>", "").replace("</reasoning>", "")
        return cleaned

    def _extract_visible_text(self, text: str) -> str:
        """Remove reasoning blocks entirely, then trim remaining visible text."""
        cleaned = text
        while True:
            lowered = cleaned.lower()
            start = lowered.find("<think>")
            if start == -1:
                break
            end = lowered.find("</think>", start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len("</think>"):]
        while True:
            lowered = cleaned.lower()
            start = lowered.find("<reasoning>")
            if start == -1:
                break
            end = lowered.find("</reasoning>", start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len("</reasoning>"):]
        return cleaned.strip()

    def _looks_like_thinking_only_length_response(self, response: NormalizedResponse) -> bool:
        if response.finish_reason != "length":
            return False
        raw_text = (response.content or "").strip()
        if not raw_text:
            return False
        visible = self._extract_visible_text(raw_text)
        if visible:
            return False
        lowered = raw_text.lower()
        return "<think>" in lowered or "<reasoning>" in lowered

    # ------------------------------------------------------------------
    # Truncated tool-call detection (Sprint 1 / PR 1.3, P0-4)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_truncated_tool_call(tool_calls: List[Any]) -> bool:
        """Heuristic for "the model ran out of tokens while emitting tool args".

        Sprint 1 / PR 1.3 — mirrors the hermes-agent detector at
        ``run_agent.py`` lines 13362-13366.  We look at each tool_call's
        ``arguments`` payload after stripping trailing whitespace.  A
        well-formed JSON object/array always ends with ``}`` or ``]``; if
        we see anything else the args were almost certainly cut mid-stream.

        Why a heuristic rather than ``json.loads``?  Because some models
        emit args in slightly non-canonical JSON (single quotes, trailing
        commas) that ``json.loads`` rejects but that downstream tools are
        happy with after a normalisation pass.  We only want to refuse
        dispatch when the *shape* looks incomplete, not when the *syntax*
        is loose.

        We also tolerate already-parsed dict/list arguments — those came
        in fully decoded by the provider so they are by definition
        non-truncated.  Empty / whitespace-only strings are ignored
        (treated as "{}" elsewhere) and never count as truncation.
        """
        for call in tool_calls:
            raw = getattr(call, "arguments", None)
            if isinstance(raw, (dict, list)):
                continue
            text = "" if raw is None else str(raw)
            stripped = text.rstrip()
            if not stripped:
                continue
            if not stripped.endswith(("}", "]")):
                return True
        return False

    def _validate_tool_call_arguments(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> "AgentEngine._ToolCallValidationOutcome":
        """Normalise tool_call args, detect truncation, decide retry vs dispatch.

        Sprint 1 / PR 1.3 — implements P0-4 entirely:

        1. **Type normalisation.**  ``arguments`` arrives as ``str`` 99% of
           the time, but providers occasionally hand us a pre-parsed
           ``dict`` / ``list`` (already-decoded JSON), or a non-string scalar
           (e.g. integer mistakenly serialised as the args field), or an
           empty string (treated as ``"{}"``).  We coerce all of these into
           their canonical string form before any further checks so the
           rest of the pipeline only sees well-typed args.
        2. **JSON parse.**  We attempt ``json.loads`` on each arg string and
           collect every (tool_name, error_message) pair that fails.  An
           empty ``invalid_json_args`` list is the dispatch-OK fast path.
        3. **Truncation heuristic.**  If *any* parse failure looks truncated
           (``_detect_truncated_tool_call``) we treat the whole response
           as a length-class failure: refuse to dispatch and either retry
           silently (if the caller has budget left) or finalise as
           TOOL_CALL_TRUNCATED.  This branch is critical because some
           routers rewrite ``finish_reason`` from ``"length"`` to
           ``"tool_calls"``, hiding the real cause from the length
           handler — the heuristic catches that case.
        4. **Pure-syntax JSON error.**  When the args don't look truncated
           but still don't parse, we fall back to silently re-issuing the
           call up to ``max_invalid_json_retries`` times.  After that we
           give up the silent path and ask the model to fix it: we append
           the assistant tool_calls message followed by one ``role=tool``
           error stub per failing call.  Hermes shows that this is the
           single most reliable recovery for cheap JSON formatting bugs;
           it doesn't poison the conversation because the assistant
           message is the *real* one the model just emitted (just badly
           formatted).  Compatible callers that want to opt out of this
           injection set ``invalid_json_recovery_enabled=False``.

        The method **mutates** ``response.tool_calls[*].arguments`` in
        place to install the normalised representation so downstream
        dispatch sees consistent types.  The mutation is safe: by the
        time we reach this point the response object is owned by this
        turn and not shared.

        ``context.metadata`` is updated with the running counters so the
        recovery decisions trail and ``EngineResult.metadata.runtime``
        view stay accurate.
        """
        invalid_json_args: list[tuple[str, str]] = []

        for call in response.tool_calls:
            raw = getattr(call, "arguments", None)

            # 1) Pre-parsed dict / list — leave alone.  These are
            # produced by ``OpenAICompatibleModel._parse_tool_call`` after
            # a successful upstream JSON parse, or by ScriptedProvider in
            # tests.  They are never truncated by definition.
            if isinstance(raw, (dict, list)):
                continue

            # 2) Non-string scalars: coerce to ``str()``.  A model can
            # technically emit ``"arguments": 42`` if the provider does
            # not enforce the schema; falling back to ``str()`` keeps the
            # downstream JSON parse honest about what we received.
            if raw is not None and not isinstance(raw, str):
                call.arguments = str(raw)
                raw = call.arguments

            # 3) Empty / whitespace-only string: treat as empty object.
            # OpenAI sometimes emits this for tools with no required
            # parameters; rejecting it would create needless noise.
            args_text = "" if raw is None else str(raw)
            if not args_text or not args_text.strip():
                call.arguments = {}
                continue

            try:
                parsed = json.loads(args_text)
            except json.JSONDecodeError as exc:
                invalid_json_args.append((call.name, str(exc)))
                # Keep the raw string on ``call.arguments`` so the
                # truncation heuristic and the error-injection path can
                # both see it.  Dispatch will not see this call —
                # ``invalid_json_args`` gates that.
                continue

            # Successful parse: store the decoded form so dispatch sees
            # the canonical dict/list payload regardless of how the wire
            # protocol delivered it.
            call.arguments = parsed if isinstance(parsed, (dict, list)) else {}

        # 4) Fast path: no JSON errors, dispatch.
        if not invalid_json_args:
            context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = 0
            return AgentEngine._ToolCallValidationOutcome(action="ok")

        # 5) Truncation guard.  If the failing args don't terminate with
        # ``}`` / ``]`` we treat this as a length-class failure regardless
        # of what ``finish_reason`` said — see the docstring for why.
        invalid_names = {name for name, _ in invalid_json_args}
        truncated = any(
            (not (str(getattr(c, "arguments", "")) or "").rstrip().endswith(("}", "]")))
            and c.name in invalid_names
            for c in response.tool_calls
        )
        if truncated and getattr(self.config, "truncated_tool_call_detection_enabled", True):
            attempts = int(context.metadata.get(TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES, 0))
            max_attempts = max(0, int(getattr(self.config, "max_truncated_tool_call_retries", 1)))
            if attempts < max_attempts:
                context.metadata[TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES] = attempts + 1
                # Reset the JSON retry counter — we burned this attempt on
                # the truncation path, not on a JSON-format mistake.
                context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = 0
                return AgentEngine._ToolCallValidationOutcome(
                    action="retry", invalid_json_args=invalid_json_args
                )
            return AgentEngine._ToolCallValidationOutcome(
                action="truncated", invalid_json_args=invalid_json_args
            )

        # 6) Pure-syntax JSON error path: silent retry, then injection.
        attempts = int(context.metadata.get(TURN_KEY_INVALID_JSON_RETRIES, 0)) + 1
        context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = attempts
        max_attempts = max(0, int(getattr(self.config, "max_invalid_json_retries", 3)))
        if attempts < max_attempts:
            return AgentEngine._ToolCallValidationOutcome(
                action="retry", invalid_json_args=invalid_json_args
            )

        # Exhausted silent retries — fall through to injection.  The
        # caller is responsible for appending these messages to history
        # and continuing the loop; we surface them here rather than
        # mutating ``messages`` so the validation routine has zero
        # side-effects on the turn message list.
        if not getattr(self.config, "invalid_json_recovery_enabled", True):
            # Operator opted out of injection — fail the turn instead.
            return AgentEngine._ToolCallValidationOutcome(
                action="truncated", invalid_json_args=invalid_json_args
            )

        injection: list[Dict[str, Any]] = []
        # Build the assistant message first so role-alternation is
        # preserved (assistant tool_calls → tool result(s)).  The args
        # we send back are the **raw** strings — that is what the model
        # actually produced and rebroadcasting them lets the model see
        # exactly what was wrong.
        injection.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": c.arguments
                            if isinstance(c.arguments, str)
                            else json.dumps(c.arguments, ensure_ascii=False),
                        },
                    }
                    for c in response.tool_calls
                ],
                "finish_reason": response.finish_reason,
                "metadata": {"_invalid_json_recovery": True},
            }
        )
        invalid_lookup = {name: err for name, err in invalid_json_args}
        for c in response.tool_calls:
            if c.name in invalid_lookup:
                err_msg = invalid_lookup[c.name]
                injection.append(
                    {
                        "role": "tool",
                        "name": c.name,
                        "tool_call_id": c.id,
                        "content": (
                            f"Error: Invalid JSON arguments. {err_msg}. "
                            "For tools with no required parameters, use an empty object: {}. "
                            "Please retry with valid JSON."
                        ),
                        "is_error": True,
                        "metadata": {"_invalid_json_recovery": True},
                    }
                )
            else:
                injection.append(
                    {
                        "role": "tool",
                        "name": c.name,
                        "tool_call_id": c.id,
                        "content": "Skipped: another tool call in this response had invalid JSON.",
                        "is_error": True,
                        "metadata": {"_invalid_json_recovery": True},
                    }
                )

        # Reset the silent-retry budget so a *future* JSON error in the
        # same turn gets its own clean count.  Without this, a model that
        # emits one bad JSON, gets injection, then emits another bad JSON
        # would skip the silent-retry path entirely and immediately
        # inject again.
        context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = 0

        return AgentEngine._ToolCallValidationOutcome(
            action="inject_error",
            invalid_json_args=invalid_json_args,
            injection_messages=injection,
        )

    def _handle_length_with_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> "AgentEngine._LengthHandlingOutcome":
        """``finish_reason="length"`` AND ``tool_calls`` non-empty.

        Sprint 1 / PR 1.3 — mirrors hermes-agent ``run_agent.py``
        11793-11819.  When a model exhausts its output budget *while*
        producing tool_calls, the canonical recovery is **not** to
        continue (the next chunk would be more tool args, not prose) but
        rather to give the model a single chance to produce complete
        arguments by re-issuing the same request.  We deliberately do
        NOT append the broken assistant message to history during this
        retry: the goal is to remove the bad attempt from the model's
        view of the conversation rather than ask it to recover from it.

        After ``max_truncated_tool_call_retries`` (default 1) we surrender
        and finalise the turn with ExitReason.TOOL_CALL_TRUNCATED.
        """
        attempts = int(context.metadata.get(TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "max_truncated_tool_call_retries", 1)))

        if attempts < max_attempts:
            context.metadata[TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES] = attempts + 1
            return AgentEngine._LengthHandlingOutcome(
                action="continue",
                # Re-issue the same request from the current message
                # state — the broken assistant tool_calls do NOT enter
                # history.
                messages=messages,
            )

        # Out of retries — finalise as TOOL_CALL_TRUNCATED.  Any visible
        # text the model emitted before the truncation is preserved on
        # ``truncated_response_prefix_parts`` so downstream finalisation
        # (or upstream observability) can still surface it.
        visible_text = self._extract_visible_text(response.content or "")
        if visible_text:
            prefix_parts = context.metadata.setdefault("truncated_response_prefix_parts", [])
            if isinstance(prefix_parts, list):
                prefix_parts.append(visible_text)
        context.metadata["partial"] = True
        context.metadata["length_exit_reason"] = "tool_call_truncated"

        rollback = self._get_messages_up_to_last_assistant(messages)
        return AgentEngine._LengthHandlingOutcome(
            action="finalize",
            messages=rollback,
            final_response=visible_text or None,
            exit_reason=ExitReason.TOOL_CALL_TRUNCATED,
        )

    @staticmethod
    def _get_messages_up_to_last_assistant(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a copy of ``messages`` with trailing continuation scaffolding removed.

        We only want to roll back the artificial continuation user prompts /
        partial assistant fragments introduced by PR 1.2.  The last fully-
        committed assistant turn from the *original* conversation should stay.
        """
        trimmed = list(messages)
        while trimmed:
            tail = trimmed[-1]
            if not isinstance(tail, dict):
                trimmed.pop()
                continue
            metadata = tail.get("metadata") if isinstance(tail.get("metadata"), dict) else {}
            if metadata.get("_length_continue_prompt") or metadata.get("partial"):
                trimmed.pop()
                continue
            break
        return trimmed

    @staticmethod
    def _pop_context_response(context: TurnContext, key: str) -> NormalizedResponse | None:
        candidate = context.metadata.pop(key, None)
        if isinstance(candidate, NormalizedResponse):
            return candidate
        return None

    def _append_assistant_tool_message(
        self,
        messages: List[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in response.tool_calls
        ]
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": tool_calls,
                "finish_reason": response.finish_reason,
            }
        )

    def _append_assistant_text_message(
        self,
        messages: List[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "finish_reason": response.finish_reason,
            }
        )

    def _append_tool_result_message(self, messages: List[Dict[str, Any]], result: ToolResult) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": result.content,
                "is_error": result.is_error,
                "metadata": dict(result.metadata),
            }
        )

    def _accumulate_usage(self, response: NormalizedResponse, context: TurnContext) -> None:
        """Add this LLM call's usage to the per-turn accumulator.

        Sprint 3 / PR 3.1.  Tolerant by design: any failure to extract /
        normalise usage degrades to a no-op rather than crashing the turn —
        accumulation is observability, not correctness-critical.

        Reads ``response.metadata["usage"]`` (the raw provider dict) and
        normalises via ``aether.runtime.usage.normalize_usage`` using the
        provider's ``provider_name`` / ``api_mode`` class attributes.
        Writes the running ``CanonicalUsage`` to
        ``context.metadata["usage_accumulator"]`` and bumps
        ``context.metadata["api_calls"]``.
        """
        try:
            raw = (response.metadata or {}).get("usage") if response else None
            provider_name = getattr(self.services.provider, "provider_name", "openai")
            api_mode = getattr(self.services.provider, "api_mode", "chat")
            this_call = normalize_usage(raw, provider=provider_name, api_mode=api_mode)
            acc = context.metadata.get("usage_accumulator")
            if not isinstance(acc, CanonicalUsage):
                acc = CanonicalUsage()
            context.metadata["usage_accumulator"] = acc.add(this_call)
            context.metadata["api_calls"] = int(context.metadata.get("api_calls", 0)) + 1
        except Exception:  # noqa: BLE001 — observability path, never crash a turn
            self.services.logger.debug(
                "usage accumulation failed; leaving accumulator unchanged",
                exc_info=True,
            )

    def _build_result(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        iterations: int,
        final_response: str | None,
        error_text: str | None,
        exit_reason: ExitReason,
        *,
        context: TurnContext,
        active_system_prompt: str | None,
    ) -> EngineResult:
        if exit_reason == ExitReason.INTERRUPTED:
            status = EngineStatus.INTERRUPTED
        elif exit_reason == ExitReason.MAX_ITERATIONS:
            status = EngineStatus.MAX_ITERATIONS
        elif exit_reason in {
            ExitReason.PROVIDER_ERROR,
            ExitReason.TOOL_ERROR,
            ExitReason.MIDDLEWARE_ERROR,
            ExitReason.UNKNOWN_TOOL,
            # Sprint 1 / PR 1.1: response-shape validation failure after
            # all retries.  Same terminal class as PROVIDER_ERROR but with
            # a distinct ExitReason so observers can branch.
            ExitReason.RESPONSE_INVALID,
            # Sprint 1 / PR 1.2: the model repeatedly hit its output budget
            # and we had to stop with a partial / rolled-back answer.
            ExitReason.LENGTH_EXHAUSTED,
            # Sprint 1 / PR 1.3: tool_call arguments were cut off
            # mid-stream and we refused to dispatch them.  Treat as
            # FAILED so callers can branch (the alternative — COMPLETED —
            # would silently look like success, hiding a real recovery
            # failure).
            ExitReason.TOOL_CALL_TRUNCATED,
            # Sprint 1.5 / phantom-tool recovery: model wrote tool
            # invocations as prose for the entire retry budget.  The
            # turn produced text but no actual work was done — the
            # user explicitly asked for action and got narration.
            # Classifying as FAILED ensures the green checkmark in
            # the CLI footer flips to a warning state by default,
            # so users notice without needing ``-v``.
            ExitReason.PHANTOM_TOOL_INTENT,
        }:
            status = EngineStatus.FAILED
        else:
            status = EngineStatus.COMPLETED

        # The ``runtime`` block here is a flat snapshot of the per-turn retry
        # counters that Sprint-1+ paths use to drive recovery.  We read them
        # straight from ``context.metadata`` (their authoritative home, since
        # Sprint 0) instead of from ``self.*`` — the latter would re-introduce
        # the multi-session leak this PR was supposed to fix.
        #
        # Sprint 3 / PR 3.1 (P1-4 + P1-11): the top-level keys ``usage`` /
        # ``api_calls`` / ``iteration_budget`` / ``exit`` / ``reasoning`` /
        # ``compaction`` form the **stable v1 metadata schema** documented in
        # ``EngineResult`` — additive only after this PR.  See that docstring
        # for consumer guarantees.
        usage_acc = context.metadata.get("usage_accumulator")
        if not isinstance(usage_acc, CanonicalUsage):
            usage_acc = CanonicalUsage()

        last_msg = messages[-1] if messages else None
        last_msg_role = (
            last_msg.get("role", "unknown") if isinstance(last_msg, dict) else "unknown"
        )
        stuck_after_tool = (
            isinstance(last_msg, dict)
            and last_msg.get("role") == "tool"
            and exit_reason in {ExitReason.MAX_ITERATIONS, ExitReason.EMPTY_RESPONSE}
        )

        # PR 3.1: ``turn`` is a flat snapshot of context.metadata for
        # downstream observability.  We exclude internal accumulator objects
        # (CanonicalUsage instances etc.) that are not JSON-serializable —
        # their normalised dict form is exposed at the top level under ``usage``.
        turn_snapshot = {
            k: v
            for k, v in context.metadata.items()
            if k not in _METADATA_INTERNAL_KEYS
        }

        metadata = {
            "request": {
                "session_id": request.session_id,
                "model_config": asdict(request.model_config),
                "system_message_present": bool(request.system_message),
                "stream_callback_present": bool(request.stream_callback),
            },
            "turn": turn_snapshot,
            "runtime": {
                "empty_response_retries": int(
                    context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)
                ),
                "provider_error_retries": int(
                    context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)
                ),
                # Sprint 1 / PR 1.3 — exposed so observability dashboards
                # and tests can confirm the truncated-tool-call detector
                # actually fired.  Both reset to 0 at turn entry.
                "truncated_tool_call_retries": int(
                    context.metadata.get(TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES, 0)
                ),
                "invalid_json_retries": int(
                    context.metadata.get(TURN_KEY_INVALID_JSON_RETRIES, 0)
                ),
                # Sprint 1.5 / P0-9 — count of synthesized phantom
                # ``ToolCall``s dispatched this turn (prose intents
                # that mapped cleanly to registered tools).  ``0``
                # for a normal turn; non-zero whenever the engine
                # had to repair a Kimi-class "tool call as prose"
                # response into structured calls inline.
                "phantom_tool_synthesized": int(
                    context.metadata.get(TURN_KEY_PHANTOM_TOOL_SYNTHESIZED, 0)
                ),
                # Sprint 1.5 / phantom-tool recovery — counts how
                # many corrective ``role=user`` nudges this turn sent
                # before the model produced a structured ``tool_calls``
                # (or PHANTOM_TOOL_INTENT exited).  ``0`` means the
                # turn never triggered the detector.
                "phantom_tool_retries": int(
                    context.metadata.get(TURN_KEY_PHANTOM_TOOL_RETRIES, 0)
                ),
            },
            # ----------------- Sprint 3 / PR 3.1: stable v1 ----------------- #
            # Aggregated token usage across every LLM call this turn.
            # Always present (zero-valued on turns with no LLM call).
            "usage": usage_acc.to_dict(),
            "api_calls": int(context.metadata.get("api_calls", 0)),
            # Iteration budget snapshot.  Filled with structured data by
            # PR 3.2; for now we surface the legacy max_iterations bound
            # so consumers can already key off the "exhausted" semantic.
            "iteration_budget": {
                "used": iterations,
                "max": int(getattr(self.config, "max_iterations", 0) or 0),
                "remaining": max(
                    0,
                    int(getattr(self.config, "max_iterations", 0) or 0) - iterations,
                ),
                "grace_consumed": False,
            },
            # Why the loop terminated, in a structured form distinct from
            # the top-level ``exit_reason`` enum (which is the canonical
            # source — ``exit`` is the auxiliary diagnostic block).
            "exit": {
                "reason": exit_reason.value,
                "last_msg_role": last_msg_role,
                "stuck_after_tool": bool(stuck_after_tool),
            },
            # Reasoning block (Sprint 5 will populate ``last_reasoning``
            # from provider-emitted thinking blocks; Sprint 3 just reserves
            # the shape so consumers can rely on its presence).
            "reasoning": {
                "last_reasoning": context.metadata.get("last_reasoning_text"),
            },
            # Compaction counters (PR 3.4..3.7 will increment these as the
            # five-tier pipeline fires; PR 3.1 reserves the shape so
            # downstream consumers can already render zero-valued footers).
            "compaction": {
                "tier1_spilled_count": int(
                    context.metadata.get("tier1_spilled_count", 0)
                ),
                "tier2_snipped_count": int(
                    context.metadata.get("tier2_snipped_count", 0)
                ),
                "tier3_cleared_count": int(
                    context.metadata.get("tier3_cleared_count", 0)
                ),
                "tier4_collapse_segments": int(
                    context.metadata.get("tier4_collapse_segments", 0)
                ),
                "tier5_summaries_generated": int(
                    context.metadata.get("tier5_summaries_generated", 0)
                ),
            },
        }

        return EngineResult(
            session_id=request.session_id,
            status=status,
            exit_reason=exit_reason,
            messages=messages,
            iterations=iterations,
            final_response=final_response,
            error=error_text,
            task_id=context.task_id,
            turn_id=context.turn_id,
            system_prompt=active_system_prompt,
            streamed=bool(context.metadata.get("streamed_output")),
            metadata=metadata,
        )
