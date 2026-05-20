"""Turn runner orchestration for AgentEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aether.agents.runtime.context_assembly import (
    ContextAssemblyInput,
    ContextAssemblyPipeline,
)
from aether.agents.runtime.session_lifecycle import (
    SessionLifecycleController,
    TurnFinalizationInput,
)
from aether.agents.runtime.tool_dispatch import (
    ToolDispatchController,
    ToolDispatchRequest,
)
from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineResult,
    ExitReason,
    LoopState,
    NormalizedResponse,
    TurnContext,
)
from aether.runtime.core.iteration_budget import IterationBudget
from aether.runtime.core.services import EngineServices


@dataclass(slots=True)
class TurnRunInput:
    request: EngineRequest


@dataclass(slots=True)
class TurnRunResult:
    result: EngineResult


class TurnRunnerAdapter(Protocol):
    """Narrow bridge to AgentEngine helpers not yet extracted."""

    def apply_turn_nudges(self, context: TurnContext) -> None: ...

    def build_stream_callback(self, request: EngineRequest, context: TurnContext) -> Any: ...

    def build_stream_silent_callback(self, request: EngineRequest, context: TurnContext) -> Any: ...

    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool: ...

    def handle_pipeline_error(self, error: Exception, state: LoopState, context: TurnContext) -> None: ...

    def pop_context_response(self, context: TurnContext, key: str) -> NormalizedResponse | None: ...

    def invoke_provider_with_recovery(
        self,
        *,
        request: EngineRequest,
        canonical_messages: list[dict[str, Any]],
        prepared_messages: list[dict[str, Any]],
        stream_callback: Any,
        stream_silent_callback: Any,
        context: TurnContext,
    ) -> Any: ...

    def resolve_terminal_exit_reason(self, hint: Any, error: Exception) -> ExitReason: ...

    def accumulate_usage(self, response: NormalizedResponse, context: TurnContext) -> None: ...

    def safe_call_hook(self, name: str, **kwargs: Any) -> Any: ...

    def handle_length_with_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> Any: ...

    def handle_length_finish_reason(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        request: EngineRequest,
        context: TurnContext,
    ) -> Any: ...

    def validate_tool_call_arguments(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> Any: ...

    def get_messages_up_to_last_assistant(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def extract_visible_text(self, content: str) -> str: ...

    def apply_pending_steer_to_tool_results(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        start_idx: int,
        context: TurnContext,
    ) -> None: ...

    def append_assistant_tool_message(
        self,
        messages: list[dict[str, Any]],
        response: NormalizedResponse,
    ) -> None: ...

    def maybe_recover_phantom_tool_intent(
        self,
        *,
        response_to_store: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> str: ...

    def dispatch_synthesized_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
        state_machine: Any,
        request: EngineRequest,
    ) -> tuple[str, ExitReason | None, str | None]: ...

    def finalize_empty_response(
        self,
        *,
        response: NormalizedResponse,
        response_to_store: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
        request: EngineRequest,
        phantom_outcome: str,
        prefix: str,
    ) -> Any: ...

    def is_continue_loop_finalization(self, finalized: Any) -> bool: ...


class LegacyTurnRunnerAdapter:
    """Bridge TurnRunner to AgentEngine during staged extraction."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def apply_turn_nudges(self, context: TurnContext) -> None:
        self._engine._apply_turn_nudges(context)  # noqa: SLF001

    def build_stream_callback(self, request: EngineRequest, context: TurnContext) -> Any:
        return self._engine._build_stream_callback(request, context)  # noqa: SLF001

    def build_stream_silent_callback(self, request: EngineRequest, context: TurnContext) -> Any:
        return self._engine._build_stream_silent_callback(request, context)  # noqa: SLF001

    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool:
        return self._engine._is_interrupted(session_id, context)  # noqa: SLF001

    def handle_pipeline_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        self._engine._handle_pipeline_error(error, state, context)  # noqa: SLF001

    def pop_context_response(self, context: TurnContext, key: str) -> NormalizedResponse | None:
        return self._engine._pop_context_response(context, key)  # noqa: SLF001

    def invoke_provider_with_recovery(self, **kwargs: Any) -> Any:
        return self._engine._invoke_provider_with_recovery(**kwargs)  # noqa: SLF001

    def resolve_terminal_exit_reason(self, hint: Any, error: Exception) -> ExitReason:
        return self._engine._resolve_terminal_exit_reason(hint, error)  # noqa: SLF001

    def accumulate_usage(self, response: NormalizedResponse, context: TurnContext) -> None:
        self._engine._accumulate_usage(response, context)  # noqa: SLF001

    def safe_call_hook(self, name: str, **kwargs: Any) -> Any:
        return self._engine._safe_call_hook(name, **kwargs)  # noqa: SLF001

    def handle_length_with_tool_calls(self, **kwargs: Any) -> Any:
        return self._engine._handle_length_with_tool_calls(**kwargs)  # noqa: SLF001

    def handle_length_finish_reason(self, **kwargs: Any) -> Any:
        return self._engine._handle_length_finish_reason(**kwargs)  # noqa: SLF001

    def validate_tool_call_arguments(self, **kwargs: Any) -> Any:
        return self._engine._validate_tool_call_arguments(**kwargs)  # noqa: SLF001

    def get_messages_up_to_last_assistant(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._engine._get_messages_up_to_last_assistant(messages)  # noqa: SLF001

    def extract_visible_text(self, content: str) -> str:
        return self._engine._extract_visible_text(content)  # noqa: SLF001

    def apply_pending_steer_to_tool_results(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        self._engine._apply_pending_steer_to_tool_results(messages, **kwargs)  # noqa: SLF001

    def append_assistant_tool_message(
        self,
        messages: list[dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        self._engine._append_assistant_tool_message(messages, response)  # noqa: SLF001

    def maybe_recover_phantom_tool_intent(self, **kwargs: Any) -> str:
        return self._engine._maybe_recover_phantom_tool_intent(**kwargs)  # noqa: SLF001

    def dispatch_synthesized_tool_calls(self, **kwargs: Any) -> tuple[str, ExitReason | None, str | None]:
        return self._engine._dispatch_synthesized_tool_calls(**kwargs)  # noqa: SLF001

    def finalize_empty_response(self, **kwargs: Any) -> Any:
        return self._engine._finalize_empty_response(**kwargs)  # noqa: SLF001

    def is_continue_loop_finalization(self, finalized: Any) -> bool:
        return self._engine._is_continue_loop_finalization(finalized)  # noqa: SLF001


class TurnRunner:
    """Execute one AgentEngine turn using injected runtime controllers."""

    def __init__(
        self,
        *,
        services: EngineServices,
        config: EngineConfig,
        session_lifecycle_controller: SessionLifecycleController,
        context_assembly_pipeline: ContextAssemblyPipeline,
        tool_dispatch_controller: ToolDispatchController,
        adapter: TurnRunnerAdapter,
    ) -> None:
        self._services = services
        self._config = config
        self._session_lifecycle_controller = session_lifecycle_controller
        self._context_assembly_pipeline = context_assembly_pipeline
        self._tool_dispatch_controller = tool_dispatch_controller
        self._adapter = adapter

    def run(self, turn_input: TurnRunInput) -> TurnRunResult:
        request = turn_input.request
        context: TurnContext | None = None
        active_system_prompt: str | None = None
        final_response: str | None = None
        error_text: str | None = None
        exit_reason = ExitReason.EMPTY_RESPONSE
        iterations = 0
        budget: IterationBudget | None = None

        try:
            turn = self._session_lifecycle_controller.prepare_turn(request)
            state_machine = turn.state_machine
            messages = turn.messages
            context = turn.context
            active_system_prompt = turn.active_system_prompt
            self._adapter.apply_turn_nudges(context)
            stream_callback_wrapped = self._adapter.build_stream_callback(request, context)
            stream_silent_callback_wrapped = self._adapter.build_stream_silent_callback(
                request,
                context,
            )

            state_machine.transition(LoopState.PREPARE)
            if self._adapter.is_interrupted(request.session_id, context):
                state_machine.transition(LoopState.INTERRUPTED)
                exit_reason = ExitReason.INTERRUPTED
            else:
                state_machine.transition(LoopState.PRE_LLM)

                budget = IterationBudget(max_total=self._config.max_iterations)
                context.metadata["_iteration_budget_obj"] = budget
                context.metadata["iteration_budget"] = budget.to_dict()

                while budget.consume():
                    context.iteration = iterations + 1

                    if self._adapter.is_interrupted(request.session_id, context):
                        state_machine.transition(LoopState.INTERRUPTED)
                        exit_reason = ExitReason.INTERRUPTED
                        break

                    try:
                        assembly_result = self._context_assembly_pipeline.assemble(
                            ContextAssemblyInput(
                                request=request,
                                messages=messages,
                                context=context,
                                iteration=context.iteration,
                            )
                        )
                    except Exception as exc:
                        self._adapter.handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break
                    messages = assembly_result.canonical_messages
                    prepared_messages = assembly_result.prepared_messages
                    hook_outcome = assembly_result.hook_outcome

                    state_machine.transition(LoopState.LLM_CALL)
                    response = hook_outcome.short_circuit_response
                    if response is None:
                        response = self._adapter.pop_context_response(context, "llm_pre_response")
                    if response is None:
                        invoke_outcome = self._adapter.invoke_provider_with_recovery(
                            request=request,
                            canonical_messages=messages,
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
                            exc = invoke_outcome.error
                            assert exc is not None
                            self._adapter.handle_pipeline_error(exc, state_machine.state, context)
                            recovered_response = self._adapter.pop_context_response(context, "llm_error_response")
                            if recovered_response is None:
                                error_text = str(exc)
                                terminal_hint = context.metadata.pop(
                                    "recovery_terminal_exit_reason",
                                    None,
                                )
                                exit_reason = self._adapter.resolve_terminal_exit_reason(
                                    terminal_hint,
                                    exc,
                                )
                                state_machine.transition(LoopState.FAILED)
                                break
                            response = recovered_response

                    state_machine.transition(LoopState.POST_LLM)
                    try:
                        response = self._services.middleware_pipeline.run_after_llm(response, context)
                    except Exception as exc:
                        self._adapter.handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break

                    self._adapter.accumulate_usage(response, context)
                    iterations += 1

                    self._adapter.safe_call_hook(
                        "post_llm_call",
                        session_id=request.session_id,
                        iteration=context.iteration,
                        response_text=response.content or "",
                        context_metadata=context.metadata,
                    )

                    length_text = (response.content or "").strip()
                    if response.finish_reason == "length" and (
                        response.tool_calls or length_text
                    ):
                        if (
                            response.tool_calls
                            and getattr(self._config, "truncated_tool_call_detection_enabled", True)
                        ):
                            handled = self._adapter.handle_length_with_tool_calls(
                                response=response,
                                messages=messages,
                                context=context,
                            )
                        else:
                            handled = self._adapter.handle_length_finish_reason(
                                response=response,
                                messages=messages,
                                request=request,
                                context=context,
                            )
                        if handled.action == "continue":
                            messages = handled.messages
                            state_machine.transition(LoopState.CHECK_EXIT)
                            if budget.exhausted:
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
                        if getattr(self._config, "truncated_tool_call_detection_enabled", True):
                            validation = self._adapter.validate_tool_call_arguments(
                                response=response,
                                messages=messages,
                                context=context,
                            )
                            if validation.action == "retry":
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if budget.exhausted:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue
                            if validation.action == "truncated":
                                rollback = self._adapter.get_messages_up_to_last_assistant(messages)
                                visible_text = self._adapter.extract_visible_text(response.content or "")
                                if visible_text:
                                    prefix_parts = context.metadata.setdefault(
                                        "truncated_response_prefix_parts",
                                        [],
                                    )
                                    if isinstance(prefix_parts, list):
                                        prefix_parts.append(visible_text)
                                context.metadata["partial"] = True
                                context.metadata.setdefault(
                                    "length_exit_reason",
                                    "tool_call_truncated",
                                )
                                messages = rollback
                                final_response = visible_text or None
                                exit_reason = ExitReason.TOOL_CALL_TRUNCATED
                                state_machine.transition(LoopState.FINALIZE)
                                break
                            if validation.action == "inject_error":
                                tool_result_start_idx = len(messages)
                                messages.extend(validation.injection_messages)
                                self._adapter.apply_pending_steer_to_tool_results(
                                    messages,
                                    session_id=request.session_id,
                                    start_idx=tool_result_start_idx,
                                    context=context,
                                )
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if budget.exhausted:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue

                        state_machine.transition(LoopState.TOOL_DISPATCH)
                        self._adapter.append_assistant_tool_message(messages, response)
                        tool_result_start_idx = len(messages)
                        state_machine.transition(LoopState.TOOL_EXECUTE)
                        tool_dispatch = self._tool_dispatch_controller.dispatch(
                            ToolDispatchRequest(
                                tool_calls=list(response.tool_calls),
                                messages=messages,
                                context=context,
                                request=request,
                                iteration=iterations + 1,
                                tool_result_start_idx=tool_result_start_idx,
                            )
                        )
                        messages = tool_dispatch.messages

                        if tool_dispatch.exit_reason == ExitReason.INTERRUPTED:
                            exit_reason = ExitReason.INTERRUPTED
                            state_machine.transition(LoopState.INTERRUPTED)
                            break

                        if tool_dispatch.exit_reason in {
                            ExitReason.MIDDLEWARE_ERROR,
                            ExitReason.TOOL_ERROR,
                            ExitReason.UNKNOWN_TOOL,
                        }:
                            error_text = tool_dispatch.error_text
                            exit_reason = tool_dispatch.exit_reason
                            state_machine.transition(LoopState.FAILED)
                            break

                        if tool_dispatch.exit_reason is not None:
                            exit_reason = tool_dispatch.exit_reason
                            state_machine.transition(LoopState.CHECK_EXIT)
                            state_machine.transition(LoopState.FINALIZE)
                            break

                        if (
                            tool_dispatch.all_tools_cheap
                            and not tool_dispatch.schema_injected
                        ):
                            budget.refund()
                            context.metadata["iteration_budget"] = budget.to_dict()

                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break

                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    prefix_parts = context.metadata.pop("truncated_response_prefix_parts", None)
                    prefix = (
                        " ".join(
                            part.strip()
                            for part in prefix_parts
                            if isinstance(part, str) and part.strip()
                        )
                        if isinstance(prefix_parts, list)
                        else ""
                    )
                    suffix = (response.content or "").strip()
                    combined_content = (
                        (prefix + " " + suffix).strip()
                        if prefix and suffix
                        else (prefix or suffix)
                    )
                    response_to_store = response
                    if combined_content != (response.content or ""):
                        response_to_store = NormalizedResponse(
                            content=combined_content,
                            tool_calls=list(response.tool_calls),
                            finish_reason=response.finish_reason,
                            metadata=dict(response.metadata),
                        )

                    phantom_outcome = self._adapter.maybe_recover_phantom_tool_intent(
                        response_to_store=response_to_store,
                        messages=messages,
                        context=context,
                    )
                    if phantom_outcome == "synthesized":
                        synth_outcome, synth_exit, synth_error = self._adapter.dispatch_synthesized_tool_calls(
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
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    if phantom_outcome == "retry":
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    finalized = self._adapter.finalize_empty_response(
                        response=response,
                        response_to_store=response_to_store,
                        messages=messages,
                        context=context,
                        request=request,
                        phantom_outcome=phantom_outcome,
                        prefix=prefix,
                    )
                    if self._adapter.is_continue_loop_finalization(finalized):
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue
                    final_response = (
                        finalized.final_response
                        if isinstance(finalized.final_response, str)
                        else None
                    )
                    exit_reason = finalized.exit_reason or ExitReason.EMPTY_RESPONSE
                    error_text = finalized.error_text
                    state_machine.transition(LoopState.FINALIZE)
                    break

            if state_machine.state not in {LoopState.FAILED, LoopState.INTERRUPTED, LoopState.FINALIZE}:
                state_machine.transition(LoopState.FINALIZE)
                if budget is not None and budget.exhausted:
                    exit_reason = ExitReason.MAX_ITERATIONS

            assert exit_reason is not None
            result = self._session_lifecycle_controller.finalize_turn(
                TurnFinalizationInput(
                    request=request,
                    messages=messages,
                    context=context,
                    final_response=final_response,
                    error_text=error_text,
                    exit_reason=exit_reason,
                    iterations=iterations,
                    budget=budget,
                    state_machine=state_machine,
                    active_system_prompt=active_system_prompt,
                )
            )
            return TurnRunResult(result=result)
        finally:
            self._session_lifecycle_controller.cleanup_after_turn(context)
