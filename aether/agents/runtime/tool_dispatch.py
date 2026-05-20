"""Tool dispatch controller for AgentEngine tool-use iterations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from aether.agents.core.tool_hardening import prepare_tool_calls
from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    LoopState,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.core.services import EngineServices
from aether.tools.base import UnknownToolError


@dataclass(slots=True)
class ToolDispatchRequest:
    tool_calls: list[ToolCall]
    messages: list[dict[str, Any]]
    context: TurnContext
    request: EngineRequest
    iteration: int
    tool_result_start_idx: int
    validate_schema: bool = True


@dataclass(slots=True)
class ToolDispatchResult:
    tool_results: list[ToolResult] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    should_continue: bool = True
    exit_reason: ExitReason | None = None
    error_text: str | None = None
    all_tools_cheap: bool = False
    dispatched_count: int = 0
    schema_injected: bool = False


class ToolDispatchAdapter(Protocol):
    def maybe_inject_schema_errors(
        self,
        *,
        dispatch_plan: Any,
        tool_calls: list[ToolCall],
        context: TurnContext,
    ) -> list[dict[str, Any]] | None: ...

    def apply_pending_steer_to_tool_results(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        start_idx: int,
        context: TurnContext,
    ) -> None: ...

    def is_interrupted(self, session_id: str, context: TurnContext) -> bool: ...

    def record_interrupt_metadata(
        self,
        context: TurnContext,
        *,
        was_in_tool_call: bool,
    ) -> None: ...

    def apply_tool_permission_gate(
        self,
        call: ToolCall,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> ToolCall | ToolResult: ...

    def handle_pipeline_error(
        self,
        error: Exception,
        state: LoopState,
        context: TurnContext,
    ) -> None: ...

    def format_unknown_tool_content(self, tool_name: str, *, context: TurnContext) -> str: ...

    def fire_post_tool_hook(
        self,
        *,
        tool_call: ToolCall,
        result: ToolResult | None,
        dispatch_error: BaseException | None,
        elapsed_ms: float,
        session_id: str,
        iteration: int,
        context: TurnContext,
    ) -> None: ...

    def accumulate_edited_paths(self, result: ToolResult | None, context: TurnContext) -> None: ...

    def maybe_mark_verifier_invoked(self, tool_call: ToolCall, context: TurnContext) -> None: ...

    def dispatch_internal_diagnostic_update(
        self,
        *,
        tool_call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> None: ...

    def record_tool_result_error(self, context: TurnContext, result: ToolResult) -> None: ...

    def append_tool_result_message(
        self,
        messages: list[dict[str, Any]],
        result: ToolResult,
    ) -> None: ...

    def is_permission_abort_result(self, result: ToolResult) -> bool: ...

    def is_cheap_tool(self, tool_name: str) -> bool: ...


class LegacyToolDispatchAdapter:
    """Bridge ToolDispatchController to existing AgentEngine helpers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def maybe_inject_schema_errors(
        self,
        *,
        dispatch_plan: Any,
        tool_calls: list[ToolCall],
        context: TurnContext,
    ) -> list[dict[str, Any]] | None:
        response = type("_ToolDispatchResponse", (), {"tool_calls": tool_calls})()
        return self._engine._maybe_inject_schema_errors(  # noqa: SLF001
            dispatch_plan=dispatch_plan,
            response=response,
            context=context,
        )

    def apply_pending_steer_to_tool_results(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        start_idx: int,
        context: TurnContext,
    ) -> None:
        self._engine._apply_pending_steer_to_tool_results(  # noqa: SLF001
            messages,
            session_id=session_id,
            start_idx=start_idx,
            context=context,
        )

    def is_interrupted(self, session_id: str, context: TurnContext) -> bool:
        return self._engine._is_interrupted(session_id, context)  # noqa: SLF001

    def record_interrupt_metadata(
        self,
        context: TurnContext,
        *,
        was_in_tool_call: bool,
    ) -> None:
        self._engine._record_interrupt_metadata(  # noqa: SLF001
            context,
            was_in_tool_call=was_in_tool_call,
        )

    def apply_tool_permission_gate(
        self,
        call: ToolCall,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> ToolCall | ToolResult:
        return self._engine._apply_tool_permission_gate(  # noqa: SLF001
            call,
            request=request,
            context=context,
        )

    def handle_pipeline_error(
        self,
        error: Exception,
        state: LoopState,
        context: TurnContext,
    ) -> None:
        self._engine._handle_pipeline_error(error, state, context)  # noqa: SLF001

    def format_unknown_tool_content(self, tool_name: str, *, context: TurnContext) -> str:
        return self._engine._format_unknown_tool_content(tool_name, context=context)  # noqa: SLF001

    def fire_post_tool_hook(
        self,
        *,
        tool_call: ToolCall,
        result: ToolResult | None,
        dispatch_error: BaseException | None,
        elapsed_ms: float,
        session_id: str,
        iteration: int,
        context: TurnContext,
    ) -> None:
        self._engine._fire_post_tool_hook(  # noqa: SLF001
            tool_call=tool_call,
            result=result,
            dispatch_error=dispatch_error,
            elapsed_ms=elapsed_ms,
            session_id=session_id,
            iteration=iteration,
            context=context,
        )

    def accumulate_edited_paths(self, result: ToolResult | None, context: TurnContext) -> None:
        self._engine._accumulate_edited_paths(result, context)  # noqa: SLF001

    def maybe_mark_verifier_invoked(self, tool_call: ToolCall, context: TurnContext) -> None:
        self._engine._maybe_mark_verifier_invoked(tool_call, context)  # noqa: SLF001

    def dispatch_internal_diagnostic_update(
        self,
        *,
        tool_call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> None:
        self._engine._dispatch_internal_diagnostic_update(  # noqa: SLF001
            tool_call=tool_call,
            result=result,
            context=context,
        )

    def record_tool_result_error(self, context: TurnContext, result: ToolResult) -> None:
        self._engine._record_tool_result_error(context, result)  # noqa: SLF001

    def append_tool_result_message(
        self,
        messages: list[dict[str, Any]],
        result: ToolResult,
    ) -> None:
        self._engine._append_tool_result_message(messages, result)  # noqa: SLF001

    def is_permission_abort_result(self, result: ToolResult) -> bool:
        return self._engine._is_permission_abort_result(result)  # noqa: SLF001

    def is_cheap_tool(self, tool_name: str) -> bool:
        return self._engine._is_cheap_tool(tool_name)  # noqa: SLF001


class ToolDispatchController:
    """Dispatch prepared model tool calls through registry and middleware."""

    def __init__(
        self,
        *,
        services: EngineServices,
        hooks: EngineHooks,
        config: EngineConfig,
        adapter: ToolDispatchAdapter,
    ) -> None:
        self._services = services
        self._hooks = hooks
        self._config = config
        self._adapter = adapter

    def dispatch(self, request: ToolDispatchRequest) -> ToolDispatchResult:
        messages = request.messages
        context = request.context
        tool_results: list[ToolResult] = []
        dispatched_count = 0

        dispatch_plan = prepare_tool_calls(
            request.tool_calls,
            registry=self._services.tool_registry,
            config=self._config,
            context=context,
        )
        self._record_dispatch_plan_metadata(dispatch_plan, context)

        if request.validate_schema and dispatch_plan.exit_reason is None:
            schema_injection = self._adapter.maybe_inject_schema_errors(
                dispatch_plan=dispatch_plan,
                tool_calls=request.tool_calls,
                context=context,
            )
            if schema_injection is not None:
                messages.extend(schema_injection)
                self._adapter.apply_pending_steer_to_tool_results(
                    messages,
                    session_id=request.request.session_id,
                    start_idx=request.tool_result_start_idx,
                    context=context,
                )
                return ToolDispatchResult(
                    tool_results=[],
                    messages=messages,
                    should_continue=True,
                    all_tools_cheap=self._all_tools_cheap(request.tool_calls),
                    dispatched_count=0,
                    schema_injected=True,
                )

        for prepared in dispatch_plan.prepared:
            call = prepared.call
            if self._adapter.is_interrupted(request.request.session_id, context):
                self._adapter.record_interrupt_metadata(context, was_in_tool_call=True)
                return ToolDispatchResult(
                    tool_results=tool_results,
                    messages=messages,
                    should_continue=False,
                    exit_reason=ExitReason.INTERRUPTED,
                    all_tools_cheap=self._all_tools_cheap(request.tool_calls),
                    dispatched_count=dispatched_count,
                )

            if prepared.synthetic_result is not None:
                synthetic = self._handle_synthetic_result(
                    prepared.synthetic_result,
                    request=request,
                    tool_results=tool_results,
                )
                if synthetic.exit_reason is not None:
                    synthetic.dispatched_count = dispatched_count
                    return synthetic
                continue

            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="tool execution did not produce a result",
                is_error=True,
            )
            tool_call: ToolCall | None = None
            permission_checked = self._adapter.apply_tool_permission_gate(
                call,
                request=request.request,
                context=context,
            )

            if isinstance(permission_checked, ToolResult):
                result = permission_checked
            else:
                try:
                    pre_tool = self._services.middleware_pipeline.run_before_tool(
                        permission_checked,
                        context,
                    )
                except Exception as exc:
                    self._adapter.handle_pipeline_error(exc, LoopState.TOOL_EXECUTE, context)
                    return self._failed_result(
                        request=request,
                        tool_results=tool_results,
                        exit_reason=ExitReason.MIDDLEWARE_ERROR,
                        error_text=str(exc),
                        dispatched_count=dispatched_count,
                    )

                if isinstance(pre_tool, ToolResult):
                    result = pre_tool
                    tool_call = None
                else:
                    tool_call = pre_tool

            dispatch_error: BaseException | None = None
            dispatch_t0 = time.perf_counter()
            if not isinstance(permission_checked, ToolResult) and tool_call is not None:
                dispatch_outcome = self._dispatch_one_tool(
                    tool_call=tool_call,
                    context=context,
                    request=request,
                    dispatch_t0=dispatch_t0,
                    tool_results=tool_results,
                    dispatched_count=dispatched_count,
                )
                if isinstance(dispatch_outcome, ToolDispatchResult):
                    return dispatch_outcome
                result, dispatch_error, dispatched_count = dispatch_outcome

            try:
                result = self._services.middleware_pipeline.run_after_tool(result, context)
            except Exception as exc:
                self._adapter.handle_pipeline_error(exc, LoopState.TOOL_EXECUTE, context)
                return self._failed_result(
                    request=request,
                    tool_results=tool_results,
                    exit_reason=ExitReason.MIDDLEWARE_ERROR,
                    error_text=str(exc),
                    dispatched_count=dispatched_count,
                )

            if tool_call is not None:
                self._adapter.accumulate_edited_paths(result, context)
                self._adapter.maybe_mark_verifier_invoked(tool_call, context)
                self._adapter.fire_post_tool_hook(
                    tool_call=tool_call,
                    result=result,
                    dispatch_error=dispatch_error,
                    elapsed_ms=(time.perf_counter() - dispatch_t0) * 1000.0,
                    session_id=request.request.session_id,
                    iteration=request.iteration,
                    context=context,
                )
                self._adapter.dispatch_internal_diagnostic_update(
                    tool_call=tool_call,
                    result=result,
                    context=context,
                )

            self._adapter.record_tool_result_error(context, result)
            self._adapter.append_tool_result_message(messages, result)
            tool_results.append(result)
            if bool(result.metadata.get("interrupted")):
                self._adapter.record_interrupt_metadata(context, was_in_tool_call=True)
            if self._adapter.is_permission_abort_result(result):
                self._adapter.record_interrupt_metadata(context, was_in_tool_call=True)
                return ToolDispatchResult(
                    tool_results=tool_results,
                    messages=messages,
                    should_continue=False,
                    exit_reason=ExitReason.INTERRUPTED,
                    all_tools_cheap=self._all_tools_cheap(request.tool_calls),
                    dispatched_count=dispatched_count,
                )

        self._adapter.apply_pending_steer_to_tool_results(
            messages,
            session_id=request.request.session_id,
            start_idx=request.tool_result_start_idx,
            context=context,
        )

        if dispatch_plan.exit_reason is not None:
            try:
                exit_reason = ExitReason(dispatch_plan.exit_reason)
            except ValueError:
                exit_reason = ExitReason.UNKNOWN_TOOL
            context.metadata["partial"] = True
            return ToolDispatchResult(
                tool_results=tool_results,
                messages=messages,
                should_continue=False,
                exit_reason=exit_reason,
                all_tools_cheap=self._all_tools_cheap(request.tool_calls),
                dispatched_count=dispatched_count,
            )

        return ToolDispatchResult(
            tool_results=tool_results,
            messages=messages,
            should_continue=True,
            all_tools_cheap=self._all_tools_cheap(request.tool_calls),
            dispatched_count=dispatched_count,
        )

    @staticmethod
    def _record_dispatch_plan_metadata(dispatch_plan: Any, context: TurnContext) -> None:
        if getattr(dispatch_plan, "repaired_count", 0):
            context.metadata["tool_names_repaired"] = (
                int(context.metadata.get("tool_names_repaired", 0))
                + int(dispatch_plan.repaired_count)
            )
        if getattr(dispatch_plan, "deduped_count", 0):
            context.metadata["tool_calls_deduped"] = (
                int(context.metadata.get("tool_calls_deduped", 0))
                + int(dispatch_plan.deduped_count)
            )
        if getattr(dispatch_plan, "capped_count", 0):
            context.metadata["tool_calls_capped"] = (
                int(context.metadata.get("tool_calls_capped", 0))
                + int(dispatch_plan.capped_count)
            )

    def _handle_synthetic_result(
        self,
        result: ToolResult,
        *,
        request: ToolDispatchRequest,
        tool_results: list[ToolResult],
    ) -> ToolDispatchResult:
        try:
            result = self._services.middleware_pipeline.run_after_tool(result, request.context)
        except Exception as exc:
            self._adapter.handle_pipeline_error(exc, LoopState.TOOL_EXECUTE, request.context)
            return self._failed_result(
                request=request,
                tool_results=tool_results,
                exit_reason=ExitReason.MIDDLEWARE_ERROR,
                error_text=str(exc),
                dispatched_count=0,
            )
        self._adapter.record_tool_result_error(request.context, result)
        self._adapter.append_tool_result_message(request.messages, result)
        tool_results.append(result)
        return ToolDispatchResult(
            tool_results=tool_results,
            messages=request.messages,
            should_continue=True,
            all_tools_cheap=self._all_tools_cheap(request.tool_calls),
        )

    def _dispatch_one_tool(
        self,
        *,
        tool_call: ToolCall,
        context: TurnContext,
        request: ToolDispatchRequest,
        dispatch_t0: float,
        tool_results: list[ToolResult],
        dispatched_count: int,
    ) -> tuple[ToolResult, BaseException | None, int] | ToolDispatchResult:
        dispatch_error: BaseException | None = None
        context.metadata.pop("tool_error_result", None)
        context.metadata["_active_tool_call"] = tool_call
        context.metadata["_tool_interrupt_behavior"] = getattr(
            self._services.tool_registry.get(tool_call.name),
            "interrupt_behavior",
            "block",
        )
        result = ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content="tool execution did not produce a result",
            is_error=True,
        )
        try:
            result = self._services.tool_registry.dispatch(tool_call, context)
            dispatched_count += 1
        except UnknownToolError as exc:
            dispatch_error = exc
            if self._config.fail_on_unknown_tool:
                self._adapter.fire_post_tool_hook(
                    tool_call=tool_call,
                    result=None,
                    dispatch_error=dispatch_error,
                    elapsed_ms=(time.perf_counter() - dispatch_t0) * 1000.0,
                    session_id=request.request.session_id,
                    iteration=request.iteration,
                    context=context,
                )
                return self._failed_result(
                    request=request,
                    tool_results=tool_results,
                    exit_reason=ExitReason.UNKNOWN_TOOL,
                    error_text=f"Unknown tool: {tool_call.name}",
                    dispatched_count=dispatched_count,
                )
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=self._adapter.format_unknown_tool_content(
                    tool_call.name,
                    context=context,
                ),
                is_error=True,
                metadata={
                    "_unknown_tool_recovery": True,
                    "_tool_error_category": "unknown_tool",
                }
                if getattr(
                    self._config,
                    "tool_error_structured_format_enabled",
                    True,
                )
                else {},
            )
        except Exception as exc:
            dispatch_error = exc
            if self._config.fail_on_tool_error:
                self._adapter.handle_pipeline_error(exc, LoopState.TOOL_EXECUTE, context)
                recovered_tool_result = context.metadata.pop("tool_error_result", None)
                if not isinstance(recovered_tool_result, ToolResult):
                    self._adapter.fire_post_tool_hook(
                        tool_call=tool_call,
                        result=None,
                        dispatch_error=dispatch_error,
                        elapsed_ms=(time.perf_counter() - dispatch_t0) * 1000.0,
                        session_id=request.request.session_id,
                        iteration=request.iteration,
                        context=context,
                    )
                    return self._failed_result(
                        request=request,
                        tool_results=tool_results,
                        exit_reason=ExitReason.TOOL_ERROR,
                        error_text=str(exc),
                        dispatched_count=dispatched_count,
                    )
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
            context.metadata.pop("_tool_interrupt_behavior", None)
        return result, dispatch_error, dispatched_count

    def _failed_result(
        self,
        *,
        request: ToolDispatchRequest,
        tool_results: list[ToolResult],
        exit_reason: ExitReason,
        error_text: str | None,
        dispatched_count: int,
    ) -> ToolDispatchResult:
        return ToolDispatchResult(
            tool_results=tool_results,
            messages=request.messages,
            should_continue=False,
            exit_reason=exit_reason,
            error_text=error_text,
            all_tools_cheap=self._all_tools_cheap(request.tool_calls),
            dispatched_count=dispatched_count,
        )

    def _all_tools_cheap(self, tool_calls: list[ToolCall]) -> bool:
        return bool(tool_calls) and all(
            self._adapter.is_cheap_tool(call.name) for call in tool_calls
        )
