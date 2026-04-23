"""Primary agent implementation for Aether harness."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.config.schema import EngineConfig
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
from aether.runtime.interrupts import InterruptController
from aether.runtime.services import EngineServices
from aether.runtime.state_machine import EngineStateMachine
from aether.tools.base import UnknownToolError
from aether.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aether.subagents.contracts import SubagentResult, SubagentTask
    from aether.subagents.manager import SubagentManager


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
    ) -> None:
        self.config = config or EngineConfig()
        self.services = EngineServices(
            provider=provider,
            tool_registry=tool_registry or ToolRegistry(),
            middleware_pipeline=middleware_pipeline or MiddlewarePipeline(),
            interrupt_controller=interrupt_controller or InterruptController(),
            logger=logger or logging.getLogger(__name__),
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
        try:
            state_machine = EngineStateMachine()
            messages = copy.deepcopy(request.messages)
            if request.user_message is not None:
                messages.append({"role": "user", "content": request.user_message})

            context = TurnContext(
                session_id=request.session_id,
                iteration=0,
                metadata=dict(request.metadata),
            )

            final_response: str | None = None
            error_text: str | None = None
            exit_reason = ExitReason.EMPTY_RESPONSE
            iterations = 0

            state_machine.transition(LoopState.PREPARE)
            if self._is_interrupted(request.session_id):
                state_machine.transition(LoopState.INTERRUPTED)
                exit_reason = ExitReason.INTERRUPTED
                return self._build_result(
                    request,
                    messages,
                    iterations,
                    final_response,
                    error_text,
                    exit_reason,
                )

            state_machine.transition(LoopState.PRE_LLM)

            while iterations < self.config.max_iterations:
                context.iteration = iterations + 1

                if self._is_interrupted(request.session_id):
                    state_machine.transition(LoopState.INTERRUPTED)
                    exit_reason = ExitReason.INTERRUPTED
                    break

                try:
                    prepared_messages = self.services.middleware_pipeline.run_before_llm(messages, context)
                except Exception as exc:
                    self._handle_pipeline_error(exc, state_machine.state, context)
                    error_text = str(exc)
                    exit_reason = ExitReason.MIDDLEWARE_ERROR
                    state_machine.transition(LoopState.FAILED)
                    break

                state_machine.transition(LoopState.LLM_CALL)
                response = self._pop_context_response(context, "llm_pre_response")
                if response is None:
                    try:
                        response = self.services.provider.generate(
                            prepared_messages,
                            self.services.tool_registry.list_descriptors(),
                            request.model_config,
                            context,
                        )
                    except Exception as exc:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        recovered_response = self._pop_context_response(context, "llm_error_response")
                        if recovered_response is None:
                            error_text = str(exc)
                            exit_reason = ExitReason.PROVIDER_ERROR
                            state_machine.transition(LoopState.FAILED)
                            break
                        response = recovered_response

                state_machine.transition(LoopState.POST_LLM)
                try:
                    response = self.services.middleware_pipeline.run_after_llm(response, context)
                except Exception as exc:
                    self._handle_pipeline_error(exc, state_machine.state, context)
                    error_text = str(exc)
                    exit_reason = ExitReason.MIDDLEWARE_ERROR
                    state_machine.transition(LoopState.FAILED)
                    break

                iterations += 1

                if response.tool_calls:
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

                    state_machine.transition(LoopState.CHECK_EXIT)
                    if iterations >= self.config.max_iterations:
                        state_machine.transition(LoopState.FINALIZE)
                        exit_reason = ExitReason.MAX_ITERATIONS
                        break

                    state_machine.transition(LoopState.PRE_LLM)
                    continue

                # No tool calls -> finalize response
                self._append_assistant_text_message(messages, response)
                final_response = response.content or ""
                exit_reason = ExitReason.TEXT_RESPONSE if final_response else ExitReason.EMPTY_RESPONSE
                state_machine.transition(LoopState.FINALIZE)
                break

            if state_machine.state not in {LoopState.FAILED, LoopState.INTERRUPTED, LoopState.FINALIZE}:
                state_machine.transition(LoopState.FINALIZE)
                if iterations >= self.config.max_iterations:
                    exit_reason = ExitReason.MAX_ITERATIONS

            if state_machine.state == LoopState.FINALIZE:
                state_machine.transition(LoopState.DONE)

            result = self._build_result(
                request,
                messages,
                iterations,
                final_response,
                error_text,
                exit_reason,
            )
            self.services.interrupt_controller.clear(request.session_id)
            return result
        finally:
            self._current_session_id = None

    def _is_interrupted(self, session_id: str) -> bool:
        return self.services.interrupt_controller.is_interrupted(session_id)

    def _handle_pipeline_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        self.services.logger.exception("AgentEngine error at %s: %s", state, error)
        self.services.middleware_pipeline.run_on_error(error, state, context)

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

    def _build_result(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        iterations: int,
        final_response: str | None,
        error_text: str | None,
        exit_reason: ExitReason,
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
        }:
            status = EngineStatus.FAILED
        else:
            status = EngineStatus.COMPLETED

        return EngineResult(
            session_id=request.session_id,
            status=status,
            exit_reason=exit_reason,
            messages=messages,
            iterations=iterations,
            final_response=final_response,
            error=error_text,
            metadata={
                "request": {
                    "session_id": request.session_id,
                    "model_config": asdict(request.model_config),
                }
            },
        )
