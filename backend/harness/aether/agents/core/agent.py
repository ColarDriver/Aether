"""Primary agent implementation for Aether harness."""

from __future__ import annotations

import copy
import json
import logging
import uuid
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
from aether.runtime.hooks import EngineHooks
from aether.runtime.interrupts import InterruptController
from aether.runtime.services import EngineServices
from aether.runtime.session_store import InMemorySessionStore, SessionStore
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
        session_store: SessionStore | None = None,
        hooks: EngineHooks | None = None,
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
        self._session_store = session_store or InMemorySessionStore()
        self._hooks = hooks or EngineHooks()

        # Cross-turn counters used by nudge policies.
        self._memory_nudge_counter = 0
        self._skill_nudge_counter = 0

        # Turn-local counters reset by _prepare_turn_entry().
        self._empty_response_retries = 0
        self._provider_error_retries = 0

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
                        try:
                            response = self.services.provider.generate(
                                prepared_messages,
                                self.services.tool_registry.list_descriptors(),
                                request.model_config,
                                context,
                                stream_callback=stream_callback_wrapped,
                            )
                        except Exception as exc:
                            self._provider_error_retries += 1
                            # Give on_error middlewares one chance to convert provider failure
                            # into a normalized assistant response.
                            self._handle_pipeline_error(exc, state_machine.state, context)
                            recovered_response = self._pop_context_response(context, "llm_error_response")
                            if recovered_response is None:
                                error_text = str(exc)
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

                    iterations += 1

                    self._safe_call_hook(
                        "post_llm_call",
                        session_id=request.session_id,
                        iteration=context.iteration,
                        response_text=response.content or "",
                        context_metadata=context.metadata,
                    )

                    if response.tool_calls:
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

                    # No tool calls -> finalize response
                    self._append_assistant_text_message(messages, response)
                    final_response = response.content or ""
                    if stream_callback_wrapped and final_response and not context.metadata.get("streamed_output"):
                        stream_callback_wrapped(final_response)
                        context.metadata["stream_fallback_emitted"] = True

                    if final_response:
                        self._empty_response_retries = 0
                        exit_reason = ExitReason.TEXT_RESPONSE
                    else:
                        self._empty_response_retries += 1
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
        # Reset per-turn counters so retries from a previous turn don't leak.
        self._empty_response_retries = 0
        self._provider_error_retries = 0

        state_machine = EngineStateMachine()
        messages = self._sanitize_messages(copy.deepcopy(request.messages))

        if request.user_message is not None:
            messages.append({"role": "user", "content": self._sanitize_text(request.user_message)})

        task_id = self._sanitize_text(str(request.metadata.get("task_id", "")).strip()) if request.metadata else ""
        if not task_id:
            task_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())

        metadata = dict(request.metadata)
        metadata.update(
            {
                "entry_prepared": True,
                "task_id": task_id,
                "turn_id": turn_id,
                "request_has_stream_callback": bool(request.stream_callback),
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

        if selected_prompt:
            messages = self._inject_system_prompt(messages, selected_prompt)

        context.metadata["system_prompt_applied"] = bool(selected_prompt)
        context.metadata["system_prompt_source"] = (
            "request_field" if requested_prompt else ("session_store" if stored_prompt else "none")
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
        context.metadata.setdefault("should_review_memory", False)
        context.metadata.setdefault("should_review_skills", False)

        interval = max(0, int(getattr(self.config, "memory_nudge_interval", 0)))
        if interval > 0:
            self._memory_nudge_counter += 1
            if self._memory_nudge_counter >= interval:
                context.metadata["should_review_memory"] = True
                self._memory_nudge_counter = 0

    def _register_skill_nudge(self, context: TurnContext) -> None:
        interval = max(0, int(getattr(self.config, "skill_nudge_interval", 0)))
        if interval <= 0:
            return

        self._skill_nudge_counter += 1
        if self._skill_nudge_counter >= interval:
            context.metadata["should_review_skills"] = True
            self._skill_nudge_counter = 0

    def _build_stream_callback(self, request: EngineRequest, context: TurnContext):
        callback = request.stream_callback
        if callback is None:
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
        }:
            status = EngineStatus.FAILED
        else:
            status = EngineStatus.COMPLETED

        metadata = {
            "request": {
                "session_id": request.session_id,
                "model_config": asdict(request.model_config),
                "system_message_present": bool(request.system_message),
                "stream_callback_present": bool(request.stream_callback),
            },
            "turn": dict(context.metadata),
            "runtime": {
                "empty_response_retries": self._empty_response_retries,
                "provider_error_retries": self._provider_error_retries,
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
