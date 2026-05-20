"""Agent run service implementation."""

from __future__ import annotations

from collections.abc import Mapping
import threading
import time
import uuid
from typing import Any, Literal

from aether.agents.middlewares.base import EngineMiddleware
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.runtime.core.contracts import (
    EngineResult,
    EngineStatus,
    ExitReason,
    LoopState,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.services.common import (
    ServiceConflictError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from aether.services.runs.builder import RunDependencyBuilder
from aether.services.runs.contracts import (
    AgentRunCancelRequest,
    AgentRunRequest,
    AgentRunResult,
    AgentRunSnapshot,
    AgentRunStatus,
    RunEventSink,
)
from aether.services.runs.events import (
    AssistantDelta,
    IterationFinished,
    IterationStarted,
    LoopStateChanged,
    ReasoningDelta,
    RunCancelled,
    RunEvent,
    RunFailed,
    RunFinished,
    RunStarted,
    RunStatusChanged,
    SilentProgress,
    TokenUsageUpdated,
    ToolFinished,
    ToolStarted,
)
from aether.services.runs.handles import RunHandle, RunRegistry
from aether.services.sessions import SessionService


class AgentRunService:
    def __init__(
        self,
        *,
        session_service: SessionService | None = None,
        builder: RunDependencyBuilder | None = None,
        registry: RunRegistry | None = None,
    ) -> None:
        self._sessions = session_service or SessionService()
        self._builder = builder or RunDependencyBuilder()
        self._registry = registry or RunRegistry()
        self._lock = threading.Lock()
        self._snapshots_by_run: dict[str, AgentRunSnapshot] = {}
        self._latest_run_by_session: dict[str, str] = {}

    def start(
        self,
        request: AgentRunRequest,
        sink: RunEventSink | None = None,
    ) -> AgentRunResult:
        record = self._sessions.resolve_record(_require_non_empty(request.session_id, "session_id"))
        _validate_record_for_run(record.session_id, record.provider, record.model)
        run_id = request.run_id or str(uuid.uuid4())
        handle = RunHandle(session_id=record.session_id, run_id=run_id)
        emitter = _RunEventEmitter(session_id=record.session_id, run_id=run_id, sink=sink)

        if not self._registry.register(handle):
            raise ServiceConflictError(
                "RUN_ALREADY_ACTIVE",
                details={"code": "RUN_ALREADY_ACTIVE", "session_id": record.session_id},
            )

        self._set_snapshot(
            AgentRunSnapshot(
                session_id=record.session_id,
                run_id=run_id,
                status=AgentRunStatus.RUNNING,
                started_at=time.time(),
            )
        )
        emitter.emit(RunStarted(session_id=record.session_id, run_id=run_id))

        try:
            provider = self._builder.build_provider(record)
            config = self._builder.build_engine_config(request.options)
            tool_registry = self._builder.build_tool_registry()
            skill_catalog = self._builder.build_skill_catalog(config)
            agent_type_registry = self._builder.build_agent_type_registry(config)
            task_store = self._builder.build_task_store(config)
            subagent_manager = self._builder.build_subagent_manager(config)
            engine = self._builder.build_engine(
                provider=provider,
                tool_registry=tool_registry,
                middleware_pipeline=MiddlewarePipeline([_RunEventMiddleware(emitter)]),
                config=config,
                hooks=_RunEventHooks(emitter),
                skill_catalog=skill_catalog,
                agent_type_registry=agent_type_registry,
                subagent_manager=subagent_manager,
                task_store=task_store,
            )
            engine_request = self._builder.build_engine_request(
                record=record,
                request=request,
                run_id=run_id,
                handle=handle,
                stream_callback=emitter.text_delta,
                stream_silent_callback=emitter.silent_delta,
                loop_state_callback=emitter.loop_state,
            )
            engine_result = engine.run_loop(engine_request)
            self._sessions.persist_run_result(
                record.session_id,
                messages=engine_result.messages,
                system_prompt=engine_result.system_prompt or record.system_prompt,
            )
            result = _result_from_engine_result(
                session_id=record.session_id,
                run_id=run_id,
                result=engine_result,
            )
            emitter.usage(result.usage)
            _emit_terminal_event(emitter, engine_result, result)
            self._set_snapshot(
                AgentRunSnapshot(
                    session_id=record.session_id,
                    run_id=run_id,
                    status=_status_from_result(result),
                    started_at=self._snapshot_started_at(run_id),
                    completed_at=time.time(),
                    result=result,
                    error=_error_message_from_result(result),
                )
            )
            return result
        except (ServiceNotFoundError, ServiceValidationError, ServiceConflictError):
            raise
        except Exception as exc:  # noqa: BLE001 - preserve gateway behavior as run result
            message = str(exc) or type(exc).__name__
            emitter.error(message)
            result = _error_result(record.session_id, run_id, exc)
            self._set_snapshot(
                AgentRunSnapshot(
                    session_id=record.session_id,
                    run_id=run_id,
                    status=AgentRunStatus.FAILED,
                    started_at=self._snapshot_started_at(run_id),
                    completed_at=time.time(),
                    result=result,
                    error=message,
                )
            )
            return result
        finally:
            self._registry.unregister(record.session_id, handle)

    def cancel(self, request: AgentRunCancelRequest) -> bool:
        return self._registry.cancel(
            request.session_id,
            run_id=request.run_id,
            reason=request.reason or "rpc-cancel",
        )

    def status(self, run_id_or_session_id: str) -> AgentRunSnapshot | None:
        key = run_id_or_session_id.strip()
        if not key:
            return None
        handle = self._registry.get_by_run(key) or self._registry.get_by_session(key)
        if handle is not None:
            existing = self._snapshots_by_run.get(handle.run_id)
            return existing or AgentRunSnapshot(
                session_id=handle.session_id,
                run_id=handle.run_id,
                status=AgentRunStatus.RUNNING,
            )
        with self._lock:
            if key in self._snapshots_by_run:
                return self._snapshots_by_run[key]
            run_id = self._latest_run_by_session.get(key)
            if run_id:
                return self._snapshots_by_run.get(run_id)
        return None

    def final_result(self, run_id_or_session_id: str) -> AgentRunResult | None:
        snapshot = self.status(run_id_or_session_id)
        return snapshot.result if snapshot is not None else None

    def _set_snapshot(self, snapshot: AgentRunSnapshot) -> None:
        with self._lock:
            self._snapshots_by_run[snapshot.run_id] = snapshot
            self._latest_run_by_session[snapshot.session_id] = snapshot.run_id

    def _snapshot_started_at(self, run_id: str) -> float | None:
        with self._lock:
            snapshot = self._snapshots_by_run.get(run_id)
        return snapshot.started_at if snapshot is not None else None


class _RunEventEmitter:
    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        sink: RunEventSink | None,
    ) -> None:
        self.session_id = session_id
        self.run_id = run_id
        self._sink = sink
        self._sequence = 0
        self._lock = threading.Lock()

    def _next_sequence(self) -> int:
        with self._lock:
            value = self._sequence
            self._sequence += 1
            return value

    def emit(self, event: RunEvent) -> None:
        if self._sink is not None:
            self._sink.emit(event)

    def text_delta(self, text: str) -> None:
        self.emit(
            AssistantDelta(
                session_id=self.session_id,
                run_id=self.run_id,
                text=text,
                sequence=self._next_sequence(),
            )
        )

    def reasoning_delta(self, text: str) -> None:
        if not text:
            return
        self.emit(
            ReasoningDelta(
                session_id=self.session_id,
                run_id=self.run_id,
                text=text,
                sequence=self._next_sequence(),
            )
        )

    def silent_delta(self, text: str) -> None:
        if not text:
            return
        self.emit(
            SilentProgress(
                session_id=self.session_id,
                run_id=self.run_id,
                chars=len(text),
                sequence=self._next_sequence(),
            )
        )

    def status(
        self,
        kind: Literal["thinking", "responding", "tool_use", "idle"],
        detail: str | None = None,
    ) -> None:
        self.emit(
            RunStatusChanged(
                session_id=self.session_id,
                run_id=self.run_id,
                kind=kind,
                detail=detail,
            )
        )

    def loop_state(self, state: LoopState) -> None:
        self.emit(
            LoopStateChanged(
                session_id=self.session_id,
                run_id=self.run_id,
                state=state.value if hasattr(state, "value") else str(state),
            )
        )

    def usage(self, usage: dict[str, Any]) -> None:
        self.emit(
            TokenUsageUpdated(
                session_id=self.session_id,
                run_id=self.run_id,
                input_tokens=_int_value(usage.get("input_tokens")),
                output_tokens=_int_value(usage.get("output_tokens")),
                cache_read_tokens=_int_value(usage.get("cache_read_tokens")),
                cache_write_tokens=_int_value(usage.get("cache_write_tokens")),
            )
        )

    def done(self, final_text: str, exit_reason: str) -> None:
        self.emit(
            RunFinished(
                session_id=self.session_id,
                run_id=self.run_id,
                final_text=final_text,
                exit_reason=exit_reason,
            )
        )

    def cancelled(self, reason: str | None, partial_text: str) -> None:
        self.emit(
            RunCancelled(
                session_id=self.session_id,
                run_id=self.run_id,
                reason=reason,
                partial_text=partial_text,
            )
        )

    def error(self, message: str) -> None:
        self.emit(
            RunFailed(
                session_id=self.session_id,
                run_id=self.run_id,
                message=message,
            )
        )


class _RunEventMiddleware(EngineMiddleware):
    def __init__(self, emitter: _RunEventEmitter) -> None:
        self._emitter = emitter

    def before_llm(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        self._emitter.status("thinking")
        self._emitter.emit(
            IterationStarted(
                session_id=self._emitter.session_id,
                run_id=self._emitter.run_id,
                iteration=_wire_iteration(context),
            )
        )
        return messages

    def after_llm(
        self,
        response: NormalizedResponse,
        context: TurnContext,
    ) -> NormalizedResponse:
        if response.content:
            self._emitter.status("responding")
        reasoning = response.metadata.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self._emitter.reasoning_delta(reasoning)
        self._emitter.emit(
            IterationFinished(
                session_id=self._emitter.session_id,
                run_id=self._emitter.run_id,
                iteration=_wire_iteration(context),
            )
        )
        return response

    def before_tool(
        self,
        call: ToolCall | ToolResult,
        context: TurnContext,
    ) -> ToolCall | ToolResult:
        self._emitter.status("tool_use", detail=getattr(call, "name", None))
        if isinstance(call, ToolCall):
            self._emitter.emit(
                ToolStarted(
                    session_id=self._emitter.session_id,
                    run_id=self._emitter.run_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.arguments or {}),
                    iteration=_wire_iteration(context),
                )
            )
        return call

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        self._emitter.emit(
            ToolFinished(
                session_id=self._emitter.session_id,
                run_id=self._emitter.run_id,
                tool_call_id=result.tool_call_id,
                tool_name=result.name,
                content=result.content,
                is_error=bool(result.is_error),
                iteration=_wire_iteration(context),
                metadata=dict(result.metadata or {}),
            )
        )
        self._emitter.status("thinking")
        return result

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        self._emitter.error(f"{state.value}: {error}")


class _RunEventHooks(EngineHooks):
    def __init__(self, emitter: _RunEventEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    def on_session_end(
        self,
        *,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        self._emitter.status("idle")


def _validate_record_for_run(session_id: str, provider: str, model: str) -> None:
    if not provider.strip():
        raise ServiceValidationError(
            f"session has no provider: {session_id}",
            details={"session_id": session_id},
        )
    if not model.strip():
        raise ServiceValidationError(
            f"session has no model: {session_id}",
            details={"session_id": session_id},
        )


def _result_from_engine_result(
    *,
    session_id: str,
    run_id: str,
    result: EngineResult,
) -> AgentRunResult:
    metadata = dict(result.metadata or {})
    interrupt = metadata.get("interrupt") if isinstance(metadata, dict) else None
    partial_text = ""
    if isinstance(interrupt, dict):
        partial_text = str(interrupt.get("partial_text") or "")
    final_text = result.final_response if result.final_response is not None else partial_text
    return AgentRunResult(
        session_id=session_id,
        run_id=run_id,
        final_text=final_text,
        exit_reason=_wire_exit_reason(result),
        usage=_usage_from_metadata(metadata),
        metadata=metadata,
    )


def _wire_exit_reason(result: EngineResult) -> str:
    if (
        result.status == EngineStatus.INTERRUPTED
        or result.exit_reason == ExitReason.INTERRUPTED
    ):
        return "cancelled"
    if (
        result.status == EngineStatus.MAX_ITERATIONS
        or result.exit_reason == ExitReason.MAX_ITERATIONS
    ):
        return "max_iterations"
    if result.status == EngineStatus.FAILED:
        return "error"
    return "done"


def _usage_from_metadata(metadata: dict[str, Any]) -> dict[str, int]:
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input_tokens": _int_value(usage.get("input_tokens")),
        "output_tokens": _int_value(usage.get("output_tokens")),
        "cache_read_tokens": _int_value(usage.get("cache_read_tokens")),
        "cache_write_tokens": _int_value(usage.get("cache_write_tokens")),
        "reasoning_tokens": _int_value(usage.get("reasoning_tokens")),
        "prompt_tokens": _int_value(usage.get("prompt_tokens")),
        "completion_tokens": _int_value(usage.get("completion_tokens")),
        "total_tokens": _int_value(usage.get("total_tokens")),
    }


def _emit_terminal_event(
    emitter: _RunEventEmitter,
    engine_result: EngineResult,
    result: AgentRunResult,
) -> None:
    if result.exit_reason == "cancelled":
        interrupt = engine_result.metadata.get("interrupt")
        reason = None
        if isinstance(interrupt, dict):
            reason = str(interrupt.get("reason") or "") or None
        emitter.cancelled(reason=reason, partial_text=result.final_text)
    elif result.exit_reason == "error":
        emitter.error(engine_result.error or engine_result.exit_reason.value)
    else:
        emitter.done(final_text=result.final_text, exit_reason=result.exit_reason)


def _error_result(session_id: str, run_id: str, exc: Exception) -> AgentRunResult:
    return AgentRunResult(
        session_id=session_id,
        run_id=run_id,
        final_text="",
        exit_reason="error",
        usage=_usage_from_metadata({}),
        metadata={
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        },
    )


def _status_from_result(result: AgentRunResult) -> AgentRunStatus:
    if result.exit_reason == "cancelled":
        return AgentRunStatus.CANCELLED
    if result.exit_reason == "error":
        return AgentRunStatus.FAILED
    return AgentRunStatus.COMPLETED


def _error_message_from_result(result: AgentRunResult) -> str | None:
    if result.exit_reason != "error":
        return None
    error = result.metadata.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        return str(message) if message is not None else None
    return None


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _wire_iteration(context: TurnContext) -> int:
    return max(0, int(context.iteration or 0) - 1)


def _require_non_empty(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceValidationError(
            f"agent run requires non-empty string '{field}'",
            details={"field": field},
        )
    return value.strip()


__all__ = ["AgentRunService"]
