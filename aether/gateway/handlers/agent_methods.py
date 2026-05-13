"""``agent.*`` RPC methods.

The gateway keeps the engine synchronous and runs ``agent.run`` through
the dispatcher's long-handler pool.  Streaming output is bridged to the
client as JSON-RPC ``event`` notifications on the same transport.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from aether.agents.core.agent import AgentEngine
from aether.agents.middlewares.base import EngineMiddleware
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.cli.sessions import (
    SessionRecord,
    load_session,
    save_session,
    update_session_from_state,
)
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.gateway.dispatcher import current_request_id, method, notify
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    Cancelled,
    Done,
    Error,
    GatewayError,
    IterationEnd,
    IterationStart,
    LoopStateChanged,
    Reasoning,
    Status,
    TextDelta,
    TokenUsage,
    ToolCall as ToolCallEvent,
    ToolResult as ToolResultEvent,
)
from aether.gateway.handlers.state import set_current_session
from aether.gateway.run_handle import RunHandle, running_runs
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
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
from aether.tools.registry import ToolRegistry


class _EventSink:
    """Small typed wrapper around ``notify("event", ...)``."""

    def __init__(self, *, session_id: str, run_id: str) -> None:
        self.session_id = session_id
        self.run_id = run_id
        self._sequence = 0
        self._lock = threading.Lock()

    def _next_sequence(self) -> int:
        with self._lock:
            value = self._sequence
            self._sequence += 1
            return value

    def emit(self, event: Any) -> None:
        notify("event", event.model_dump(mode="json", exclude_none=True))

    def text_delta(self, text: str) -> None:
        self.emit(
            TextDelta(
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
            Reasoning(
                session_id=self.session_id,
                run_id=self.run_id,
                text=text,
                sequence=self._next_sequence(),
            )
        )

    def silent_delta(self, _text: str) -> None:
        return None

    def status(self, kind: str, detail: str | None = None) -> None:
        self.emit(
            Status(
                session_id=self.session_id,
                run_id=self.run_id,
                kind=kind,  # type: ignore[arg-type]
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
            TokenUsage(
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
            Done(
                session_id=self.session_id,
                run_id=self.run_id,
                final_text=final_text,
                exit_reason=exit_reason,
            )
        )

    def cancelled(self, reason: str | None, partial_text: str) -> None:
        self.emit(
            Cancelled(
                session_id=self.session_id,
                run_id=self.run_id,
                reason=reason,
                partial_text=partial_text,
            )
        )

    def error(self, message: str) -> None:
        self.emit(
            Error(
                session_id=self.session_id,
                run_id=self.run_id,
                message=message,
            )
        )


class _GatewayEventMiddleware(EngineMiddleware):
    """Translate engine middleware callbacks into agent event notifications."""

    def __init__(self, sink: _EventSink) -> None:
        self._sink = sink

    def before_llm(self, messages: list[dict], context: TurnContext) -> list[dict]:
        iteration = _wire_iteration(context)
        self._sink.status("thinking")
        self._sink.emit(
            IterationStart(
                session_id=self._sink.session_id,
                run_id=self._sink.run_id,
                iteration=iteration,
            )
        )
        return messages

    def after_llm(
        self,
        response: NormalizedResponse,
        context: TurnContext,
    ) -> NormalizedResponse:
        if response.content:
            self._sink.status("responding")
        reasoning = response.metadata.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self._sink.reasoning_delta(reasoning)
        self._sink.emit(
            IterationEnd(
                session_id=self._sink.session_id,
                run_id=self._sink.run_id,
                iteration=_wire_iteration(context),
            )
        )
        return response

    def before_tool(
        self,
        call: ToolCall | ToolResult,
        context: TurnContext,
    ) -> ToolCall | ToolResult:
        self._sink.status("tool_use", detail=getattr(call, "name", None))
        if isinstance(call, ToolCall):
            self._sink.emit(
                ToolCallEvent(
                    session_id=self._sink.session_id,
                    run_id=self._sink.run_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.arguments or {}),
                    iteration=_wire_iteration(context),
                )
            )
        return call

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        self._sink.emit(
            ToolResultEvent(
                session_id=self._sink.session_id,
                run_id=self._sink.run_id,
                tool_call_id=result.tool_call_id,
                tool_name=result.name,
                content=result.content,
                is_error=bool(result.is_error),
                iteration=_wire_iteration(context),
            )
        )
        self._sink.status("thinking")
        return result

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        self._sink.error(f"{state.value}: {error}")


class _GatewayEventHooks(EngineHooks):
    """Translate engine lifecycle hooks into agent event notifications."""

    def __init__(self, sink: _EventSink) -> None:
        super().__init__()
        self._sink = sink

    def on_session_end(
        self,
        *,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        self._sink.status("idle")


def agent_run(params: dict[str, Any] | None) -> dict[str, Any]:
    run_params = _parse_run_params(params)
    session_id = run_params["session_id"]
    run_id = _run_id_from_request()
    sink = _EventSink(session_id=session_id, run_id=run_id)
    handle = RunHandle(session_id=session_id, run_id=run_id)

    record = load_session(session_id)
    if record is None:
        raise GatewayError(
            f"session not found: {session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": session_id},
        )
    _validate_record_for_run(record)
    set_current_session(session_id)

    if not running_runs.register(handle):
        raise GatewayError(
            "RUN_ALREADY_ACTIVE",
            code=ERROR_APPLICATION,
            data={"code": "RUN_ALREADY_ACTIVE", "session_id": session_id},
        )

    try:
        provider = _build_provider_for_record(record)
        config = _build_engine_config(run_params.get("max_iterations"))
        tool_registry = _build_tool_registry()
        engine = AgentEngine(
            provider,
            tool_registry=tool_registry,
            middleware_pipeline=MiddlewarePipeline([_GatewayEventMiddleware(sink)]),
            config=config,
            hooks=_GatewayEventHooks(sink),
        )
        request = EngineRequest(
            session_id=session_id,
            user_message=run_params["user_message"],
            system_message=(
                run_params["system_override"]
                if run_params["system_override"] is not None
                else record.system_prompt
            ),
            stream_callback=sink.text_delta,
            stream_silent_callback=sink.silent_delta,
            messages=list(record.messages),
            model_config=ModelCallConfig(temperature=run_params["temperature"]),
            metadata={"run_id": run_id, "_loop_state_callback": sink.loop_state},
            approval_prompter=None,
            tool_permission_prompter=None,
            interrupt_signal=handle.interrupt_signal,
        )
        result = engine.run_loop(request)
        _persist_result(record, result)
        response = _response_from_result(result)
        usage = response.get("usage")
        if isinstance(usage, dict):
            sink.usage(usage)
        _emit_terminal_event(sink, result, response)
        return response
    except GatewayError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as run result, not worker crash
        sink.error(str(exc) or type(exc).__name__)
        return _error_response(exc)
    finally:
        running_runs.unregister(session_id, handle)


def agent_cancel(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="agent.cancel")
    running_runs.cancel(session_id)
    return {"ok": True}


def _parse_run_params(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="agent.run")
    user_message = _require_str(params, "user_message", where="agent.run")
    body = params or {}

    max_iterations = body.get("max_iterations")
    if max_iterations is not None:
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise GatewayError(
                "agent.run requires positive integer 'max_iterations'",
                code=ERROR_INVALID_PARAMS,
            )

    temperature = body.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise GatewayError(
            "agent.run requires numeric or null 'temperature'",
            code=ERROR_INVALID_PARAMS,
        )

    system_override = body.get("system_override")
    if system_override is not None and not isinstance(system_override, str):
        raise GatewayError(
            "agent.run requires string or null 'system_override'",
            code=ERROR_INVALID_PARAMS,
        )

    return {
        "session_id": session_id,
        "user_message": user_message,
        "max_iterations": max_iterations,
        "temperature": float(temperature) if temperature is not None else None,
        "system_override": system_override,
    }


def _require_str(params: dict[str, Any] | None, key: str, *, where: str) -> str:
    if not params or not isinstance(params.get(key), str) or not params[key].strip():
        raise GatewayError(
            f"{where} requires non-empty string '{key}'",
            code=ERROR_INVALID_PARAMS,
        )
    return params[key].strip()


def _run_id_from_request() -> str:
    request_id = current_request_id()
    if isinstance(request_id, (str, int)):
        return str(request_id)
    return str(uuid.uuid4())


def _validate_record_for_run(record: SessionRecord) -> None:
    if not record.provider.strip():
        raise GatewayError(
            f"session has no provider: {record.session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": record.session_id},
        )
    if not record.model.strip():
        raise GatewayError(
            f"session has no model: {record.session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": record.session_id},
        )


def _build_provider_for_record(record: SessionRecord) -> ModelProvider:
    from aether.cli.providers import build_provider

    return build_provider(
        record.provider,
        model=record.model,
        base_url=record.base_url,
    )


def _build_engine_config(max_iterations: Any) -> EngineConfig:
    config = EngineConfig()
    if isinstance(max_iterations, int):
        config.max_iterations = max_iterations
    return config


def _build_tool_registry() -> ToolRegistry | None:
    return None


def _persist_result(record: SessionRecord, result: EngineResult) -> None:
    update_session_from_state(
        record,
        messages=result.messages,
        provider=record.provider,
        model=record.model,
        base_url=record.base_url,
        system_prompt=result.system_prompt or record.system_prompt,
    )
    save_session(record)


def _response_from_result(result: EngineResult) -> dict[str, Any]:
    metadata = dict(result.metadata or {})
    interrupt = metadata.get("interrupt") if isinstance(metadata, dict) else None
    partial_text = ""
    if isinstance(interrupt, dict):
        partial_text = str(interrupt.get("partial_text") or "")
    final_text = result.final_response if result.final_response is not None else partial_text
    return {
        "final_text": final_text,
        "exit_reason": _wire_exit_reason(result),
        "usage": _usage_from_metadata(metadata),
        "metadata": metadata,
    }


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
    sink: _EventSink,
    result: EngineResult,
    response: dict[str, Any],
) -> None:
    exit_reason = response["exit_reason"]
    final_text = str(response.get("final_text") or "")
    if exit_reason == "cancelled":
        interrupt = result.metadata.get("interrupt")
        reason = None
        if isinstance(interrupt, dict):
            reason = str(interrupt.get("reason") or "") or None
        sink.cancelled(reason=reason, partial_text=final_text)
    elif exit_reason == "error":
        sink.error(result.error or result.exit_reason.value)
    else:
        sink.done(final_text=final_text, exit_reason=exit_reason)


def _error_response(exc: Exception) -> dict[str, Any]:
    return {
        "final_text": "",
        "exit_reason": "error",
        "usage": _usage_from_metadata({}),
        "metadata": {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        },
    }


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _wire_iteration(context: TurnContext) -> int:
    return max(0, int(context.iteration or 0) - 1)


def register() -> None:
    """Register ``agent.*`` handlers on the dispatcher.  Idempotent."""
    method("agent.run", long=True)(agent_run)
    method("agent.cancel", long=False)(agent_cancel)


def reset_agent_runs_for_tests() -> None:
    running_runs.clear()


__all__ = [
    "agent_cancel",
    "agent_run",
    "register",
    "reset_agent_runs_for_tests",
]
