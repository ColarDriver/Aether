"""``agent.*`` RPC methods backed by AgentRunService."""

from __future__ import annotations

import os
import uuid
from typing import Any

from aether.config.schema import EngineConfig
from aether.gateway.dispatcher import current_request_id, method, notify
from aether.gateway.handlers.prompter_bridge import (
    GatewayPrompter,
    GatewayToolPermissionPrompter,
)
from aether.gateway.handlers.run_event_adapter import service_event_to_gateway_payload
from aether.gateway.handlers.service_errors import service_error_to_gateway
from aether.gateway.handlers.state import (
    get_current_session,
    set_current_session,
)
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)
from aether.models.provider.base import ModelProvider
from aether.services.common import (
    ServiceConflictError,
    ServiceError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from aether.services.runs import (
    AgentRunCancelRequest,
    AgentRunOptions,
    AgentRunRequest,
    AgentRunResult,
    AgentRunService,
    RunEvent,
    RunEventSink,
    RunRegistry,
)
from aether.services.runs.builder import RunDependencyBuilder
from aether.services.sessions import SessionService
from aether.tools.registry import ToolRegistry


class _GatewayRunEventSink:
    def emit(self, event: RunEvent) -> None:
        payload = service_event_to_gateway_payload(event)
        if payload is not None:
            notify("event", payload)


_RUN_REGISTRY = RunRegistry()


def _new_run_service() -> AgentRunService:
    return AgentRunService(
        session_service=SessionService(
            current_getter=get_current_session,
            current_setter=set_current_session,
        ),
        builder=RunDependencyBuilder(
            provider_factory=lambda record: _build_provider_for_record(record),
            config_factory=lambda options: _build_config_from_options(options),
            tool_registry_factory=lambda: _build_tool_registry(),
        ),
        registry=_RUN_REGISTRY,
    )


_RUN_SERVICE = _new_run_service()


def agent_run(params: dict[str, Any] | None) -> dict[str, Any]:
    run_params = _parse_run_params(params)
    session_id = run_params["session_id"]
    run_id = _run_id_from_request()
    set_current_session(session_id)
    request = AgentRunRequest(
        session_id=session_id,
        user_message=run_params["user_message"],
        run_id=run_id,
        options=AgentRunOptions(
            max_iterations=run_params["max_iterations"],
            temperature=run_params["temperature"],
            max_tokens=run_params["max_tokens"],
            disable_builtin_tools=run_params["disable_builtin_tools"],
            system_override=run_params["system_override"],
        ),
        approval_prompter=GatewayPrompter(session_id=session_id, run_id=run_id),
        tool_permission_prompter=GatewayToolPermissionPrompter(run_id=run_id),
    )
    try:
        result = _RUN_SERVICE.start(request, sink=_GatewayRunEventSink())
    except ServiceConflictError as exc:
        raise GatewayError(
            exc.message,
            code=ERROR_APPLICATION,
            data=exc.details or {"code": "RUN_ALREADY_ACTIVE", "session_id": session_id},
        ) from exc
    except ServiceNotFoundError as exc:
        raise GatewayError(
            exc.message,
            code=ERROR_APPLICATION,
            data=exc.details or {"session_id": session_id},
        ) from exc
    except ServiceValidationError as exc:
        if "session has no " in exc.message:
            raise GatewayError(
                exc.message,
                code=ERROR_APPLICATION,
                data=exc.details or {"session_id": session_id},
            ) from exc
        raise service_error_to_gateway(exc) from exc
    except ServiceError as exc:
        raise service_error_to_gateway(exc) from exc
    return _response_from_service_result(result)


def agent_cancel(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="agent.cancel")
    _RUN_SERVICE.cancel(AgentRunCancelRequest(session_id=session_id))
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

    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise GatewayError(
                "agent.run requires positive integer 'max_tokens'",
                code=ERROR_INVALID_PARAMS,
            )

    disable_builtin_tools = body.get("disable_builtin_tools")
    if disable_builtin_tools is not None and not isinstance(disable_builtin_tools, bool):
        raise GatewayError(
            "agent.run requires boolean 'disable_builtin_tools'",
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
        "max_tokens": max_tokens,
        "disable_builtin_tools": bool(disable_builtin_tools) if disable_builtin_tools is not None else None,
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


def _build_provider_for_record(record: Any) -> ModelProvider:
    from aether.cli.providers import build_provider

    return build_provider(
        record.provider,
        model=record.model,
        api_key=os.getenv("AETHER_API_KEY"),
        base_url=record.base_url,
    )


def _build_engine_config(max_iterations: Any) -> EngineConfig:
    config = EngineConfig()
    if isinstance(max_iterations, int):
        config.max_iterations = max_iterations
    config.tool_permissions_enabled = True
    config.skill_listing_enabled = True
    config.agent_type_registry_enabled = True
    return config


def _build_config_from_options(options: AgentRunOptions) -> EngineConfig:
    config = _build_engine_config(options.max_iterations)
    if options.disable_builtin_tools is True:
        config.use_builtin_tools = False
    return config


def _build_tool_registry() -> ToolRegistry | None:
    return None


def _response_from_service_result(result: AgentRunResult) -> dict[str, Any]:
    return {
        "final_text": result.final_text,
        "exit_reason": result.exit_reason,
        "usage": dict(result.usage or {}),
        "metadata": dict(result.metadata or {}),
    }


def register() -> None:
    method("agent.run", long=True)(agent_run)
    method("agent.cancel", long=False)(agent_cancel)


def reset_agent_runs_for_tests() -> None:
    global _RUN_REGISTRY, _RUN_SERVICE
    _RUN_REGISTRY = RunRegistry()
    _RUN_SERVICE = _new_run_service()


__all__ = [
    "_build_engine_config",
    "_build_provider_for_record",
    "_build_tool_registry",
    "_parse_run_params",
    "agent_cancel",
    "agent_run",
    "register",
    "reset_agent_runs_for_tests",
]
