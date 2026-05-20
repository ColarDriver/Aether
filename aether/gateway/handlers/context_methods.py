"""``context.*`` RPC methods for context compression status/control."""

from __future__ import annotations

import copy
from typing import Any, Protocol

from aether import AgentEngine
from aether.agents.runtime.context_assembly import LegacyContextAssemblyAdapter
from aether.cli.sessions import SessionRecord, load_session, save_session
from aether.gateway.dispatcher import method
from aether.gateway.handlers.agent_methods import (
    _build_engine_config,
    _build_provider_for_record,
)
from aether.gateway.handlers.state import get_current_session
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)
from aether.runtime.context import (
    CompressionLifecycleService,
    CompressionRequest,
    CompressionResult,
    DefaultContextEngine,
)
from aether.runtime.core.contracts import TurnContext
from aether.services.compact import estimate_messages_tokens


class _CompressionService(Protocol):
    def compress(self, request: CompressionRequest) -> CompressionResult: ...


_CONTEXT_STATUS: dict[str, dict[str, Any]] = {}


def context_status(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _resolve_session_id(params, where="context.status")
    record = _require_session(session_id, where="context.status")
    status = _CONTEXT_STATUS.get(session_id)
    if status is None:
        status = _default_status(record)
    return dict(status)


def context_compress(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _resolve_session_id(params, where="context.compress")
    record = _require_session(session_id, where="context.compress")
    params = params or {}
    focus = params.get("focus")
    if focus is not None and not isinstance(focus, str):
        raise GatewayError(
            "context.compress requires string or null 'focus'",
            code=ERROR_INVALID_PARAMS,
        )
    force = params.get("force", True)
    if not isinstance(force, bool):
        raise GatewayError(
            "context.compress requires boolean 'force'",
            code=ERROR_INVALID_PARAMS,
        )

    if len(record.messages) < 4:
        metadata = {
            "status": "skipped",
            "trigger_reason": "manual",
            "source_message_count": len(record.messages),
            "result_message_count": len(record.messages),
            "reason": "not_enough_context",
            "source_tokens": estimate_messages_tokens(record.messages),
            "result_tokens": estimate_messages_tokens(record.messages),
        }
        return _store_status(
            record,
            status="skipped",
            metadata=metadata,
            error=None,
        )

    _validate_record_for_compression(record)
    before_messages = copy.deepcopy(record.messages)
    source_tokens = estimate_messages_tokens(before_messages)
    context = TurnContext(session_id=session_id, iteration=0, metadata={})
    service = _build_compression_service(record)
    result = service.compress(
        CompressionRequest(
            messages=before_messages,
            context=context,
            trigger_reason="manual",
            force=force,
            focus=focus.strip() if isinstance(focus, str) and focus.strip() else None,
        )
    )
    result_tokens = estimate_messages_tokens(result.messages)
    metadata = {
        **dict(result.metadata),
        "trigger_reason": "manual",
        "source_tokens": source_tokens,
        "result_tokens": result_tokens,
    }
    if result.status == "compressed":
        record.messages = result.messages
        save_session(record)
    return _store_status(
        record,
        status=result.status,
        metadata=metadata,
        error=result.error,
    )


def _resolve_session_id(params: dict[str, Any] | None, *, where: str) -> str:
    body = params or {}
    session_id = body.get("session_id")
    if session_id is None:
        session_id = get_current_session()
    if not isinstance(session_id, str) or not session_id.strip():
        raise GatewayError(
            f"{where} requires non-empty string 'session_id'",
            code=ERROR_INVALID_PARAMS,
        )
    return session_id.strip()


def _require_session(session_id: str, *, where: str) -> SessionRecord:
    record = load_session(session_id)
    if record is None:
        raise GatewayError(
            f"session not found: {session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": session_id, "where": where},
        )
    return record


def _validate_record_for_compression(record: SessionRecord) -> None:
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


def _build_compression_service(record: SessionRecord) -> _CompressionService:
    provider = _build_provider_for_record(record)
    config = _build_engine_config(None)
    config.use_builtin_tools = False
    config.compression_enabled = True
    engine = AgentEngine(provider, config=config)
    return CompressionLifecycleService(
        context_engine=DefaultContextEngine(
            adapter=LegacyContextAssemblyAdapter(engine),
        )
    )


def _default_status(record: SessionRecord) -> dict[str, Any]:
    message_count = len(record.messages)
    tokens = estimate_messages_tokens(record.messages)
    return {
        "session_id": record.session_id,
        "context_engine": "default",
        "compression_count": 0,
        "last_compression": None,
        "message_count": message_count,
        "token_estimate": tokens,
    }


def _store_status(
    record: SessionRecord,
    *,
    status: str,
    metadata: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    previous = _CONTEXT_STATUS.get(record.session_id) or _default_status(record)
    compression_count = int(previous.get("compression_count", 0) or 0)
    if status == "compressed":
        compression_count += 1
    envelope = {
        "session_id": record.session_id,
        "context_engine": "default",
        "compression_count": compression_count,
        "last_compression": dict(metadata),
        "message_count": len(record.messages),
        "token_estimate": estimate_messages_tokens(record.messages),
        "status": status,
        "error": error,
    }
    _CONTEXT_STATUS[record.session_id] = envelope
    return dict(envelope)


def reset_context_status_for_tests() -> None:
    _CONTEXT_STATUS.clear()


def register() -> None:
    method("context.status", long=False)(context_status)
    method("context.compress", long=True)(context_compress)


__all__ = [
    "context_compress",
    "context_status",
    "register",
    "reset_context_status_for_tests",
]
