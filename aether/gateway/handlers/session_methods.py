"""``session.*`` RPC methods.

Thin wrappers around :mod:`aether.cli.sessions`.  Each handler converts
the on-disk :class:`~aether.cli.sessions.SessionRecord` into the wire
``SessionInfo`` shape (epoch floats instead of ISO strings) and
forwards every other piece of behaviour to the existing module.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from aether.cli.sessions import (
    SessionRecord,
    delete_session,
    list_sessions,
    load_session,
    save_session,
)
from aether.gateway.dispatcher import method
from aether.gateway.handlers.schemas import SessionInfo, TranscriptMessage
from aether.gateway.handlers.state import (
    get_current_session,
    set_current_session,
)
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)


# ── Conversion helpers ──────────────────────────────────────────────


def _iso_to_epoch(iso: str) -> float:
    """Tolerantly convert an ISO-8601 timestamp to epoch seconds.

    Accepts the ``Z`` suffix used by :func:`aether.cli.sessions._now_iso`
    and falls back to ``0.0`` on anything unparseable so a stale or
    malformed record still round-trips through the API.
    """
    if not iso:
        return 0.0
    text = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _to_info(record: SessionRecord) -> dict[str, Any]:
    return SessionInfo(
        session_id=record.session_id,
        created_at=_iso_to_epoch(record.created_at),
        updated_at=_iso_to_epoch(record.updated_at),
        provider=record.provider,
        model=record.model,
        base_url=record.base_url,
        system_prompt=record.system_prompt,
        message_count=len(record.messages),
        summary=record.first_user_message or None,
    ).model_dump(mode="json", exclude_none=True)


_ALLOWED_ROLES = frozenset({"user", "assistant", "system", "tool"})


def _to_transcript(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role")
    if role not in _ALLOWED_ROLES:
        # Tolerate weird historical records — they still need to
        # round-trip somewhere visible to the user, so coerce.
        role = "user"
    content = msg.get("content")
    text = content if isinstance(content, str) else None
    name = msg.get("name") if isinstance(msg.get("name"), str) else None
    tool_call_id = (
        msg.get("tool_call_id")
        if isinstance(msg.get("tool_call_id"), str)
        else None
    )
    metadata = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else None
    return TranscriptMessage(
        role=role,
        text=text,
        name=name,
        tool_call_id=tool_call_id,
        metadata=metadata,
    ).model_dump(mode="json", exclude_none=True)


def _require_str(params: dict[str, Any] | None, key: str, *, where: str) -> str:
    if not params or not isinstance(params.get(key), str) or not params[key].strip():
        raise GatewayError(
            f"{where} requires non-empty string '{key}'",
            code=ERROR_INVALID_PARAMS,
        )
    return params[key].strip()


# ── Method handlers ────────────────────────────────────────────────


def session_create(params: dict[str, Any] | None) -> dict[str, Any]:
    provider = _require_str(params, "provider", where="session.create")
    model = _require_str(params, "model", where="session.create")
    params = params or {}

    base_url = params.get("base_url") if isinstance(params.get("base_url"), str) else None
    system_prompt = params.get("system") if isinstance(params.get("system"), str) else None
    requested_session_id = params.get("session_id")
    if requested_session_id is not None:
        if not isinstance(requested_session_id, str) or not requested_session_id.strip():
            raise GatewayError(
                "session.create requires non-empty string 'session_id'",
                code=ERROR_INVALID_PARAMS,
            )
        session_id = requested_session_id.strip()
    else:
        session_id = str(uuid.uuid4())

    record = SessionRecord.new(
        session_id=session_id,
        provider=provider,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
    )
    save_session(record)
    set_current_session(record.session_id)
    return {"session_id": record.session_id, "info": _to_info(record)}


def session_list(params: dict[str, Any] | None) -> dict[str, Any]:
    records = list_sessions()
    if params and isinstance(params.get("limit"), int):
        limit = params["limit"]
        if limit > 0:
            records = records[:limit]
    return {"sessions": [_to_info(r) for r in records]}


def session_resume(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="session.resume")
    record = _resolve_session_record(session_id)
    if record is None:
        raise GatewayError(
            f"session not found: {session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": session_id},
        )
    set_current_session(session_id)
    return {
        "info": _to_info(record),
        "messages": [_to_transcript(m) for m in record.messages],
    }


def session_update(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="session.update")
    record = load_session(session_id)
    if record is None:
        raise GatewayError(
            f"session not found: {session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": session_id},
        )
    params = params or {}

    if "provider" in params:
        record.provider = _optional_non_empty_str(
            params,
            "provider",
            where="session.update",
        )
    if "model" in params:
        record.model = _optional_non_empty_str(
            params,
            "model",
            where="session.update",
        )
    if "base_url" in params:
        value = params.get("base_url")
        if value is not None and not isinstance(value, str):
            raise GatewayError(
                "session.update requires string or null 'base_url'",
                code=ERROR_INVALID_PARAMS,
            )
        record.base_url = value
    if "system" in params:
        value = params.get("system")
        if value is not None and not isinstance(value, str):
            raise GatewayError(
                "session.update requires string or null 'system'",
                code=ERROR_INVALID_PARAMS,
            )
        record.system_prompt = value

    save_session(record)
    set_current_session(record.session_id)
    return {"session_id": record.session_id, "info": _to_info(record)}


def session_delete(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_str(params, "session_id", where="session.delete")
    deleted = delete_session(session_id)
    if deleted and get_current_session() == session_id:
        set_current_session(None)
    return {"deleted": bool(deleted)}


def session_current(_params: dict[str, Any] | None) -> dict[str, Any]:
    current = get_current_session()
    if current is None:
        return {"session_id": None}
    record = load_session(current)
    if record is None:
        # In-memory binding pointed at a session that no longer exists
        # on disk (deleted out of band, AETHER_HOME swapped …).  Clear
        # the stale binding so the next call returns the truthful null.
        set_current_session(None)
        return {"session_id": None}
    return {"session_id": current, "info": _to_info(record)}


def _resolve_session_record(session_id_or_prefix: str) -> SessionRecord | None:
    """Match Python TUI resume semantics: exact id first, then unique prefix."""
    record = load_session(session_id_or_prefix)
    if record is not None:
        return record

    records = list_sessions()
    matches = [r for r in records if r.session_id.startswith(session_id_or_prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise GatewayError(
            f"ambiguous session prefix {session_id_or_prefix!r}: "
            f"matches {len(matches)} records",
            code=ERROR_APPLICATION,
            data={"session_id": session_id_or_prefix, "matches": len(matches)},
        )
    return None


def _optional_non_empty_str(params: dict[str, Any], key: str, *, where: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GatewayError(
            f"{where} requires non-empty string '{key}'",
            code=ERROR_INVALID_PARAMS,
        )
    return value.strip()


def register() -> None:
    """Register the ``session.*`` method handlers on the dispatcher.

    Called by :func:`aether.gateway.handlers.register_handler_methods`
    at gateway boot, and by tests after they wipe the registry.
    Idempotent — re-registering is allowed and overwrites in place.
    """
    method("session.create", long=False)(session_create)
    method("session.list", long=True)(session_list)
    method("session.resume", long=True)(session_resume)
    method("session.update", long=False)(session_update)
    method("session.delete", long=False)(session_delete)
    method("session.current", long=False)(session_current)


__all__ = [
    "register",
    "session_create",
    "session_current",
    "session_delete",
    "session_list",
    "session_resume",
    "session_update",
]
