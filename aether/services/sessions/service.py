"""Session service implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
import json
import uuid
from pathlib import Path
from typing import Any

from aether.cli.sessions import (
    SessionRecord,
    delete_session,
    list_sessions,
    load_session,
    save_session,
    update_session_from_state,
)
from aether.runtime.session.plan_artifact import clear_plan
from aether.runtime.session.session_state import clear_mode, get_mode, set_mode
from aether.services.common import (
    ServiceConflictError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from aether.services.sessions.contracts import (
    SessionCreateRequest,
    SessionCurrentResult,
    SessionDeleteRequest,
    SessionExportRequest,
    SessionExportResult,
    SessionInfo,
    SessionListResult,
    SessionRenameRequest,
    SessionResumeRequest,
    SessionUpdateRequest,
    TranscriptMessage,
    TranscriptToolCall,
)

CurrentGetter = Callable[[], str | None]
CurrentSetter = Callable[[str | None], None]

_ALLOWED_ROLES = frozenset({"user", "assistant", "system", "tool"})
_LOCAL_CURRENT_SESSION: str | None = None


def _get_local_current_session() -> str | None:
    return _LOCAL_CURRENT_SESSION


def _set_local_current_session(session_id: str | None) -> None:
    global _LOCAL_CURRENT_SESSION
    _LOCAL_CURRENT_SESSION = session_id


class SessionService:
    def __init__(
        self,
        *,
        session_dir: Path | None = None,
        current_getter: CurrentGetter | None = None,
        current_setter: CurrentSetter | None = None,
    ) -> None:
        self._session_dir = session_dir
        self._current_getter = current_getter or _get_local_current_session
        self._current_setter = current_setter or _set_local_current_session

    def create(self, request: SessionCreateRequest) -> SessionInfo:
        provider = _require_non_empty(request.provider, "provider")
        model = _require_non_empty(request.model, "model")
        session_id = request.session_id.strip() if isinstance(request.session_id, str) and request.session_id.strip() else str(uuid.uuid4())
        record = SessionRecord.new(
            session_id=session_id,
            provider=provider,
            model=model,
            base_url=request.base_url,
            system_prompt=request.system_prompt,
        )
        clear_mode(record.session_id)
        clear_plan(record.session_id)
        record.mode = "agent"
        save_session(record, base=self._session_dir)
        self._current_setter(record.session_id)
        return self.info(record)

    def list(self, *, limit: int | None = None) -> SessionListResult:
        records = list_sessions(base=self._session_dir)
        if isinstance(limit, int) and limit > 0:
            records = records[:limit]
        return SessionListResult([self.info(record) for record in records])

    def resume(self, request: SessionResumeRequest) -> SessionCurrentResult:
        key = _require_non_empty(request.session_id_or_prefix, "session_id")
        record = self.resolve_record(key)
        self._current_setter(record.session_id)
        set_mode(record.session_id, getattr(record, "mode", "agent"))
        return SessionCurrentResult(
            session_id=record.session_id,
            info=self.info(record),
            messages=[message_to_transcript(message) for message in record.messages],
        )

    def current(self) -> SessionCurrentResult | None:
        current = self._current_getter()
        if current is None:
            return None
        record = load_session(current, base=self._session_dir)
        if record is None:
            self._current_setter(None)
            return None
        return SessionCurrentResult(
            session_id=record.session_id,
            info=self.info(record),
            messages=[],
        )

    def update(self, request: SessionUpdateRequest) -> SessionInfo:
        session_id = _require_non_empty(request.session_id, "session_id")
        record = load_session(session_id, base=self._session_dir)
        if record is None:
            raise ServiceNotFoundError(
                f"session not found: {session_id}",
                details={"session_id": session_id},
            )
        if request.provider is not None:
            record.provider = _require_non_empty(request.provider, "provider")
        if request.model is not None:
            record.model = _require_non_empty(request.model, "model")
        if request.update_base_url:
            record.base_url = request.base_url
        if request.update_system_prompt:
            record.system_prompt = request.system_prompt
        save_session(record, base=self._session_dir)
        self._current_setter(record.session_id)
        return self.info(record)

    def delete(self, request: SessionDeleteRequest) -> bool:
        session_id = _require_non_empty(request.session_id, "session_id")
        deleted = delete_session(session_id, base=self._session_dir)
        if deleted and self._current_getter() == session_id:
            self._current_setter(None)
        return deleted

    def rename(self, request: SessionRenameRequest) -> SessionInfo:
        session_id = _require_non_empty(request.session_id, "session_id")
        new_session_id = _require_non_empty(request.new_session_id, "new_session_id")
        record = load_session(session_id, base=self._session_dir)
        if record is None:
            raise ServiceNotFoundError(
                f"session not found: {session_id}",
                details={"session_id": session_id},
            )
        if load_session(new_session_id, base=self._session_dir) is not None:
            raise ServiceConflictError(
                f"session already exists: {new_session_id}",
                details={"session_id": new_session_id},
            )
        delete_session(session_id, base=self._session_dir)
        record.session_id = new_session_id
        save_session(record, base=self._session_dir)
        if self._current_getter() == session_id:
            self._current_setter(new_session_id)
        return self.info(record)

    def export(self, request: SessionExportRequest) -> SessionExportResult:
        record = self.resolve_record(
            _require_non_empty(request.session_id_or_prefix, "session_id")
        )
        return SessionExportResult(session_id=record.session_id, data=record.to_json())

    def transcript(self, session_id_or_prefix: str) -> list[TranscriptMessage]:
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        return [message_to_transcript(message) for message in record.messages]

    def persist_run_result(
        self,
        session_id: str,
        *,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> SessionInfo:
        record = self.resolve_record(_require_non_empty(session_id, "session_id"))
        update_session_from_state(
            record,
            messages=messages,
            provider=record.provider,
            model=record.model,
            base_url=record.base_url,
            system_prompt=system_prompt or record.system_prompt,
        )
        save_session(record, base=self._session_dir)
        return self.info(record)

    def resolve_record(self, session_id_or_prefix: str) -> SessionRecord:
        record = load_session(session_id_or_prefix, base=self._session_dir)
        if record is not None:
            return record
        records = list_sessions(base=self._session_dir)
        matches = [item for item in records if item.session_id.startswith(session_id_or_prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ServiceConflictError(
                f"ambiguous session prefix {session_id_or_prefix!r}: matches {len(matches)} records",
                details={"session_id": session_id_or_prefix, "matches": len(matches)},
            )
        raise ServiceNotFoundError(
            f"session not found: {session_id_or_prefix}",
            details={"session_id": session_id_or_prefix},
        )

    def info(self, record: SessionRecord) -> SessionInfo:
        mode = get_mode(record.session_id)
        if mode == "agent" and getattr(record, "mode", "agent") == "plan":
            mode = "plan"
        return SessionInfo(
            session_id=record.session_id,
            created_at=iso_to_epoch(record.created_at),
            updated_at=iso_to_epoch(record.updated_at),
            provider=record.provider,
            model=record.model,
            base_url=record.base_url,
            system_prompt=record.system_prompt,
            message_count=len(record.messages),
            summary=record.first_user_message or None,
            mode=mode,
        )


def iso_to_epoch(iso: str) -> float:
    if not iso:
        return 0.0
    text = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def message_to_transcript(msg: dict[str, Any]) -> TranscriptMessage:
    role = msg.get("role")
    if role not in _ALLOWED_ROLES:
        role = "user"
    content = msg.get("content")
    return TranscriptMessage(
        role=role,
        text=content if isinstance(content, str) else None,
        name=msg.get("name") if isinstance(msg.get("name"), str) else None,
        tool_call_id=msg.get("tool_call_id") if isinstance(msg.get("tool_call_id"), str) else None,
        tool_calls=extract_tool_calls(msg) if role == "assistant" else [],
        is_error=bool(msg.get("is_error")) if role == "tool" else False,
        metadata=msg.get("metadata") if isinstance(msg.get("metadata"), dict) else None,
    )


def extract_tool_calls(msg: dict[str, Any]) -> list[TranscriptToolCall]:
    raw = msg.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out: list[TranscriptToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        call_id = item.get("id")
        function = item.get("function")
        if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments_raw = function.get("arguments")
        if isinstance(arguments_raw, dict):
            arguments = arguments_raw
        elif isinstance(arguments_raw, str):
            try:
                parsed = json.loads(arguments_raw) if arguments_raw else {}
            except json.JSONDecodeError:
                arguments = {"__raw__": arguments_raw}
            else:
                arguments = parsed if isinstance(parsed, dict) else {"__raw__": arguments_raw}
        else:
            arguments = {}
        out.append(TranscriptToolCall(id=call_id, name=name, arguments=arguments))
    return out


def session_info_to_dict(info: SessionInfo) -> dict[str, Any]:
    return {key: value for key, value in asdict(info).items() if value is not None}


def transcript_message_to_dict(message: TranscriptMessage) -> dict[str, Any]:
    data = asdict(message)
    data["tool_calls"] = [asdict(call) for call in message.tool_calls]
    return data


def _require_non_empty(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceValidationError(
            f"session requires non-empty string '{field}'",
            details={"field": field},
        )
    return value.strip()


__all__ = [
    "SessionService",
    "extract_tool_calls",
    "iso_to_epoch",
    "message_to_transcript",
    "session_info_to_dict",
    "transcript_message_to_dict",
]
