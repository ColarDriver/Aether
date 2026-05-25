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
from aether.runtime.session.session_state import SessionMode, clear_mode, get_mode, set_mode
from aether.runtime.tools.tool_result_storage import cleanup_session_spills
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
    TranscriptAttachment,
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

    def search(self, query: str, *, limit: int | None = 50) -> SessionListResult:
        needle = (query or "").strip().lower()
        records = list_sessions(base=self._session_dir)
        if needle:
            records = [record for record in records if _record_matches_query(record, needle)]
        if isinstance(limit, int) and limit > 0:
            records = records[:limit]
        return SessionListResult([self.info(record) for record in records])

    def detail(self, session_id_or_prefix: str) -> SessionCurrentResult:
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        return SessionCurrentResult(
            session_id=record.session_id,
            info=self.info(record),
            messages=[message_to_transcript(message) for message in record.messages],
        )

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
        if deleted:
            clear_mode(session_id)
            clear_plan(session_id)
            cleanup_session_spills(session_id=session_id, max_age_seconds=0)
        return deleted

    def set_session_mode(self, session_id_or_prefix: str, mode: str) -> SessionInfo:
        session_mode = _require_session_mode(mode)
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        set_mode(record.session_id, session_mode)
        record.mode = session_mode
        save_session(record, base=self._session_dir)
        return self.info(record)

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
    metadata = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else None
    text, attachments = extract_message_content(content, metadata=metadata)
    return TranscriptMessage(
        role=role,
        text=text,
        name=msg.get("name") if isinstance(msg.get("name"), str) else None,
        tool_call_id=msg.get("tool_call_id") if isinstance(msg.get("tool_call_id"), str) else None,
        tool_calls=extract_tool_calls(msg) if role == "assistant" else [],
        attachments=attachments if role == "user" else [],
        is_error=bool(msg.get("is_error")) if role == "tool" else False,
        metadata=metadata,
    )


def extract_message_content(
    content: Any,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[str | None, list[TranscriptAttachment]]:
    attachments = extract_attachments_from_metadata(metadata)
    if isinstance(content, str):
        return content, attachments
    if not isinstance(content, list):
        return None, attachments

    text_parts: list[str] = []
    content_attachments: list[TranscriptAttachment] = []
    for item in content:
        if isinstance(item, str):
            if item:
                text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        block_type = item.get("type")
        if block_type in {"text", "input_text", "output_text"}:
            text = _first_string(item.get("text"), item.get("content"))
            if text:
                text_parts.append(text)
            continue
        attachment = attachment_from_content_block(item)
        if attachment is not None:
            content_attachments.append(attachment)

    text = "\n".join(part for part in text_parts if part)
    return (text if text else None), [*attachments, *content_attachments]


def extract_attachments_from_metadata(metadata: dict[str, Any] | None) -> list[TranscriptAttachment]:
    if not metadata:
        return []
    for key in ("attachments", "displayAttachments", "display_attachments"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            return [attachment for item in raw if (attachment := attachment_from_record(item)) is not None]
    return []


def attachment_from_content_block(block: dict[str, Any]) -> TranscriptAttachment | None:
    block_type = block.get("type")
    if block_type in {"image", "input_image", "image_url"}:
        return image_attachment_from_block(block)
    if block_type in {"file", "document"}:
        return file_attachment_from_block(block)
    return None


def attachment_from_record(value: Any) -> TranscriptAttachment | None:
    if not isinstance(value, dict):
        return None
    kind = _first_string(value.get("type"), value.get("kind"))
    name = _first_string(value.get("name"), value.get("filename"))
    path = _first_string(value.get("path"), value.get("file_path"), value.get("filePath"))
    url = _first_string(value.get("url"), value.get("previewUrl"), value.get("preview_url"))
    mime_type = _first_string(value.get("mimeType"), value.get("mime_type"), value.get("media_type"))
    data = _first_string(value.get("data"))
    if not any((kind, name, path, url, mime_type, data)):
        return None
    if kind == "image":
        attachment_type = "image"
    elif kind == "text":
        attachment_type = "text"
    else:
        attachment_type = "image" if _first_string(data, url) else "file"
    return TranscriptAttachment(
        type=attachment_type,
        name=name or _name_from_path(path) or attachment_type,
        path=path,
        url=url,
        mime_type=mime_type,
        data=data,
        is_directory=bool(value.get("isDirectory") or value.get("is_directory")),
        line_start=_int_or_none(value.get("lineStart"), value.get("line_start")),
        line_end=_int_or_none(value.get("lineEnd"), value.get("line_end")),
        note=_first_string(value.get("note")),
        quote=_first_string(value.get("quote")),
    )


def image_attachment_from_block(block: dict[str, Any]) -> TranscriptAttachment:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    image_url = block.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url if isinstance(image_url, str) else None
    mime_type = _first_string(
        block.get("mimeType"),
        block.get("mime_type"),
        block.get("media_type"),
        source.get("media_type") if isinstance(source, dict) else None,
    )
    data = _first_string(
        block.get("data"),
        source.get("data") if isinstance(source, dict) else None,
        url if isinstance(url, str) and url.startswith("data:") else None,
    )
    external_url = url if isinstance(url, str) and not url.startswith("data:") else None
    return TranscriptAttachment(
        type="image",
        name=_first_string(block.get("name"), block.get("filename")) or "image",
        url=external_url,
        mime_type=mime_type,
        data=data,
    )


def file_attachment_from_block(block: dict[str, Any]) -> TranscriptAttachment:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    path = _first_string(
        block.get("path"),
        block.get("file_path"),
        block.get("filePath"),
        source.get("path") if isinstance(source, dict) else None,
    )
    mime_type = _first_string(block.get("mimeType"), block.get("mime_type"), block.get("media_type"))
    return TranscriptAttachment(
        type="file",
        name=_first_string(block.get("name"), block.get("filename")) or _name_from_path(path) or "file",
        path=path,
        mime_type=mime_type,
        data=_first_string(block.get("data"), source.get("data") if isinstance(source, dict) else None),
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


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _name_from_path(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).name or path


def _int_or_none(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _record_matches_query(record: SessionRecord, needle: str) -> bool:
    fields = [
        record.session_id,
        record.provider,
        record.model,
        record.first_user_message,
        record.system_prompt,
    ]
    if any(needle in str(value or "").lower() for value in fields):
        return True
    for message in record.messages:
        if needle in _message_search_text(message).lower():
            return True
    return False


def _message_search_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


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


def _require_session_mode(value: str | None) -> str:
    text = _require_non_empty(value, "mode")
    allowed = {mode.value for mode in SessionMode}
    if text not in allowed:
        raise ServiceValidationError(
            f"unsupported session mode: {text!r}",
            details={"mode": text, "allowed": sorted(allowed)},
        )
    return text


__all__ = [
    "SessionService",
    "extract_tool_calls",
    "iso_to_epoch",
    "message_to_transcript",
    "session_info_to_dict",
    "transcript_message_to_dict",
]
