"""Session service implementation."""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import uuid
from pathlib import Path
from typing import Any

from aether.cli.sessions import (
    SessionRecord,
    assistant_turn_count,
    delete_session,
    first_user_message,
    list_sessions,
    load_session,
    save_session,
    update_session_from_state,
)
from aether.runtime.session.plan_artifact import clear_plan, read_plan, write_plan
from aether.runtime.session.session_state import (
    SessionMode,
    SessionPermissionMode,
    clear_cwd,
    clear_mode,
    clear_permission_mode,
    get_cwd,
    get_mode,
    get_permission_mode as get_runtime_permission_mode,
    set_cwd,
    set_mode,
    set_permission_mode as set_runtime_permission_mode,
)
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
    SessionForkRequest,
    SessionForkResult,
    SessionImportRequest,
    SessionImportResult,
    SessionInfo,
    SessionListResult,
    SessionMessageAction,
    SessionMessageActionsResult,
    SessionRenameRequest,
    SessionResumeRequest,
    SessionRewindRequest,
    SessionRewindResult,
    SessionTurnCheckpoint,
    SessionTurnCheckpointDiffResult,
    SessionTurnCheckpointsResult,
    SessionTurnCodeSnapshot,
    SessionTurnTarget,
    SessionUpdateRequest,
    TranscriptAttachment,
    TranscriptMessage,
    TranscriptToolCall,
)

CurrentGetter = Callable[[], str | None]
CurrentSetter = Callable[[str | None], None]

_ALLOWED_ROLES = frozenset({"user", "assistant", "system", "tool"})
_SAFE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_LOCAL_CURRENT_SESSION: str | None = None


@dataclass(slots=True)
class _CompletedTurn:
    message_index: int
    message_id: str
    user_message_index: int
    user_message_count: int
    content: str | None
    response_end_index: int


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
        clear_cwd(record.session_id)
        clear_permission_mode(record.session_id)
        clear_plan(record.session_id)
        record.mode = "agent"
        record.metadata["permission_mode"] = SessionPermissionMode.DEFAULT.value
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
            clear_cwd(session_id)
            clear_permission_mode(session_id)
            clear_plan(session_id)
            cleanup_session_spills(session_id=session_id, max_age_seconds=0)
        return deleted

    def set_session_mode(self, session_id_or_prefix: str, mode: str) -> SessionInfo:
        session_mode = _require_session_mode(mode)
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        set_mode(record.session_id, session_mode)
        record.mode = session_mode
        current_permission_mode = _record_permission_mode(record, session_mode=session_mode)
        if session_mode == SessionMode.PLAN.value:
            record.metadata["permission_mode"] = SessionPermissionMode.PLAN.value
            set_runtime_permission_mode(record.session_id, SessionPermissionMode.PLAN.value)
        elif current_permission_mode == SessionPermissionMode.PLAN.value:
            record.metadata["permission_mode"] = SessionPermissionMode.DEFAULT.value
            clear_permission_mode(record.session_id)
        save_session(record, base=self._session_dir)
        return self.info(record)

    def permission_mode(self, session_id_or_prefix: str) -> str:
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        mode = get_mode(record.session_id)
        if mode == SessionMode.AGENT.value and getattr(record, "mode", SessionMode.AGENT.value) == SessionMode.PLAN.value:
            mode = SessionMode.PLAN.value
        return _record_permission_mode(record, session_mode=mode)

    def set_permission_mode(self, session_id_or_prefix: str, permission_mode: str) -> SessionInfo:
        value = _require_permission_mode(permission_mode)
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        record.metadata["permission_mode"] = value
        set_runtime_permission_mode(record.session_id, value)
        if value == SessionPermissionMode.PLAN.value:
            set_mode(record.session_id, SessionMode.PLAN.value)
            record.mode = SessionMode.PLAN.value
        else:
            set_mode(record.session_id, SessionMode.AGENT.value)
            record.mode = SessionMode.AGENT.value
        save_session(record, base=self._session_dir)
        return self.info(record)

    def rename(self, request: SessionRenameRequest) -> SessionInfo:
        session_id = _require_non_empty(request.session_id, "session_id")
        new_session_id = _require_safe_session_id(request.new_session_id, "new_session_id")
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
        mode = get_mode(session_id)
        cwd = get_cwd(session_id)
        permission_mode = _record_permission_mode(record)
        plan = read_plan(session_id)
        delete_session(session_id, base=self._session_dir)
        record.session_id = new_session_id
        save_session(record, base=self._session_dir)
        clear_mode(session_id)
        clear_cwd(session_id)
        clear_permission_mode(session_id)
        set_mode(new_session_id, mode)
        set_runtime_permission_mode(new_session_id, permission_mode)
        if cwd is not None:
            set_cwd(new_session_id, cwd)
        clear_plan(session_id)
        if plan is not None:
            write_plan(new_session_id, plan)
        if self._current_getter() == session_id:
            self._current_setter(new_session_id)
        return self.info(record)

    def fork(self, request: SessionForkRequest) -> SessionForkResult:
        source = self.resolve_record(
            _require_non_empty(request.session_id_or_prefix, "session_id")
        )
        message_index = _resolve_message_index(
            source,
            message_index=request.message_index,
            target_user_message_id=request.target_user_message_id,
            user_message_index=request.user_message_index,
            expected_content=request.expected_content,
        )
        if not isinstance(message_index, int) or isinstance(message_index, bool):
            raise ServiceValidationError(
                "message_index must be an integer",
                details={"message_index": message_index},
            )
        if message_index < 0 or message_index >= len(source.messages):
            raise ServiceValidationError(
                "message_index is outside the session transcript",
                details={
                    "message_index": message_index,
                    "message_count": len(source.messages),
                },
            )
        fork_id = (
            request.new_session_id.strip()
            if isinstance(request.new_session_id, str) and request.new_session_id.strip()
            else str(uuid.uuid4())
        )
        if load_session(fork_id, base=self._session_dir) is not None:
            raise ServiceConflictError(
                f"session already exists: {fork_id}",
                details={"session_id": fork_id},
            )
        copied_messages = copy.deepcopy(source.messages[: message_index + 1])
        fork = SessionRecord.new(
            session_id=fork_id,
            provider=source.provider,
            model=source.model,
            base_url=source.base_url,
            system_prompt=source.system_prompt,
        )
        fork.messages = copied_messages
        fork.mode = "agent"
        fork.metadata["permission_mode"] = SessionPermissionMode.DEFAULT.value
        fork.turn_count = assistant_turn_count(copied_messages)
        fork.first_user_message = first_user_message(copied_messages)[:200]
        clear_mode(fork.session_id)
        clear_permission_mode(fork.session_id)
        clear_plan(fork.session_id)
        save_session(fork, base=self._session_dir)
        self._current_setter(fork.session_id)
        info = self.info(fork)
        return SessionForkResult(
            source_session_id=source.session_id,
            forked_from_index=message_index,
            messages_copied=len(copied_messages),
            info=info,
            messages=[message_to_transcript(message) for message in fork.messages],
        )

    def rewind(self, request: SessionRewindRequest) -> SessionRewindResult:
        record = self.resolve_record(
            _require_non_empty(request.session_id_or_prefix, "session_id")
        )
        message_index = _resolve_message_index(
            record,
            message_index=request.message_index,
            target_user_message_id=request.target_user_message_id,
            user_message_index=request.user_message_index,
            expected_content=request.expected_content,
            rewind_before_target=request.rewind_before_target,
        )
        if not isinstance(message_index, int) or isinstance(message_index, bool):
            raise ServiceValidationError(
                "message_index must be an integer",
                details={"message_index": message_index},
            )
        if message_index < -1 or message_index >= len(record.messages):
            raise ServiceValidationError(
                "message_index is outside the session transcript",
                details={
                    "message_index": message_index,
                    "message_count": len(record.messages),
                    "minimum": -1,
                },
            )
        original_count = len(record.messages)
        record.messages = copy.deepcopy(record.messages[: message_index + 1])
        record.turn_count = assistant_turn_count(record.messages)
        record.first_user_message = first_user_message(record.messages)[:200]
        save_session(record, base=self._session_dir)
        self._current_setter(record.session_id)
        info = self.info(record)
        return SessionRewindResult(
            session_id=record.session_id,
            rewound_to_index=message_index,
            messages_kept=len(record.messages),
            messages_removed=max(0, original_count - len(record.messages)),
            info=info,
            messages=[message_to_transcript(message) for message in record.messages],
        )

    def turn_checkpoints(self, session_id_or_prefix: str) -> SessionTurnCheckpointsResult:
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        checkpoints: list[SessionTurnCheckpoint] = []
        for turn in _completed_turns(record.messages):
            messages = record.messages[turn.message_index + 1 : turn.response_end_index]
            checkpoint = _workspace_checkpoint_from_messages(messages)
            files_changed = _files_changed_for_turn(messages, checkpoint)
            insertions, deletions = _diff_stats_for_turn(messages)
            checkpoint_id = _checkpoint_id(checkpoint)
            if not checkpoint_id and not files_changed and insertions == 0 and deletions == 0:
                continue
            checkpoints.append(
                SessionTurnCheckpoint(
                    target=SessionTurnTarget(
                        target_user_message_id=turn.message_id,
                        user_message_index=turn.user_message_index,
                        user_message_count=turn.user_message_count,
                        message_index=turn.message_index,
                        content=turn.content,
                    ),
                    code=SessionTurnCodeSnapshot(
                        available=True,
                        files_changed=files_changed,
                        insertions=insertions,
                        deletions=deletions,
                        checkpoint_id=checkpoint_id,
                    ),
                    work_dir=_first_string_from_mapping(checkpoint, "root", "cwd", "work_dir", "workDir"),
                    conversation={
                        "messages_removed": max(0, len(record.messages) - turn.response_end_index),
                        "removed_message_ids": [
                            _message_stable_id(message, index)
                            for index, message in enumerate(record.messages[turn.response_end_index :], start=turn.response_end_index)
                        ],
                    },
                )
            )
        return SessionTurnCheckpointsResult(session_id=record.session_id, checkpoints=checkpoints)

    def turn_checkpoint_diff(
        self,
        session_id_or_prefix: str,
        *,
        path: str,
        target_user_message_id: str | None = None,
        user_message_index: int | None = None,
    ) -> SessionTurnCheckpointDiffResult:
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        requested_path = _require_non_empty(path, "path")
        turn = _resolve_turn_target(
            record.messages,
            target_user_message_id=target_user_message_id,
            user_message_index=user_message_index,
        )
        if turn is None:
            raise ServiceNotFoundError(
                "turn checkpoint target not found",
                details={
                    "session_id": record.session_id,
                    "target_user_message_id": target_user_message_id,
                    "user_message_index": user_message_index,
                },
            )
        messages = record.messages[turn.message_index + 1 : turn.response_end_index]
        checkpoint = _workspace_checkpoint_from_messages(messages)
        diff = _diff_for_path_in_messages(messages, requested_path)
        target = SessionTurnTarget(
            target_user_message_id=turn.message_id,
            user_message_index=turn.user_message_index,
            user_message_count=turn.user_message_count,
            message_index=turn.message_index,
            content=turn.content,
        )
        work_dir = _first_string_from_mapping(checkpoint, "root", "cwd", "work_dir", "workDir")
        checkpoint_id = _checkpoint_id(checkpoint)
        if diff:
            return SessionTurnCheckpointDiffResult(
                session_id=record.session_id,
                state="ok",
                target=target,
                path=requested_path,
                diff=diff,
                work_dir=work_dir,
                checkpoint_id=checkpoint_id,
            )
        return SessionTurnCheckpointDiffResult(
            session_id=record.session_id,
            state="missing",
            target=target,
            path=requested_path,
            work_dir=work_dir,
            checkpoint_id=checkpoint_id,
            error="No diff metadata is available for this file in the selected turn.",
        )

    def message_actions(self, session_id_or_prefix: str, message_index: int) -> SessionMessageActionsResult:
        record = self.resolve_record(_require_non_empty(session_id_or_prefix, "session_id"))
        if not isinstance(message_index, int) or isinstance(message_index, bool):
            raise ServiceValidationError("message_index must be an integer", details={"message_index": message_index})
        if message_index < 0 or message_index >= len(record.messages):
            raise ServiceValidationError(
                "message_index is outside the session transcript",
                details={"message_index": message_index, "message_count": len(record.messages)},
            )
        message = record.messages[message_index]
        role = str(message.get("role") or "user")
        turn = _turn_for_message_index(record.messages, message_index)
        checkpoint_id = None
        target_id = None
        user_message_index = None
        if turn is not None:
            checkpoint_id = _checkpoint_id(_workspace_checkpoint_from_messages(record.messages[turn.message_index + 1 : turn.response_end_index]))
            target_id = turn.message_id
            user_message_index = turn.user_message_index
        actions = [
            SessionMessageAction(name="quote", supported=True, label="Quote message"),
            SessionMessageAction(
                name="fork",
                supported=True,
                label="Fork transcript",
            ),
            SessionMessageAction(
                name="rewind",
                supported=True,
                label="Rewind transcript",
                destructive=True,
            ),
            SessionMessageAction(
                name="retry",
                supported=turn is not None,
                label="Retry from prompt" if turn is not None else "Retry",
                reason=None if turn is not None else "No completed user turn owns this message.",
                checkpoint_id=checkpoint_id,
                destructive=True,
            ),
            SessionMessageAction(
                name="rewind_restore",
                supported=checkpoint_id is not None,
                label="Rewind and restore workspace",
                reason=None if checkpoint_id is not None else "No checkpoint metadata is available for this turn.",
                checkpoint_id=checkpoint_id,
                destructive=True,
            ),
            SessionMessageAction(
                name="undo_run",
                supported=turn is not None and checkpoint_id is not None,
                label="Undo run",
                reason=None if turn is not None and checkpoint_id is not None else "Undo run requires a completed checkpointed turn.",
                checkpoint_id=checkpoint_id,
                destructive=True,
            ),
        ]
        return SessionMessageActionsResult(
            session_id=record.session_id,
            message_index=message_index,
            role=role,
            target_user_message_id=target_id,
            user_message_index=user_message_index,
            actions=actions,
        )

    def export(self, request: SessionExportRequest) -> SessionExportResult:
        record = self.resolve_record(
            _require_non_empty(request.session_id_or_prefix, "session_id")
        )
        return SessionExportResult(session_id=record.session_id, data=record.to_json())

    def import_session(self, request: SessionImportRequest) -> SessionImportResult:
        if not isinstance(request.data, dict):
            raise ServiceValidationError(
                "session import requires a JSON object",
                details={"field": "data"},
            )
        record = _record_from_import_data(request.data)
        source_session_id = record.session_id or None
        target_session_id = (
            _require_safe_session_id(request.new_session_id, "new_session_id")
            if isinstance(request.new_session_id, str) and request.new_session_id.strip()
            else _safe_or_generated_session_id(record.session_id)
        )
        existing = load_session(target_session_id, base=self._session_dir)
        overwritten = existing is not None
        if overwritten and not request.overwrite:
            raise ServiceConflictError(
                f"session already exists: {target_session_id}",
                details={"session_id": target_session_id},
            )

        record.session_id = target_session_id
        _normalize_imported_record(record)
        if overwritten:
            delete_session(target_session_id, base=self._session_dir)
            clear_mode(target_session_id)
            clear_cwd(target_session_id)
            clear_permission_mode(target_session_id)
            clear_plan(target_session_id)
            cleanup_session_spills(session_id=target_session_id, max_age_seconds=0)
        save_session(record, base=self._session_dir)
        set_mode(record.session_id, record.mode)
        set_runtime_permission_mode(record.session_id, _record_permission_mode(record))
        if request.make_current:
            self._current_setter(record.session_id)
        info = self.info(record)
        return SessionImportResult(
            source_session_id=source_session_id,
            overwritten=overwritten,
            info=info,
            messages=[message_to_transcript(message) for message in record.messages],
        )

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

    def persist_context_status(self, session_id: str, status: dict[str, Any]) -> SessionInfo:
        record = self.resolve_record(_require_non_empty(session_id, "session_id"))
        record.metadata["context_status"] = dict(status)
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
            permission_mode=_record_permission_mode(record, session_mode=mode),
            cwd=get_cwd(record.session_id),
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


def _completed_turns(messages: list[dict[str, Any]]) -> list[_CompletedTurn]:
    user_message_count = sum(1 for message in messages if message.get("role") == "user")
    turns: list[_CompletedTurn] = []
    current: _CompletedTurn | None = None
    user_message_index = -1
    has_response = False
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            if current is not None and has_response:
                current.response_end_index = index
                turns.append(current)
            user_message_index += 1
            text, _attachments = extract_message_content(
                message.get("content"),
                metadata=message.get("metadata") if isinstance(message.get("metadata"), dict) else None,
            )
            current = _CompletedTurn(
                message_index=index,
                message_id=_message_stable_id(message, index),
                user_message_index=user_message_index,
                user_message_count=user_message_count,
                content=text,
                response_end_index=len(messages),
            )
            has_response = False
            continue
        if current is not None and message.get("role") in {"assistant", "tool"}:
            has_response = True
    if current is not None and has_response:
        current.response_end_index = len(messages)
        turns.append(current)
    return turns


def _turn_for_message_index(messages: list[dict[str, Any]], message_index: int) -> _CompletedTurn | None:
    for turn in _completed_turns(messages):
        if turn.message_index <= message_index < turn.response_end_index:
            return turn
    return None


def _resolve_message_index(
    record: SessionRecord,
    *,
    message_index: int | None,
    target_user_message_id: str | None,
    user_message_index: int | None,
    expected_content: str | None,
    rewind_before_target: bool = False,
) -> int:
    if message_index is not None:
        if expected_content is not None:
            _assert_expected_content(record, message_index, expected_content)
        return message_index
    target = _resolve_turn_target(
        record.messages,
        target_user_message_id=target_user_message_id,
        user_message_index=user_message_index,
    )
    if target is None:
        raise ServiceValidationError(
            "message target is required",
            details={
                "target_user_message_id": target_user_message_id,
                "user_message_index": user_message_index,
            },
        )
    if expected_content is not None and target.content != expected_content:
        raise ServiceConflictError(
            "message content changed",
            details={
                "target_user_message_id": target.message_id,
                "user_message_index": target.user_message_index,
            },
        )
    return target.message_index - 1 if rewind_before_target else target.message_index


def _resolve_turn_target(
    messages: list[dict[str, Any]],
    *,
    target_user_message_id: str | None,
    user_message_index: int | None,
) -> _CompletedTurn | None:
    turns = _completed_turns(messages)
    if isinstance(target_user_message_id, str) and target_user_message_id.strip():
        wanted = target_user_message_id.strip()
        for turn in turns:
            if turn.message_id == wanted:
                return turn
    if isinstance(user_message_index, int) and not isinstance(user_message_index, bool):
        for turn in turns:
            if turn.user_message_index == user_message_index:
                return turn
    return None


def _assert_expected_content(record: SessionRecord, message_index: int, expected_content: str) -> None:
    if message_index < 0 or message_index >= len(record.messages):
        return
    message = record.messages[message_index]
    if message.get("role") != "user":
        return
    text, _attachments = extract_message_content(
        message.get("content"),
        metadata=message.get("metadata") if isinstance(message.get("metadata"), dict) else None,
    )
    if text != expected_content:
        raise ServiceConflictError(
            "message content changed",
            details={"message_index": message_index},
        )


def _message_stable_id(message: dict[str, Any], index: int) -> str:
    for value in (
        message.get("id"),
        message.get("message_id"),
        message.get("messageId"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        for key in ("id", "message_id", "messageId", "uuid"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"message-{index}"


def _workspace_checkpoint_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        metadata = message.get("metadata")
        if not isinstance(metadata, dict):
            continue
        checkpoint = _dict_from_unknown(metadata.get("workspace_checkpoint"))
        if checkpoint:
            return checkpoint
        turn = metadata.get("turn")
        if isinstance(turn, dict):
            checkpoint = _dict_from_unknown(turn.get("workspace_checkpoint"))
            if checkpoint:
                return checkpoint
    return {}


def _checkpoint_id(checkpoint: dict[str, Any]) -> str | None:
    value = checkpoint.get("checkpoint_id") or checkpoint.get("checkpointId")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _files_changed_for_turn(messages: list[dict[str, Any]], checkpoint: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in checkpoint.get("files") if isinstance(checkpoint.get("files"), list) else []:
        if isinstance(item, dict):
            path = item.get("path")
            if isinstance(path, str) and path.strip():
                paths.append(path.strip())
    for message in messages:
        paths.extend(_paths_from_message(message))
    return _unique_strings(paths)


def _paths_from_message(message: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        for key in ("path", "file_path", "filePath"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
        for key in ("edited_paths", "editedPaths", "paths", "files_changed", "filesChanged"):
            value = metadata.get(key)
            if isinstance(value, list):
                paths.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    for call in extract_tool_calls(message):
        for key in ("path", "file_path", "filePath"):
            value = call.arguments.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
    return paths


def _diff_for_path_in_messages(messages: list[dict[str, Any]], requested_path: str) -> str | None:
    tool_call_paths: dict[str, list[str]] = {}
    for message in messages:
        for call in extract_tool_calls(message):
            tool_call_paths[call.id] = _paths_from_mapping(call.arguments)

    diffs: list[str] = []
    for message in messages:
        candidate_paths = _paths_from_message(message)
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str):
            candidate_paths.extend(tool_call_paths.get(tool_call_id, []))
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path_matches = any(_paths_match(candidate, requested_path) for candidate in candidate_paths)

        diff = _diff_from_mapping(metadata) if isinstance(metadata, dict) else None
        if diff and (path_matches or _diff_mentions_path(diff, requested_path)):
            diffs.append(diff)
            continue

        text, _attachments = extract_message_content(
            message.get("content"),
            metadata=metadata if isinstance(metadata, dict) else None,
        )
        if text and _looks_like_unified_diff(text) and (path_matches or _diff_mentions_path(text, requested_path)):
            diffs.append(text)

    if not diffs:
        return None
    return "\n\n".join(_unique_preserve_strings(diffs))


def _paths_from_mapping(mapping: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("path", "file_path", "filePath", "old_path", "oldPath"):
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    for key in ("edited_paths", "editedPaths", "paths", "files_changed", "filesChanged"):
        value = mapping.get(key)
        if isinstance(value, list):
            paths.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    return paths


def _diff_from_mapping(mapping: dict[str, Any]) -> str | None:
    for key in ("diff", "unified_diff", "unifiedDiff", "patch", "content_diff", "contentDiff"):
        value = mapping.get(key)
        if isinstance(value, str) and _looks_like_unified_diff(value):
            return value
    for key in ("result", "data", "metadata"):
        nested = mapping.get(key)
        if isinstance(nested, dict):
            diff = _diff_from_mapping(nested)
            if diff:
                return diff
    return None


def _looks_like_unified_diff(value: str) -> bool:
    if "@@" not in value:
        return False
    return any(line.startswith(("--- ", "+++ ", "diff --git ")) for line in value.splitlines())


def _diff_mentions_path(diff: str, requested_path: str) -> bool:
    wanted = _normalize_path_for_compare(requested_path)
    if not wanted:
        return False
    for line in diff.splitlines():
        if not line.startswith(("--- ", "+++ ", "diff --git ")):
            continue
        for token in line.split():
            cleaned = token
            if cleaned in {"---", "+++", "diff", "--git", "a/", "b/"}:
                continue
            cleaned = cleaned.removeprefix("a/").removeprefix("b/")
            if _paths_match(cleaned, wanted):
                return True
    return False


def _paths_match(candidate: str, requested: str) -> bool:
    left = _normalize_path_for_compare(candidate)
    right = _normalize_path_for_compare(requested)
    if not left or not right:
        return False
    if left == right:
        return True
    return left.endswith("/" + right) or right.endswith("/" + left)


def _normalize_path_for_compare(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return ""
    normalized = re.sub(r"^[ab]/", "", normalized)
    return normalized.rstrip("/")


def _unique_preserve_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        marker = value.strip()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def _diff_stats_for_turn(messages: list[dict[str, Any]]) -> tuple[int, int]:
    insertions = 0
    deletions = 0
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if "@@" not in text:
            continue
        for line in text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                insertions += 1
            elif line.startswith("-"):
                deletions += 1
    return insertions, deletions


def _first_string_from_mapping(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dict_from_unknown(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
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


def _record_from_import_data(data: dict[str, Any]) -> SessionRecord:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    if not isinstance(payload, dict):
        raise ServiceValidationError(
            "session import requires a JSON object",
            details={"field": "data"},
        )
    try:
        return SessionRecord.from_json(payload)
    except (TypeError, ValueError) as exc:
        raise ServiceValidationError(
            "session import payload is not a valid Aether session record",
            details={"error": str(exc)},
        ) from exc


def _normalize_imported_record(record: SessionRecord) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    record.created_at = record.created_at or now
    record.updated_at = record.updated_at or record.created_at
    record.provider = _require_non_empty(record.provider, "provider")
    record.model = _require_non_empty(record.model, "model")
    record.mode = _require_session_mode(record.mode)
    record.messages = [dict(message) for message in record.messages if isinstance(message, dict)]
    record.turn_count = assistant_turn_count(record.messages)
    record.first_user_message = first_user_message(record.messages)[:200]
    if not isinstance(record.metadata, dict):
        record.metadata = {}


def _safe_or_generated_session_id(value: str | None) -> str:
    if isinstance(value, str) and value.strip():
        try:
            return _require_safe_session_id(value, "session_id")
        except ServiceValidationError:
            pass
    return str(uuid.uuid4())


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


def _require_safe_session_id(value: str | None, field: str) -> str:
    text = _require_non_empty(value, field)
    if not _SAFE_SESSION_ID_RE.fullmatch(text):
        raise ServiceValidationError(
            f"session id contains unsafe characters: {text!r}",
            details={"field": field},
        )
    return text


def _require_session_mode(value: str | None) -> str:
    text = _require_non_empty(value, "mode")
    allowed = {mode.value for mode in SessionMode}
    if text not in allowed:
        raise ServiceValidationError(
            f"unsupported session mode: {text!r}",
            details={"mode": text, "allowed": sorted(allowed)},
        )
    return text


def _require_permission_mode(value: str | None) -> str:
    text = _require_non_empty(value, "permission_mode")
    allowed = {mode.value for mode in SessionPermissionMode}
    if text not in allowed:
        raise ServiceValidationError(
            f"unsupported permission mode: {text!r}",
            details={"permission_mode": text, "allowed": sorted(allowed)},
        )
    return text


def _record_permission_mode(record: SessionRecord, *, session_mode: str | None = None) -> str:
    if session_mode == SessionMode.PLAN.value:
        return SessionPermissionMode.PLAN.value
    live = get_runtime_permission_mode(record.session_id)
    if live != SessionPermissionMode.DEFAULT.value:
        return live
    metadata = getattr(record, "metadata", {})
    stored = metadata.get("permission_mode") if isinstance(metadata, dict) else None
    if isinstance(stored, str) and stored in {mode.value for mode in SessionPermissionMode}:
        if stored != SessionPermissionMode.DEFAULT.value:
            set_runtime_permission_mode(record.session_id, stored)
        return stored
    return SessionPermissionMode.DEFAULT.value


__all__ = [
    "SessionService",
    "extract_tool_calls",
    "iso_to_epoch",
    "message_to_transcript",
    "session_info_to_dict",
    "transcript_message_to_dict",
]
