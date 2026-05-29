"""Durable prompt status records for the local web console."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from threading import RLock
import time
import uuid
from typing import Any, Callable

from aether.web.serializers import to_jsonable

PromptStatus = str


@dataclass(slots=True)
class WebPromptRecord:
    prompt_id: str
    run_id: str
    session_id: str
    kind: str
    frame: dict[str, Any]
    created_at: float
    expires_at: float | None
    status: PromptStatus
    process_instance_id: str
    resolution: dict[str, Any] | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "WebPromptRecord":
        return cls(
            prompt_id=_str(payload.get("prompt_id")),
            run_id=_str(payload.get("run_id")),
            session_id=_str(payload.get("session_id")),
            kind=_str(payload.get("kind")) or "prompt",
            frame=_dict(payload.get("frame")),
            created_at=_float(payload.get("created_at")) or time.time(),
            expires_at=_float(payload.get("expires_at")),
            status=_str(payload.get("status")) or "pending",
            process_instance_id=_str(payload.get("process_instance_id")),
            resolution=_dict_or_none(payload.get("resolution")),
            reason=_str(payload.get("reason")) or None,
        )


class WebPromptStore:
    """JSON-backed prompt status store.

    The provider future itself is intentionally not durable. This store only
    records enough state for the browser to distinguish an active in-process
    prompt from a stale historical prompt after backend restart.
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        process_instance_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.process_instance_id = process_instance_id or str(uuid.uuid4())
        self._path = _store_path(root)
        self._clock = clock or time.time
        self._lock = RLock()

    @property
    def path(self) -> Path:
        return self._path

    def put_pending(self, prompt_id: str, frame: dict[str, Any]) -> WebPromptRecord:
        now = self._clock()
        payload = _dict(frame.get("payload"))
        deadline_ms = _float(payload.get("deadline_ms"))
        expires_at = now + deadline_ms / 1000.0 if deadline_ms and deadline_ms > 0 else None
        record = WebPromptRecord(
            prompt_id=prompt_id,
            run_id=_str(payload.get("run_id")),
            session_id=_str(payload.get("session_id")) or _str(_dict(payload.get("request")).get("session_id")),
            kind=_prompt_kind(frame, payload),
            frame=to_jsonable(frame),
            created_at=now,
            expires_at=expires_at,
            status="pending",
            process_instance_id=self.process_instance_id,
        )
        with self._lock:
            records = self._load()
            records[prompt_id] = record
            self._save(records)
        return record

    def update_status(
        self,
        prompt_id: str,
        status: PromptStatus,
        *,
        resolution: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> WebPromptRecord | None:
        with self._lock:
            records = self._load()
            record = records.get(prompt_id)
            if record is None:
                return None
            record.status = status
            if resolution is not None:
                record.resolution = to_jsonable(resolution)
            if reason is not None:
                record.reason = reason
            records[prompt_id] = record
            self._save(records)
            return record

    def get(self, prompt_id: str) -> WebPromptRecord | None:
        with self._lock:
            return self._load().get(prompt_id)

    def terminal_replay_records(self, *, max_age_seconds: float = 24 * 60 * 60, limit: int = 50) -> list[WebPromptRecord]:
        now = self._clock()
        cutoff = now - max(0.0, max_age_seconds)
        with self._lock:
            records = [
                record
                for record in self._load().values()
                if record.status in {"stale", "expired", "disconnected"}
                and bool(record.frame)
                and record.created_at >= cutoff
            ]
        records.sort(key=lambda record: record.created_at)
        if limit > 0 and len(records) > limit:
            records = records[-limit:]
        return records

    def mark_orphaned_stale(self) -> int:
        now = self._clock()
        changed = 0
        with self._lock:
            records = self._load()
            for record in records.values():
                if record.status != "pending":
                    continue
                if record.process_instance_id == self.process_instance_id:
                    continue
                if record.expires_at is not None and record.expires_at <= now:
                    record.status = "expired"
                    record.reason = "Prompt expired while the backend process was not available."
                else:
                    record.status = "stale"
                    record.reason = "Backend restarted before this prompt was resolved."
                changed += 1
            if changed:
                self._save(records)
        return changed

    def _load(self) -> dict[str, WebPromptRecord]:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        prompts = payload.get("prompts")
        if not isinstance(prompts, dict):
            return {}
        records: dict[str, WebPromptRecord] = {}
        for prompt_id, raw in prompts.items():
            if not isinstance(prompt_id, str) or not isinstance(raw, dict):
                continue
            try:
                record = WebPromptRecord.from_json(raw)
            except (TypeError, ValueError):
                continue
            if record.prompt_id:
                records[record.prompt_id] = record
        return records

    def _save(self, records: dict[str, WebPromptRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "prompts": {
                prompt_id: record.to_json()
                for prompt_id, record in sorted(records.items())
            },
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self._path)


def _store_path(root: str | Path | None) -> Path:
    if root is not None:
        candidate = Path(root).expanduser()
        if candidate.suffix:
            return candidate
        return candidate / "web_prompts" / "prompts.json"
    return _aether_home() / "web_prompts" / "prompts.json"


def _aether_home() -> Path:
    return Path(os.getenv("AETHER_HOME", Path.home() / ".aether")).expanduser()


def _prompt_kind(frame: dict[str, Any], payload: dict[str, Any]) -> str:
    if frame.get("type") == "permission.requested":
        return "permission"
    if frame.get("type") == "approval.requested":
        return _str(payload.get("kind")) or "approval"
    return _str(frame.get("type")) or "prompt"


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


__all__ = ["WebPromptRecord", "WebPromptStore"]
