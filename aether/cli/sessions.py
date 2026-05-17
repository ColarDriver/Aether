"""Persistent session storage for the Aether REPL.

Every turn the REPL serialises the running ``ReplState`` to
``$AETHER_HOME/sessions/<session_id>.json`` (default
``~/.aether/sessions/``).  Writes are atomic — we write to a temp file
in the same directory and ``os.replace`` it into place — so a crash
mid-write can never leave a half-baked file.

The ``/resume`` slash command and the ``aether --resume`` flag both go
through this module:

    list_sessions()        → all saved records, newest first
    load_session(id)       → one record, or None
    save_session(record)   → atomic write
    update_session_from_state(...)   → refresh record from a live REPL
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SESSION_FORMAT_VERSION = 1


@dataclass(slots=True)
class SessionRecord:
    """On-disk representation of a single conversation."""

    session_id: str
    created_at: str  # ISO-8601 UTC ('Z' suffix)
    updated_at: str  # ISO-8601 UTC
    provider: str = ""
    model: str = ""
    base_url: str | None = None
    system_prompt: str | None = None
    mode: str = "agent"
    messages: list[dict[str, Any]] = field(default_factory=list)
    first_user_message: str = ""
    turn_count: int = 0
    version: int = SESSION_FORMAT_VERSION

    # ------- factories --------------------------------------------------

    @classmethod
    def new(
        cls,
        *,
        session_id: str,
        provider: str = "",
        model: str = "",
        base_url: str | None = None,
        system_prompt: str | None = None,
    ) -> "SessionRecord":
        now = _now_iso()
        return cls(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            provider=provider,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
        )

    # ------- (de)serialisation ----------------------------------------

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SessionRecord":
        # Tolerate older / partial payloads — defaults fill the gaps.
        return cls(
            session_id=str(data.get("session_id") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or data.get("created_at") or ""),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            base_url=data.get("base_url"),
            system_prompt=data.get("system_prompt"),
            mode=_normalise_mode(data.get("mode")),
            messages=list(data.get("messages") or []),
            first_user_message=str(data.get("first_user_message") or ""),
            turn_count=int(data.get("turn_count") or 0),
            version=int(data.get("version") or 0),
        )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_mode(value: Any) -> str:
    text = str(value or "agent")
    return text if text in {"agent", "plan"} else "agent"


def default_session_dir() -> Path:
    base = Path(os.getenv("AETHER_HOME", Path.home() / ".aether"))
    return base / "sessions"


def session_file(session_id: str, *, base: Path | None = None) -> Path:
    return (base or default_session_dir()) / f"{session_id}.json"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_session(record: SessionRecord, *, base: Path | None = None) -> Path:
    """Atomically write *record* to disk; return the resulting path."""
    target = session_file(record.session_id, base=base)
    target.parent.mkdir(parents=True, exist_ok=True)
    record.updated_at = _now_iso()
    payload = json.dumps(record.to_json(), ensure_ascii=False, indent=2)

    # Write to a sibling temp file then atomically rename.  This avoids
    # leaving a half-written ``<id>.json`` if the process is killed
    # mid-write.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        tmp.write(payload)
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except OSError:
            # fsync is best-effort — some filesystems (tmpfs etc.) reject it.
            pass
        tmp.close()
        os.replace(tmp.name, target)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return target


def load_session(
    session_id: str, *, base: Path | None = None
) -> SessionRecord | None:
    path = session_file(session_id, base=base)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return SessionRecord.from_json(data)


def list_sessions(*, base: Path | None = None) -> list[SessionRecord]:
    """Return every saved record sorted by ``updated_at`` (newest first)."""
    root = base or default_session_dir()
    if not root.is_dir():
        return []
    out: list[SessionRecord] = []
    for path in root.glob("*.json"):
        if path.name.startswith("."):
            continue  # skip in-flight tmp files
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rec = SessionRecord.from_json(data)
            if not rec.session_id:
                continue
            out.append(rec)
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda r: r.updated_at or "", reverse=True)
    return out


def delete_session(session_id: str, *, base: Path | None = None) -> bool:
    path = session_file(session_id, base=base)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Helpers used by the REPL / slash commands
# ---------------------------------------------------------------------------

def first_user_message(messages: list[dict[str, Any]]) -> str:
    """Pick the first user-authored utterance for preview snippets."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text", "")).strip()
    return ""


def assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    """Count how many model responses are in this conversation."""
    return sum(1 for m in messages if m.get("role") == "assistant")


def update_session_from_state(
    record: SessionRecord,
    *,
    messages: list[dict[str, Any]],
    provider: str,
    model: str,
    base_url: str | None,
    system_prompt: str | None,
) -> SessionRecord:
    """Refresh *record* in-place from a live REPL state."""
    record.provider = provider
    record.model = model
    record.base_url = base_url
    record.system_prompt = system_prompt
    record.messages = messages
    record.turn_count = assistant_turn_count(messages)
    if not record.first_user_message:
        record.first_user_message = first_user_message(messages)[:200]
    return record


# ---------------------------------------------------------------------------
# Pretty-printing for picker rows / CLI listings
# ---------------------------------------------------------------------------

def format_relative_time(iso: str) -> str:
    """Human-readable "x ago" string, falling back to the raw ISO if unparseable."""
    if not iso:
        return ""
    try:
        when = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    now = _dt.datetime.now(_dt.timezone.utc)
    delta = now - when
    secs = int(delta.total_seconds())
    if secs < 0:
        secs = 0
    if secs < 5:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86_400:
        return f"{secs // 3600}h ago"
    if secs < 86_400 * 7:
        return f"{secs // 86_400}d ago"
    return when.strftime("%Y-%m-%d %H:%M")


def format_session_preview(r: SessionRecord, *, max_chars: int = 60) -> str:
    """One-line preview snippet ('5 turns · openai/gpt-5  · "fix bug …"')."""
    preview = (r.first_user_message or "(no messages yet)").strip()
    if len(preview) > max_chars:
        preview = preview[: max_chars - 1] + "…"
    bits = [f"{r.turn_count} turn{'s' if r.turn_count != 1 else ''}"]
    if r.provider or r.model:
        bits.append(f"{r.provider}/{r.model}".strip("/"))
    bits.append(f"\u201c{preview}\u201d")
    return "  ·  ".join(bits)
