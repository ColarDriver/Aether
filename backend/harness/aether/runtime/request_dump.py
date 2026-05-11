"""Debug dump helper for failed provider requests."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|cookie|token|secret|password)",
    re.IGNORECASE,
)


def dump_api_request_debug(
    api_kwargs: dict[str, Any],
    *,
    model: str,
    provider: str,
    base_url: str | None,
    reason: str,
    error: Exception | None,
    dump_dir: Path,
    session_id: str | None = None,
    max_content_chars: int = 4000,
) -> Path:
    """Write a redacted JSON request snapshot and return the path."""

    dump_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    safe_reason = _safe_filename(reason or "unknown")
    safe_session = _safe_filename(session_id or "session")
    path = dump_dir / f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}_{safe_session}_{safe_reason}_{time.time_ns()}.json"
    payload = {
        "ts": now.isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "reason": reason,
        "error": str(error) if error is not None else None,
        "kwargs": redact_for_dump(api_kwargs, max_content_chars=max_content_chars),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def redact_for_dump(value: Any, *, max_content_chars: int = 4000) -> Any:
    return _redact(value, key_hint="", max_content_chars=max_content_chars)


def _redact(value: Any, *, key_hint: str, max_content_chars: int) -> Any:
    if _SENSITIVE_KEY_RE.search(key_hint):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(key): _redact(
                item,
                key_hint=str(key),
                max_content_chars=max_content_chars,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _redact(item, key_hint=key_hint, max_content_chars=max_content_chars)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _redact(item, key_hint=key_hint, max_content_chars=max_content_chars)
            for item in value
        ]
    if isinstance(value, str):
        if key_hint == "content" and len(value) > max_content_chars:
            return {
                "_truncated": True,
                "original_length": len(value),
                "preview": value[:max_content_chars],
            }
        return value
    return value


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe[:80] or "unknown"
