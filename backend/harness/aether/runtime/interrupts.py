"""Thread-safe interrupt controller."""

from __future__ import annotations

from threading import RLock
from typing import Dict, Optional


class InterruptController:
    """Session-scoped interrupt flags."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._reasons: Dict[str, Optional[str]] = {}

    def request(self, session_id: str, reason: str | None = None) -> None:
        with self._lock:
            self._reasons[session_id] = reason

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._reasons.pop(session_id, None)

    def is_interrupted(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._reasons

    def reason(self, session_id: str) -> str | None:
        with self._lock:
            return self._reasons.get(session_id)
