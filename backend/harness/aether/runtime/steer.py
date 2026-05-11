"""Thread-safe inbox for asynchronous user steering text."""

from __future__ import annotations

from threading import RLock


class SteerInbox:
    """Store pending `/steer` text per session.

    The inbox is deliberately tiny: callers append text from UI/gateway
    threads, and the agent loop drains it at safe role-preserving boundaries.
    """

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}
        self._lock = RLock()

    def append(self, session_id: str, text: str) -> bool:
        session = (session_id or "").strip()
        cleaned = (text or "").strip()
        if not session or not cleaned:
            return False
        with self._lock:
            existing = self._pending.get(session)
            self._pending[session] = f"{existing}\n{cleaned}" if existing else cleaned
        return True

    def drain(self, session_id: str) -> str | None:
        session = (session_id or "").strip()
        if not session:
            return None
        with self._lock:
            return self._pending.pop(session, None)

    def put_back(self, session_id: str, text: str) -> None:
        session = (session_id or "").strip()
        cleaned = (text or "").strip()
        if not session or not cleaned:
            return
        with self._lock:
            existing = self._pending.get(session)
            # `text` was drained before any concurrently appended value, so it
            # must stay first to preserve user-visible ordering.
            self._pending[session] = f"{cleaned}\n{existing}" if existing else cleaned

    def clear(self, session_id: str) -> None:
        session = (session_id or "").strip()
        if not session:
            return
        with self._lock:
            self._pending.pop(session, None)
