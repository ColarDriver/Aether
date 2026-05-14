"""Process-local state shared across gateway handlers.

Currently tracks the "current session" — set on the most recent
``session.create`` / ``session.resume`` call, cleared on
``session.delete`` of the same id.  The tracker is in-memory only:
gateway restarts lose the binding (callers can re-resume the session
to restore it).

A simple ``threading.Lock`` guards the value so concurrent long
handlers cannot race. In the current stdio-only setup there is only
one inbound caller at a time, but the lock is cheap and preserves a
future multi-peer path.
"""

from __future__ import annotations

import threading
from typing import Optional


_current_session_lock = threading.Lock()
_current_session_id: Optional[str] = None


def set_current_session(session_id: Optional[str]) -> None:
    """Replace the current-session tracker.  Pass ``None`` to clear."""
    global _current_session_id
    with _current_session_lock:
        _current_session_id = session_id


def get_current_session() -> Optional[str]:
    """Return the current session id, or ``None`` if nothing is bound."""
    with _current_session_lock:
        return _current_session_id


def reset_state_for_tests() -> None:
    """Drop the tracker so each test starts clean.  Test-only."""
    set_current_session(None)


__all__ = [
    "get_current_session",
    "reset_state_for_tests",
    "set_current_session",
]
