"""Per-session lightweight state shared across tools within one process.

Sprint 3.5 / PR 3.5.7 introduces this module to back the *session mode*
state used by ``EnterPlanModeTool`` / ``ExitPlanModeTool`` and the
plan-mode write-tool gate in :mod:`aether.tools.registry`.

Why a module-level ``dict`` (and not ``TurnContext.metadata``)?

The mode must persist **across turns** within the same session — once
the model enters plan mode, every subsequent turn (and tool dispatch
within those turns) needs to see ``mode == "plan"`` until ``ExitPlanMode``
flips it back.  ``TurnContext`` is reconstructed each turn, so it is the
wrong owner.  An ``AgentEngine`` is shared across many sessions in some
deployments, so attribute storage on the engine would also be wrong.

A process-level mapping keyed by ``session_id`` is the simplest design
that survives across turns while remaining isolated between sessions.
The caller is responsible for cleaning up via :func:`clear_mode` when
the session ends — Aether's CLI does this implicitly by never re-using
session ids.

Thread-safety: writes are uncommon (only on plan-mode transitions), but
the dict is wrapped behind a ``threading.Lock`` so concurrent CLI repls
sharing a process behave deterministically.
"""

from __future__ import annotations

import threading
from enum import Enum


class SessionMode(str, Enum):
    AGENT = "agent"
    PLAN = "plan"


_DEFAULT_MODE: str = SessionMode.AGENT.value
_SESSION_MODE: dict[str, str] = {}
_LOCK = threading.Lock()


__all__ = [
    "SessionMode",
    "get_mode",
    "set_mode",
    "clear_mode",
    "all_sessions",
    "_DEFAULT_MODE",
]


def get_mode(session_id: str) -> str:
    """Return the mode for ``session_id`` or ``"agent"`` if unset."""
    if not session_id:
        return _DEFAULT_MODE
    with _LOCK:
        return _SESSION_MODE.get(session_id, _DEFAULT_MODE)


def set_mode(session_id: str, mode: str | SessionMode) -> None:
    """Persist ``mode`` for ``session_id``.

    Accepts the enum or its string value.  Unknown strings are stored
    verbatim — the caller is expected to validate via ``SessionMode``.
    """
    if not session_id:
        raise ValueError("session_id is required")
    value = mode.value if isinstance(mode, SessionMode) else str(mode)
    with _LOCK:
        _SESSION_MODE[session_id] = value


def clear_mode(session_id: str) -> None:
    """Drop any stored mode for ``session_id``.  Idempotent."""
    if not session_id:
        return
    with _LOCK:
        _SESSION_MODE.pop(session_id, None)


def all_sessions() -> dict[str, str]:
    """Snapshot of every session->mode pair.  For tests / observability."""
    with _LOCK:
        return dict(_SESSION_MODE)
