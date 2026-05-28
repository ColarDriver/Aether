"""Per-session lightweight state shared across tools within one process.

Backs the session-mode state used by ``EnterPlanModeTool`` /
``ExitPlanModeTool`` and the plan-mode write-tool gate in
:mod:`aether.tools.registry`.

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
from pathlib import Path


class SessionMode(str, Enum):
    AGENT = "agent"
    PLAN = "plan"


class SessionPermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"
    DONT_ASK = "dontAsk"


_DEFAULT_MODE: str = SessionMode.AGENT.value
_DEFAULT_PERMISSION_MODE: str = SessionPermissionMode.DEFAULT.value
_SESSION_MODE: dict[str, str] = {}
_SESSION_PERMISSION_MODE: dict[str, str] = {}
_SESSION_CWD: dict[str, str] = {}
_LOCK = threading.Lock()


__all__ = [
    "SessionMode",
    "SessionPermissionMode",
    "get_mode",
    "set_mode",
    "clear_mode",
    "get_permission_mode",
    "set_permission_mode",
    "clear_permission_mode",
    "get_cwd",
    "set_cwd",
    "clear_cwd",
    "all_sessions",
    "_DEFAULT_MODE",
    "_DEFAULT_PERMISSION_MODE",
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


def get_permission_mode(session_id: str) -> str:
    """Return the tool permission preset for ``session_id``."""
    if not session_id:
        return _DEFAULT_PERMISSION_MODE
    with _LOCK:
        return _SESSION_PERMISSION_MODE.get(session_id, _DEFAULT_PERMISSION_MODE)


def set_permission_mode(session_id: str, mode: str | SessionPermissionMode) -> None:
    """Persist the tool permission preset for ``session_id``."""
    if not session_id:
        raise ValueError("session_id is required")
    value = mode.value if isinstance(mode, SessionPermissionMode) else str(mode)
    with _LOCK:
        _SESSION_PERMISSION_MODE[session_id] = value


def clear_permission_mode(session_id: str) -> None:
    """Drop any stored permission preset for ``session_id``.  Idempotent."""
    if not session_id:
        return
    with _LOCK:
        _SESSION_PERMISSION_MODE.pop(session_id, None)


def all_sessions() -> dict[str, str]:
    """Snapshot of every session->mode pair.  For tests / observability."""
    with _LOCK:
        return dict(_SESSION_MODE)


# ---------------------------------------------------------------------------
# Session CWD tracking
# ---------------------------------------------------------------------------
#
# Mirrors open-claude-code's ``bootstrap/state.ts`` cwd state + the
# ``utils/Shell.ts`` ``pwd -P`` round-trip pattern.  ``ShellTool``
# appends ``; pwd -P > <capture>`` to every command, reads the result
# after the subprocess exits, and stores the captured path here so the
# next shell call (and the path-aware tools) default to the same
# directory.  Without this, every ``cd /workspace/foo`` would be
# silently undone by the next subprocess Popen.
#
# The stored path is the *resolved* physical path (``Path.resolve()``)
# so symlink-aware comparisons (``Path.samefile``) are easy and
# subsequent ``pwd -P`` captures don't flip-flop on the same logical
# directory.


def get_cwd(session_id: str) -> str | None:
    """Return the tracked CWD for ``session_id`` or ``None`` if unset.

    ``None`` means the caller (typically ``ShellTool._resolve_cwd``)
    should fall through to its own configured default
    (``default_cwd``) and then to the process CWD.
    """
    if not session_id:
        return None
    with _LOCK:
        return _SESSION_CWD.get(session_id)


def set_cwd(session_id: str, cwd: str | Path) -> None:
    """Persist a CWD for ``session_id``.

    Resolves to the physical path (``Path.resolve``) so symlinked
    targets compare equal to their ``pwd -P`` form on the next turn.
    Silently no-ops on empty session_id (subagent contexts where the
    parent owns the CWD).
    """
    if not session_id:
        return
    if not cwd:
        return
    resolved = str(Path(str(cwd)).expanduser().resolve())
    with _LOCK:
        _SESSION_CWD[session_id] = resolved


def clear_cwd(session_id: str) -> None:
    """Drop any tracked CWD for ``session_id``.  Idempotent."""
    if not session_id:
        return
    with _LOCK:
        _SESSION_CWD.pop(session_id, None)
