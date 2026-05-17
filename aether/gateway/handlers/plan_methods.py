"""``plan.*`` RPC methods — Sprint 12 PR 12.1.

Read and write the per-session ``agent`` / ``plan`` mode that backs the
``/plan`` slash command and the engine's plan-mode reminder injection.

This module deliberately stays a thin wrapper over
:mod:`aether.runtime.session.session_state`: the session state map is
the single source of truth.  The engine pre-permission gate
(``aether.tools.registry.WRITE_TOOLS_BLOCKED_IN_PLAN``) reads the same
state, so changing mode through these RPCs immediately affects which
tools can run.

``plan.current`` returns a complete plan envelope (mode + plan
metadata).  The first version only fills the ``mode`` and reports an
empty artifact; PR 12.4 will fill ``plan_path`` / ``has_plan`` /
``plan_content`` after the plan artifact module lands.
"""

from __future__ import annotations

from typing import Any

from aether.gateway.dispatcher import method
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)
from aether.runtime.session.session_state import SessionMode, get_mode, set_mode


_SUPPORTED_MODES: frozenset[str] = frozenset({m.value for m in SessionMode})


def _require_session_id(params: dict[str, Any] | None, *, where: str) -> str:
    if not params or not isinstance(params.get("session_id"), str):
        raise GatewayError(
            f"{where} requires non-empty string 'session_id'",
            code=ERROR_INVALID_PARAMS,
        )
    value = params["session_id"].strip()
    if not value:
        raise GatewayError(
            f"{where} requires non-empty string 'session_id'",
            code=ERROR_INVALID_PARAMS,
        )
    return value


def _require_session_exists(session_id: str, *, where: str) -> None:
    """Resolve the session id against the on-disk session store.

    PR 12.1 doesn't create or modify sessions itself; we only refuse
    operations whose target session id is missing from disk so callers
    don't silently flip mode on a session that no longer exists.
    """
    # Local import — ``aether.cli.sessions`` pulls in a fair amount of
    # filesystem code we don't want to pay for at module import time.
    from aether.cli.sessions import load_session

    if load_session(session_id) is None:
        raise GatewayError(
            f"unknown session: {session_id}",
            code=ERROR_APPLICATION,
            data={"session_id": session_id},
        )


def plan_mode_get(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_session_id(params, where="plan.mode_get")
    _require_session_exists(session_id, where="plan.mode_get")
    return {"session_id": session_id, "mode": get_mode(session_id)}


def plan_mode_set(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_session_id(params, where="plan.mode_set")
    _require_session_exists(session_id, where="plan.mode_set")
    if not params or not isinstance(params.get("mode"), str):
        raise GatewayError(
            "plan.mode_set requires non-empty string 'mode'",
            code=ERROR_INVALID_PARAMS,
        )
    mode = params["mode"].strip()
    if mode not in _SUPPORTED_MODES:
        raise GatewayError(
            f"unsupported plan mode: {mode!r}; expected one of "
            f"{sorted(_SUPPORTED_MODES)}",
            code=ERROR_INVALID_PARAMS,
            data={"mode": mode},
        )
    set_mode(session_id, mode)
    _persist_session_mode(session_id, mode)
    return {"session_id": session_id, "mode": mode}


def plan_current(params: dict[str, Any] | None) -> dict[str, Any]:
    session_id = _require_session_id(params, where="plan.current")
    _require_session_exists(session_id, where="plan.current")
    mode = get_mode(session_id)
    plan_path: str | None
    has_plan: bool
    plan_content: str | None
    try:
        # PR 12.4: read the on-disk plan artifact if the module is
        # present.  We tolerate ImportError so PR 12.1 can land before
        # PR 12.4 without breaking the wire contract.
        from aether.runtime.session.plan_artifact import (  # pyright: ignore[reportMissingImports]
            get_plan_path,
            read_plan,
        )
    except ImportError:
        plan_path = None
        has_plan = False
        plan_content = None
    else:
        path = get_plan_path(session_id)
        plan_path = str(path)
        content = read_plan(session_id)
        if content is None:
            has_plan = False
            plan_content = None
        else:
            has_plan = True
            plan_content = content
    return {
        "session_id": session_id,
        "mode": mode,
        "plan_path": plan_path,
        "has_plan": has_plan,
        "plan_content": plan_content,
    }


def plan_clear(params: dict[str, Any] | None) -> dict[str, Any]:
    """Clear the current session's plan artifact and reset mode to agent."""
    session_id = _require_session_id(params, where="plan.clear")
    _require_session_exists(session_id, where="plan.clear")
    plan_path: str | None = None
    try:
        from aether.runtime.session.plan_artifact import clear_plan, get_plan_path
    except ImportError:  # pragma: no cover - PR 12.1 compatibility
        pass
    else:
        plan_path = str(get_plan_path(session_id))
        clear_plan(session_id)
    set_mode(session_id, SessionMode.AGENT)
    _persist_session_mode(session_id, SessionMode.AGENT.value)
    return {
        "session_id": session_id,
        "mode": SessionMode.AGENT.value,
        "plan_path": plan_path,
        "has_plan": False,
        "plan_content": None,
    }


def register() -> None:
    """Register the ``plan.*`` method handlers on the dispatcher.

    Idempotent — re-registering overwrites in place.
    """
    method("plan.mode_get", long=False)(plan_mode_get)
    method("plan.mode_set", long=False)(plan_mode_set)
    method("plan.current", long=False)(plan_current)
    method("plan.clear", long=False)(plan_clear)


__all__ = [
    "plan_mode_get",
    "plan_mode_set",
    "plan_current",
    "plan_clear",
    "register",
]


def _persist_session_mode(session_id: str, mode: str) -> None:
    from aether.cli.sessions import load_session, save_session

    record = load_session(session_id)
    if record is None:
        return
    record.mode = mode
    save_session(record)
