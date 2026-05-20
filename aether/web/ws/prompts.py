"""Browser prompt bridge for run approvals and tool permissions."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError
from dataclasses import asdict, is_dataclass
import itertools
from typing import Any, Callable, cast

from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionMode,
    ToolPermissionRequest,
    ToolPermissionRule,
)
from aether.web.serializers import to_jsonable

FrameSender = Callable[[dict[str, Any]], None]

_DEFAULT_PROMPT_TIMEOUT = 24 * 60 * 60.0


class WebPromptDisconnected(Exception):
    """Raised when the browser socket disappears while a prompt is pending."""


class WebPromptBroker:
    def __init__(self, *, send_frame: FrameSender) -> None:
        self._send_frame = send_frame
        self._counter = itertools.count(1)
        self._pending: dict[str, Future[dict[str, Any]]] = {}

    def request_approval(
        self,
        *,
        kind: str,
        session_id: str,
        run_id: str,
        plan_text: str | None = None,
        plan_path: str | None = None,
        questions: list[dict[str, Any]] | None = None,
        timeout: float = _DEFAULT_PROMPT_TIMEOUT,
    ) -> dict[str, Any]:
        prompt_id = self._new_prompt_id("approval")
        future: Future[dict[str, Any]] = Future()
        self._pending[prompt_id] = future
        self._send_frame(
            {
                "type": "approval.requested",
                "payload": {
                    "prompt_id": prompt_id,
                    "kind": kind,
                    "session_id": session_id,
                    "run_id": run_id,
                    "plan_text": plan_text,
                    "plan_path": plan_path,
                    "questions": questions or [],
                    "deadline_ms": int(timeout * 1000),
                },
            }
        )
        return self._wait(prompt_id, future, timeout=timeout)

    def request_permission(
        self,
        *,
        run_id: str,
        request: ToolPermissionRequest,
        timeout: float = _DEFAULT_PROMPT_TIMEOUT,
    ) -> ToolPermissionDecision:
        prompt_id = self._new_prompt_id("permission")
        future: Future[dict[str, Any]] = Future()
        self._pending[prompt_id] = future
        self._send_frame(
            {
                "type": "permission.requested",
                "payload": {
                    "prompt_id": prompt_id,
                    "run_id": run_id,
                    "request": _permission_request_to_payload(request),
                    "deadline_ms": int(timeout * 1000),
                },
            }
        )
        try:
            payload = self._wait(prompt_id, future, timeout=timeout)
        except TimeoutError:
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                feedback="permission prompt timed out",
                source="timeout",
            )
        except WebPromptDisconnected:
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.ABORT,
                feedback="browser disconnected",
                source="web",
            )
        return _decision_from_payload(payload.get("decision", payload))

    def resolve(self, prompt_id: str, payload: dict[str, Any]) -> bool:
        future = self._pending.pop(prompt_id, None)
        if future is None:
            return False
        future.set_result(payload)
        return True

    def reject_all(self, message: str = "browser disconnected") -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(WebPromptDisconnected(message))

    def _new_prompt_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._counter)}"

    def _wait(
        self,
        prompt_id: str,
        future: Future[dict[str, Any]],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        try:
            return future.result(timeout=timeout)
        finally:
            self._pending.pop(prompt_id, None)


class WebApprovalPrompter:
    def __init__(self, *, broker: WebPromptBroker, session_id: str, run_id: str) -> None:
        self._broker = broker
        self._session_id = session_id
        self._run_id = run_id

    def is_interactive(self) -> bool:
        return True

    def confirm_plan(
        self,
        plan: str,
        *,
        context: Any | None = None,
        plan_path: str | None = None,
    ) -> dict[str, Any]:
        del context
        try:
            return self._broker.request_approval(
                kind="plan",
                session_id=self._session_id,
                run_id=self._run_id,
                plan_text=plan,
                plan_path=plan_path,
            )
        except (TimeoutError, WebPromptDisconnected):
            return {"confirmed": False}

    def ask_questions(
        self,
        questions: list[dict[str, Any]],
        *,
        timeout: float | None = None,
        plan_path: str | None = None,
    ) -> dict[str, Any]:
        try:
            result = self._broker.request_approval(
                kind="questions",
                session_id=self._session_id,
                run_id=self._run_id,
                questions=[to_jsonable(question) for question in questions],
                plan_path=plan_path,
                timeout=float(timeout if timeout is not None else _DEFAULT_PROMPT_TIMEOUT),
            )
        except (TimeoutError, WebPromptDisconnected):
            return {}
        answers = result.get("answers")
        return dict(answers) if isinstance(answers, dict) else {}


class WebToolPermissionPrompter:
    def __init__(self, *, broker: WebPromptBroker, run_id: str) -> None:
        self._broker = broker
        self._run_id = run_id

    def is_interactive(self) -> bool:
        return True

    def request_tool_permission(
        self,
        request: ToolPermissionRequest,
        *,
        timeout: float | None = None,
    ) -> ToolPermissionDecision:
        return self._broker.request_permission(
            run_id=self._run_id,
            request=request,
            timeout=float(timeout if timeout is not None else _DEFAULT_PROMPT_TIMEOUT),
        )


def _permission_request_to_payload(request: ToolPermissionRequest) -> dict[str, Any]:
    return {
        "session_id": request.session_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "arguments": to_jsonable(request.arguments),
        "category": request.category,
        "risk": request.risk,
        "preview": to_jsonable(request.preview) if request.preview is not None else None,
        "reason": request.reason,
        "allow_session": request.allow_session,
    }


def _decision_from_payload(payload: Any) -> ToolPermissionDecision:
    if is_dataclass(payload):
        payload = asdict(cast(Any, payload))
    if not isinstance(payload, dict):
        payload = {}
    raw_type = payload.get("type") or payload.get("decision") or "deny"
    decision_type = _decision_type(raw_type)
    updated_arguments = payload.get("updated_arguments")
    feedback = payload.get("feedback")
    return ToolPermissionDecision(
        type=decision_type,
        updated_arguments=updated_arguments if isinstance(updated_arguments, dict) else None,
        feedback=str(feedback) if feedback is not None else None,
        rule=_rule_from_payload(payload.get("rule")),
        source="web",
    )


def _decision_type(raw: Any) -> ToolPermissionDecisionType:
    text = str(raw).strip().lower()
    aliases = {
        "allow": ToolPermissionDecisionType.ALLOW_ONCE,
        "allow_once": ToolPermissionDecisionType.ALLOW_ONCE,
        "allow_session": ToolPermissionDecisionType.ALLOW_SESSION,
        "deny": ToolPermissionDecisionType.DENY,
        "abort": ToolPermissionDecisionType.ABORT,
    }
    return aliases.get(text, ToolPermissionDecisionType.DENY)


def _rule_from_payload(payload: Any) -> ToolPermissionRule | None:
    if payload is None:
        return None
    if is_dataclass(payload):
        payload = asdict(cast(Any, payload))
    if not isinstance(payload, dict):
        return None
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return None
    try:
        behavior = ToolPermissionMode(str(payload.get("behavior", "allow")))
    except Exception:
        behavior = ToolPermissionMode.ALLOW
    return ToolPermissionRule(
        tool_name=tool_name,
        behavior=behavior,
        scope=str(payload.get("scope") or "session"),
        path_prefix=payload.get("path_prefix") if isinstance(payload.get("path_prefix"), str) else None,
        command_prefix=payload.get("command_prefix") if isinstance(payload.get("command_prefix"), str) else None,
        reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
    )


async def sleep_forever() -> None:
    await asyncio.Event().wait()


__all__ = [
    "WebApprovalPrompter",
    "WebPromptBroker",
    "WebPromptDisconnected",
    "WebToolPermissionPrompter",
]
