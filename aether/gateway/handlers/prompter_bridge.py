"""Gateway-backed prompter implementations for engine interaction hooks."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from aether.gateway import reverse_rpc
from aether.gateway.protocol import (
    ApprovalQuestion,
    ApprovalRequest,
    PermissionPreview,
    PermissionRequest,
    PermissionToolRequest,
)
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionMode,
    ToolPermissionPreview,
    ToolPermissionRequest,
    ToolPermissionRule,
)


class GatewayPromptDisconnected(BaseException):
    """Raised when the peer disappears while a prompt is pending."""


class GatewayPrompter:
    """Prompter protocol implementation backed by reverse RPC."""

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        request_timeout: float = 600.0,
    ) -> None:
        self.session_id = session_id
        self.run_id = run_id
        self.request_timeout = float(request_timeout)

    def is_interactive(self) -> bool:
        return True

    def confirm_plan(self, plan: str, *, context: Any | None = None) -> bool:
        timeout = self.request_timeout
        params = ApprovalRequest(
            kind="plan",
            session_id=self.session_id,
            run_id=self.run_id,
            tool_call_id=None,
            plan_text=plan,
            deadline_ms=_deadline_ms(timeout),
        ).model_dump(mode="json", exclude_none=False)
        try:
            result = reverse_rpc.call("approval.request", params, timeout=timeout)
        except OSError as exc:
            raise GatewayPromptDisconnected("peer disconnected") from exc
        return bool(result.get("confirmed"))

    def ask_questions(
        self,
        questions: list[Mapping[str, Any]],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        effective_timeout = float(timeout if timeout is not None else self.request_timeout)
        wire_questions = [_question_to_wire(q) for q in questions]
        params = ApprovalRequest(
            kind="questions",
            session_id=self.session_id,
            run_id=self.run_id,
            tool_call_id=None,
            questions=wire_questions,
            deadline_ms=_deadline_ms(effective_timeout),
        ).model_dump(mode="json", exclude_none=False)
        try:
            result = reverse_rpc.call(
                "approval.request",
                params,
                timeout=effective_timeout,
            )
        except OSError as exc:
            raise GatewayPromptDisconnected("peer disconnected") from exc
        answers = result.get("answers")
        return dict(answers) if isinstance(answers, dict) else {}


class GatewayToolPermissionPrompter:
    """ToolPermissionPrompter implementation backed by reverse RPC."""

    def __init__(
        self,
        *,
        run_id: str,
        request_timeout: float = 120.0,
    ) -> None:
        self.run_id = run_id
        self.request_timeout = float(request_timeout)

    def is_interactive(self) -> bool:
        return True

    def request_tool_permission(
        self,
        request: ToolPermissionRequest,
        *,
        timeout: float | None = None,
    ) -> ToolPermissionDecision:
        effective_timeout = float(timeout if timeout is not None else self.request_timeout)
        params = PermissionRequest(
            session_id=request.session_id,
            run_id=self.run_id,
            request=_permission_request_to_wire(request),
            deadline_ms=_deadline_ms(effective_timeout),
        ).model_dump(mode="json", exclude_none=False)
        try:
            result = reverse_rpc.call(
                "permission.request",
                params,
                timeout=effective_timeout,
            )
        except TimeoutError:
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                feedback="permission prompt timed out",
                source="timeout",
            )
        except OSError as exc:
            raise GatewayPromptDisconnected("peer disconnected") from exc
        return _decision_from_wire(result)


def _question_to_wire(question: Mapping[str, Any]) -> ApprovalQuestion:
    qid = str(question.get("id") or "")
    text = str(question.get("text") or question.get("prompt") or "")
    raw_options = question.get("options") or []
    options: list[str] = []
    if isinstance(raw_options, list):
        for item in raw_options:
            if isinstance(item, Mapping):
                value = item.get("label") or item.get("id")
                if value is not None:
                    options.append(str(value))
            elif item is not None:
                options.append(str(item))
    kind = "select" if options else "open"
    return ApprovalQuestion(id=qid, text=text, kind=kind, options=options)


def _permission_request_to_wire(
    request: ToolPermissionRequest,
) -> PermissionToolRequest:
    preview = _preview_to_wire(request.preview) if request.preview is not None else None
    return PermissionToolRequest(
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        arguments=dict(request.arguments or {}),
        category=request.category,
        risk=request.risk,
        preview=preview,
        reason=request.reason,
        allow_session=bool(request.allow_session),
    )


def _preview_to_wire(preview: ToolPermissionPreview) -> PermissionPreview:
    return PermissionPreview(
        title=preview.title,
        subtitle=preview.subtitle,
        body=preview.body,
        diff=preview.diff,
        path=preview.path,
        command=preview.command,
        metadata=dict(preview.metadata or {}),
    )


def _decision_from_wire(payload: dict[str, Any]) -> ToolPermissionDecision:
    raw_type = payload.get("type")
    try:
        decision_type = ToolPermissionDecisionType(str(raw_type))
    except Exception:
        decision_type = ToolPermissionDecisionType.DENY

    updated_arguments = payload.get("updated_arguments")
    if not isinstance(updated_arguments, dict):
        updated_arguments = None

    feedback = payload.get("feedback")
    rule = _rule_from_wire(payload.get("rule"))
    return ToolPermissionDecision(
        type=decision_type,
        updated_arguments=updated_arguments,
        feedback=str(feedback) if feedback is not None else None,
        rule=rule,
        source="gateway",
    )


def _rule_from_wire(payload: Any) -> ToolPermissionRule | None:
    if payload is None:
        return None
    if isinstance(payload, ToolPermissionRule):
        return payload
    if is_dataclass(payload):
        payload = asdict(payload)
    if not isinstance(payload, Mapping):
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
        path_prefix=(
            str(payload["path_prefix"]) if payload.get("path_prefix") is not None else None
        ),
        command_prefix=(
            str(payload["command_prefix"])
            if payload.get("command_prefix") is not None
            else None
        ),
        reason=str(payload["reason"]) if payload.get("reason") is not None else None,
    )


def _deadline_ms(timeout: float) -> int:
    return max(0, int(timeout * 1000))


__all__ = [
    "GatewayPromptDisconnected",
    "GatewayPrompter",
    "GatewayToolPermissionPrompter",
]
