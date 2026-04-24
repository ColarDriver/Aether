"""Guardrail middleware for tool-call policy enforcement."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from aether.agents.middlewares.common import RuntimeMiddlewareBase
from aether.guardrails.provider import (
    GuardrailDecision,
    GuardrailProvider,
    GuardrailReason,
    GuardrailRequest,
)
from aether.runtime.contracts import ToolCall, ToolResult, TurnContext

logger = logging.getLogger(__name__)


class GuardrailMiddleware(RuntimeMiddlewareBase):
    """Evaluate tool calls against a GuardrailProvider before execution."""

    def __init__(
        self,
        provider: GuardrailProvider,
        *,
        fail_closed: bool = True,
        passport: str | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger=logger_ or logger)
        self.provider = provider
        self.fail_closed = fail_closed
        self.passport = passport

    def before_tool(
        self,
        call: ToolCall | ToolResult,
        context: TurnContext,
    ) -> ToolCall | ToolResult:
        # Respect prior middleware short-circuits (already converted to ToolResult).
        if isinstance(call, ToolResult):
            return call
        # Build a normalized policy request from the runtime tool call/context.
        request = self._build_request(call, context)
        decision = self._evaluate(request)
        # Allow path: continue normal tool dispatch unchanged.
        if decision.allow:
            return call

        # Deny path: convert policy decision into a synthetic tool error result
        # so the agent can continue with an alternative strategy.
        result = self._build_denied_result(call, decision)
        self.logger.warning(
            "Guardrail denied tool call: session=%s tool=%s policy=%s code=%s",
            context.session_id,
            call.name,
            decision.policy_id,
            decision.reasons[0].code if decision.reasons else "oap.denied",
        )
        # Emit a structured event for downstream audit/observability hooks.
        self._append_event(
            context,
            {
                "type": "guardrail_denied",
                "tool": call.name,
                "tool_call_id": call.id,
                "policy_id": decision.policy_id,
                "code": result.metadata.get("code"),
            },
        )
        return result

    def _build_request(self, call: ToolCall, context: TurnContext) -> GuardrailRequest:
        # Attach execution metadata so provider-side policy can use session/iteration scoping.
        return GuardrailRequest(
            tool_name=call.name,
            tool_input=dict(call.arguments),
            agent_id=self.passport,
            session_id=context.session_id,
            iteration=context.iteration,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        try:
            return self.provider.evaluate(request)
        except Exception as exc:
            self.logger.exception("Guardrail provider error: tool=%s", request.tool_name)
            # fail-open: preserve availability if guardrail backend is down.
            if not self.fail_closed:
                return GuardrailDecision(allow=True, policy_id="guardrail_fail_open")
            # fail-closed: block execution when policy evaluation cannot be trusted.
            return GuardrailDecision(
                allow=False,
                policy_id="guardrail_fail_closed",
                reasons=[
                    GuardrailReason(
                        code="oap.evaluator_error",
                        message=f"guardrail provider error (fail-closed): {self._error_detail(exc)}",
                    )
                ],
            )

    def _build_denied_result(self, call: ToolCall, decision: GuardrailDecision) -> ToolResult:
        # Always provide a deterministic denial reason/code to the model loop.
        reason = decision.reasons[0] if decision.reasons else GuardrailReason(
            code="oap.denied",
            message="blocked by guardrail policy",
        )
        content = (
            f"Guardrail denied: tool '{call.name}' was blocked ({reason.code}). "
            f"Reason: {reason.message}. Choose an alternative approach."
        )
        return self._tool_error_result(
            call,
            code=reason.code,
            message=content,
            metadata={"policy_id": decision.policy_id or "unknown"},
        )
