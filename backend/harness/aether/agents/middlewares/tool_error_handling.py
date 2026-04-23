"""Tool error middleware and runtime middleware builders for Aether."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aether.agents.middlewares.base import EngineMiddleware
from aether.agents.middlewares.common import RuntimeMiddlewareBase
from aether.runtime.contracts import LoopState, ToolCall, ToolResult, TurnContext

if TYPE_CHECKING:
    from aether.guardrails.provider import GuardrailProvider

logger = logging.getLogger(__name__)


class ToolErrorHandlingMiddleware(RuntimeMiddlewareBase):
    """Convert tool execution failures into error ToolResults when possible."""

    def __init__(self, *, logger_: logging.Logger | None = None) -> None:
        super().__init__(logger=logger_ or logger)

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        if not result.is_error:
            return result

        content = result.content.strip()
        if len(content) > self.max_error_detail_chars:
            content = content[: self.max_error_detail_chars - 3] + "..."

        metadata = dict(result.metadata)
        metadata.setdefault("code", "tool.error")
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            content=content,
            is_error=True,
            metadata=metadata,
        )

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        if state != LoopState.TOOL_EXECUTE:
            return

        call = context.metadata.get("_active_tool_call")
        if not isinstance(call, ToolCall):
            self.logger.exception("Tool execution failed but active tool call context is missing", exc_info=error)
            return

        detail = self._error_detail(error)
        self.logger.exception(
            "Tool execution failed: session=%s name=%s id=%s detail=%s",
            context.session_id,
            call.name,
            call.id,
            detail,
            exc_info=error,
        )

        context.metadata["tool_error_result"] = self._tool_error_result(
            call,
            code="tool.execution_error",
            message=(
                f"Error: Tool '{call.name}' failed with {error.__class__.__name__}: {detail}. "
                "Continue with available context, or choose an alternative tool."
            ),
            metadata={"exception_type": error.__class__.__name__},
        )
        self._append_event(
            context,
            {
                "type": "tool_error",
                "tool": call.name,
                "tool_call_id": call.id,
                "exception_type": error.__class__.__name__,
                "detail": detail,
            },
        )


def _build_runtime_middlewares(
    *,
    guardrail_provider: "GuardrailProvider | None" = None,
    guardrail_fail_closed: bool = True,
    guardrail_passport: str | None = None,
) -> list[EngineMiddleware]:
    """Build the core middleware stack for AgentEngine."""
    from aether.agents.middlewares.llm_error import LLMErrorHandlingMiddleware
    from aether.agents.middlewares.token_usage import TokenUsageMiddleware

    middlewares: list[EngineMiddleware] = [
        LLMErrorHandlingMiddleware(),
        TokenUsageMiddleware(),
    ]

    if guardrail_provider is not None:
        from aether.guardrails.middleaware import GuardrailMiddleware

        middlewares.append(
            GuardrailMiddleware(
                guardrail_provider,
                fail_closed=guardrail_fail_closed,
                passport=guardrail_passport,
            )
        )

    middlewares.append(ToolErrorHandlingMiddleware())
    return middlewares


def build_lead_runtime_middlewares(
    *,
    lazy_init: bool = True,  # noqa: ARG001 - kept for compatibility with prior signature
    guardrail_provider: "GuardrailProvider | None" = None,
    guardrail_fail_closed: bool = True,
    guardrail_passport: str | None = None,
) -> list[EngineMiddleware]:
    """Build runtime middlewares for lead agent execution."""
    return _build_runtime_middlewares(
        guardrail_provider=guardrail_provider,
        guardrail_fail_closed=guardrail_fail_closed,
        guardrail_passport=guardrail_passport,
    )


def build_subagent_runtime_middlewares(
    *,
    lazy_init: bool = True,  # noqa: ARG001 - kept for compatibility with prior signature
    guardrail_provider: "GuardrailProvider | None" = None,
    guardrail_fail_closed: bool = True,
    guardrail_passport: str | None = None,
) -> list[EngineMiddleware]:
    """Build runtime middlewares for subagent execution."""
    return _build_runtime_middlewares(
        guardrail_provider=guardrail_provider,
        guardrail_fail_closed=guardrail_fail_closed,
        guardrail_passport=guardrail_passport,
    )
