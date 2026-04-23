"""Middleware hook contracts."""

from __future__ import annotations

from aether.runtime.contracts import LoopState, NormalizedResponse, ToolCall, ToolResult, TurnContext


class EngineMiddleware:
    """Synchronous chain hooks for the loop engine."""

    def before_llm(self, messages: list[dict], context: TurnContext) -> list[dict]:
        return messages

    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        return response

    def before_tool(self, call: ToolCall, context: TurnContext) -> ToolCall:
        return call

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        return result

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        return None
