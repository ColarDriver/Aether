"""Middleware pipeline runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from aether.agents.middlewares.base import EngineMiddleware
from aether.runtime.contracts import LoopState, NormalizedResponse, ToolCall, ToolResult, TurnContext


@dataclass(slots=True)
class MiddlewarePipeline:
    middlewares: List[EngineMiddleware] = field(default_factory=list)
    logger: logging.Logger | None = None

    def add(self, middleware: EngineMiddleware) -> None:
        self.middlewares.append(middleware)

    def run_before_llm(self, messages: list[dict], context: TurnContext) -> list[dict]:
        current = messages
        for middleware in self.middlewares:
            current = middleware.before_llm(current, context)
        return current

    def run_after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        current = response
        for middleware in self.middlewares:
            current = middleware.after_llm(current, context)
        return current

    def run_before_tool(self, call: ToolCall, context: TurnContext) -> ToolCall:
        current = call
        for middleware in self.middlewares:
            current = middleware.before_tool(current, context)
        return current

    def run_after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        current = result
        for middleware in self.middlewares:
            current = middleware.after_tool(current, context)
        return current

    def run_on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        for middleware in self.middlewares:
            try:
                middleware.on_error(error, state, context)
            except Exception as hook_error:  # pragma: no cover - defensive path
                if self.logger:
                    self.logger.exception("Middleware on_error failed: %s", hook_error)
