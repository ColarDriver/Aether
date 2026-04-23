"""Shared helpers for Aether runtime middlewares."""

from __future__ import annotations

import logging
from typing import Any

from aether.agents.middlewares.base import EngineMiddleware
from aether.runtime.contracts import ToolCall, ToolResult, TurnContext


class RuntimeMiddlewareBase(EngineMiddleware):
    """Base middleware with common logging and result helpers."""

    max_error_detail_chars: int = 500

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def _error_detail(self, exc: BaseException) -> str:
        detail = str(exc).strip() or exc.__class__.__name__
        if len(detail) > self.max_error_detail_chars:
            detail = detail[: self.max_error_detail_chars - 3] + "..."
        return detail

    def _tool_error_result(
        self,
        call: ToolCall,
        *,
        code: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        payload = {"code": code}
        if metadata:
            payload.update(metadata)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=message,
            is_error=True,
            metadata=payload,
        )

    def _append_event(self, context: TurnContext, event: dict[str, Any]) -> None:
        events = context.metadata.setdefault("events", [])
        if isinstance(events, list):
            events.append(event)
