"""Middleware for logging token usage from normalized provider responses."""

from __future__ import annotations

import logging
from typing import Any

from aether.agents.middlewares.common import RuntimeMiddlewareBase
from aether.runtime.contracts import NormalizedResponse, TurnContext

logger = logging.getLogger(__name__)


class TokenUsageMiddleware(RuntimeMiddlewareBase):
    """Log token usage metadata when providers return it."""

    def __init__(self, *, logger_: logging.Logger | None = None) -> None:
        super().__init__(logger=logger_ or logger)

    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        usage = self._extract_usage(response.metadata)
        if usage:
            self.logger.info(
                "LLM token usage: input=%s output=%s total=%s",
                usage.get("input_tokens", usage.get("prompt_tokens", "?")),
                usage.get("output_tokens", usage.get("completion_tokens", "?")),
                usage.get("total_tokens", "?"),
            )
        return response

    @staticmethod
    def _extract_usage(metadata: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(metadata, dict):
            return None

        usage = metadata.get("usage")
        if isinstance(usage, dict):
            return usage

        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            return token_usage

        llm_output = metadata.get("llm_output")
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage")
            if isinstance(usage, dict):
                return usage

        return None
