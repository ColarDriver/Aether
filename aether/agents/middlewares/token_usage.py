"""Middleware for logging token usage from normalized provider responses.

this middleware shares the canonical normalisation
path with the engine's per-turn accumulator (see
``aether.runtime.observability.usage.normalize_usage``) so log output stays consistent
with what ends up in ``EngineResult.metadata['usage']``.
"""

from __future__ import annotations

import logging
from typing import Any

from aether.agents.middlewares.common import RuntimeMiddlewareBase
from aether.runtime.core.contracts import NormalizedResponse, TurnContext
from aether.runtime.observability.usage import CanonicalUsage, normalize_usage

logger = logging.getLogger(__name__)


class TokenUsageMiddleware(RuntimeMiddlewareBase):
    """Log canonical-form token usage when providers return any."""

    def __init__(self, *, logger_: logging.Logger | None = None) -> None:
        super().__init__(logger=logger_ or logger)

    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        raw_usage = self._extract_usage(response.metadata)
        if raw_usage is None:
            return response

        # Best-effort provider hint for the parser dispatch.  The engine
        # injects ``active_provider_name`` / ``active_provider_api_mode``
        # via _accumulate_usage's caller; we honour them when present and
        # fall back to ``"openai"`` (the most permissive parser) otherwise.
        provider_name = (
            context.metadata.get("active_provider_name") or "openai"
        )
        api_mode = context.metadata.get("active_provider_api_mode") or "chat"

        usage: CanonicalUsage = normalize_usage(
            raw_usage,
            provider=str(provider_name),
            api_mode=str(api_mode),
        )
        if usage.total_tokens > 0:
            self.logger.info(
                "LLM token usage: input=%d output=%d cache_read=%d cache_write=%d "
                "reasoning=%d total=%d",
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
            )
        return response

    @staticmethod
    def _extract_usage(metadata: dict[str, Any]) -> dict[str, Any] | None:
        """Pull a raw usage dict out of a NormalizedResponse.metadata.

        Provider-side normalisation already puts the canonical raw dict
        under ``metadata["usage"]``, but alternate metadata shapes
        (LangChain-shaped callbacks, scripted test fixtures) sometimes
        use ``token_usage`` or ``llm_output.token_usage`` instead. We
        keep those probes so ad-hoc test responses still surface in the
        log.
        """
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
            inner = llm_output.get("token_usage")
            if isinstance(inner, dict):
                return inner

        return None
