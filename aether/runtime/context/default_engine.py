"""Default context engine backed by Aether's existing compaction helpers."""

from __future__ import annotations

from typing import Any, Protocol

from aether.runtime.context.engine import ContextEngineResult
from aether.runtime.core.contracts import TurnContext


class DefaultContextEngineAdapter(Protocol):
    """Legacy bridge used while compaction details still live on AgentEngine."""

    def maybe_compact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> Any | None: ...

    def apply_collapse_view(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...


class DefaultContextEngine:
    """Context engine that preserves current compaction/collapse behavior."""

    name = "default"

    def __init__(self, *, adapter: DefaultContextEngineAdapter) -> None:
        self._adapter = adapter

    def should_compress_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> bool:
        del messages, context
        # The existing compaction pipeline owns threshold decisions. Returning
        # True here preserves the old behavior where preflight always called
        # into ``_maybe_compact_messages`` exactly once per turn.
        return True

    def compact_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> ContextEngineResult:
        raw = self._adapter.maybe_compact_messages(
            messages,
            context=context,
            trigger_reason=trigger_reason,
        )
        if raw is None:
            return ContextEngineResult(
                messages=messages,
                changed=False,
                reason=trigger_reason,
                metadata={"status": "skipped"},
            )

        compressed_messages = getattr(raw, "compressed_messages", messages)
        tiers_run = list(getattr(raw, "tiers_run", []) or [])
        tokens_before = getattr(raw, "tokens_before", None)
        tokens_after = getattr(raw, "tokens_after", None)
        exhausted = bool(getattr(raw, "exhausted", False))
        changed = bool(tiers_run) or compressed_messages is not messages

        return ContextEngineResult(
            messages=compressed_messages,
            changed=changed,
            reason=trigger_reason,
            metadata={
                "status": "compressed" if changed else "skipped",
                "trigger_reason": trigger_reason,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tiers_run": tiers_run,
                "exhausted": exhausted,
            },
            raw_result=raw,
        )

    def apply_provider_projection(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._adapter.apply_collapse_view(messages, context)


__all__ = [
    "DefaultContextEngine",
    "DefaultContextEngineAdapter",
]
