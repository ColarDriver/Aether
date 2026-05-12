"""Tier 5 autocompact gate and circuit breaker."""

from __future__ import annotations

from typing import Any

from aether.runtime.core.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.llm_fork import LLMForkSummarizer


class AutoCompactor:
    """Fork an LLM summariser when all Tier 5 gate conditions pass."""

    name = "tier5_autocompact"

    def __init__(
        self,
        *,
        config: Any,
        summarizer: LLMForkSummarizer,
        logger: Any,
    ) -> None:
        self.config = config
        self.summarizer = summarizer
        self.logger = logger

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict[str, Any]], int]:
        if turn_context.metadata.get("_compaction_in_progress"):
            return messages, 0
        if not getattr(self.config, "compression_enabled", False):
            return messages, 0
        if not getattr(self.config, "autocompact_enabled", True):
            return messages, 0
        if turn_context.metadata.get("collapse_owns_headroom"):
            return messages, 0

        threshold_tokens = int(
            ctx.model_window * float(getattr(self.config, "compression_autocompact_pct", 0.85))
        )
        if ctx.pre_compaction_tokens < threshold_tokens:
            return messages, 0

        max_failures = max(0, int(getattr(self.config, "compression_max_failures", 3)))
        if ctx.consecutive_failures >= max_failures:
            self.logger.warning(
                "tier5_autocompact skipped: circuit breaker tripped (%d failures)",
                ctx.consecutive_failures,
            )
            return messages, 0

        turn_context.metadata["_compaction_in_progress"] = True
        try:
            compacted = self.summarizer.summarise(
                messages,
                model=ctx.model,
                turn_context=turn_context,
            )
        except Exception as exc:  # noqa: BLE001
            # Read the *latest* failure count from turn_context.metadata
            # rather than incrementing the snapshot we took at pipeline
            # entry: today only one tier (this one) ever writes the
            # counter so the snapshot is fine, but reading-from-source
            # makes the increment correct under future tiers that might
            # also report failures (and matches how every other
            # ``compaction_*`` counter in the pipeline is bumped).
            current_failures = int(
                turn_context.metadata.get("compaction_consecutive_failures", 0)
            )
            turn_context.metadata["compaction_consecutive_failures"] = (
                current_failures + 1
            )
            self.logger.warning("tier5_autocompact failed: %s", exc)
            return messages, 0
        finally:
            turn_context.metadata.pop("_compaction_in_progress", None)

        if compacted is messages:
            return messages, 0
        turn_context.metadata["compaction_consecutive_failures"] = 0
        turn_context.metadata["tier5_summaries_generated"] = (
            int(turn_context.metadata.get("tier5_summaries_generated", 0)) + 1
        )
        return compacted, max(0, len(messages) - len(compacted))
