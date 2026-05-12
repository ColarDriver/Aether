"""Five-tier context compaction pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from aether.runtime.core.contracts import TurnContext


@dataclass(slots=True)
class CompactionContext:
    """Runtime snapshot passed through compaction tiers."""

    session_id: str
    model: str
    model_window: int
    pre_compaction_tokens: int
    target_pct: float
    trigger_reason: str
    consecutive_failures: int = 0
    tier_outcomes: list[dict[str, Any]] = field(default_factory=list)


class CompactorTier(Protocol):
    name: str

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict[str, Any]], int]:
        ...


@dataclass(slots=True)
class CompactionResult:
    compressed_messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    tiers_run: list[str]
    exhausted: bool


class CompactionPipeline:
    """Run compaction tiers in order, re-estimating after each tier."""

    def __init__(
        self,
        *,
        tiers: list[CompactorTier],
        token_estimator: Callable[[list[dict[str, Any]]], int],
        config: Any,
        logger: Any,
    ) -> None:
        # All four instance attributes are public-by-convention and
        # named consistently (no leading underscore on any of them).
        # The estimator was historically ``_estimate``; renamed in PR
        # 3.4 cleanup so callers (and ``hasattr`` checks in tests /
        # debuggers) can find it without guessing whether the field is
        # private.  Pass-through alias ``_estimate`` is *not* kept —
        # nothing outside this module reaches in for it (verified by
        # repo-wide grep at rename time).
        self.tiers = list(tiers)
        self.token_estimator = token_estimator
        self.config = config
        self.logger = logger

    def maybe_compress(
        self,
        messages: list[dict[str, Any]],
        *,
        turn_context: TurnContext,
        model: str,
        model_window: int,
        trigger_reason: str,
    ) -> CompactionResult:
        tokens_before = self.token_estimator(messages)
        if not getattr(self.config, "compression_enabled", False):
            return CompactionResult(
                compressed_messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tiers_run=[],
                exhausted=False,
            )

        target_pct = float(getattr(self.config, "compression_pre_llm_pct", 0.85))
        target_tokens = int(model_window * target_pct)
        if trigger_reason == "preflight" and tokens_before <= target_tokens:
            return CompactionResult(
                compressed_messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tiers_run=[],
                exhausted=False,
            )

        ctx = CompactionContext(
            session_id=turn_context.session_id,
            model=model,
            model_window=model_window,
            pre_compaction_tokens=tokens_before,
            target_pct=target_pct,
            trigger_reason=trigger_reason,
            consecutive_failures=int(
                turn_context.metadata.get("compaction_consecutive_failures", 0)
            ),
        )

        current = messages
        tiers_run: list[str] = []
        # Short-circuit on the per-tier threshold check is only correct
        # for ``preflight`` triggers — there the goal is "keep estimated
        # prompt under target_tokens".  When the trigger is
        # ``context_overflow`` / ``payload_too_large`` the *server* has
        # already declared the prompt too large; our local estimator is
        # only a rough proxy and may say "OK" while the server still
        # rejects.  In that case we want every tier (including the
        # expensive Tier 5) to get a chance to free more tokens before
        # surfacing ``COMPRESSION_EXHAUSTED``.
        respect_threshold_break = trigger_reason == "preflight"
        for tier in self.tiers:
            tier_before = self.token_estimator(current)
            new_messages, freed = tier.maybe_run(current, ctx, turn_context)
            tier_after = self.token_estimator(new_messages)
            outcome = {
                "name": tier.name,
                "tokens_before": tier_before,
                "tokens_after": tier_after,
                "freed_reported": freed,
                "freed_actual": max(0, tier_before - tier_after),
            }
            ctx.tier_outcomes.append(outcome)
            self.logger.info(
                "compact[%s] %s: %d -> %d tokens (-%d)",
                trigger_reason,
                tier.name,
                tier_before,
                tier_after,
                outcome["freed_actual"],
            )
            current = new_messages
            tiers_run.append(tier.name)
            if respect_threshold_break and tier_after <= target_tokens:
                break

        tokens_after = self.token_estimator(current)
        # Single canonical key for tier outcomes — matches the public
        # contract documented in
        # ``docs/sprint-3-compaction-pipeline/04_pr3_4_tier5_autocompact.md``
        # (T-C1 row).  An earlier draft of this file also wrote a
        # ``compaction_tier_outcomes`` mirror; that duplicate has been
        # dropped because it was never the canonical name and pushed
        # consumers toward picking the wrong key.
        turn_context.metadata.setdefault("tier_outcomes", []).extend(ctx.tier_outcomes)
        turn_context.metadata["compaction_last_trigger"] = trigger_reason
        turn_context.metadata["compaction_last_tokens_before"] = tokens_before
        turn_context.metadata["compaction_last_tokens_after"] = tokens_after
        turn_context.metadata["compaction_last_tiers_run"] = list(tiers_run)

        return CompactionResult(
            compressed_messages=current,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tiers_run=tiers_run,
            exhausted=tokens_after > target_tokens,
        )
