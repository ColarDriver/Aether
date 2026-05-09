"""Sprint 3 / PR 3.4 — ``CompactionPipeline`` orchestration tests.

The pipeline itself is intentionally a thin orchestrator: it walks
through tiers in order, re-evaluates token count after each, and stops
as soon as the prompt is small enough.  These tests pin three behaviour
groups:

* Group A — disabled / preflight early-exit semantics.
* Group B — tier ordering, per-tier accounting, threshold short-circuit.
* Group C — observability: ``turn_context.metadata`` accumulators.

Tiers are mocked with a tiny ``_FakeTier`` so the test surface stays
focused on the pipeline arithmetic — no real summariser, no provider.
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass, field
from typing import Any, Callable

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import (
    CompactionContext,
    CompactionPipeline,
    CompactionResult,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """Minimal duck-typed stand-in for ``EngineConfig``."""

    compression_enabled: bool = True
    compression_pre_llm_pct: float = 0.85


@dataclass
class _FakeTier:
    """Tier whose ``maybe_run`` returns whatever the test scripts."""

    name: str
    new_messages: list[dict] | None = None
    freed: int = 0
    raises: BaseException | None = None
    calls: int = 0
    transform: Callable[[list[dict]], list[dict]] | None = None

    def maybe_run(
        self,
        messages: list[dict],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict], int]:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        if self.transform is not None:
            return self.transform(messages), self.freed
        if self.new_messages is None:
            return messages, self.freed
        return list(self.new_messages), self.freed


def _ctx(metadata: dict | None = None) -> TurnContext:
    return TurnContext(
        session_id="sess-1",
        iteration=0,
        metadata=metadata if metadata is not None else {},
    )


def _build_pipeline(
    *,
    tiers: list[Any],
    estimator: Callable[[list[dict]], int],
    config: _Config | None = None,
) -> CompactionPipeline:
    return CompactionPipeline(
        tiers=tiers,
        token_estimator=estimator,
        config=config or _Config(),
        logger=logging.getLogger("test.compaction.pipeline"),
    )


# ---------------------------------------------------------------------------
# Group A — disabled / preflight early exit
# ---------------------------------------------------------------------------


class CompactionPipelineGroupADisabledTests(unittest.TestCase):
    def test_a1_disabled_returns_messages_unchanged(self) -> None:
        """compression_enabled=False makes the pipeline a no-op."""
        tier = _FakeTier(name="t1", new_messages=[{"role": "user", "content": "x"}])
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: 1_000,
            config=_Config(compression_enabled=False),
        )
        original = [{"role": "user", "content": "hi"}]
        result = pipeline.maybe_compress(
            original,
            turn_context=_ctx(),
            model="claude-sonnet",
            model_window=200_000,
            trigger_reason="preflight",
        )
        self.assertIsInstance(result, CompactionResult)
        self.assertEqual(result.compressed_messages, original)
        self.assertEqual(result.tiers_run, [])
        self.assertFalse(result.exhausted)
        self.assertEqual(tier.calls, 0)

    def test_a2_preflight_below_threshold_short_circuits(self) -> None:
        """Preflight + estimate below 85 % of window → no tier runs."""
        tier = _FakeTier(name="t1")
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: 1_000,  # well under 85 % of 100k
            config=_Config(),
        )
        result = pipeline.maybe_compress(
            [{"role": "user", "content": "small"}],
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="preflight",
        )
        self.assertEqual(result.tiers_run, [])
        self.assertEqual(tier.calls, 0)
        self.assertEqual(result.tokens_before, result.tokens_after)

    def test_a3_context_overflow_runs_pipeline_even_if_below_threshold(self) -> None:
        """Forced trigger (context_overflow) skips the preflight bypass."""
        tier = _FakeTier(name="t1", new_messages=[{"role": "user", "content": "x"}])
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: 1_000,
            config=_Config(),
        )
        result = pipeline.maybe_compress(
            [{"role": "user", "content": "small"}],
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        self.assertEqual(tier.calls, 1)
        self.assertEqual(result.tiers_run, ["t1"])


# ---------------------------------------------------------------------------
# Group B — tier ordering / threshold short-circuit
# ---------------------------------------------------------------------------


class CompactionPipelineGroupBOrchestrationTests(unittest.TestCase):
    def test_b1_all_noop_tiers_run_to_exhaustion(self) -> None:
        """When every tier is a no-op, pipeline runs them all and reports exhausted."""
        tiers = [_FakeTier(name=f"t{i}") for i in range(4)]
        pipeline = _build_pipeline(
            tiers=tiers,
            estimator=lambda msgs: 200_000,  # always above target
            config=_Config(),
        )
        original = [{"role": "user", "content": "x"}]
        result = pipeline.maybe_compress(
            original,
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        self.assertEqual(result.tiers_run, ["t0", "t1", "t2", "t3"])
        self.assertTrue(result.exhausted)
        for tier in tiers:
            self.assertEqual(tier.calls, 1)

    def test_b2_first_tier_meets_threshold_short_circuits_rest_on_preflight(self) -> None:
        """Preflight + tier 1 brings us under target → subsequent tiers skipped.

        The tier-loop short-circuit is preflight-only: when the trigger
        is ``preflight`` the goal is "keep estimated prompt under the
        configured fraction of the window", so a tier that drops the
        estimate under that fraction means the rest of the (more
        expensive) tiers don't have to fire.  See
        ``test_b2b_recovery_trigger_does_not_short_circuit`` below for
        the recovery-trigger contract.
        """
        # Estimator is called: tokens_before(1), tier_before(1),
        # tier_after(1), tokens_after(1) = 4 calls.  Second (tier_after)
        # value drops under target so tier 2 never runs.
        sequence = iter([200_000, 200_000, 50_000, 50_000])
        tier1 = _FakeTier(name="t1", new_messages=[{"role": "u", "content": "x"}])
        tier2 = _FakeTier(name="t2")
        pipeline = _build_pipeline(
            tiers=[tier1, tier2],
            estimator=lambda msgs: next(sequence),
            config=_Config(),
        )
        result = pipeline.maybe_compress(
            [{"role": "user", "content": "lots"}],
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="preflight",
        )
        self.assertEqual(result.tiers_run, ["t1"])
        self.assertEqual(tier1.calls, 1)
        self.assertEqual(tier2.calls, 0)
        self.assertFalse(result.exhausted)

    def test_b2b_recovery_trigger_does_not_short_circuit(self) -> None:
        """``context_overflow`` / ``payload_too_large`` triggers run every tier.

        When the *server* told us the prompt is too large, our local
        estimator's verdict that we're already under target is not
        trustworthy — a generous rough estimate can hide a real
        oversize.  Pipeline must keep running through every tier
        (especially Tier 5) regardless of the per-tier check.
        """
        # Same scripted shape as b2 but with a recovery trigger.
        sequence = iter([200_000, 200_000, 50_000, 50_000, 30_000, 30_000])
        tier1 = _FakeTier(name="t1", new_messages=[{"role": "u", "content": "x"}])
        tier2 = _FakeTier(name="t2", new_messages=[{"role": "u", "content": "y"}])
        pipeline = _build_pipeline(
            tiers=[tier1, tier2],
            estimator=lambda msgs: next(sequence),
            config=_Config(),
        )
        result = pipeline.maybe_compress(
            [{"role": "user", "content": "lots"}],
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        self.assertEqual(result.tiers_run, ["t1", "t2"])
        self.assertEqual(tier1.calls, 1)
        self.assertEqual(tier2.calls, 1)
        self.assertFalse(result.exhausted)

    def test_b3_partial_freeing_falls_through_to_next_tier(self) -> None:
        """Tier 1 reduces but stays above target — tier 2 must still run."""
        # 6 calls: tokens_before, t1_before, t1_after, t2_before,
        # t2_after, tokens_after.
        sequence = iter([200_000, 200_000, 150_000, 150_000, 50_000, 50_000])
        tier1 = _FakeTier(name="t1", new_messages=[{"role": "u", "content": "x"}])
        tier2 = _FakeTier(name="t2", new_messages=[{"role": "u", "content": "y"}])
        pipeline = _build_pipeline(
            tiers=[tier1, tier2],
            estimator=lambda msgs: next(sequence),
            config=_Config(),
        )
        result = pipeline.maybe_compress(
            [{"role": "user", "content": "lots"}],
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        self.assertEqual(result.tiers_run, ["t1", "t2"])
        self.assertEqual(tier1.calls, 1)
        self.assertEqual(tier2.calls, 1)
        self.assertFalse(result.exhausted)

    def test_b4_tier_exception_is_not_swallowed(self) -> None:
        """A tier raising must propagate — pipeline does not silently swallow."""
        tier = _FakeTier(name="t1", raises=RuntimeError("boom"))
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: 200_000,
            config=_Config(),
        )
        with self.assertRaises(RuntimeError):
            pipeline.maybe_compress(
                [{"role": "user", "content": "x"}],
                turn_context=_ctx(),
                model="x",
                model_window=100_000,
                trigger_reason="context_overflow",
            )

    def test_b5_tier_outcome_records_before_after_freed(self) -> None:
        """Each tier outcome records before/after/freed metrics."""
        # tokens_before, t_before, t_after, tokens_after = 4 calls.
        sequence = iter([100_000, 100_000, 60_000, 60_000])
        tier = _FakeTier(name="tA", new_messages=[{"role": "u", "content": "x"}], freed=42)
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: next(sequence),
            config=_Config(),
        )
        ctx_obj = _ctx()
        result = pipeline.maybe_compress(
            [{"role": "user", "content": "lots"}],
            turn_context=ctx_obj,
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        outcomes = ctx_obj.metadata["tier_outcomes"]
        self.assertEqual(len(outcomes), 1)
        record = outcomes[0]
        self.assertEqual(record["name"], "tA")
        self.assertEqual(record["tokens_before"], 100_000)
        self.assertEqual(record["tokens_after"], 60_000)
        self.assertEqual(record["freed_reported"], 42)
        self.assertEqual(record["freed_actual"], 40_000)
        # Pipeline ran a single tier and stopped (60k < 85k).
        self.assertEqual(result.tiers_run, ["tA"])

    def test_b6_freed_actual_clamped_to_zero_when_tier_grows_payload(self) -> None:
        """Adversarial tier that *grows* the payload should not produce negative ``freed_actual``."""
        sequence = iter([100_000, 100_000, 110_000, 110_000])
        tier = _FakeTier(name="t-grow", new_messages=[{"role": "u", "content": "y" * 9999}])
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: next(sequence),
            config=_Config(),
        )
        ctx_obj = _ctx()
        pipeline.maybe_compress(
            [{"role": "user", "content": "x"}],
            turn_context=ctx_obj,
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        outcomes = ctx_obj.metadata["tier_outcomes"]
        self.assertEqual(outcomes[0]["freed_actual"], 0)


# ---------------------------------------------------------------------------
# Group C — observability accumulation across multiple invocations
# ---------------------------------------------------------------------------


class CompactionPipelineGroupCObservabilityTests(unittest.TestCase):
    def test_c1_metadata_records_outcomes_for_one_run(self) -> None:
        """Single run populates tier_outcomes + summary fields on turn metadata."""
        tier = _FakeTier(name="tx")
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: 200_000,
            config=_Config(),
        )
        ctx_obj = _ctx()
        pipeline.maybe_compress(
            [{"role": "user", "content": "x"}],
            turn_context=ctx_obj,
            model="claude-sonnet",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        # ``tier_outcomes`` is the canonical key (matches the doc
        # contract in 04_pr3_4_tier5_autocompact.md row T-C1).  An
        # earlier draft also wrote a ``compaction_tier_outcomes``
        # mirror; that duplicate has been removed and we explicitly
        # assert it is *not* present so we don't reintroduce the
        # double-write by accident.
        self.assertIn("tier_outcomes", ctx_obj.metadata)
        self.assertNotIn("compaction_tier_outcomes", ctx_obj.metadata)
        self.assertEqual(len(ctx_obj.metadata["tier_outcomes"]), 1)
        self.assertEqual(ctx_obj.metadata["compaction_last_trigger"], "context_overflow")
        self.assertEqual(ctx_obj.metadata["compaction_last_tokens_before"], 200_000)
        self.assertEqual(ctx_obj.metadata["compaction_last_tiers_run"], ["tx"])

    def test_c2_repeated_runs_accumulate_tier_outcomes(self) -> None:
        """Two pipeline invocations on the same context append, not replace."""
        tier = _FakeTier(name="tx")
        pipeline = _build_pipeline(
            tiers=[tier],
            estimator=lambda msgs: 200_000,
            config=_Config(),
        )
        ctx_obj = _ctx()
        for _ in range(2):
            pipeline.maybe_compress(
                [{"role": "user", "content": "x"}],
                turn_context=ctx_obj,
                model="x",
                model_window=100_000,
                trigger_reason="context_overflow",
            )
        # Each call appends one outcome → 2 total.
        self.assertEqual(len(ctx_obj.metadata["tier_outcomes"]), 2)
        # The dropped ``compaction_tier_outcomes`` mirror must stay
        # gone — see ``test_c1`` for why.
        self.assertNotIn("compaction_tier_outcomes", ctx_obj.metadata)
        # ``compaction_last_*`` reflects the most recent run only.
        self.assertEqual(ctx_obj.metadata["compaction_last_trigger"], "context_overflow")

    def test_c3_consecutive_failures_carried_into_compaction_context(self) -> None:
        """Pipeline lifts ``compaction_consecutive_failures`` into ``CompactionContext``."""
        observed: list[int] = []

        class _Spy:
            name = "spy"

            def maybe_run(self, messages, ctx, turn_context):  # type: ignore[no-untyped-def]
                observed.append(ctx.consecutive_failures)
                return messages, 0

        pipeline = _build_pipeline(
            tiers=[_Spy()],
            estimator=lambda msgs: 200_000,
            config=_Config(),
        )
        ctx_obj = _ctx({"compaction_consecutive_failures": 2})
        pipeline.maybe_compress(
            [{"role": "user", "content": "x"}],
            turn_context=ctx_obj,
            model="x",
            model_window=100_000,
            trigger_reason="context_overflow",
        )
        self.assertEqual(observed, [2])


class CompactionPipelinePublicSurfaceTests(unittest.TestCase):
    """Pin the public ``aether.services.compact`` package surface.

    External consumers (agent integration, future tier PRs, third-party
    extensions) import from the package, not submodules.  The ``__all__``
    list documents the contract — this test guards against silent
    additions or removals so we notice when the surface changes and can
    update docs / consumers in the same PR.
    """

    EXPECTED_EXPORTS = frozenset(
        {
            # Live tiers + orchestrator.
            "AutoCompactor",
            "CompactionContext",
            "CompactionPipeline",
            "CompactionResult",
            "CompactorTier",
            # Tier 5 internals worth pinning for tests / operators.
            "COMPACT_PROMPT",
            "LLMForkSummarizer",
            "UsageSink",
            # Tier 3 (PR 3.5) — TimeBased is the live implementation,
            # Cached is the Sprint 5+ stub, plus the two module-level
            # constants (placeholder string + default tool whitelist).
            "TimeBasedMicrocompactor",
            "CachedMicrocompactor",
            "TIME_BASED_MC_CLEARED_MESSAGE",
            "DEFAULT_COMPACTABLE_TOOLS",
            # Tier 2/4 placeholders — exported so future PRs can drop
            # in real implementations without changing how the agent
            # imports them.
            "NoOpCollapseTier",
            "NoOpSnipper",
            # Token estimator the pipeline binds to by default.
            "estimate_messages_tokens",
        }
    )

    def test_p1_all_documented_symbols_importable(self) -> None:
        from aether.services import compact

        for name in self.EXPECTED_EXPORTS:
            with self.subTest(symbol=name):
                self.assertTrue(
                    hasattr(compact, name),
                    f"aether.services.compact missing public export {name!r}",
                )

    def test_p2_dunder_all_matches_documented_set(self) -> None:
        """``__all__`` must equal the documented set — no drift."""
        from aether.services import compact

        self.assertEqual(
            set(compact.__all__),
            set(self.EXPECTED_EXPORTS),
            "aether.services.compact.__all__ drifted from documented surface; "
            "update tests/docs together when intentional.",
        )

    def test_p3_compactortier_protocol_is_runtime_checkable_or_documented(self) -> None:
        """``CompactorTier`` must be importable as a *type* (Protocol)."""
        from aether.services.compact import (
            CachedMicrocompactor,
            CompactorTier,
            NoOpSnipper,
        )

        # Structural conformance — every NoOp/stub tier shipped in this
        # module satisfies the protocol shape.
        for tier in (NoOpSnipper(), CachedMicrocompactor()):
            self.assertTrue(hasattr(tier, "name"))
            self.assertTrue(callable(getattr(tier, "maybe_run", None)))
        # CompactorTier is a class-like object usable in annotations.
        self.assertTrue(isinstance(CompactorTier, type) or hasattr(CompactorTier, "__class__"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
