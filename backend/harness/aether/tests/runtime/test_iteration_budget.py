"""Unit tests for :mod:`aether.runtime.core.iteration_budget`.

Sprint 3 / PR 3.2 — covers the four behaviour groups laid out in
``docs/sprint-3-compaction-pipeline/02_pr3_2_iteration_budget.md``
section 5.1:

* Group A — basic consume / refund semantics
* Group B — grace_call one-shot contract
* Group C — to_dict() serialisation shape
* Group D — boundary cases (max=0, max=1, large max)
"""

from __future__ import annotations

import json
import time
import unittest

from aether.runtime.core.iteration_budget import IterationBudget


class TestGroupABasicSemantics(unittest.TestCase):
    """Group A — consume / refund interactions."""

    def test_a1_initial_state(self) -> None:
        """T-A1: fresh budget has full headroom and no usage."""
        budget = IterationBudget(max_total=5)
        self.assertEqual(budget.remaining, 5)
        self.assertFalse(budget.exhausted)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.consume_count, 0)
        self.assertEqual(budget.refund_count, 0)
        self.assertFalse(budget.grace_consumed)

    def test_a2_consume_until_exhausted(self) -> None:
        """T-A2: consume() succeeds max_total times then refuses."""
        budget = IterationBudget(max_total=5)
        for i in range(5):
            self.assertTrue(budget.consume(), f"consume {i + 1}/5 should succeed")
        self.assertTrue(budget.exhausted)
        self.assertEqual(budget.remaining, 0)
        self.assertFalse(
            budget.consume(),
            "consume #6 must return False once exhausted",
        )
        self.assertEqual(budget.consume_count, 5)
        self.assertEqual(budget.used, 5)

    def test_a3_refund_after_consume(self) -> None:
        """T-A3: refund cancels the most recent consume."""
        budget = IterationBudget(max_total=3)
        budget.consume()
        budget.refund()
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.consume_count, 1)
        self.assertEqual(budget.refund_count, 1)
        self.assertEqual(budget.remaining, 3)
        self.assertFalse(budget.exhausted)

    def test_a4_consume_refund_consume_recovers_slot(self) -> None:
        """T-A4: refund frees a slot that another consume can claim."""
        budget = IterationBudget(max_total=5)
        for _ in range(5):
            budget.consume()
        self.assertTrue(budget.exhausted)
        budget.refund()
        self.assertFalse(budget.exhausted)
        self.assertTrue(budget.consume(), "post-refund consume must succeed")
        self.assertEqual(budget.used, 5)
        self.assertFalse(
            budget.consume(),
            "we are exhausted again, no more slots",
        )

    def test_a5_refund_on_fresh_budget_is_noop(self) -> None:
        """T-A5: refund() with used==0 is silent and stays at zero."""
        budget = IterationBudget(max_total=5)
        budget.refund()
        budget.refund()
        budget.refund()
        self.assertEqual(budget.used, 0)
        self.assertEqual(
            budget.refund_count,
            0,
            "refund_count must not increment when there's nothing to refund",
        )

    def test_a6_refund_after_exhaustion_unlocks_one_slot(self) -> None:
        """T-A6: refund after exhaustion must restore a usable slot."""
        budget = IterationBudget(max_total=2)
        budget.consume()
        budget.consume()
        self.assertTrue(budget.exhausted)
        budget.refund()
        self.assertEqual(budget.used, 1)
        self.assertEqual(budget.remaining, 1)
        self.assertFalse(budget.exhausted)
        self.assertTrue(budget.consume())


class TestGroupBGraceCall(unittest.TestCase):
    """Group B — grace_call one-shot bonus contract."""

    def test_b1_first_grace_call_grants(self) -> None:
        """T-B1: first grace_call() returns True and flips the flag."""
        budget = IterationBudget(max_total=3)
        for _ in range(3):
            budget.consume()
        self.assertTrue(budget.grace_call())
        self.assertTrue(budget.grace_consumed)

    def test_b2_second_grace_call_refuses(self) -> None:
        """T-B2: subsequent grace_call() invocations all return False."""
        budget = IterationBudget(max_total=1)
        budget.consume()
        self.assertTrue(budget.grace_call())
        for _ in range(5):
            self.assertFalse(budget.grace_call())
        self.assertTrue(budget.grace_consumed)

    def test_b3_grace_does_not_change_used_counts(self) -> None:
        """T-B3: grace_call() leaves used / consume_count untouched."""
        budget = IterationBudget(max_total=2)
        budget.consume()
        budget.consume()
        snapshot_used = budget.used
        snapshot_consume_count = budget.consume_count
        budget.grace_call()
        self.assertEqual(budget.used, snapshot_used)
        self.assertEqual(budget.consume_count, snapshot_consume_count)

    def test_b4_grace_does_not_unlock_consume(self) -> None:
        """T-B4: grace is a side-channel; main loop stays terminated."""
        budget = IterationBudget(max_total=2)
        budget.consume()
        budget.consume()
        budget.grace_call()
        self.assertFalse(
            budget.consume(),
            "grace must not re-open the main consume() path",
        )
        self.assertTrue(budget.exhausted)

    def test_b5_grace_callable_before_exhaustion(self) -> None:
        """Grace works whenever called — exhaustion is not a precondition.

        Documented behaviour: ``grace_call`` is a one-shot guard the
        engine can use defensively even mid-loop (e.g. to mark a
        summary path as already taken).  We don't test the engine's
        choice here, only that the dataclass itself accepts the call.
        """
        budget = IterationBudget(max_total=10)
        self.assertTrue(budget.grace_call())
        self.assertFalse(budget.grace_call())
        self.assertEqual(
            budget.used,
            0,
            "grace must not pre-consume slots when called early",
        )


class TestGroupCSerialization(unittest.TestCase):
    """Group C — to_dict() schema and JSON-friendliness."""

    def test_c1_to_dict_field_completeness(self) -> None:
        """T-C1: every documented field is present."""
        budget = IterationBudget(max_total=4)
        snapshot = budget.to_dict()
        expected_keys = {
            "used",
            "max",
            "remaining",
            "grace_consumed",
            "consume_count",
            "refund_count",
        }
        self.assertEqual(set(snapshot.keys()), expected_keys)
        self.assertEqual(snapshot["max"], 4)
        self.assertEqual(snapshot["used"], 0)
        self.assertEqual(snapshot["remaining"], 4)
        self.assertFalse(snapshot["grace_consumed"])

    def test_c2_to_dict_is_json_serialisable(self) -> None:
        """T-C2: json.dumps doesn't choke on the snapshot."""
        budget = IterationBudget(max_total=3)
        budget.consume()
        budget.refund()
        budget.consume()
        budget.consume()
        budget.grace_call()
        encoded = json.dumps(budget.to_dict())
        decoded = json.loads(encoded)
        self.assertEqual(decoded["used"], 2)
        self.assertEqual(decoded["consume_count"], 3)
        self.assertEqual(decoded["refund_count"], 1)
        self.assertTrue(decoded["grace_consumed"])

    def test_c3_to_dict_reflects_current_state(self) -> None:
        """T-C3: snapshot stays in sync with each mutation."""
        budget = IterationBudget(max_total=10)
        budget.consume()
        snap1 = budget.to_dict()
        self.assertEqual(snap1["used"], 1)
        budget.consume()
        budget.consume()
        snap2 = budget.to_dict()
        self.assertEqual(snap2["used"], 3)
        budget.refund()
        snap3 = budget.to_dict()
        self.assertEqual(snap3["used"], 2)
        self.assertEqual(snap3["refund_count"], 1)
        self.assertEqual(snap3["remaining"], 8)


class TestGroupDBoundaries(unittest.TestCase):
    """Group D — extreme parameter values."""

    def test_d1_zero_budget(self) -> None:
        """T-D1: max_total=0 makes consume always fail; grace still works."""
        budget = IterationBudget(max_total=0)
        self.assertTrue(budget.exhausted)
        self.assertEqual(budget.remaining, 0)
        self.assertFalse(budget.consume())
        self.assertTrue(budget.grace_call())
        self.assertFalse(budget.grace_call())

    def test_d2_unit_budget(self) -> None:
        """T-D2: max_total=1 supports consume + refund + reconsume."""
        budget = IterationBudget(max_total=1)
        self.assertTrue(budget.consume())
        self.assertTrue(budget.exhausted)
        self.assertFalse(budget.consume())
        budget.refund()
        self.assertFalse(budget.exhausted)
        self.assertTrue(budget.consume())
        self.assertTrue(budget.exhausted)

    def test_d3_large_budget_is_fast(self) -> None:
        """T-D3: 1M consumes complete well under the half-second budget.

        This is a smoke test for accidental O(n^2) behaviour in the
        dataclass — counters are integers, so the only way this
        could fail is a refactor that loops over ``used`` somewhere.
        """
        budget = IterationBudget(max_total=10**6)
        start = time.perf_counter()
        for _ in range(10**6):
            budget.consume()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.5, f"consume loop took {elapsed:.3f}s")
        self.assertTrue(budget.exhausted)
        self.assertFalse(budget.consume())


if __name__ == "__main__":
    unittest.main()
