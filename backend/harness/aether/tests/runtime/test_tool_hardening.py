"""Sprint 2 / PR 2.3 — tool dispatch hardening tests.

Coverage layout:

1. ``repair_tool_name`` unit tests pin the four-stage repair pipeline
   (exact / prefix-strip / normalise / Levenshtein) and the tied-distance
   refusal that prevents the engine from dispatching the wrong tool when
   two registered names are equally close to the typo'd input.

2. ``prepare_tool_calls`` unit tests pin the orchestrator end-to-end:
   * dedup keys are insertion-order-independent
   * cap fires only for the configured delegate-class names
   * cap counter persists across iterations within a single turn
   * unrepairable names produce role=tool synthetic results without
     dispatching
   * the per-turn unrepairable-name retry counter triggers
     ``ExitReason.INVALID_TOOL_REPEATED`` after the configured budget

3. ``AgentEngine`` integration tests assert the full happy/sad paths
   end-to-end so a future refactor of the orchestrator surface can't
   silently break the engine wiring.
"""

from __future__ import annotations

import unittest
from typing import Any, Iterable, List

from aether import AgentEngine
from aether.agents.core.tool_hardening import (
    DEFAULT_DELEGATE_TOOL_NAMES,
    TURN_KEY_DELEGATE_CALLS,
    TURN_KEY_INVALID_TOOL_RETRIES,
    prepare_tool_calls,
    repair_tool_name,
)
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _NoopExecutor(ToolExecutor):
    """ToolExecutor double — records calls and returns a fixed string.

    Argument validation is intentionally weak (we only care about
    dispatch routing in these tests; argument schema enforcement is
    its own existing test surface).
    """

    def __init__(self, name: str, description: str = "noop") -> None:
        self._descriptor = ToolDescriptor(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}},
            required=(),
        )
        self.calls: List[ToolCall] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def validate(self, call: ToolCall) -> None:
        return None

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"ok:{call.name}",
            is_error=False,
            metadata={},
        )


def _registry(*names: str) -> ToolRegistry:
    """Build a ``ToolRegistry`` with one no-op executor per name."""
    reg = ToolRegistry()
    for name in names:
        reg.register(_NoopExecutor(name))
    return reg


def _ctx() -> TurnContext:
    return TurnContext(session_id="t", iteration=0, metadata={})


# ---------------------------------------------------------------------------
# repair_tool_name
# ---------------------------------------------------------------------------


class RepairToolNameTests(unittest.TestCase):
    def test_exact_match_returns_as_is(self) -> None:
        reg = _registry("read_file", "write_file")
        self.assertEqual(repair_tool_name("read_file", reg), "read_file")

    def test_case_normalisation_returns_canonical_name(self) -> None:
        reg = _registry("read_file")
        self.assertEqual(repair_tool_name("readFile", reg), "read_file")
        self.assertEqual(repair_tool_name("ReadFile", reg), "read_file")
        self.assertEqual(repair_tool_name("READ_FILE", reg), "read_file")

    def test_dash_underscore_swap(self) -> None:
        reg = _registry("read_file")
        self.assertEqual(repair_tool_name("read-file", reg), "read_file")

    def test_namespace_prefix_strip(self) -> None:
        reg = _registry("read_file")
        self.assertEqual(repair_tool_name("tool_read_file", reg), "read_file")
        self.assertEqual(repair_tool_name("functions.read_file", reg), "read_file")
        self.assertEqual(repair_tool_name("mcp__filesystem__read_file", reg), "read_file")

    def test_levenshtein_distance_one_typo(self) -> None:
        reg = _registry("read_file", "write_file", "list_dir")
        self.assertEqual(repair_tool_name("reed_file", reg), "read_file")
        self.assertEqual(repair_tool_name("read_fie", reg), "read_file")

    def test_levenshtein_distance_two_typo(self) -> None:
        # ``raed_file`` swaps two adjacent characters → distance 2
        # (e↔a as two substitutions in the standard DP without
        # transposition shortcut).  Should still repair.
        reg = _registry("read_file")
        self.assertEqual(repair_tool_name("raed_file", reg), "read_file")

    def test_distance_above_threshold_returns_none(self) -> None:
        reg = _registry("read_file")
        # "completely_different" is many edits away from "read_file".
        self.assertIsNone(repair_tool_name("completely_different", reg))

    def test_tied_distance_refuses_to_guess(self) -> None:
        # Two registered names equally distant from the input — refuse
        # rather than risk dispatching the wrong tool.  ``fead_file``
        # is distance 1 from BOTH ``read_file`` (f→r) and ``head_file``
        # (f→h), so the classifier must surface unknown so the model
        # picks one explicitly.
        reg = _registry("read_file", "head_file")
        self.assertIsNone(repair_tool_name("fead_file", reg))

    def test_empty_name_returns_none(self) -> None:
        reg = _registry("read_file")
        self.assertIsNone(repair_tool_name("", reg))

    def test_does_not_match_distance_one_on_meaningful_difference(self) -> None:
        # "read_files" → "read_file" is distance 1 but represents a
        # *different* tool concept (list vs single).  We accept the
        # match because Levenshtein can't tell semantic difference;
        # this test pins the current behaviour and serves as a
        # regression marker if we ever switch to a smarter algorithm
        # (e.g. token-level similarity).
        reg = _registry("read_file")
        self.assertEqual(repair_tool_name("read_files", reg), "read_file")


# ---------------------------------------------------------------------------
# prepare_tool_calls — orchestrator
# ---------------------------------------------------------------------------


def _calls(*specs: tuple[str, str, dict | None]) -> list[ToolCall]:
    """Build a list of ``ToolCall`` from ``(id, name, args)`` tuples."""
    return [
        ToolCall(id=cid, name=name, arguments=args or {})
        for cid, name, args in specs
    ]


class PrepareToolCallsTests(unittest.TestCase):
    def test_passthrough_for_already_valid_calls(self) -> None:
        reg = _registry("read_file", "write_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                ("c1", "read_file", {"path": "/a"}),
                ("c2", "write_file", {"path": "/b", "content": "x"}),
            ),
            registry=reg,
            config=EngineConfig(),
            context=ctx,
        )
        self.assertEqual(len(plan.prepared), 2)
        for prepared in plan.prepared:
            self.assertIsNone(prepared.synthetic_result)
            self.assertIsNone(prepared.repaired_from)
        self.assertIsNone(plan.exit_reason)

    def test_repair_pipeline_rewrites_name(self) -> None:
        reg = _registry("read_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(("c1", "readFile", {"path": "/a"})),
            registry=reg,
            config=EngineConfig(),
            context=ctx,
        )
        self.assertEqual(plan.repaired_count, 1)
        self.assertEqual(plan.prepared[0].call.name, "read_file")
        self.assertEqual(plan.prepared[0].repaired_from, "readFile")
        self.assertIsNone(plan.prepared[0].synthetic_result)

    def test_unrepairable_name_produces_synthetic_error(self) -> None:
        reg = _registry("read_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(("c1", "totally_not_a_tool", None)),
            registry=reg,
            config=EngineConfig(invalid_tool_max_retries=3),
            context=ctx,
        )
        self.assertEqual(plan.unresolved_count, 1)
        synth = plan.prepared[0].synthetic_result
        self.assertIsNotNone(synth)
        assert synth is not None
        self.assertTrue(synth.is_error)
        self.assertIn("Unknown tool", synth.content)
        self.assertIn("read_file", synth.content)
        self.assertEqual(ctx.metadata[TURN_KEY_INVALID_TOOL_RETRIES], 1)

    def test_invalid_tool_budget_exhaustion_pins_terminal(self) -> None:
        reg = _registry("read_file")
        ctx = _ctx()
        ctx.metadata[TURN_KEY_INVALID_TOOL_RETRIES] = 2  # one short of cap
        plan = prepare_tool_calls(
            _calls(("c1", "totally_not_a_tool", None)),
            registry=reg,
            config=EngineConfig(invalid_tool_max_retries=3),
            context=ctx,
        )
        self.assertEqual(plan.exit_reason, ExitReason.INVALID_TOOL_REPEATED.value)

    def test_invalid_tool_counter_only_bumps_once_per_iteration(self) -> None:
        # Two unknown names in the same iteration is still "one
        # confused turn" — the counter should advance once, not twice.
        reg = _registry("read_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                ("c1", "totally_not_a_tool", None),
                ("c2", "another_unknown", None),
            ),
            registry=reg,
            config=EngineConfig(invalid_tool_max_retries=3),
            context=ctx,
        )
        self.assertEqual(ctx.metadata[TURN_KEY_INVALID_TOOL_RETRIES], 1)
        self.assertEqual(plan.unresolved_count, 2)

    def test_delegate_cap_replaces_overflow_with_synthetic_error(self) -> None:
        # 5 delegate calls in one batch with a cap of 3 → first 3
        # dispatch, last 2 become error stubs.
        reg = _registry("delegate_task")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                *(
                    (f"c{i}", "delegate_task", {"goal": f"task {i}"})
                    for i in range(5)
                )
            ),
            registry=reg,
            config=EngineConfig(max_delegate_calls_per_turn=3),
            context=ctx,
        )
        dispatched = [p for p in plan.prepared if p.synthetic_result is None]
        capped = [p for p in plan.prepared if p.synthetic_result is not None]
        self.assertEqual(len(dispatched), 3)
        self.assertEqual(len(capped), 2)
        self.assertEqual(plan.capped_count, 2)
        for prepared in capped:
            assert prepared.synthetic_result is not None
            self.assertTrue(prepared.synthetic_result.is_error)
            self.assertIn("delegate cap", prepared.synthetic_result.content)

    def test_delegate_cap_persists_across_iterations(self) -> None:
        # The cap is per *turn*, not per iteration.  When the metadata
        # already shows 3 delegates dispatched earlier this turn, the
        # next iteration's first delegate call should still be blocked.
        reg = _registry("delegate_task")
        ctx = _ctx()
        ctx.metadata[TURN_KEY_DELEGATE_CALLS] = 4
        plan = prepare_tool_calls(
            _calls(("c1", "delegate_task", {"goal": "more"})),
            registry=reg,
            config=EngineConfig(max_delegate_calls_per_turn=4),
            context=ctx,
        )
        self.assertEqual(plan.capped_count, 1)
        assert plan.prepared[0].synthetic_result is not None
        self.assertTrue(plan.prepared[0].synthetic_result.is_error)

    def test_delegate_cap_skips_non_delegate_tools(self) -> None:
        # 5 read_file calls with the cap at 1: cap should NOT fire,
        # but dedup should collapse identical-arg duplicates.
        reg = _registry("read_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                ("c1", "read_file", {"path": "/a"}),
                ("c2", "read_file", {"path": "/b"}),
                ("c3", "read_file", {"path": "/c"}),
                ("c4", "read_file", {"path": "/d"}),
                ("c5", "read_file", {"path": "/e"}),
            ),
            registry=reg,
            config=EngineConfig(max_delegate_calls_per_turn=1),
            context=ctx,
        )
        self.assertEqual(plan.capped_count, 0)
        self.assertEqual(plan.deduped_count, 0)

    def test_dedup_collapses_identical_args(self) -> None:
        # 3 reads of the same path → first dispatches, two become stubs.
        reg = _registry("read_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                ("c1", "read_file", {"path": "/etc/hosts"}),
                ("c2", "read_file", {"path": "/etc/hosts"}),
                ("c3", "read_file", {"path": "/etc/hosts"}),
            ),
            registry=reg,
            config=EngineConfig(),
            context=ctx,
        )
        dispatched = [p for p in plan.prepared if p.synthetic_result is None]
        stubs = [p for p in plan.prepared if p.synthetic_result is not None]
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(len(stubs), 2)
        self.assertEqual(plan.deduped_count, 2)
        for stub_call in stubs:
            assert stub_call.synthetic_result is not None
            # Dedup stubs are NOT errors — they just say "already done".
            self.assertFalse(stub_call.synthetic_result.is_error)
            self.assertIn("Duplicate of earlier call", stub_call.synthetic_result.content)
            self.assertIn("c1", stub_call.synthetic_result.content)

    def test_dedup_key_is_argument_order_independent(self) -> None:
        reg = _registry("call_api")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                ("c1", "call_api", {"a": 1, "b": 2}),
                ("c2", "call_api", {"b": 2, "a": 1}),
            ),
            registry=reg,
            config=EngineConfig(),
            context=ctx,
        )
        self.assertEqual(plan.deduped_count, 1)
        self.assertIsNone(plan.prepared[0].synthetic_result)
        self.assertIsNotNone(plan.prepared[1].synthetic_result)

    def test_dedup_disabled_passes_duplicates_through(self) -> None:
        reg = _registry("read_file")
        ctx = _ctx()
        plan = prepare_tool_calls(
            _calls(
                ("c1", "read_file", {"path": "/a"}),
                ("c2", "read_file", {"path": "/a"}),
            ),
            registry=reg,
            config=EngineConfig(tool_dedup_enabled=False),
            context=ctx,
        )
        self.assertEqual(plan.deduped_count, 0)
        for prepared in plan.prepared:
            self.assertIsNone(prepared.synthetic_result)


# ---------------------------------------------------------------------------
# AgentEngine end-to-end
# ---------------------------------------------------------------------------


class _ScriptedProvider(ModelProvider):
    """Minimal provider yielding a fixed script of NormalizedResponses."""

    def __init__(self, script: Iterable[NormalizedResponse]) -> None:
        self._script = list(script)
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        # Cross-PR hotfix: PR 3.1 (sprint-3) added this kwarg to the
        # base ``ModelProvider.generate`` contract; this test file was
        # authored against the pre-PR-3.1 signature.  Accept it so the
        # engine's call site (which now always passes both callbacks)
        # doesn't trip a TypeError before tool-hardening runs.
        stream_silent_callback: Any = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        self.calls += 1
        if not self._script:
            raise RuntimeError("script exhausted")
        return self._script.pop(0)


class ToolHardeningEngineIntegrationTests(unittest.TestCase):
    def test_repair_dispatches_to_canonical_name(self) -> None:
        # Model emits ``readFile`` — engine must repair to
        # ``read_file`` and dispatch successfully.
        registry = _registry("read_file")
        executor = registry.get("read_file")
        provider = _ScriptedProvider([
            NormalizedResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="readFile", arguments={"path": "/a"})
                ],
                finish_reason="tool_calls",
            ),
            NormalizedResponse(content="done", finish_reason="stop"),
        ])
        engine = AgentEngine(provider, tool_registry=registry)
        result = engine.run_turn(EngineRequest(session_id="s1", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0].name, "read_file")

    def test_unrepairable_after_three_iterations_terminates_with_invalid_tool_repeated(self) -> None:
        # Model emits ``never_seen_tool`` for three iterations; the
        # synth ToolResult is appended each iteration, so the script
        # has the LLM keep retrying with the same bad name.
        registry = _registry("read_file")
        provider = _ScriptedProvider([
            NormalizedResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"c{i}", name="never_seen_tool", arguments={})
                ],
                finish_reason="tool_calls",
            )
            for i in range(3)
        ])
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(invalid_tool_max_retries=3, max_iterations=10),
        )
        result = engine.run_turn(EngineRequest(session_id="s2", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.INVALID_TOOL_REPEATED)
        self.assertTrue(result.metadata["turn"]["partial"])

    def test_dedup_stub_does_not_invoke_executor(self) -> None:
        # Three identical read_file calls in one batch; executor should
        # see exactly one dispatch.
        registry = _registry("read_file")
        executor = registry.get("read_file")
        provider = _ScriptedProvider([
            NormalizedResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"c{i}", name="read_file", arguments={"path": "/a"})
                    for i in range(3)
                ],
                finish_reason="tool_calls",
            ),
            NormalizedResponse(content="done", finish_reason="stop"),
        ])
        engine = AgentEngine(provider, tool_registry=registry)
        result = engine.run_turn(EngineRequest(session_id="s3", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(executor.calls), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
