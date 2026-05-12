"""Sprint 3 / PR 3.4 — ``LLMForkSummarizer`` unit tests.

The summariser is the heaviest piece of the Tier 5 path: it forks a
*stateless* provider call (no tools, low temperature, bounded output)
to compress the middle slice of the conversation while preserving the
configured head + tail messages verbatim.  These tests pin three
behaviour groups:

* Group G — basic generation: head/middle/tail slicing, empty-output
  failure, ``tools=[]`` enforcement, deterministic temperature.
* Group H — ``_format_message_for_summary`` helper: each block kind
  flattens to a readable form for the summariser excerpt.
* Group I — boundary marker invariants: ``_aether_meta`` keys land on
  the synthetic system + user messages so downstream consumers (e.g.
  ``/context`` rendering) can locate the compaction boundary later.
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from typing import Any, List

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.services.compact.llm_fork import (
    COMPACT_PROMPT,
    LLMForkSummarizer,
    _format_message_for_summary,
)
from aether.tools.base import ToolDescriptor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """Duck-typed ``EngineConfig`` slice the summariser consults."""

    compression_protect_first_n: int = 2
    compression_protect_last_n: int = 6
    compression_target_summary_tokens: int = 4_000


class _RecordingProvider(ModelProvider):
    """Captures every ``generate`` call for assertions in tests."""

    provider_name = "openai"
    api_mode = "chat"

    def __init__(
        self,
        *,
        response: NormalizedResponse,
    ) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "config": config,
                "context": context,
                "stream_callback": stream_callback,
                "stream_silent_callback": stream_silent_callback,
            }
        )
        return self._response


def _ctx() -> TurnContext:
    return TurnContext(
        session_id="sess-y",
        iteration=3,
        metadata={"existing": "value"},
        task_id="task-1",
        turn_id="turn-1",
    )


def _build(
    *,
    response_text: str = "## Earlier conversation summary\n\n* compressed",
    config: _Config | None = None,
    response: NormalizedResponse | None = None,
    usage_sink: Any = None,
) -> tuple[LLMForkSummarizer, _RecordingProvider, _Config]:
    resp = response if response is not None else NormalizedResponse(content=response_text)
    provider = _RecordingProvider(response=resp)
    cfg = config or _Config()
    return (
        LLMForkSummarizer(
            provider=provider,
            config=cfg,
            logger=logging.getLogger("test.llm_fork"),
            usage_sink=usage_sink,
        ),
        provider,
        cfg,
    )


def _ten_messages() -> list[dict]:
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first user task"},
        *[{"role": "assistant", "content": f"step {i}"} for i in range(8)],
    ]


# ---------------------------------------------------------------------------
# Group G — basic generation
# ---------------------------------------------------------------------------


class LLMForkGroupGGenerationTests(unittest.TestCase):
    def test_g1_compresses_middle_keeps_head_and_tail(self) -> None:
        """10 messages → head(2) + merged-summary + tail(6) = 9.

        We deliberately emit a single merged ``user`` message instead of
        a ``[system: boundary] + [user: summary]`` pair so the result
        survives providers (Anthropic Messages, AWS Bedrock, vLLM) that
        reject mid-conversation ``system`` blocks.  See the long
        comment in ``LLMForkSummarizer.summarise`` for context.
        """
        summariser, provider, _ = _build()
        out = summariser.summarise(_ten_messages(), model="claude", turn_context=_ctx())
        # 2 head + 1 merged + 6 tail = 9.
        self.assertEqual(len(out), 9)
        # Head preserved verbatim.
        self.assertEqual(out[0]["content"], "you are helpful")
        self.assertEqual(out[1]["content"], "first user task")
        # Merged synthetic message: user role, both boundary prefix and
        # summary text in the same body.
        merged = out[2]
        self.assertEqual(merged["role"], "user")
        self.assertTrue(merged["content"].startswith("[compact_boundary]"))
        self.assertIn("Earlier conversation summary", merged["content"])
        # Tail preserved verbatim.
        self.assertEqual(out[-1]["content"], "step 7")
        # Provider was called exactly once.
        self.assertEqual(len(provider.calls), 1)

    def test_g2_too_few_messages_returns_input_unchanged(self) -> None:
        """When ``len(messages) <= protect_first + protect_last`` bail."""
        summariser, provider, _ = _build()
        msgs = [{"role": "user", "content": "only one"}]
        out = summariser.summarise(msgs, model="x", turn_context=_ctx())
        self.assertIs(out, msgs)
        self.assertEqual(provider.calls, [])

    def test_g3_empty_summary_text_raises(self) -> None:
        """An empty ``response.content`` raises so the gate increments failures."""
        summariser, _, _ = _build(response_text="   ")
        with self.assertRaises(RuntimeError):
            summariser.summarise(_ten_messages(), model="x", turn_context=_ctx())

    def test_g4_long_summary_accepted_unchanged(self) -> None:
        """Output beyond ``target_tokens`` is not truncated by the summariser."""
        long_text = "## Earlier conversation summary\n\n" + "word " * 5_000
        summariser, _, _ = _build(response_text=long_text)
        out = summariser.summarise(_ten_messages(), model="x", turn_context=_ctx())
        # The merged synthetic message still preserves the full summary
        # text after the boundary prefix — the summariser itself does
        # not enforce ``target_tokens`` as a hard cap.
        merged = out[2]
        self.assertIn(long_text.strip(), merged["content"])
        self.assertTrue(merged["content"].startswith("[compact_boundary]"))

    def test_g5_summariser_called_with_empty_tools(self) -> None:
        """Sub-call must be ``tools=[]`` so the model focuses on prose."""
        summariser, provider, _ = _build()
        summariser.summarise(_ten_messages(), model="x", turn_context=_ctx())
        self.assertEqual(provider.calls[0]["tools"], [])

    def test_g6_summariser_called_with_zero_temperature(self) -> None:
        """Determinism: the fork uses ``temperature=0.0`` and a bounded budget."""
        summariser, provider, cfg = _build()
        summariser.summarise(_ten_messages(), model="x", turn_context=_ctx())
        config = provider.calls[0]["config"]
        self.assertIsInstance(config, ModelCallConfig)
        self.assertEqual(config.temperature, 0.0)
        # Budget is target + 1000 buffer — see llm_fork.py.
        self.assertEqual(
            config.max_tokens,
            cfg.compression_target_summary_tokens + 1000,
        )
        # The active model name flows through ``extra``.
        self.assertEqual(config.extra.get("model"), "x")

    def test_g7_summariser_isolates_turn_context(self) -> None:
        """Sub-call uses a fresh ``TurnContext`` with the compaction marker."""
        summariser, provider, _ = _build()
        parent_turn = _ctx()
        summariser.summarise(_ten_messages(), model="x", turn_context=parent_turn)
        sub_ctx = provider.calls[0]["context"]
        self.assertIsInstance(sub_ctx, TurnContext)
        self.assertIsNot(sub_ctx, parent_turn)
        self.assertEqual(sub_ctx.session_id, parent_turn.session_id)
        self.assertTrue(sub_ctx.metadata.get("_compaction_in_progress"))
        self.assertEqual(sub_ctx.metadata.get("query_source"), "compact")
        # Parent metadata not polluted by the sub-call.
        self.assertEqual(parent_turn.metadata.get("existing"), "value")
        self.assertNotIn("_compaction_in_progress", parent_turn.metadata)

    def test_g8_compact_prompt_template_includes_target_tokens(self) -> None:
        """Sanity check on the static prompt: it is a format string with target_tokens."""
        rendered = COMPACT_PROMPT.format(target_tokens=1234)
        self.assertIn("1234", rendered)
        # The user-facing instruction header is preserved.
        self.assertIn("Earlier conversation summary", rendered)

    def test_g9_protect_last_zero_keeps_only_head_plus_synthetic(self) -> None:
        """``protect_last_n=0`` keeps head + merged-summary, drops tail."""
        summariser, _, _ = _build(
            config=_Config(
                compression_protect_first_n=2,
                compression_protect_last_n=0,
            ),
        )
        out = summariser.summarise(_ten_messages(), model="x", turn_context=_ctx())
        # 2 head + 1 merged synthetic = 3 (no tail).
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["content"], "you are helpful")
        self.assertEqual(out[1]["content"], "first user task")
        self.assertEqual(out[2]["role"], "user")
        self.assertTrue(out[2]["content"].startswith("[compact_boundary]"))


# ---------------------------------------------------------------------------
# Group H — message-format helper
# ---------------------------------------------------------------------------


class LLMForkGroupHFormatHelperTests(unittest.TestCase):
    def test_h1_string_content_renders_role_then_body(self) -> None:
        out = _format_message_for_summary({"role": "user", "content": "hi there"})
        self.assertEqual(out, "### USER\nhi there")

    def test_h2_list_content_flattens_each_block(self) -> None:
        """tool_use input is JSON-serialised (then truncated) so huge payloads can't blow up the excerpt."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "do thing"},
                {"type": "tool_use", "name": "shell", "input": {"cmd": "ls"}},
                {"type": "tool_result", "tool_use_id": "x", "content": "out"},
            ],
        }
        out = _format_message_for_summary(msg)
        self.assertTrue(out.startswith("### ASSISTANT"))
        self.assertIn("do thing", out)
        # Tool input now JSON-encoded (double quotes, deterministic shape).
        self.assertIn('[tool_use: shell({"cmd": "ls"})]', out)
        self.assertIn("[tool_result: out]", out)

    def test_h3_none_content_renders_role_and_none_body(self) -> None:
        out = _format_message_for_summary({"role": "tool", "content": None})
        self.assertEqual(out, "### TOOL\nNone")

    def test_h4_image_block_renders_compact_marker(self) -> None:
        msg = {
            "role": "user",
            "content": [{"type": "image", "source": {"x": 1}}],
        }
        out = _format_message_for_summary(msg)
        self.assertEqual(out, "### USER\n[image]")

    def test_h5_thinking_block_renders_with_text(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm"}],
        }
        out = _format_message_for_summary(msg)
        self.assertEqual(out, "### ASSISTANT\n[thinking: hmm]")

    def test_h6_unknown_block_falls_back_to_json_dump(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "novel", "data": [1, 2]}],
        }
        out = _format_message_for_summary(msg)
        self.assertIn("\"type\": \"novel\"", out)
        self.assertIn("[1, 2]", out)


# ---------------------------------------------------------------------------
# Group I — boundary marker metadata
# ---------------------------------------------------------------------------


class LLMForkGroupIBoundaryMarkerTests(unittest.TestCase):
    """Boundary + summary share one ``user`` message — assert all four
    metadata keys land on it together."""

    def test_i1_merged_message_carries_both_aether_meta_flags(self) -> None:
        summariser, _, _ = _build()
        out = summariser.summarise(_ten_messages(), model="claude", turn_context=_ctx())
        merged = out[2]
        self.assertEqual(merged.get("role"), "user")
        meta = merged.get("_aether_meta")
        self.assertIsNotNone(meta)
        # Both observability flags surface on the merged message so
        # downstream consumers (CLI ``/context`` renderer, transcript
        # exporters) can locate the boundary even after providers
        # collapse adjacent ``user`` messages.
        self.assertTrue(meta.get("compact_boundary"))
        self.assertTrue(meta.get("compact_summary"))
        self.assertEqual(meta.get("compacted_messages"), len(_ten_messages()) - 2 - 6)
        self.assertEqual(meta.get("model"), "claude")

    def test_i2_boundary_text_prefix_is_the_first_line(self) -> None:
        """The ``[compact_boundary]`` marker prefixes the body so consumers
        without ``_aether_meta`` access can still find the boundary by
        scanning the visible content (e.g. transcript text dumps)."""
        summariser, _, _ = _build()
        out = summariser.summarise(_ten_messages(), model="claude", turn_context=_ctx())
        merged = out[2]
        self.assertTrue(merged["content"].startswith("[compact_boundary]"))
        # Boundary marker is on the first line; summary text follows.
        first_line, rest = merged["content"].split("\n", 1)
        self.assertIn("Compacted", first_line)
        self.assertIn("Earlier conversation summary", rest)


class LLMForkGroupJTruncationTests(unittest.TestCase):
    """Cover the per-block truncation that prevents huge tool payloads
    from defeating the whole point of Tier 5."""

    def test_j1_huge_tool_use_input_is_truncated(self) -> None:
        """A 10 KB tool_use input never escapes the per-block budget."""
        big_payload = {"content": "x" * 10_000, "path": "/tmp/foo"}
        msg = {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "write_file", "input": big_payload}],
        }
        out = _format_message_for_summary(msg)
        # Marker present.
        self.assertIn("[tool_use: write_file(", out)
        self.assertIn("chars truncated]", out)
        # Length stays within a small multiple of the input cap (200ish
        # chars + a little marker overhead) — the 10 KB payload does
        # NOT make it through.
        self.assertLess(len(out), 600)

    def test_j2_huge_tool_result_string_is_truncated(self) -> None:
        big_output = "y" * 10_000
        msg = {
            "role": "tool",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": big_output}],
        }
        out = _format_message_for_summary(msg)
        self.assertIn("[tool_result:", out)
        self.assertIn("chars truncated]", out)
        self.assertLess(len(out), 800)

    def test_j3_nested_tool_result_block_list_is_truncated(self) -> None:
        """tool_result with list content (Anthropic shape) is also bounded."""
        msg = {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": [
                        {"type": "text", "text": "z" * 5_000},
                        {"type": "text", "text": "w" * 5_000},
                    ],
                },
            ],
        }
        out = _format_message_for_summary(msg)
        self.assertIn("[tool_result:", out)
        self.assertIn("chars truncated]", out)
        self.assertLess(len(out), 1_000)

    def test_j4_small_inputs_pass_through_unchanged(self) -> None:
        """Inputs under the budget are not touched (no truncation marker)."""
        msg = {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "ls", "input": {"path": "/"}}],
        }
        out = _format_message_for_summary(msg)
        self.assertNotIn("chars truncated", out)
        self.assertIn('{"path": "/"}', out)


class LLMForkGroupKUsageBridgeTests(unittest.TestCase):
    """Cover the parent-turn usage bridging fix.

    Without bridging, every Tier 5 fork silently drops its prompt and
    completion tokens from ``metadata['usage']`` — a turn that spent
    real money on summarisation looks free in the cost dashboard.
    """

    def test_k1_fork_increments_compaction_fork_api_calls(self) -> None:
        """A successful fork always bumps the fork-only API counter,
        even when no ``usage_sink`` is wired."""
        summariser, _, _ = _build()  # no usage_sink
        ctx = _ctx()
        out = summariser.summarise(_ten_messages(), model="claude", turn_context=ctx)
        # Fork was actually issued.
        self.assertEqual(len(out), 9)
        # The fork-only counter advanced exactly once.
        self.assertEqual(ctx.metadata.get("compaction_fork_api_calls"), 1)

    def test_k2_usage_sink_called_with_response_and_parent_context(self) -> None:
        """The sink fires with ``(response, parent_turn_context)`` so
        the agent's ``_accumulate_usage`` can fold fork tokens into the
        per-turn accumulator."""
        captured: list[tuple[Any, TurnContext]] = []

        def sink(resp: NormalizedResponse, parent_ctx: TurnContext) -> None:
            captured.append((resp, parent_ctx))

        # Provider returns a response with a usage payload — exactly
        # what the agent's accumulator is expecting in the real path.
        resp = NormalizedResponse(
            content="## Earlier conversation summary\n\n* compressed",
            metadata={"usage": {"prompt_tokens": 1234, "completion_tokens": 567}},
        )
        summariser, _, _ = _build(response=resp, usage_sink=sink)
        ctx = _ctx()
        summariser.summarise(_ten_messages(), model="claude", turn_context=ctx)

        self.assertEqual(len(captured), 1)
        forwarded_resp, forwarded_ctx = captured[0]
        # Same response object the provider returned (fork usage stays
        # in the raw provider shape so the agent's normalize_usage can
        # reuse the same parser path it uses for main-call responses).
        self.assertIs(forwarded_resp, resp)
        # Critically: the *parent* turn context, not the inner
        # ``summarizer_context``.  Otherwise fork usage lands on a
        # throwaway accumulator and disappears at end of fork.
        self.assertIs(forwarded_ctx, ctx)

    def test_k3_sink_failure_does_not_break_compaction(self) -> None:
        """An exception out of the sink degrades to "usage missed",
        not "compaction failed"."""
        def boom(_resp: NormalizedResponse, _ctx: TurnContext) -> None:
            raise RuntimeError("downstream accumulator is broken")

        summariser, _, _ = _build(usage_sink=boom)
        ctx = _ctx()
        # Must not raise — the fork itself succeeded, only observability
        # plumbing failed.
        out = summariser.summarise(_ten_messages(), model="claude", turn_context=ctx)
        self.assertEqual(len(out), 9)
        # Counter still bumped (so dashboards see *that* a fork ran).
        self.assertEqual(ctx.metadata.get("compaction_fork_api_calls"), 1)

    def test_k4_no_sink_still_bumps_counter_only(self) -> None:
        """Without a sink we leak token-cost detail (acceptable for
        bare-bones tests / future callers) but ``compaction_fork_api_calls``
        still ticks so the omission is *visible*."""
        resp = NormalizedResponse(
            content="## Earlier conversation summary\n\n* compressed",
            metadata={"usage": {"prompt_tokens": 999, "completion_tokens": 111}},
        )
        summariser, _, _ = _build(response=resp, usage_sink=None)
        ctx = _ctx()
        summariser.summarise(_ten_messages(), model="claude", turn_context=ctx)

        # No accumulator was written (no sink) — but the counter is
        # there so anyone debugging "where did the tokens go?" sees a
        # fork happened.
        self.assertEqual(ctx.metadata.get("compaction_fork_api_calls"), 1)
        self.assertNotIn("usage_accumulator", ctx.metadata)

    def test_k5_multiple_forks_increment_counter(self) -> None:
        """Two summarise calls in the same turn → counter at 2."""
        summariser, _, _ = _build()
        ctx = _ctx()
        # Each call needs a fresh source list because summarise returns
        # a new list (we don't reuse output as input here, just verify
        # the counter behaviour).
        summariser.summarise(_ten_messages(), model="claude", turn_context=ctx)
        summariser.summarise(_ten_messages(), model="claude", turn_context=ctx)
        self.assertEqual(ctx.metadata.get("compaction_fork_api_calls"), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
