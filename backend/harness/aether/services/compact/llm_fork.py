"""LLM-fork summariser used by Tier 5 autocompact."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, TurnContext


# Signature for the optional callback that lets the summariser hand its
# fork response back to the parent for usage accumulation.  Kept narrow
# (response + parent context) so callers can wire it to whatever
# accumulator they already have without exposing the agent internals to
# the compact services layer.
UsageSink = Callable[[NormalizedResponse, TurnContext], None]


COMPACT_PROMPT = (
    "You are a conversation summariser. Read the conversation excerpt above "
    "and produce a compact summary that preserves:\n"
    "1. The user's original task and sub-tasks.\n"
    "2. Files, commands, tools, and data the assistant touched.\n"
    "3. Decisions made and the reasoning behind them.\n"
    "4. Open questions and pending work.\n\n"
    "Format the result as Markdown under the heading "
    "'## Earlier conversation summary'. Do not call tools. Output summary "
    "text only. Aim for {target_tokens} tokens or fewer."
)

# Per-block payload truncation limit when building the summariser excerpt.
# A single ``write_file`` tool_use can carry hundreds of KB of content;
# replaying the full payload into the fork prompt defeats the purpose of
# Tier 5 (we'd be paying full input cost to summarise something we're
# trying to compress) and can itself trigger a downstream context
# overflow on the fork call.  Keep the marker plus the head/tail of the
# payload for context — the summariser only needs enough to identify
# *what* the tool did, not the verbatim payload.
_MAX_TOOL_INPUT_REPR_CHARS = 240
_MAX_TOOL_RESULT_REPR_CHARS = 480


def _truncate_repr(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` chars with a ``...[N truncated]`` marker."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - 32)
    head = text[:keep]
    return f"{head}...[{len(text) - keep} chars truncated]"


class LLMForkSummarizer:
    """One-shot provider call that replaces the middle of history."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        config: Any,
        logger: Any,
        usage_sink: Optional[UsageSink] = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.logger = logger
        # ``usage_sink`` is the bridge that lets fork-call usage land on
        # the parent turn's accumulator.  See the long comment in
        # ``summarise`` for why this is *not* optional in production
        # wiring (only kept ``Optional`` for narrow unit tests that
        # don't care about cost reporting).
        self.usage_sink = usage_sink

    def summarise(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        turn_context: TurnContext,
    ) -> list[dict[str, Any]]:
        protect_first = max(0, int(getattr(self.config, "compression_protect_first_n", 2)))
        protect_last = max(0, int(getattr(self.config, "compression_protect_last_n", 6)))
        target_tokens = max(
            1,
            int(getattr(self.config, "compression_target_summary_tokens", 4000)),
        )

        if len(messages) <= protect_first + protect_last:
            return messages

        head = list(messages[:protect_first])
        tail = list(messages[-protect_last:] if protect_last else [])
        middle = messages[protect_first:-protect_last if protect_last else None]
        if not middle:
            return messages

        excerpt = "\n\n".join(_format_message_for_summary(msg) for msg in middle)
        instruction = COMPACT_PROMPT.format(target_tokens=target_tokens)
        summarizer_context = TurnContext(
            session_id=turn_context.session_id,
            iteration=turn_context.iteration,
            metadata={
                "query_source": "compact",
                "_compaction_in_progress": True,
                "compaction_parent_turn_id": turn_context.turn_id,
            },
            task_id=turn_context.task_id,
            turn_id=turn_context.turn_id,
        )
        response = self.provider.generate(
            [
                {
                    "role": "user",
                    "content": f"{excerpt}\n\n---\n\n{instruction}",
                }
            ],
            [],
            ModelCallConfig(
                temperature=0.0,
                max_tokens=target_tokens + 1000,
                extra={"model": model},
            ),
            summarizer_context,
            stream_callback=None,
            stream_silent_callback=None,
        )

        # Bridge fork-call usage back to the parent turn.  Without this
        # the prompt/completion tokens spent re-summarising the middle
        # slice are silently dropped from ``metadata["usage"]`` —
        # observability shows the turn as cheaper than it really was,
        # rate-limit / quota accounting under-reports, and any cost
        # dashboard mis-attributes spend.  We always bump a fork-only
        # counter (regardless of whether ``usage_sink`` is wired) so
        # operators can at least see *that* a fork happened even in
        # bare-bones unit-test wiring.
        turn_context.metadata["compaction_fork_api_calls"] = (
            int(turn_context.metadata.get("compaction_fork_api_calls", 0)) + 1
        )
        if self.usage_sink is not None:
            try:
                self.usage_sink(response, turn_context)
            except Exception:  # noqa: BLE001 — observability path, never crash a fork
                self.logger.debug(
                    "compaction fork usage_sink raised; usage will be under-reported",
                    exc_info=True,
                )

        summary_text = (response.content or "").strip()
        if not summary_text:
            raise RuntimeError("summarizer returned empty text")

        # We deliberately emit a single ``user`` message instead of the
        # ``[system: boundary] + [user: summary]`` pair the early sketch
        # used: many providers (Anthropic Messages, AWS Bedrock, vLLM
        # with strict role-alternation) reject mid-conversation
        # ``system`` messages and either drop them silently or fold them
        # into the system prompt — both outcomes destroy the boundary
        # marker.  Wrapping the boundary as a prose prefix inside the
        # summary user message keeps the text content the model sees,
        # the role contract every provider accepts, and the
        # ``_aether_meta`` carry-on for downstream consumers in one slot.
        boundary_prefix = (
            f"[compact_boundary] Compacted {len(middle)} messages via Tier 5 LLM fork."
        )
        summary_msg = {
            "role": "user",
            "content": f"{boundary_prefix}\n\n{summary_text}",
            "_aether_meta": {
                "compact_boundary": True,
                "compact_summary": True,
                "compacted_messages": len(middle),
                "model": model,
            },
        }
        return head + [summary_msg] + tail


def _format_message_for_summary(msg: dict[str, Any]) -> str:
    role = str(msg.get("role", "?")).upper()
    content = msg.get("content")
    if isinstance(content, str):
        body = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type == "tool_use":
                    raw_input = block.get("input")
                    input_repr = _truncate_repr(
                        json.dumps(raw_input, ensure_ascii=False, default=str)
                        if not isinstance(raw_input, str)
                        else raw_input,
                        _MAX_TOOL_INPUT_REPR_CHARS,
                    )
                    parts.append(
                        f"[tool_use: {block.get('name')}({input_repr})]"
                    )
                elif block_type == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        parts.append(
                            f"[tool_result: {_truncate_repr(inner, _MAX_TOOL_RESULT_REPR_CHARS)}]"
                        )
                    elif isinstance(inner, list):
                        # Recurse: tool_result content can itself be a
                        # list of blocks (Anthropic-style).  Each block
                        # gets the same per-block treatment.
                        nested = "\n".join(
                            _format_block_inline(b) for b in inner
                        )
                        parts.append(
                            f"[tool_result:\n{_truncate_repr(nested, _MAX_TOOL_RESULT_REPR_CHARS)}]"
                        )
                    elif inner is not None:
                        parts.append(
                            "[tool_result: "
                            + _truncate_repr(
                                json.dumps(inner, ensure_ascii=False, default=str),
                                _MAX_TOOL_RESULT_REPR_CHARS,
                            )
                            + "]"
                        )
                    else:
                        parts.append("[tool_result: <empty>]")
                elif block_type in {"image", "document"}:
                    parts.append(f"[{block_type}]")
                elif block_type in {"thinking", "redacted_thinking"}:
                    parts.append(
                        f"[{block_type}: "
                        + _truncate_repr(
                            str(block.get("thinking", "")),
                            _MAX_TOOL_RESULT_REPR_CHARS,
                        )
                        + "]"
                    )
                else:
                    parts.append(
                        _truncate_repr(
                            json.dumps(block, ensure_ascii=False, default=str),
                            _MAX_TOOL_RESULT_REPR_CHARS,
                        )
                    )
            else:
                parts.append(_truncate_repr(str(block), _MAX_TOOL_RESULT_REPR_CHARS))
        body = "\n".join(part for part in parts if part)
    else:
        body = str(content)
    return f"### {role}\n{body}"


def _format_block_inline(block: Any) -> str:
    """Inline single-block formatter used inside nested tool_result lists."""
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "text":
            return _truncate_repr(
                str(block.get("text", "")), _MAX_TOOL_RESULT_REPR_CHARS
            )
        return _truncate_repr(
            json.dumps(block, ensure_ascii=False, default=str),
            _MAX_TOOL_RESULT_REPR_CHARS,
        )
    return _truncate_repr(str(block), _MAX_TOOL_RESULT_REPR_CHARS)
