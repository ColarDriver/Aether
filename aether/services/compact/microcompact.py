"""Tier 3 microcompact — clear stale tool results when the cache is cold.

Two intended paths (mirroring claude-code's ``microCompact.ts``):

  * :class:`TimeBasedMicrocompactor` (this file, **active** when Tier 3
    is enabled).  Detects "user came back after a long break" via the
    elapsed time since the last assistant message, and replaces all but
    the most recent compactable ``tool_result`` payloads with a
    placeholder string.  Mutates messages directly because the cache is
    assumed cold by the time the gap exceeds threshold — rewriting the
    prefix is unavoidable anyway.

  * :class:`CachedMicrocompactor` (stub here, real implementation lands
    with a future prompt-cache path). For warm-cache scenarios the planned
    implementation will use provider-specific ``cache_edits`` to delete
    tool results server-side without invalidating the cached prefix —
    so it does NOT mutate local messages.

The two paths are mutually exclusive: time-based wins when the cache is
cold (rewriting is free), cached path otherwise.  The pipeline today
only wires :class:`TimeBasedMicrocompactor`; :class:`CachedMicrocompactor`
ships as a no-op stub so future PRs can swap it in without churning
exports.

Reference: ``microCompact.ts:253-292`` in the bundled claude-code
references (``tmp/claude-code-references``).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from aether.runtime.core.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext


# Mirrors ``microCompact.ts`` line 36 — keep the exact literal so any
# downstream consumer (transcript exporter, ``/context`` view, future
# observability) that pattern-matches sees the identical marker.
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"


# Default ``microcompact_compactable_tools`` value mirrored on
# ``EngineConfig`` — re-exported here so callers that build their own
# config (tests, future schema migrations) have a single source of
# truth.  ``write_file`` is intentionally omitted: its result is just a
# "wrote N bytes" confirmation, clearing it adds noise without saving
# meaningful tokens.  ``web_search`` / ``web_fetch`` will join when the
# corresponding builtins land.
DEFAULT_COMPACTABLE_TOOLS: tuple[str, ...] = (
    "read_file",
    "shell",
    "grep",
    "glob",
    "list_dir",
)


class TimeBasedMicrocompactor:
    """Tier 3 — clear old tool results when the model has been idle.

    Triggered ONLY when **all** of the following hold:

    * ``compression_enabled`` is ``True`` (master switch);
    * the gap since the last assistant message exceeds
      ``microcompact_gap_threshold_minutes``;
    * there are strictly more than ``microcompact_keep_recent``
      compactable tool calls in history.

    Does NOT consult cache state — assumes cache is cold by the time
    the gap exceeds threshold.  Future :class:`CachedMicrocompactor`
    handles the warm-cache path without rewriting the prefix.

    Conforms to the ``CompactorTier`` protocol declared in
    :mod:`aether.services.compact.compactor`.
    """

    name: str = "tier3_microcompact"

    def __init__(self, *, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,  # noqa: ARG002 — protocol signature
        turn_context: TurnContext,
    ) -> tuple[list[dict[str, Any]], int]:
        if not getattr(self.config, "compression_enabled", False):
            return messages, 0
        if not messages:
            return messages, 0

        gap_minutes = self._compute_gap_minutes(messages, turn_context)
        threshold = float(
            getattr(self.config, "microcompact_gap_threshold_minutes", 5.0)
        )
        if gap_minutes is None or gap_minutes < threshold:
            return messages, 0

        # Floor to 1 — zero would leave the model with no working tool
        # context at all on the next call, which is worse than "old"
        # context and contradicts the placeholder's intent.
        keep_recent = max(
            1, int(getattr(self.config, "microcompact_keep_recent", 5))
        )
        compactable_tool_ids = self._collect_compactable_tool_ids(messages)
        if len(compactable_tool_ids) <= keep_recent:
            # Nothing old enough to clear without touching the
            # ``keep_recent`` window.  Bail without mutating.
            return messages, 0

        clear_set = set(compactable_tool_ids[:-keep_recent])

        new_messages, tokens_freed = self._clear_tool_results(
            messages, clear_set
        )
        if tokens_freed == 0:
            # All targeted blocks were already at the placeholder
            # (idempotent re-run).  Treat as no-op so the pipeline's
            # short-circuit accounting stays honest.
            return messages, 0

        # Observability — counter accumulates across multiple Tier 3
        # invocations within the same turn (e.g. preflight + recovery).
        turn_context.metadata["tier3_cleared_count"] = (
            int(turn_context.metadata.get("tier3_cleared_count", 0))
            + len(clear_set)
        )
        self.logger.info(
            "tier3: gap=%.1fmin > %.1fmin; cleared %d tool results, "
            "kept last %d, freed ~%d tokens",
            gap_minutes,
            threshold,
            len(clear_set),
            keep_recent,
            tokens_freed,
        )

        return new_messages, tokens_freed

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _compute_gap_minutes(
        self,
        messages: list[dict[str, Any]],
        turn_context: TurnContext,
    ) -> Optional[float]:
        """Minutes since the last assistant message; ``None`` if unknown.

        Resolution order:

        1. ``_aether_meta.timestamp`` on the most recent assistant
           message — written by :meth:`AgentEngine._assistant_aether_meta`
           on every assistant message we produce.
        2. ``turn_context.metadata['session_record_meta']
           ['last_assistant_timestamp']`` — fallback for replays /
           sessions hydrated from disk where per-message meta is lost.

        Returns ``None`` if neither source produces a valid float.  The
        gate then short-circuits, so missing-timestamp is "tier 3 does
        nothing", never "tier 3 crashes".
        """
        # Walk in reverse; bail at the first assistant message
        # encountered (matches ``microCompact.ts`` semantics — "the most
        # recent assistant turn is the one we measure from, regardless
        # of whether it has a timestamp or not").
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            meta = msg.get("_aether_meta") or {}
            ts = meta.get("timestamp") if isinstance(meta, dict) else None
            if isinstance(ts, (int, float)):
                # Negative gap (clock skew, system clock rewind) clamps
                # to 0 so the threshold check stays well-defined.
                return max(0.0, (time.time() - float(ts)) / 60.0)
            break

        sess_meta = turn_context.metadata.get("session_record_meta")
        if isinstance(sess_meta, dict):
            ts = sess_meta.get("last_assistant_timestamp")
            if isinstance(ts, (int, float)):
                return max(0.0, (time.time() - float(ts)) / 60.0)

        return None

    def _collect_compactable_tool_ids(
        self, messages: list[dict[str, Any]]
    ) -> list[str]:
        """Walk messages; return tool_use IDs of *compactable* calls in encounter order.

        "Compactable" is defined by
        :attr:`config.microcompact_compactable_tools` (case-folded +
        underscore-normalised + namespace-stripped via the same
        ``_normalize_name`` helper the phantom-tool repair path uses,
        so ``ReadFile`` / ``read-file`` / ``mcp__router__read_file``
        all match a ``read_file`` config entry).
        """
        # Local import — avoids a runtime cycle between
        # ``aether.services.compact`` and ``aether.agents.core`` at
        # module import time, while still routing through the canonical
        # name normaliser.
        from aether.agents.core.phantom_tool import _normalize_name

        compactable_set = {
            _normalize_name(n)
            for n in getattr(
                self.config,
                "microcompact_compactable_tools",
                DEFAULT_COMPACTABLE_TOOLS,
            )
            if isinstance(n, str) and n
        }
        if not compactable_set:
            return []

        ids: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                tool_name = _normalize_name(block.get("name") or "")
                if tool_name not in compactable_set:
                    continue
                tool_id = block.get("id")
                if isinstance(tool_id, str) and tool_id:
                    ids.append(tool_id)
        return ids

    def _clear_tool_results(
        self,
        messages: list[dict[str, Any]],
        clear_set: set[str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Replace cleared ``tool_result`` payloads with the placeholder.

        Returns ``(new_messages, tokens_freed_estimate)``.  Tokens freed
        is computed via :func:`estimate_messages_tokens` before/after —
        the same heuristic the pipeline uses to drive its own
        accounting, so per-tier numbers compose correctly.

        Idempotent: blocks already carrying the placeholder are left
        alone and contribute 0 freed tokens (so a no-op re-run cleanly
        short-circuits at the caller).
        """
        # Local import — same rationale as
        # ``_collect_compactable_tool_ids``: defers loading the
        # estimator until first use, avoids any boot-time module cycle.
        from aether.services.compact.token_estimation import (
            estimate_messages_tokens,
        )

        if not clear_set:
            return messages, 0

        before_tokens = estimate_messages_tokens(messages)
        new_messages: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue
            # Tool results live in ``user`` messages (Anthropic-style
            # tool-result blocks) and on standalone ``tool`` messages
            # (OpenAI-style flat ``role=tool``).  We touch both.
            role = msg.get("role")
            if role == "tool":
                # Flat OpenAI shape: ``content`` is a string and the
                # tool ID is on ``tool_call_id``.
                if (
                    isinstance(msg.get("content"), str)
                    and msg.get("tool_call_id") in clear_set
                    and msg["content"] != TIME_BASED_MC_CLEARED_MESSAGE
                ):
                    new_msg = dict(msg)
                    new_msg["content"] = TIME_BASED_MC_CLEARED_MESSAGE
                    new_messages.append(new_msg)
                else:
                    new_messages.append(msg)
                continue

            if role != "user":
                new_messages.append(msg)
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                new_messages.append(msg)
                continue

            new_content: list[Any] = []
            touched = False
            for block in content:
                if not (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") in clear_set
                ):
                    new_content.append(block)
                    continue

                inner = block.get("content")
                if (
                    isinstance(inner, str)
                    and inner == TIME_BASED_MC_CLEARED_MESSAGE
                ):
                    # Already cleared on a previous pass — leave alone
                    # so freed-token accounting stays honest.
                    new_content.append(block)
                    continue

                new_block = dict(block)
                new_block["content"] = TIME_BASED_MC_CLEARED_MESSAGE
                new_content.append(new_block)
                touched = True

            if touched:
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)

        after_tokens = estimate_messages_tokens(new_messages)
        return new_messages, max(0, before_tokens - after_tokens)


class CachedMicrocompactor:
    """Stub for the cache-edit path.

    When prompt-cache support lands, this class will:

    * Track ``tool_use_ids`` that have been "registered" with the
      provider's prompt cache.
    * On warm cache, emit a ``cache_edits`` block on the next request
      instructing the server to delete specific tool_results, without
      invalidating the cached prefix.

    For now :meth:`maybe_run` is a no-op so callers can already wire it
    into the pipeline (and exports stay stable) without breaking.
    Conforms to the ``CompactorTier`` protocol.
    """

    name: str = "tier3_microcompact_cached"

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,  # noqa: ARG002 — protocol signature
        turn_context: TurnContext,  # noqa: ARG002 — protocol signature
    ) -> tuple[list[dict[str, Any]], int]:
        return messages, 0
