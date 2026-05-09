"""Tier 2 snip placeholder.

Tier 2 is reserved for *snipping* — non-summarising, non-LLM removal
of clearly-redundant content (duplicate tool_result payloads, empty
turns, repeated assistant heartbeat text, etc.).  The real
implementation will land in a future PR; this no-op shim keeps the
``CompactionPipeline`` wiring complete so downstream tiers (and the
documented tier ordering) don't shift when Tier 2 is filled in.
"""

from __future__ import annotations

from typing import Any

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext


class NoOpSnipper:
    """No-op Tier 2 implementation.

    Conforms to the ``CompactorTier`` protocol.  Always returns the
    incoming messages unchanged with ``freed=0`` so the pipeline's
    per-tier accounting stays honest.
    """

    name: str = "tier2_snip"

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,  # noqa: ARG002 — protocol signature
        turn_context: TurnContext,  # noqa: ARG002 — protocol signature
    ) -> tuple[list[dict[str, Any]], int]:
        return messages, 0
