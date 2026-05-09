"""Tier 4 context-collapse placeholder.

Tier 4 is reserved for *context collapse* — bulk replacement of long
file-read tool_results with placeholder pointers ("file X read at
turn N; reference rather than re-include"), aggressive truncation of
old assistant chain-of-thought, etc.  More aggressive than Tier 3,
still no LLM round-trip.  The real implementation lands in a future
PR; this no-op shim keeps the pipeline ordering stable so the
documented tier numbers don't shift when Tier 4 is filled in.

Note: when the real Tier 4 ships it will set
``turn_context.metadata['collapse_owns_headroom'] = True`` to suppress
Tier 5 in the same pass — see the gate condition in
``AutoCompactor.maybe_run``.
"""

from __future__ import annotations

from typing import Any

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext


class NoOpCollapseTier:
    """No-op Tier 4 implementation.

    Conforms to the ``CompactorTier`` protocol.  Always returns the
    incoming messages unchanged with ``freed=0`` so the pipeline's
    per-tier accounting stays honest.
    """

    name: str = "tier4_collapse"

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,  # noqa: ARG002 — protocol signature
        turn_context: TurnContext,  # noqa: ARG002 — protocol signature
    ) -> tuple[list[dict[str, Any]], int]:
        return messages, 0
