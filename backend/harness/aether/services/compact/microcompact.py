"""Tier 3 microcompact placeholder.

Tier 3 is reserved for *microcompaction* — small-scope structural
rewrites (e.g. merging adjacent assistant text turns, dropping
``thinking`` blocks once the matching ``tool_use`` has already been
executed) that don't need an LLM round-trip but require more than the
trivial deduping Tier 2 does.  The real implementation lands in a
future PR; this no-op shim keeps the pipeline ordering stable so
documented tier numbers don't shift when Tier 3 is filled in.
"""

from __future__ import annotations

from typing import Any

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext


class NoOpMicrocompactor:
    """No-op Tier 3 implementation.

    Conforms to the ``CompactorTier`` protocol.  Always returns the
    incoming messages unchanged with ``freed=0`` so the pipeline's
    per-tier accounting stays honest.
    """

    name: str = "tier3_microcompact"

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,  # noqa: ARG002 — protocol signature
        turn_context: TurnContext,  # noqa: ARG002 — protocol signature
    ) -> tuple[list[dict[str, Any]], int]:
        return messages, 0
