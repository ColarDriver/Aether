"""Context compaction services.

Public surface for the five-tier compaction pipeline.  External code
(agent integration, tests, future tier implementations) should import
from this module — not from individual submodules — so we have a
single observable contract for what counts as "the compaction API".

Tiers are numbered to match
``docs/sprint-3-compaction-pipeline/04_pr3_4_tier5_autocompact.md``:

* Tier 2 — :class:`NoOpSnipper` (placeholder, future PR)
* Tier 3 — :class:`NoOpMicrocompactor` (placeholder, future PR)
* Tier 4 — :class:`NoOpCollapseTier` (placeholder, future PR)
* Tier 5 — :class:`AutoCompactor` (live; LLM-fork summariser)

The :class:`CompactorTier` Protocol is exported so tier 1 / Tier 2-4
implementations in other PRs can declare conformance with a static
type-check (``isinstance`` works too thanks to ``runtime_checkable``
on the Protocol).  :data:`COMPACT_PROMPT` is exported so tests and
operators can pin the exact instruction sent to the summariser fork.
"""

from aether.services.compact.autocompact import AutoCompactor
from aether.services.compact.collapse import NoOpCollapseTier
from aether.services.compact.compactor import (
    CompactionContext,
    CompactionPipeline,
    CompactionResult,
    CompactorTier,
)
from aether.services.compact.llm_fork import (
    COMPACT_PROMPT,
    LLMForkSummarizer,
    UsageSink,
)
from aether.services.compact.microcompact import NoOpMicrocompactor
from aether.services.compact.snip import NoOpSnipper
from aether.services.compact.token_estimation import estimate_messages_tokens

__all__ = [
    "AutoCompactor",
    "COMPACT_PROMPT",
    "CompactionContext",
    "CompactionPipeline",
    "CompactionResult",
    "CompactorTier",
    "LLMForkSummarizer",
    "NoOpCollapseTier",
    "NoOpMicrocompactor",
    "NoOpSnipper",
    "UsageSink",
    "estimate_messages_tokens",
]
