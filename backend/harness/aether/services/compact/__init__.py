"""Context compaction services.

Public surface for the five-tier compaction pipeline.  External code
(agent integration, tests, future tier implementations) should import
from this module — not from individual submodules — so we have a
single observable contract for what counts as "the compaction API".

Tiers are numbered to match
``docs/sprint-3-compaction-pipeline/04_pr3_4_tier5_autocompact.md``:

* Tier 2 — :class:`NoOpSnipper` (placeholder, future PR)
* Tier 3 — :class:`TimeBasedMicrocompactor` (live; PR 3.5)
  — :class:`CachedMicrocompactor` (stub, Sprint 5+ cache_edits path)
  — :data:`TIME_BASED_MC_CLEARED_MESSAGE` (placeholder string the
    Tier 3 implementation writes into cleared ``tool_result`` blocks)
  — :data:`DEFAULT_COMPACTABLE_TOOLS` (default value for the
    ``microcompact_compactable_tools`` config field)
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
from aether.services.compact.microcompact import (
    DEFAULT_COMPACTABLE_TOOLS,
    TIME_BASED_MC_CLEARED_MESSAGE,
    CachedMicrocompactor,
    TimeBasedMicrocompactor,
)
from aether.services.compact.snip import NoOpSnipper
from aether.services.compact.token_estimation import estimate_messages_tokens

__all__ = [
    "AutoCompactor",
    "COMPACT_PROMPT",
    "CachedMicrocompactor",
    "CompactionContext",
    "CompactionPipeline",
    "CompactionResult",
    "CompactorTier",
    "DEFAULT_COMPACTABLE_TOOLS",
    "LLMForkSummarizer",
    "NoOpCollapseTier",
    "NoOpSnipper",
    "TIME_BASED_MC_CLEARED_MESSAGE",
    "TimeBasedMicrocompactor",
    "UsageSink",
    "estimate_messages_tokens",
]
