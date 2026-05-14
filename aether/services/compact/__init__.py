"""Context compaction services.

Public surface for the five-tier compaction pipeline. External code
(agent integration, tests, future tier implementations) should import
from this module — not from individual submodules — so we have a
single observable contract for what counts as "the compaction API".

Tiers:

* Tier 2 — :class:`Snipper` plus the
  :data:`SNIP_DUPE_RULE_TOOLS` whitelist consumed by Rule 1
* Tier 3 — :class:`TimeBasedMicrocompactor`
  — :class:`CachedMicrocompactor` (stub for a future cache-edits path)
  — :data:`TIME_BASED_MC_CLEARED_MESSAGE` (placeholder string the
    Tier 3 implementation writes into cleared ``tool_result`` blocks)
  — :data:`DEFAULT_COMPACTABLE_TOOLS` (default value for the
    ``microcompact_compactable_tools`` config field)
* Tier 4 — :class:`ContextCollapseTier` plus
  :class:`CollapseSegment` and :class:`CollapseStore` for the
  projection-store data model.  Default ``compress_collapse_enabled=False``
  in :class:`EngineConfig` keeps it dormant until operators turn it on.
* Tier 5 — :class:`AutoCompactor` (LLM-fork summariser)

The :class:`CompactorTier` Protocol is exported so other tier
implementations can declare conformance with a static type check
(``isinstance`` works too thanks to ``runtime_checkable`` on the
Protocol). :data:`COMPACT_PROMPT` is exported so tests and operators
can pin the exact instruction sent to the summarizer fork.
"""

from aether.services.compact.autocompact import AutoCompactor
from aether.services.compact.collapse import (
    CollapseSegment,
    CollapseStore,
    ContextCollapseTier,
)
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
from aether.services.compact.snip import SNIP_DUPE_RULE_TOOLS, Snipper
from aether.services.compact.token_estimation import estimate_messages_tokens

__all__ = [
    "AutoCompactor",
    "COMPACT_PROMPT",
    "CachedMicrocompactor",
    "CollapseSegment",
    "CollapseStore",
    "CompactionContext",
    "CompactionPipeline",
    "CompactionResult",
    "CompactorTier",
    "ContextCollapseTier",
    "DEFAULT_COMPACTABLE_TOOLS",
    "LLMForkSummarizer",
    "SNIP_DUPE_RULE_TOOLS",
    "Snipper",
    "TIME_BASED_MC_CLEARED_MESSAGE",
    "TimeBasedMicrocompactor",
    "UsageSink",
    "estimate_messages_tokens",
]
