"""Token budgeting helpers for memory retrieval and rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .contracts import MemoryBlock, MemoryBundle


DEFAULT_MIN_MEMORY_TOKENS = 300
DEFAULT_ESTIMATED_CHARS_PER_TOKEN = 4


@dataclass(slots=True, frozen=True)
class MemoryBudget:
    """Resolved memory budget for one provider request."""

    base_budget: int
    effective_budget: int
    skipped_reason: str | None = None


def estimate_text_tokens(text: str, *, chars_per_token: int = DEFAULT_ESTIMATED_CHARS_PER_TOKEN) -> int:
    """Return a conservative local token estimate for plain text."""

    if not text:
        return 0
    denominator = max(1, chars_per_token)
    return max(1, (len(text) + denominator - 1) // denominator)


def resolve_memory_token_budget(
    *,
    model_window: int,
    estimated_prompt_tokens: int,
    memory_token_budget_pct: float,
    memory_token_budget_max: int,
    compression_threshold_pct: float = 0.85,
    min_memory_tokens: int = DEFAULT_MIN_MEMORY_TOKENS,
) -> MemoryBudget:
    """Resolve a bounded memory budget for a single outbound provider call."""

    if model_window <= 0 or memory_token_budget_pct <= 0 or memory_token_budget_max <= 0:
        return MemoryBudget(base_budget=0, effective_budget=0, skipped_reason="disabled")

    base_budget = min(
        int(model_window * memory_token_budget_pct),
        int(memory_token_budget_max),
    )
    threshold_tokens = int(model_window * compression_threshold_pct)
    remaining_before_threshold = max(0, threshold_tokens - max(0, estimated_prompt_tokens))
    effective_budget = min(base_budget, int(remaining_before_threshold * 0.5))
    if effective_budget < min_memory_tokens:
        return MemoryBudget(
            base_budget=max(0, base_budget),
            effective_budget=max(0, effective_budget),
            skipped_reason="budget_too_small",
        )
    return MemoryBudget(base_budget=base_budget, effective_budget=effective_budget)


def trim_text_to_token_budget(text: str, token_budget: int) -> str:
    """Trim text to an estimated token budget, preserving paragraph boundaries when possible."""

    if token_budget <= 0:
        return ""
    if estimate_text_tokens(text) <= token_budget:
        return text

    max_chars = max(1, token_budget * DEFAULT_ESTIMATED_CHARS_PER_TOKEN)
    suffix = "\n[memory truncated]"
    raw_limit = max(1, max_chars - len(suffix))
    candidate = text[:raw_limit].rstrip()
    paragraph_cut = candidate.rfind("\n\n")
    if paragraph_cut >= max(80, raw_limit // 2):
        candidate = candidate[:paragraph_cut].rstrip()
    return f"{candidate}{suffix}"


def pack_memory_blocks(
    blocks: Iterable[MemoryBlock],
    *,
    token_budget: int,
    block_token_max: int,
    reserve_pct: float = 0.10,
) -> MemoryBundle:
    """Return a bundle containing the highest-value blocks that fit the budget."""

    if token_budget <= 0:
        return MemoryBundle.skipped("budget_too_small")

    reserve = int(token_budget * max(0.0, min(0.9, reserve_pct)))
    usable_budget = max(0, token_budget - reserve)
    if usable_budget <= 0:
        return MemoryBundle.skipped("budget_too_small")

    packed: list[MemoryBlock] = []
    used = 0
    sorted_blocks = sorted(blocks, key=lambda block: block.relevance, reverse=True)
    for block in sorted_blocks:
        per_block_budget = min(block.token_estimate, block_token_max, usable_budget - used)
        if per_block_budget <= 0:
            break
        candidate = block
        if block.token_estimate > per_block_budget:
            trimmed = trim_text_to_token_budget(block.text, per_block_budget)
            if not trimmed.strip():
                continue
            candidate = replace(
                block,
                text=trimmed,
                token_estimate=estimate_text_tokens(trimmed),
            )
        if used + candidate.token_estimate > usable_budget:
            continue
        packed.append(candidate)
        used += candidate.token_estimate

    if not packed:
        return MemoryBundle.skipped("no_relevant_blocks")
    return MemoryBundle.from_blocks(tuple(packed))
