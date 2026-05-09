"""Canonical token usage representation, shared across providers.

Sprint 3 / PR 3.1 (P1-4 + P1-11):
    Three model providers (OpenAI-compatible, Anthropic, Codex/Responses)
    return token usage in slightly different shapes.  Downstream consumers
    (CLI footer, SDK, billing, Sprint 3 compaction triggers) need ONE
    stable representation they can rely on.

This module provides:

    * ``CanonicalUsage`` — a frozen dataclass with five primary fields
      and three derived totals (``prompt_tokens / completion_tokens /
      total_tokens``) kept as ``@property`` so they always stay in sync.

    * ``normalize_usage(raw, provider, api_mode)`` — tolerant conversion
      from a provider-native usage dict to ``CanonicalUsage``.  Returns
      ``CanonicalUsage()`` (all zeros) on any malformed input rather
      than raising — callers shouldn't have to guard against partial
      provider responses.

    * ``sum_usages(iterable)`` — fold a sequence of CanonicalUsage into
      one (used by the engine to accumulate per-LLM-call usage across
      a turn).

The semantics of each field follow Anthropic + OpenAI conventions
combined:

    input_tokens       — non-cached prompt tokens billed at full rate
    output_tokens      — completion tokens (always billed)
    cache_read_tokens  — prompt tokens served from cache (cheap rate)
    cache_write_tokens — prompt tokens being inserted into cache (premium)
    reasoning_tokens   — model-internal reasoning tokens (Codex / o1-style)

Aliases preserved for backwards compatibility with legacy logging:

    prompt_tokens     = input_tokens + cache_read_tokens + cache_write_tokens
    completion_tokens = output_tokens
    total_tokens      = prompt_tokens + completion_tokens + reasoning_tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class CanonicalUsage:
    """Provider-neutral token usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def add(self, other: "CanonicalUsage") -> "CanonicalUsage":
        """Return a new CanonicalUsage with field-wise sum."""
        return CanonicalUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly dict including the three derived totals.

        We materialise the totals here (rather than relying on consumers
        to reconstruct) because the dict is what ends up in
        ``EngineResult.metadata["usage"]`` and we want CLI / SDK / billing
        to see the same number without each computing it themselves.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def normalize_usage(
    raw: Any,
    *,
    provider: str = "openai",
    api_mode: str = "chat",
) -> CanonicalUsage:
    """Convert a provider-native usage dict to CanonicalUsage.

    Tolerant: returns ``CanonicalUsage()`` on any malformed input rather
    than raising.  The caller is responsible for logging if zero usage
    is suspicious in their context.

    Recognised provider strings:
      * ``"anthropic"`` with ``api_mode="messages"`` → Anthropic Messages API
      * ``"codex"``                                  → OpenAI Responses API
      * any other (``"openai"`` / ``"kimi"`` / ``"zhipu"`` / ...) →
        OpenAI Chat Completions compatible parsing

    Unknown ``provider`` strings fall through to the OpenAI-compatible
    parser as best effort — the openai-compatible schema is the most
    forgiving of partial / extra fields.
    """
    if not isinstance(raw, dict):
        return CanonicalUsage()

    if provider == "anthropic" and api_mode == "messages":
        return _normalize_anthropic_messages(raw)
    if provider == "codex":
        return _normalize_codex(raw)
    return _normalize_openai_compatible(raw)


def sum_usages(usages: Iterable[CanonicalUsage]) -> CanonicalUsage:
    """Fold a sequence of CanonicalUsage into one."""
    total = CanonicalUsage()
    for u in usages:
        total = total.add(u)
    return total


# --------------------------------------------------------------------------- #
# Provider-specific parsers                                                   #
# --------------------------------------------------------------------------- #


def _safe_int(value: Any) -> int:
    """Coerce *value* to non-negative int; return 0 on any failure.

    Provider responses sometimes deliver numeric fields as strings
    ("100"), as None, or omit them entirely.  We accept all of these
    and produce 0 rather than crashing the engine on a logging-only path.
    """
    if value is None:
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _normalize_openai_compatible(raw: dict) -> CanonicalUsage:
    """OpenAI Chat Completions / Responses API + all openai-compatible.

    Schema (post-2024 with cache + reasoning details):

        {
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "prompt_tokens_details": {
            "cached_tokens": int,        # ← cache_read
            "audio_tokens": int          # ignored
          },
          "completion_tokens_details": {
            "reasoning_tokens": int,     # ← reasoning (o1)
            "audio_tokens": int          # ignored
          }
        }

    Important quirk: OpenAI's ``prompt_tokens`` includes cached tokens.
    We split them into ``input_tokens = prompt_tokens - cached_tokens``
    + ``cache_read_tokens = cached_tokens`` so downstream billing math
    can apply the discounted rate to cache_read separately.
    """
    cache_read = 0
    details = raw.get("prompt_tokens_details")
    if isinstance(details, dict):
        cache_read = _safe_int(details.get("cached_tokens"))

    reasoning = 0
    cd = raw.get("completion_tokens_details")
    if isinstance(cd, dict):
        reasoning = _safe_int(cd.get("reasoning_tokens"))

    prompt = _safe_int(raw.get("prompt_tokens"))
    completion = _safe_int(raw.get("completion_tokens"))
    # OpenAI's prompt_tokens already includes cached tokens; subtract so
    # input_tokens reflects what's billed at the full prompt rate.
    input_tokens = max(0, prompt - cache_read)

    return CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=completion,
        cache_read_tokens=cache_read,
        cache_write_tokens=0,  # OpenAI doesn't expose write separately
        reasoning_tokens=reasoning,
    )


def _normalize_anthropic_messages(raw: dict) -> CanonicalUsage:
    """Anthropic Messages API.

    Schema:

        {
          "input_tokens": int,
          "output_tokens": int,
          "cache_read_input_tokens": int,
          "cache_creation_input_tokens": int   # ← cache_write
        }

    Anthropic's ``input_tokens`` does NOT include cache reads or writes —
    each is exposed separately, so no subtraction is needed.
    Reasoning ("thinking") tokens are returned inline in the response
    body, not in the usage block, so we leave reasoning_tokens at 0.
    """
    return CanonicalUsage(
        input_tokens=_safe_int(raw.get("input_tokens")),
        output_tokens=_safe_int(raw.get("output_tokens")),
        cache_read_tokens=_safe_int(raw.get("cache_read_input_tokens")),
        cache_write_tokens=_safe_int(raw.get("cache_creation_input_tokens")),
        reasoning_tokens=0,
    )


def _normalize_codex(raw: dict) -> CanonicalUsage:
    """OpenAI Responses API (used by Codex / o-series).

    Schema:

        {
          "input_tokens": int,
          "output_tokens": int,
          "input_tokens_details": { "cached_tokens": int },
          "output_tokens_details": { "reasoning_tokens": int }
        }

    Unlike the Chat Completions schema, Responses API's ``input_tokens``
    excludes cached tokens — they're already split.  We don't subtract.
    """
    cache_read = 0
    itd = raw.get("input_tokens_details")
    if isinstance(itd, dict):
        cache_read = _safe_int(itd.get("cached_tokens"))

    reasoning = 0
    otd = raw.get("output_tokens_details")
    if isinstance(otd, dict):
        reasoning = _safe_int(otd.get("reasoning_tokens"))

    return CanonicalUsage(
        input_tokens=_safe_int(raw.get("input_tokens")),
        output_tokens=_safe_int(raw.get("output_tokens")),
        cache_read_tokens=cache_read,
        cache_write_tokens=0,
        reasoning_tokens=reasoning,
    )
