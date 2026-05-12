"""Token-usage normalisation tests — Sprint 3 / PR 3.1.

These pin the conversion from each provider's native usage shape to
``CanonicalUsage`` plus the field-arithmetic invariants the engine,
CLI, and Sprint-3 compaction triggers all rely on.

Test groups:

  A — OpenAI Chat Completions / openai-compatible
  B — Anthropic Messages
  C — Codex / OpenAI Responses API
  D — CanonicalUsage maths (frozen + add + sum + to_dict)
  E — Unknown / openai-compatible fallback
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from aether.runtime.observability.usage import (
    CanonicalUsage,
    normalize_usage,
    sum_usages,
)


# ----------------------------------------------------------------------- #
# Group A — OpenAI Chat Completions / openai-compatible                   #
# ----------------------------------------------------------------------- #


class OpenAICompatibleUsageTests(unittest.TestCase):
    """T-A1..T-A8 — see docs/sprint-3-compaction-pipeline/01_pr3_1_*.md § 5.1."""

    def test_T_A1_basic_no_cache_no_reasoning(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        u = normalize_usage(raw, provider="openai")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)
        self.assertEqual(u.cache_read_tokens, 0)
        self.assertEqual(u.reasoning_tokens, 0)

    def test_T_A2_cache_read_subtracted_from_prompt(self) -> None:
        # OpenAI's prompt_tokens already includes cached_tokens —
        # we split so input_tokens reflects what's billed at the full rate.
        raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 80},
        }
        u = normalize_usage(raw, provider="openai")
        self.assertEqual(u.input_tokens, 20)
        self.assertEqual(u.cache_read_tokens, 80)
        self.assertEqual(u.output_tokens, 50)
        self.assertEqual(u.prompt_tokens, 100)  # alias re-sums to the original

    def test_T_A3_reasoning_tokens(self) -> None:
        raw = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
        u = normalize_usage(raw, provider="openai")
        self.assertEqual(u.reasoning_tokens, 30)
        # reasoning is added on top of prompt + completion in total
        self.assertEqual(u.total_tokens, 100 + 50 + 30)

    def test_T_A4_empty_dict(self) -> None:
        self.assertEqual(normalize_usage({}, provider="openai"), CanonicalUsage())

    def test_T_A5_none(self) -> None:
        self.assertEqual(normalize_usage(None, provider="openai"), CanonicalUsage())

    def test_T_A6_wrong_type(self) -> None:
        self.assertEqual(
            normalize_usage("not a dict", provider="openai"),
            CanonicalUsage(),
        )

    def test_T_A7_string_numeric_values(self) -> None:
        u = normalize_usage({"prompt_tokens": "100", "completion_tokens": "50"})
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)

    def test_T_A8_none_inside_details_is_safe(self) -> None:
        raw = {"prompt_tokens_details": {"cached_tokens": None}}
        u = normalize_usage(raw, provider="openai")
        self.assertEqual(u, CanonicalUsage())

    def test_T_A9_negative_value_clamped_to_zero(self) -> None:
        # _safe_int clamps to non-negative — defensive against bad provider responses
        u = normalize_usage({"prompt_tokens": -10, "completion_tokens": 50})
        self.assertEqual(u.input_tokens, 0)
        self.assertEqual(u.output_tokens, 50)


# ----------------------------------------------------------------------- #
# Group B — Anthropic Messages                                            #
# ----------------------------------------------------------------------- #


class AnthropicMessagesUsageTests(unittest.TestCase):
    """T-B1..T-B3."""

    def test_T_B1_basic_no_cache(self) -> None:
        raw = {"input_tokens": 100, "output_tokens": 50}
        u = normalize_usage(raw, provider="anthropic", api_mode="messages")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)
        self.assertEqual(u.cache_read_tokens, 0)
        self.assertEqual(u.cache_write_tokens, 0)
        # Anthropic doesn't return reasoning in usage — left at 0
        self.assertEqual(u.reasoning_tokens, 0)

    def test_T_B2_cache_read_and_write_separate(self) -> None:
        # Unlike OpenAI, Anthropic's input_tokens is already separate from cache
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        }
        u = normalize_usage(raw, provider="anthropic", api_mode="messages")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.cache_read_tokens, 80)
        self.assertEqual(u.cache_write_tokens, 20)

    def test_T_B3_anthropic_provider_with_wrong_api_mode_fallbacks_to_openai(self) -> None:
        # provider=anthropic but schema is OpenAI shape → should fallback parser
        raw = {"prompt_tokens": 100, "completion_tokens": 50}
        u = normalize_usage(raw, provider="anthropic", api_mode="text")
        # OpenAI parser used: prompt_tokens → input_tokens
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)


# ----------------------------------------------------------------------- #
# Group C — Codex / Responses API                                         #
# ----------------------------------------------------------------------- #


class CodexUsageTests(unittest.TestCase):
    """T-C1..T-C2."""

    def test_T_C1_reasoning_tokens(self) -> None:
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 30},
        }
        u = normalize_usage(raw, provider="codex")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)
        self.assertEqual(u.reasoning_tokens, 30)

    def test_T_C2_input_tokens_not_subtracted_for_cache(self) -> None:
        # Responses API's input_tokens already EXCLUDES cached — don't subtract
        raw = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 60},
        }
        u = normalize_usage(raw, provider="codex")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.cache_read_tokens, 60)
        self.assertEqual(u.output_tokens, 50)


# ----------------------------------------------------------------------- #
# Group D — CanonicalUsage maths                                          #
# ----------------------------------------------------------------------- #


class CanonicalUsageMathTests(unittest.TestCase):
    """T-D1..T-D6."""

    def test_T_D1_derived_aliases(self) -> None:
        u = CanonicalUsage(
            input_tokens=10, output_tokens=20, cache_read_tokens=5, reasoning_tokens=3
        )
        self.assertEqual(u.prompt_tokens, 15)  # 10 + 5 + 0
        self.assertEqual(u.completion_tokens, 20)
        self.assertEqual(u.total_tokens, 38)  # 15 + 20 + 3

    def test_T_D2_add_field_wise(self) -> None:
        a = CanonicalUsage(input_tokens=10)
        b = CanonicalUsage(input_tokens=20, output_tokens=5)
        c = a.add(b)
        self.assertEqual(c.input_tokens, 30)
        self.assertEqual(c.output_tokens, 5)
        # add() must not mutate operands
        self.assertEqual(a.input_tokens, 10)
        self.assertEqual(b.input_tokens, 20)

    def test_T_D3_sum_usages_empty_returns_zero(self) -> None:
        self.assertEqual(sum_usages([]), CanonicalUsage())

    def test_T_D4_sum_usages_equivalent_to_chained_add(self) -> None:
        u1 = CanonicalUsage(input_tokens=1, output_tokens=2)
        u2 = CanonicalUsage(input_tokens=3, output_tokens=4)
        u3 = CanonicalUsage(input_tokens=5, output_tokens=6, reasoning_tokens=10)
        self.assertEqual(sum_usages([u1, u2, u3]), u1.add(u2).add(u3))

    def test_T_D5_frozen_dataclass(self) -> None:
        u = CanonicalUsage()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            u.input_tokens = 100  # type: ignore[misc]

    def test_T_D6_to_dict_includes_derived_totals(self) -> None:
        u = CanonicalUsage(input_tokens=10, output_tokens=20, cache_read_tokens=5)
        d = u.to_dict()
        # Primary fields
        self.assertEqual(d["input_tokens"], 10)
        self.assertEqual(d["output_tokens"], 20)
        self.assertEqual(d["cache_read_tokens"], 5)
        self.assertEqual(d["cache_write_tokens"], 0)
        self.assertEqual(d["reasoning_tokens"], 0)
        # Derived totals materialised in the dict
        self.assertEqual(d["prompt_tokens"], 15)
        self.assertEqual(d["completion_tokens"], 20)
        self.assertEqual(d["total_tokens"], 35)
        # JSON-friendly
        self.assertEqual(json.loads(json.dumps(d))["total_tokens"], 35)


# ----------------------------------------------------------------------- #
# Group E — Unknown / openai-compatible fallback                          #
# ----------------------------------------------------------------------- #


class UnknownProviderFallbackTests(unittest.TestCase):
    """T-E1..T-E4 — kimi / zhipu / unknown / wrong-api-mode fallbacks."""

    def test_T_E1_kimi_uses_openai_parser(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50}
        u = normalize_usage(raw, provider="kimi")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)

    def test_T_E2_zhipu_uses_openai_parser(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50}
        u = normalize_usage(raw, provider="zhipu")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)

    def test_T_E3_unknown_provider_uses_openai_parser(self) -> None:
        raw = {"prompt_tokens": 100, "completion_tokens": 50}
        u = normalize_usage(raw, provider="unknown_xyz_v3")
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 50)

    def test_T_E4_anthropic_with_text_mode_falls_back(self) -> None:
        # api_mode != "messages" → openai-compatible path
        raw = {"prompt_tokens": 42, "completion_tokens": 10}
        u = normalize_usage(raw, provider="anthropic", api_mode="text")
        self.assertEqual(u.input_tokens, 42)
        self.assertEqual(u.output_tokens, 10)


if __name__ == "__main__":
    unittest.main()
