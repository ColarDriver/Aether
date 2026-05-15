"""Unit tests for the sectional system-prompt assembler."""

from __future__ import annotations

import unittest

from aether.agents.core.system_prompt import (
    SystemPromptOptions,
    augment_system_prompt,
    augment_system_with_tool_contract,
)
from aether.tools.base import ToolDescriptor


def _descriptors(*names: str) -> list[ToolDescriptor]:
    return [ToolDescriptor(name=name) for name in names]


class SystemPromptSectionsTests(unittest.TestCase):
    def test_default_includes_all_sections(self) -> None:
        result = augment_system_prompt(
            "You are helpful",
            _descriptors("shell", "read_file"),
        )
        assert result is not None
        self.assertIn("<tool_use_contract>", result)
        self.assertIn("<verification_directive>", result)
        self.assertIn("<verifier_gate>", result)
        self.assertIn("<faithful_reporting>", result)
        contract_idx = result.index("<tool_use_contract>")
        verif_idx = result.index("<verification_directive>")
        verifier_idx = result.index("<verifier_gate>")
        faithful_idx = result.index("<faithful_reporting>")
        user_idx = result.index("You are helpful")
        self.assertLess(contract_idx, verif_idx)
        self.assertLess(verif_idx, verifier_idx)
        self.assertLess(verifier_idx, faithful_idx)
        self.assertLess(faithful_idx, user_idx)

    def test_disable_verification_drops_block(self) -> None:
        result = augment_system_prompt(
            "user prompt",
            _descriptors("shell"),
            SystemPromptOptions(include_verification_directive=False),
        )
        assert result is not None
        self.assertIn("<tool_use_contract>", result)
        self.assertNotIn("<verification_directive>", result)
        self.assertIn("<faithful_reporting>", result)

    def test_disable_faithful_drops_block(self) -> None:
        result = augment_system_prompt(
            "user prompt",
            _descriptors("shell"),
            SystemPromptOptions(include_faithful_reporting=False),
        )
        assert result is not None
        self.assertIn("<verification_directive>", result)
        self.assertNotIn("<faithful_reporting>", result)

    def test_disable_verifier_gate_drops_block(self) -> None:
        result = augment_system_prompt(
            "user prompt",
            _descriptors("shell"),
            SystemPromptOptions(include_verifier_gate=False),
        )
        assert result is not None
        self.assertIn("<verification_directive>", result)
        self.assertNotIn("<verifier_gate>", result)

    def test_disable_tool_contract_drops_block(self) -> None:
        result = augment_system_prompt(
            "user prompt",
            _descriptors("shell"),
            SystemPromptOptions(include_tool_contract=False),
        )
        assert result is not None
        self.assertNotIn("<tool_use_contract>", result)
        self.assertIn("<verification_directive>", result)

    def test_empty_tool_list_drops_contract_keeps_other_blocks(self) -> None:
        result = augment_system_prompt(
            "user prompt",
            _descriptors(),  # no tools to advertise
            SystemPromptOptions(),
        )
        assert result is not None
        self.assertNotIn("<tool_use_contract>", result)
        self.assertIn("<verification_directive>", result)
        self.assertIn("<faithful_reporting>", result)

    def test_all_disabled_returns_input_unchanged(self) -> None:
        result = augment_system_prompt(
            "user prompt",
            _descriptors("shell"),
            SystemPromptOptions(
                include_tool_contract=False,
                include_verification_directive=False,
                include_faithful_reporting=False,
                include_verifier_gate=False,
            ),
        )
        self.assertEqual(result, "user prompt")

    def test_no_user_prompt_with_all_disabled_returns_none(self) -> None:
        result = augment_system_prompt(
            None,
            _descriptors("shell"),
            SystemPromptOptions(
                include_tool_contract=False,
                include_verification_directive=False,
                include_faithful_reporting=False,
                include_verifier_gate=False,
            ),
        )
        self.assertIsNone(result)

    def test_preserves_user_system_message(self) -> None:
        result = augment_system_prompt(
            "You are an Aether dev assistant.",
            _descriptors("shell"),
        )
        assert result is not None
        # User text appears verbatim with blank line before it.
        self.assertTrue(result.endswith("\n\nYou are an Aether dev assistant."))

    def test_no_user_prompt_omits_trailing_section(self) -> None:
        result = augment_system_prompt(None, _descriptors("shell"))
        assert result is not None
        # Should NOT have a trailing blank line followed by user text.
        self.assertTrue(result.endswith("</faithful_reporting>"))

    def test_blank_user_prompt_treated_as_empty(self) -> None:
        # Whitespace-only user prompt should not produce a dangling
        # blank section at the end.
        result = augment_system_prompt("   ", _descriptors("shell"))
        assert result is not None
        self.assertTrue(result.endswith("</faithful_reporting>"))


class BackwardsCompatibleAliasTests(unittest.TestCase):
    def test_alias_returns_full_default_assembly(self) -> None:
        # Existing callers use the narrower function name; they should
        # now receive all sections by default.
        alias = augment_system_with_tool_contract("ok", _descriptors("shell"))
        explicit = augment_system_prompt("ok", _descriptors("shell"), SystemPromptOptions())
        self.assertEqual(alias, explicit)

    def test_alias_preserves_empty_tools_behavior(self) -> None:
        # When no tools are registered the contract block is suppressed
        # (the directive blocks still appear — they don't depend on
        # tools). Verifies the "empty descriptors -> no tool contract"
        # property survives the refactor.
        result = augment_system_with_tool_contract("ok", _descriptors())
        assert result is not None
        self.assertNotIn("<tool_use_contract>", result)


class SectionContentTests(unittest.TestCase):
    def test_verification_directive_mentions_diagnostics_block(self) -> None:
        # The directive names <diagnostics> so the model knows to
        # honour those blocks when they appear.
        result = augment_system_prompt(None, _descriptors())
        assert result is not None
        self.assertIn("<diagnostics>", result)

    def test_faithful_reporting_bans_dishonest_summaries(self) -> None:
        result = augment_system_prompt(None, _descriptors())
        assert result is not None
        # Two anchor phrases that future prompt edits must preserve to
        # keep the behaviour intent.
        self.assertIn("all tests pass", result)
        self.assertIn("accurate report", result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
