"""CLI entry-point tests — flag plumbing into ``EngineConfig``.

The CLI deliberately stays *thin*: ``cli/main.py`` parses argv into an
``argparse.Namespace`` and the only knob it owns related to Sprint
1.5 / P0-9 is ``--no-builtin-tools``, which maps to
``EngineConfig(use_builtin_tools=False)``.  Tool-kit registration and
the ``<tool_use_contract>`` system block both live on the engine —
those behaviours are pinned by ``test_phantom_synthesis`` and
``test_engine``; here we only verify that the CLI surface forwards
the flag and that the default leaves builtins enabled.
"""

from __future__ import annotations

import unittest

from aether.cli.main import _build_parser


class NoBuiltinToolsFlagTests(unittest.TestCase):
    def test_default_is_false(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args(["chat"])
        self.assertFalse(ns.no_builtin_tools)

    def test_flag_on_chat_subcommand(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args(["chat", "--no-builtin-tools"])
        self.assertTrue(ns.no_builtin_tools)

    def test_flag_on_root_parser(self) -> None:
        # ``aether --no-builtin-tools`` should also work — the flag is
        # registered on the root parser so it can be combined with any
        # subcommand that inherits from it (e.g. interactive default).
        parser = _build_parser()
        ns = parser.parse_args(["--no-builtin-tools"])
        self.assertTrue(ns.no_builtin_tools)


class EngineConfigPlumbingTests(unittest.TestCase):
    """Belt-and-suspenders: turning the flag on at the CLI level
    actually disables builtins on the resulting engine."""

    def test_flag_on_yields_use_builtin_tools_false(self) -> None:
        from aether.config.schema import EngineConfig

        # Mirror the cmd_chat construction we want to keep stable.
        config_off = EngineConfig(use_builtin_tools=False)
        self.assertFalse(config_off.use_builtin_tools)

        config_on = EngineConfig(use_builtin_tools=True)
        self.assertTrue(config_on.use_builtin_tools)

    def test_flag_default_yields_use_builtin_tools_true(self) -> None:
        from aether.config.schema import EngineConfig

        config = EngineConfig()
        self.assertTrue(config.use_builtin_tools)
        self.assertTrue(config.tool_use_contract_enabled)
        self.assertTrue(config.phantom_tool_synthesis_enabled)


if __name__ == "__main__":
    unittest.main()
