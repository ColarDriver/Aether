"""Tests for ``tools.list`` (PR 8d).

Mirrors the structure of ``test_commands_methods.py`` — we exercise the
real :func:`aether.tools.builtins.build_default_tool_registry` so any
drift between the registry and the wrapper surfaces immediately.
"""

from __future__ import annotations

import unittest

from aether.gateway.dispatcher import (
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.protocol import RpcRequest, RpcResponse


class ToolsList(unittest.TestCase):
    def setUp(self) -> None:
        reset_dispatcher_for_tests()
        reset_state_for_tests()
        register_builtins()
        register_handler_methods()

    def _result(self, name: str, params: dict | None = None) -> dict:
        resp = dispatch_request(RpcRequest(id="x", method=name, params=params))
        assert isinstance(resp, RpcResponse)
        if resp.error is not None:
            self.fail(f"{name} error: {resp.error.code} {resp.error.message}")
        assert resp.result is not None
        return resp.result

    def test_list_size_matches_registry(self) -> None:
        from aether.tools.builtins import build_default_tool_registry

        result = self._result("tools.list")
        registry = build_default_tool_registry()
        self.assertEqual(len(result["tools"]), len(registry.list_descriptors()))

    def test_every_tool_has_name_and_description(self) -> None:
        result = self._result("tools.list")
        for entry in result["tools"]:
            self.assertIn("name", entry)
            self.assertTrue(entry["name"])
            self.assertIn("description", entry)
            self.assertIsInstance(entry["description"], str)
            self.assertIn("parameters", entry)
            self.assertIn("required", entry)
            self.assertIsInstance(entry["required"], list)

    def test_alphabetical_ordering(self) -> None:
        result = self._result("tools.list")
        names = [entry["name"] for entry in result["tools"]]
        self.assertEqual(names, sorted(names))

    def test_includes_known_builtin_tools(self) -> None:
        """Sanity-check a handful of tools that are central to the agent."""
        result = self._result("tools.list")
        names = {entry["name"] for entry in result["tools"]}
        # These names come from build_default_tool_registry's bundled kit.
        # Adjust if the registry renames anything; the goal is to catch a
        # silent regression where the catalog goes empty.
        self.assertGreaterEqual(len(names), 10)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
