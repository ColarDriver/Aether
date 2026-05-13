"""Tests for ``commands.catalog`` (PR 3).

These tests pull the actual REGISTRY from ``aether.cli.commands`` so a
drift between the registry and the catalog wrapper is caught
automatically.  We deliberately don't hard-code the count of commands
— the slash registry changes occasionally, but the wrapper must keep
the count and metadata aligned.
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


class CommandsCatalog(unittest.TestCase):
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

    def test_catalog_size_matches_registry(self) -> None:
        from aether.cli.commands import REGISTRY

        result = self._result("commands.catalog")
        self.assertEqual(len(result["commands"]), len(REGISTRY))

    def test_every_registry_command_present_in_catalog(self) -> None:
        from aether.cli.commands import REGISTRY

        result = self._result("commands.catalog")
        catalog_names = {entry["name"] for entry in result["commands"]}
        registry_names = set(REGISTRY.keys())
        self.assertEqual(catalog_names, registry_names)

    def test_entries_carry_name_and_description(self) -> None:
        result = self._result("commands.catalog")
        for entry in result["commands"]:
            self.assertIn("name", entry)
            self.assertTrue(entry["name"].startswith("/"))
            self.assertIn("description", entry)
            self.assertIsInstance(entry["description"], str)
            self.assertTrue(entry["description"])

    def test_catalog_is_alphabetical_by_name(self) -> None:
        result = self._result("commands.catalog")
        names = [entry["name"] for entry in result["commands"]]
        self.assertEqual(names, sorted(names))

    def test_session_commands_categorised(self) -> None:
        result = self._result("commands.catalog")
        by_name = {entry["name"]: entry for entry in result["commands"]}
        for cmd in ("/new", "/session", "/sessions", "/resume", "/model"):
            self.assertEqual(
                by_name[cmd].get("category"),
                "session",
                msg=f"{cmd} should be categorised 'session'",
            )

    def test_control_commands_categorised(self) -> None:
        result = self._result("commands.catalog")
        by_name = {entry["name"]: entry for entry in result["commands"]}
        self.assertEqual(by_name["/interrupt"]["category"], "control")

    def test_help_is_local(self) -> None:
        result = self._result("commands.catalog")
        by_name = {entry["name"]: entry for entry in result["commands"]}
        self.assertEqual(by_name["/help"]["category"], "local")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
