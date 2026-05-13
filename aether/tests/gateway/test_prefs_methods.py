"""Tests for the ``prefs.*`` gateway methods (PR 3)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from aether.gateway.dispatcher import (
    _LONG_METHODS,
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.protocol import (
    ERROR_INVALID_PARAMS,
    RpcRequest,
    RpcResponse,
)
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)


class _PrefsMethodsCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ, {"AETHER_HOME": self._tmp.name}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        reset_state_for_tests()
        register_builtins()
        register_handler_methods()

        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)

    def _call(self, name: str, params: dict | None = None) -> RpcResponse:
        self._buf.seek(0)
        self._buf.truncate(0)
        resp = dispatch_request(
            RpcRequest(id=name, method=name, params=params),
            transport=self._sink,
        )
        if resp is not None:
            return resp
        if name not in _LONG_METHODS:
            self.fail(f"{name}: None response but not registered long")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self._buf.getvalue().strip():
            time.sleep(0.01)
        return RpcResponse.model_validate(json.loads(self._buf.getvalue().strip()))

    def _result(self, name: str, params: dict | None = None) -> dict:
        resp = self._call(name, params)
        if resp.error is not None:
            self.fail(f"{name} error: {resp.error.code} {resp.error.message}")
        assert resp.result is not None
        return resp.result


class GetSetRoundTrip(_PrefsMethodsCase):
    def test_unknown_key_returns_null(self) -> None:
        result = self._result("prefs.get", {"key": "never.set"})
        self.assertIsNone(result["value"])

    def test_simple_unknown_key_round_trip(self) -> None:
        self._result("prefs.set", {"key": "ui.tip_seen", "value": "true"})
        result = self._result("prefs.get", {"key": "ui.tip_seen"})
        self.assertEqual(result["value"], "true")

    def test_set_then_get_persists_across_load(self) -> None:
        from aether.cli.prefs import load_prefs

        self._result(
            "prefs.set",
            {"key": "last_model_by_provider.claude", "value": "claude-haiku-4-5-20251001"},
        )
        # Independently load from disk to confirm atomic write happened.
        prefs = load_prefs()
        self.assertEqual(
            prefs.last_model_by_provider["claude"], "claude-haiku-4-5-20251001"
        )


class NestedKeySyntax(_PrefsMethodsCase):
    def test_set_individual_provider_model(self) -> None:
        self._result(
            "prefs.set",
            {"key": "last_model_by_provider.openai", "value": "gpt-4o"},
        )
        self._result(
            "prefs.set",
            {"key": "last_model_by_provider.claude", "value": "claude-sonnet-4-6"},
        )

        result = self._result(
            "prefs.get", {"key": "last_model_by_provider.openai"}
        )
        self.assertEqual(result["value"], "gpt-4o")

        all_pref = self._result("prefs.get", {"key": "last_model_by_provider"})
        self.assertEqual(
            all_pref["value"],
            {"openai": "gpt-4o", "claude": "claude-sonnet-4-6"},
        )

    def test_set_with_null_value_unsets_provider_slot(self) -> None:
        self._result(
            "prefs.set",
            {"key": "last_model_by_provider.openai", "value": "gpt-4o"},
        )
        self._result(
            "prefs.set",
            {"key": "last_model_by_provider.openai", "value": None},
        )
        result = self._result(
            "prefs.get", {"key": "last_model_by_provider.openai"}
        )
        self.assertIsNone(result["value"])

    def test_bulk_replace_last_model_dict(self) -> None:
        self._result(
            "prefs.set",
            {
                "key": "last_model_by_provider",
                "value": {"claude": "x", "openai": "y"},
            },
        )
        result = self._result("prefs.get", {"key": "last_model_by_provider"})
        self.assertEqual(result["value"], {"claude": "x", "openai": "y"})

    def test_bulk_replace_rejects_non_dict(self) -> None:
        resp = self._call(
            "prefs.set",
            {"key": "last_model_by_provider", "value": "not-a-dict"},
        )
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)

    def test_dotted_key_without_provider_segment_rejected(self) -> None:
        resp = self._call(
            "prefs.set",
            {"key": "last_model_by_provider.", "value": "x"},
        )
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)


class AllReturnsFullBlob(_PrefsMethodsCase):
    def test_all_has_version_and_provider_map(self) -> None:
        self._result(
            "prefs.set",
            {"key": "last_model_by_provider.claude", "value": "claude-sonnet-4-6"},
        )
        result = self._result("prefs.all")
        prefs = result["prefs"]
        self.assertIn("version", prefs)
        self.assertEqual(
            prefs["last_model_by_provider"], {"claude": "claude-sonnet-4-6"}
        )

    def test_unknown_round_trip_fields_surface(self) -> None:
        self._result("prefs.set", {"key": "ui.banner_seen", "value": "1"})
        result = self._result("prefs.all")
        self.assertEqual(result["prefs"]["ui.banner_seen"], "1")


class InvalidParams(_PrefsMethodsCase):
    def test_get_requires_string_key(self) -> None:
        for params in (None, {}, {"key": ""}, {"key": 42}):
            resp = self._call("prefs.get", params)
            assert resp.error is not None
            self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)

    def test_set_requires_string_key(self) -> None:
        for params in (None, {}, {"key": "", "value": 1}):
            resp = self._call("prefs.set", params)
            assert resp.error is not None
            self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)

    def test_version_is_read_only(self) -> None:
        resp = self._call("prefs.set", {"key": "version", "value": 99})
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
