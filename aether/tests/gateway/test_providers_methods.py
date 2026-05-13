"""Tests for the ``providers.*`` gateway methods (PR 3)."""

from __future__ import annotations

import io
import json
import time
import unittest

from aether.gateway.dispatcher import (
    _LONG_METHODS,
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.protocol import (
    ERROR_APPLICATION,
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


class _ProvidersCase(unittest.TestCase):
    def setUp(self) -> None:
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


class ListProviders(_ProvidersCase):
    def test_returns_all_known_providers(self) -> None:
        result = self._result("providers.list")
        names = [p["name"] for p in result["providers"]]
        # Aligned with aether/cli/providers.py:_DEFAULTS.
        self.assertEqual(sorted(names), ["claude", "codex", "openai"])

    def test_each_entry_has_required_fields(self) -> None:
        result = self._result("providers.list")
        for entry in result["providers"]:
            self.assertIn("name", entry)
            self.assertIn("display_name", entry)
            self.assertIn("requires_api_key", entry)
            self.assertIsInstance(entry["requires_api_key"], bool)

    def test_openai_advertises_default_base_url(self) -> None:
        result = self._result("providers.list")
        openai = next(p for p in result["providers"] if p["name"] == "openai")
        self.assertEqual(
            openai["default_base_url"], "https://api.openai.com/v1"
        )


class ListModels(_ProvidersCase):
    def test_known_provider_returns_models(self) -> None:
        result = self._result("providers.models", {"provider": "claude"})
        self.assertGreater(len(result["models"]), 0)
        ids = [m["id"] for m in result["models"]]
        self.assertIn("claude-sonnet-4-6", ids)

    def test_alias_resolves_to_canonical(self) -> None:
        # ``anthropic`` is an alias for ``claude``.
        a = self._result("providers.models", {"provider": "anthropic"})
        b = self._result("providers.models", {"provider": "claude"})
        self.assertEqual(a["models"], b["models"])

    def test_unknown_provider_raises_application_error(self) -> None:
        resp = self._call("providers.models", {"provider": "gibberish"})
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_APPLICATION)

    def test_missing_provider_param_rejected(self) -> None:
        for params in (None, {}, {"provider": ""}, {"provider": 1}):
            resp = self._call("providers.models", params)
            assert resp.error is not None
            self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)

    def test_model_entries_have_required_fields(self) -> None:
        result = self._result("providers.models", {"provider": "claude"})
        for entry in result["models"]:
            self.assertIn("id", entry)
            self.assertIn("display_name", entry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
