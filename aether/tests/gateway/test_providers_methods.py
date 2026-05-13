"""Tests for the ``providers.*`` gateway methods (PR 3)."""

from __future__ import annotations

import io
import json
import os
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


class LiveModelDiscovery(_ProvidersCase):
    """The picker must reflect what the provider's `/v1/models` returns
    when the call succeeds, and silently fall back to the static catalog
    on any failure (no API key, network error, malformed response). We
    monkey-patch the provider class rather than the gateway to keep
    coverage close to the real call path.
    """

    def setUp(self) -> None:
        super().setUp()
        self._provider_ids: list[str] = []
        self._provider_error: Exception | None = None

        class _FakeProvider:
            def list_models(inner_self) -> list[str]:
                if self._provider_error is not None:
                    raise self._provider_error
                return list(self._provider_ids)

        self._fake_provider = _FakeProvider()
        self._build_provider = mock.Mock(return_value=self._fake_provider)
        self._build_provider_patch = mock.patch(
            "aether.gateway.handlers.providers_methods.build_provider",
            self._build_provider,
        )
        self._build_provider_patch.start()
        self.addCleanup(self._build_provider_patch.stop)
        self._openai_patch = mock.patch(
            "aether.gateway.handlers.providers_methods._live_list_models_openai",
            self._fake_openai_discovery,
        )
        self._openai_patch.start()
        self.addCleanup(self._openai_patch.stop)

    def _fake_openai_discovery(
        self,
        base_url: str | None,
        discovery: dict[str, object],
    ) -> tuple[list[str], dict[str, object]]:
        if self._provider_error is not None:
            return [], {
                "kind": "static",
                "reason": "list_models_error",
                "error": str(self._provider_error),
                **({"base_url": base_url} if base_url else {}),
            }
        if not self._provider_ids:
            return [], {
                "kind": "static",
                "reason": "empty_response",
                **({"base_url": base_url} if base_url else {}),
            }
        cleaned = sorted(set(self._provider_ids))
        discovery["count"] = len(cleaned)
        return cleaned, discovery

    def test_live_ids_replace_static_catalog_when_provider_responds(self) -> None:
        self._provider_ids = [
            "kimi-k2",
            "kimi-k2-thinking",
            "kimi-k2.5",
            "kimi-k2.6",
            "gpt-4o",
        ]
        result = self._result("providers.models", {"provider": "openai"})
        ids = [entry["id"] for entry in result["models"]]
        self.assertIn("kimi-k2.6", ids)
        self.assertIn("gpt-4o", ids)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(result["discovery"]["kind"], "live")
        self.assertEqual(result["discovery"]["count"], 5)

    def test_falls_back_to_static_catalog_when_provider_returns_empty(self) -> None:
        self._provider_ids = []
        result = self._result("providers.models", {"provider": "openai"})
        ids = [entry["id"] for entry in result["models"]]
        # Static catalog still contains gpt-4o; verify presence rather than
        # exact equality so future catalog tweaks do not break this test.
        self.assertIn("gpt-4o", ids)
        self.assertEqual(result["discovery"]["kind"], "static")
        self.assertEqual(result["discovery"]["reason"], "empty_response")

    def test_discovery_reports_list_models_error(self) -> None:
        self._provider_error = RuntimeError("upstream timed out")
        result = self._result("providers.models", {"provider": "openai"})
        self.assertEqual(result["discovery"]["kind"], "static")
        self.assertEqual(result["discovery"]["reason"], "list_models_error")
        self.assertIn("upstream timed out", result["discovery"]["error"])

    def test_forwards_aether_api_key_to_provider_factory(self) -> None:
        self._provider_ids = ["claude-custom"]
        with mock.patch.dict(os.environ, {"AETHER_API_KEY": "sk-test"}, clear=False):
            self._result("providers.models", {"provider": "claude"})
        self._build_provider.assert_called()
        self.assertEqual(self._build_provider.call_args.kwargs["api_key"], "sk-test")


class SuggestBaseUrl(unittest.TestCase):
    """Detect the most common /model "looks fine, chat 404s" trap: the
    user's `OPENAI_BASE_URL` is missing the `/v1` prefix that the chat
    endpoint expects, but discovery finds models at `…/v1/models` anyway.
    """

    def test_no_suggestion_when_base_url_already_matches_api_root(self) -> None:
        from aether.gateway.handlers.providers_methods import _suggest_base_url

        self.assertIsNone(
            _suggest_base_url(
                "http://endpoint.local/v1",
                "http://endpoint.local/v1/models",
            )
        )

    def test_suggests_v1_when_base_url_is_root(self) -> None:
        from aether.gateway.handlers.providers_methods import _suggest_base_url

        self.assertEqual(
            _suggest_base_url(
                "http://endpoint.local",
                "http://endpoint.local/v1/models",
            ),
            "http://endpoint.local/v1",
        )

    def test_suggests_full_root_when_base_url_is_unset(self) -> None:
        from aether.gateway.handlers.providers_methods import _suggest_base_url

        self.assertEqual(
            _suggest_base_url(None, "http://endpoint.local/v1/models"),
            "http://endpoint.local/v1",
        )

    def test_returns_none_for_non_models_probes(self) -> None:
        from aether.gateway.handlers.providers_methods import _suggest_base_url

        self.assertIsNone(
            _suggest_base_url(
                "http://endpoint.local",
                "http://endpoint.local/something-else",
            )
        )


class CandidateUrls(unittest.TestCase):
    def test_walks_common_probes(self) -> None:
        from aether.gateway.handlers.providers_methods import _candidate_urls

        urls = _candidate_urls("http://endpoint.local")
        self.assertEqual(
            urls,
            [
                "http://endpoint.local/models",
                "http://endpoint.local/v1/models",
                "http://endpoint.local/api/models",
            ],
        )

    def test_appends_models_for_v1_only_base_url(self) -> None:
        from aether.gateway.handlers.providers_methods import _candidate_urls

        urls = _candidate_urls("http://endpoint.local/v1")
        # /models on the stripped base is `/v1/models`, which is the same
        # URL appended for `/v1` bases — make sure we don't duplicate.
        self.assertEqual(urls.count("http://endpoint.local/v1/models"), 1)


class ExtractModelIds(unittest.TestCase):
    """Tolerant parser must handle every common /models response shape so
    custom OpenAI-compatible servers (vLLM, Ollama, Kimi-style) don't
    silently fall through to the static catalog."""

    def test_canonical_openai_shape(self) -> None:
        from aether.gateway.handlers.providers_methods import _extract_model_ids

        self.assertEqual(
            _extract_model_ids({"data": [{"id": "gpt-4o"}, {"id": "gpt-4.1"}]}),
            ["gpt-4o", "gpt-4.1"],
        )

    def test_models_key_with_objects(self) -> None:
        from aether.gateway.handlers.providers_methods import _extract_model_ids

        self.assertEqual(
            _extract_model_ids({"models": [{"id": "kimi-k2"}, {"id": "kimi-k2.6"}]}),
            ["kimi-k2", "kimi-k2.6"],
        )

    def test_top_level_array_of_strings(self) -> None:
        from aether.gateway.handlers.providers_methods import _extract_model_ids

        self.assertEqual(_extract_model_ids(["a", "b"]), ["a", "b"])

    def test_data_array_of_strings(self) -> None:
        from aether.gateway.handlers.providers_methods import _extract_model_ids

        self.assertEqual(_extract_model_ids({"data": ["a", "b"]}), ["a", "b"])

    def test_object_with_name_field(self) -> None:
        from aether.gateway.handlers.providers_methods import _extract_model_ids

        self.assertEqual(
            _extract_model_ids({"data": [{"name": "foo"}]}),
            ["foo"],
        )

    def test_unknown_shape_returns_empty(self) -> None:
        from aether.gateway.handlers.providers_methods import _extract_model_ids

        self.assertEqual(_extract_model_ids({"weird": True}), [])
        self.assertEqual(_extract_model_ids("not an object"), [])
        self.assertEqual(_extract_model_ids(None), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
