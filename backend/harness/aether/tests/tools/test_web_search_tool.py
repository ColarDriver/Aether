"""Unit tests for ``aether.tools.builtins.web_search``.

Sprint 3.5 / PR-2 (PR 3.5.5).
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

import httpx

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.tools.builtins.web_search import WebSearchTool


def _ctx(*, config: EngineConfig | None = None) -> TurnContext:
    return TurnContext(
        session_id="ses-search",
        iteration=0,
        metadata={"_engine_config": config or EngineConfig(web_search_api_key="test-key")},
    )


class _StubResponse:
    def __init__(self, *, status_code: int = 200, json_payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._json = json_payload or {}
        self.headers: dict[str, str] = {}
        self._text = ""

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self  # type: ignore[arg-type]
            )


class _StubClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "_StubClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((url, params or {}))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _brave_payload(*, count: int) -> dict[str, Any]:
    return {
        "web": {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.test/{i}",
                    "description": f"Snippet for result {i}.",
                }
                for i in range(count)
            ]
        }
    }


class HappyPathTests(unittest.TestCase):
    def test_c2_five_results_formatted_as_markdown(self) -> None:
        client = _StubClient(_StubResponse(json_payload=_brave_payload(count=5)))
        tool = WebSearchTool(client_factory=lambda: client)
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "python asyncio"}),
            _ctx(),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertIn("Web search: python asyncio", result.content)
        self.assertIn("Found 5 results", result.content)
        for i in range(5):
            self.assertIn(f"https://example.test/{i}", result.content)
        self.assertEqual(result.metadata["result_count"], 5)
        self.assertEqual(client.calls[0][1]["q"], "python asyncio")

    def test_c3_zero_results_returns_no_results_message(self) -> None:
        client = _StubClient(_StubResponse(json_payload={"web": {"results": []}}))
        tool = WebSearchTool(client_factory=lambda: client)
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "blank"}),
            _ctx(),
        )
        self.assertFalse(result.is_error)
        self.assertIn("No results found", result.content)

    def test_c5_max_results_one_passes_one(self) -> None:
        client = _StubClient(_StubResponse(json_payload=_brave_payload(count=10)))
        tool = WebSearchTool(client_factory=lambda: client)
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "x", "max_results": 1}),
            _ctx(),
        )
        self.assertEqual(result.metadata["max_results"], 1)
        self.assertEqual(client.calls[0][1]["count"], 1)
        self.assertIn("Found 1 results", result.content)
        self.assertNotIn("https://example.test/1", result.content)

    def test_c6_oversize_max_results_clipped(self) -> None:
        client = _StubClient(_StubResponse(json_payload=_brave_payload(count=20)))
        tool = WebSearchTool(client_factory=lambda: client)
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "x", "max_results": 100}),
            _ctx(),
        )
        self.assertEqual(result.metadata["max_results"], 20)
        self.assertEqual(client.calls[0][1]["count"], 20)


class FailureTests(unittest.TestCase):
    def test_c1_no_api_key_returns_error(self) -> None:
        cfg = EngineConfig(web_search_api_key=None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_API_KEY", None)
            tool = WebSearchTool(client_factory=lambda: _StubClient(_StubResponse()))
            result = tool.execute(
                ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
                _ctx(config=cfg),
            )
        self.assertTrue(result.is_error)
        self.assertIn("API key", result.content)

    def test_c1b_env_var_satisfies_key(self) -> None:
        cfg = EngineConfig(web_search_api_key=None)
        client = _StubClient(_StubResponse(json_payload=_brave_payload(count=1)))
        with patch.dict(os.environ, {"BRAVE_API_KEY": "from-env"}):
            tool = WebSearchTool(client_factory=lambda: client)
            result = tool.execute(
                ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
                _ctx(config=cfg),
            )
        self.assertFalse(result.is_error)

    def test_c4_4xx_returns_error(self) -> None:
        client = _StubClient(_StubResponse(status_code=403))
        tool = WebSearchTool(client_factory=lambda: client)
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("HTTP 403", result.content)

    def test_c7_timeout_returns_error(self) -> None:
        tool = WebSearchTool(client_factory=lambda: _StubClient(httpx.TimeoutException("slow")))
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)

    def test_c8_disabled_via_config(self) -> None:
        cfg = EngineConfig(web_search_enabled=False, web_search_api_key="test")
        tool = WebSearchTool(client_factory=lambda: _StubClient(_StubResponse()))
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)

    def test_c9_unknown_provider_rejected(self) -> None:
        cfg = EngineConfig(web_search_provider="serpapi", web_search_api_key="test")
        tool = WebSearchTool(client_factory=lambda: _StubClient(_StubResponse()))
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "x"}),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("not implemented", result.content)

    def test_c10_empty_query_rejected(self) -> None:
        tool = WebSearchTool(client_factory=lambda: _StubClient(_StubResponse()))
        result = tool.execute(
            ToolCall(id="c1", name="web_search", arguments={"query": "  "}),
            _ctx(),
        )
        self.assertTrue(result.is_error)


class DescriptorTests(unittest.TestCase):
    def test_c11_descriptor_shape(self) -> None:
        d = WebSearchTool().descriptor
        self.assertEqual(d.name, "web_search")
        self.assertEqual(d.parameters["properties"]["max_results"]["maximum"], 20)
        self.assertEqual(d.parameters["properties"]["max_results"]["minimum"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
