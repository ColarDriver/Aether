"""Unit tests for ``aether.tools.builtins.web_fetch``.

Sprint 3.5 / PR-2 (PR 3.5.5).
"""

from __future__ import annotations

import socket
import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.tools.builtins.web_fetch import WebFetchTool


def _addrinfo(addr: str = "1.2.3.4") -> list:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (addr, 0))]


def _ctx(session_id: str = "ses-A", *, config: EngineConfig | None = None) -> TurnContext:
    return TurnContext(
        session_id=session_id,
        iteration=0,
        metadata={"_engine_config": config or EngineConfig()},
    )


class _StubResponse:
    def __init__(self, *, status_code: int, body: bytes, content_type: str = "text/html") -> None:
        self.status_code = status_code
        self.content = body
        self.headers = {"content-type": content_type}


class _StubClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[str] = []

    def __enter__(self) -> "_StubClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str) -> Any:
        self.calls.append(url)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@contextmanager
def _fake_client(response: Any):
    stub = _StubClient(response)
    try:
        yield (lambda: stub), stub
    finally:
        pass


class WebFetchHappyPathTests(unittest.TestCase):
    def test_b1_simple_html_returns_markdown_with_status(self) -> None:
        body = b"<html><body><h1>Hi</h1><p>world</p></body></html>"
        with _fake_client(_StubResponse(status_code=200, body=body)) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="c1", name="web_fetch", arguments={"url": "https://example.test/", "prompt": "find headings"}),
                    _ctx(),
                )
        self.assertFalse(result.is_error, result.content)
        self.assertIn("HTTP status: 200", result.content)
        self.assertIn("find headings", result.content)
        self.assertIn("Hi", result.content)
        self.assertEqual(result.metadata["status"], 200)

    def test_b6_json_content_type_pretty_printed(self) -> None:
        body = b'{"ok": true, "items": [1, 2, 3]}'
        with _fake_client(_StubResponse(status_code=200, body=body, content_type="application/json")) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="c1", name="web_fetch", arguments={"url": "https://example.test/api", "prompt": "summarise"}),
                    _ctx(),
                )
        self.assertIn("```json", result.content)
        self.assertIn('"ok": true', result.content)

    def test_b6b_plain_text_content_type_returned_verbatim(self) -> None:
        body = b"plain text body"
        with _fake_client(_StubResponse(status_code=200, body=body, content_type="text/plain")) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="c1", name="web_fetch", arguments={"url": "https://example.test/", "prompt": "x"}),
                    _ctx(),
                )
        self.assertIn("plain text body", result.content)

    def test_b5_5xx_returns_status_inline(self) -> None:
        body = b"<html>err</html>"
        with _fake_client(_StubResponse(status_code=503, body=body)) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="c1", name="web_fetch", arguments={"url": "https://example.test/", "prompt": "x"}),
                    _ctx(),
                )
        self.assertFalse(result.is_error)
        self.assertIn("HTTP status: 503", result.content)

    def test_b3_oversize_html_spills_to_disk(self) -> None:
        big = b"<html><body>" + (b"X" * 200_000) + b"</body></html>"
        with _fake_client(_StubResponse(status_code=200, body=big)) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="cBig", name="web_fetch", arguments={"url": "https://example.test/", "prompt": "x"}),
                    _ctx(session_id="ses-spill"),
                )
        self.assertIn("output truncated", result.content)
        self.assertIn("saved to", result.content)


class WebFetchSafetyTests(unittest.TestCase):
    def test_b2_loopback_url_refused_no_request_made(self) -> None:
        with _fake_client(_StubResponse(status_code=200, body=b"")) as (factory, stub):
            tool = WebFetchTool(client_factory=factory)
            result = tool.execute(
                ToolCall(id="c1", name="web_fetch", arguments={"url": "http://127.0.0.1/", "prompt": "x"}),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("refused", result.content)
        self.assertEqual(stub.calls, [])

    def test_b2b_file_scheme_refused(self) -> None:
        tool = WebFetchTool(client_factory=lambda: _StubClient(_StubResponse(status_code=200, body=b"")))
        result = tool.execute(
            ToolCall(id="c1", name="web_fetch", arguments={"url": "file:///etc/passwd", "prompt": "x"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)


class WebFetchValidationTests(unittest.TestCase):
    def _tool(self) -> WebFetchTool:
        return WebFetchTool(client_factory=lambda: _StubClient(_StubResponse(status_code=200, body=b"")))

    def test_b9_missing_url_rejected(self) -> None:
        tool = self._tool()
        result = tool.execute(ToolCall(id="c1", name="web_fetch", arguments={"prompt": "x"}), _ctx())
        self.assertTrue(result.is_error)

    def test_b10_missing_prompt_rejected(self) -> None:
        tool = self._tool()
        result = tool.execute(ToolCall(id="c1", name="web_fetch", arguments={"url": "https://x.test/"}), _ctx())
        self.assertTrue(result.is_error)

    def test_b11_disabled_via_config(self) -> None:
        cfg = EngineConfig(web_fetch_enabled=False)
        tool = self._tool()
        result = tool.execute(
            ToolCall(id="c1", name="web_fetch", arguments={"url": "https://x.test/", "prompt": "p"}),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("disabled", result.content)


class WebFetchErrorTests(unittest.TestCase):
    def test_b4_timeout_returns_structured_error(self) -> None:
        with _fake_client(httpx.TimeoutException("slow")) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="c1", name="web_fetch", arguments={"url": "https://x.test/", "prompt": "x"}),
                    _ctx(),
                )
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)

    def test_b12_generic_http_error_surfaced(self) -> None:
        with _fake_client(httpx.ConnectError("nope")) as (factory, _):
            tool = WebFetchTool(client_factory=factory)
            with patch("socket.getaddrinfo", return_value=_addrinfo()):
                result = tool.execute(
                    ToolCall(id="c1", name="web_fetch", arguments={"url": "https://x.test/", "prompt": "x"}),
                    _ctx(),
                )
        self.assertTrue(result.is_error)
        self.assertIn("fetch failed", result.content)


class WebFetchDescriptorTests(unittest.TestCase):
    def test_b13_descriptor_shape(self) -> None:
        tool = WebFetchTool()
        d = tool.descriptor
        self.assertEqual(d.name, "web_fetch")
        self.assertEqual(d.parameters["type"], "object")
        self.assertIn("url", d.parameters["properties"])
        self.assertIn("prompt", d.parameters["properties"])
        self.assertEqual(sorted(d.required), ["prompt", "url"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
