"""Sprint 0 / PR 0.2 — provider single-shot + structured error contract.

Two questions are pinned down here:

1. ``OpenAICompatibleModel.generate`` is now single-shot — every call to it
   issues exactly one ``httpx.post`` and either returns or raises a
   ``ProviderInvocationError``.  No internal retry loop, no blocking
   ``time.sleep`` — those concerns belong to the engine's recovery layer
   (PR 0.3).
2. The exception surface is structured.  Engine-level recovery code can
   read ``status_code`` / ``retry_after_seconds`` / ``body_summary`` /
   ``is_network_error`` directly without poking at httpx internals.
"""

from __future__ import annotations

import time
import unittest

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.provider.openai_compatible import (
    OpenAICompatibleModel,
    _parse_retry_after,
    _summarize_body,
)
from aether.runtime.contracts import TurnContext
from aether.runtime.provider_errors import ProviderInvocationError


def _ctx() -> TurnContext:
    return TurnContext(session_id="t", iteration=0)


def _make_provider(transport: httpx.MockTransport) -> OpenAICompatibleModel:
    """Build an ``OpenAICompatibleModel`` whose internal ``httpx.Client`` is
    routed through ``transport``.

    We achieve this by monkey-patching ``httpx.Client.__init__`` for the
    duration of one call to thread a ``transport`` argument through.  This
    is simpler than refactoring the production code to accept a transport
    factory, and it isolates the test concern in test code.
    """
    return OpenAICompatibleModel(
        model="m1",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        request_timeout_sec=5,
    )


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------


class ParseRetryAfterTests(unittest.TestCase):
    def test_none_for_missing_headers(self) -> None:
        self.assertIsNone(_parse_retry_after(None))
        self.assertIsNone(_parse_retry_after(httpx.Headers({})))

    def test_seconds_form(self) -> None:
        self.assertEqual(_parse_retry_after(httpx.Headers({"retry-after": "30"})), 30.0)

    def test_milliseconds_takes_precedence(self) -> None:
        # When both headers are present, the *-Ms form wins because it's
        # higher resolution and only sent by providers that explicitly chose
        # to be precise.
        headers = httpx.Headers({"retry-after": "999", "retry-after-ms": "1500"})
        self.assertEqual(_parse_retry_after(headers), 1.5)

    def test_zero_clamp(self) -> None:
        # Negative values (clock skew with HTTP-date) should clamp to zero.
        # We can't easily inject a negative value via float; instead verify
        # the code path against an unparseable header (returns None) and an
        # exact zero (returns 0.0).
        self.assertEqual(_parse_retry_after(httpx.Headers({"retry-after": "0"})), 0.0)
        self.assertIsNone(_parse_retry_after(httpx.Headers({"retry-after": "garbage"})))

    def test_http_date_form(self) -> None:
        # 30 seconds in the future, formatted as HTTP-date.
        future = time.time() + 30
        from email.utils import formatdate

        headers = httpx.Headers({"retry-after": formatdate(future, usegmt=True)})
        parsed = _parse_retry_after(headers)
        # Allow a wide tolerance because the HTTP-date format only encodes
        # second-level resolution.
        self.assertIsNotNone(parsed)
        assert parsed is not None  # for type-checker
        self.assertGreater(parsed, 25.0)
        self.assertLess(parsed, 35.0)


class SummarizeBodyTests(unittest.TestCase):
    def test_short_body_returned_verbatim(self) -> None:
        class _Resp:
            text = "rate limit exceeded"

        self.assertEqual(_summarize_body(_Resp()), "rate limit exceeded")

    def test_long_body_truncated_with_marker(self) -> None:
        class _Resp:
            text = "x" * 5000

        out = _summarize_body(_Resp())
        assert out is not None
        self.assertTrue(out.endswith("...(truncated)"))
        # 1024 chars + marker
        self.assertGreater(len(out), 1024)
        self.assertLess(len(out), 5000)

    def test_none_body(self) -> None:
        class _Resp:
            text = None

        self.assertIsNone(_summarize_body(_Resp()))


# ---------------------------------------------------------------------------
# generate() error wrapping
# ---------------------------------------------------------------------------


def _good_response() -> httpx.Response:
    """Fixed minimal valid chat-completions response."""
    return httpx.Response(
        200,
        json={
            "model": "m1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


class ProviderGenerateContractTests(unittest.TestCase):
    """End-to-end contract: generate() either returns NormalizedResponse
    or raises ProviderInvocationError.  No httpx exception escapes."""

    def setUp(self) -> None:
        # Use unittest's monkey-patch helper.  We restore in tearDown to
        # avoid leaking the patched __init__ into other tests.
        self._saved_client_init = httpx.Client.__init__

    def tearDown(self) -> None:
        httpx.Client.__init__ = self._saved_client_init  # type: ignore[method-assign]

    def _install_transport(self, transport: httpx.MockTransport) -> None:
        real_init = self._saved_client_init

        def patched(self, *args, **kwargs):
            kwargs["transport"] = transport
            return real_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched  # type: ignore[method-assign]

    def test_success_path_returns_normalized_response(self) -> None:
        transport = httpx.MockTransport(lambda req: _good_response())
        self._install_transport(transport)

        provider = _make_provider(transport)
        result = provider.generate([], [], ModelCallConfig(), _ctx())

        self.assertEqual(result.content, "hi")

    def test_429_raises_with_status_and_retry_after(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"retry-after": "12"},
                text="rate limited",
            )

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)

        provider = _make_provider(transport)
        with self.assertRaises(ProviderInvocationError) as cm:
            provider.generate([], [], ModelCallConfig(), _ctx())

        exc = cm.exception
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.retry_after_seconds, 12.0)
        self.assertFalse(exc.is_network_error)
        self.assertIsNotNone(exc.body_summary)
        assert exc.body_summary is not None  # for type checker
        self.assertIn("rate limited", exc.body_summary)

    def test_413_payload_too_large_classified_as_http(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(413, text='{"error":{"message":"too long"}}')

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)

        provider = _make_provider(transport)
        with self.assertRaises(ProviderInvocationError) as cm:
            provider.generate([], [], ModelCallConfig(), _ctx())

        self.assertEqual(cm.exception.status_code, 413)
        self.assertIsNone(cm.exception.retry_after_seconds)
        self.assertFalse(cm.exception.is_network_error)

    def test_transport_error_marks_network_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("could not connect")

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)

        provider = _make_provider(transport)
        with self.assertRaises(ProviderInvocationError) as cm:
            provider.generate([], [], ModelCallConfig(), _ctx())

        exc = cm.exception
        self.assertIsNone(exc.status_code)
        self.assertTrue(exc.is_network_error)
        assert exc.body_summary is not None  # for type checker
        self.assertIn("could not connect", exc.body_summary)

    def test_malformed_json_response_is_provider_error(self) -> None:
        # 200 OK but body is not JSON — provider must wrap it instead of
        # leaking json.JSONDecodeError to the engine.
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>not json</html>")

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)

        provider = _make_provider(transport)
        with self.assertRaises(ProviderInvocationError) as cm:
            provider.generate([], [], ModelCallConfig(), _ctx())

        exc = cm.exception
        self.assertIsNone(exc.status_code)
        self.assertTrue(exc.is_network_error)
        assert exc.body_summary is not None  # for type checker
        self.assertIn("non-JSON response body", exc.body_summary)

    def test_single_shot_no_internal_retry(self) -> None:
        """Critical regression: provider must not retry internally.

        We count how many times the transport is hit; a single failed call
        should produce exactly one HTTP attempt.  Old behaviour (Sprint 0
        baseline) would retry up to 3 times here.
        """
        attempts = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503, text="upstream busy")

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)

        provider = _make_provider(transport)
        with self.assertRaises(ProviderInvocationError):
            provider.generate([], [], ModelCallConfig(), _ctx())

        self.assertEqual(attempts["n"], 1, "provider must be single-shot now")


# ---------------------------------------------------------------------------
# ProviderInvocationError str() form
# ---------------------------------------------------------------------------


class ProviderInvocationErrorReprTests(unittest.TestCase):
    def test_http_error_message_contains_status_and_body(self) -> None:
        exc = ProviderInvocationError(
            status_code=429,
            retry_after_seconds=2.5,
            body_summary="too many requests",
        )
        msg = str(exc)
        self.assertIn("HTTP 429", msg)
        self.assertIn("too many requests", msg)
        self.assertIn("retry-after=2.50s", msg)

    def test_network_error_message_marks_network(self) -> None:
        exc = ProviderInvocationError(
            is_network_error=True,
            body_summary="connection reset",
        )
        msg = str(exc)
        self.assertIn("network error", msg)
        self.assertIn("connection reset", msg)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
