"""Sprint 1 / PR 1.1 — SSE streaming + stale-stream detection.

Pins down three pieces of behaviour:

1. ``_parse_sse_stream`` — pure SSE event-stream parser.  Forwards
   per-chunk content via ``stream_callback``, accumulates tool-call
   argument fragments, ignores keep-alive/comment lines, recovers from
   garbled JSON frames, and reconstitutes a final ``NormalizedResponse``
   with a complete content string and parsed tool calls.
2. ``OpenAICompatibleModel.generate`` — chooses streaming when a
   ``stream_callback`` is supplied, falls back to non-streaming when
   the SSE stream stalls, and persistently disables streaming on the
   instance after a stall (so the second call goes straight to
   non-streaming without a wasted round-trip).
3. ``StreamStallError`` — surfaces structured info (``stalled_after_seconds``)
   that downstream observers / recovery strategies can read.
"""

from __future__ import annotations

import unittest

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.provider.openai_compatible import (
    OpenAICompatibleModel,
    _parse_sse_stream,
)
from aether.runtime.contracts import TurnContext
from aether.runtime.provider_errors import (
    ProviderInvocationError,
    StreamStallError,
)


def _ctx() -> TurnContext:
    return TurnContext(session_id="t", iteration=0, metadata={})


def _make_provider() -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        model="m1",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        request_timeout_sec=5,
    )


# ---------------------------------------------------------------------------
# _parse_sse_stream — pure parser tests
# ---------------------------------------------------------------------------


class ParseSSEStreamTests(unittest.TestCase):
    def test_forwards_each_content_chunk_in_order(self) -> None:
        # Three content chunks separated by SSE blank-line delimiters.
        # The parser must invoke stream_callback once per chunk, in order,
        # and accumulate them into the final content string.
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":", "}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"world!"},"finish_reason":"stop"}]}',
            "",
            "data: [DONE]",
        ]
        seen: list[str] = []

        result = _parse_sse_stream(
            iter(lines),
            stream_callback=seen.append,
            fallback_model="m1",
        )

        self.assertEqual(seen, ["Hello", ", ", "world!"])
        self.assertEqual(result.content, "Hello, world!")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_calls, [])
        self.assertTrue(result.metadata["streamed"])

    def test_ignores_comment_and_event_lines(self) -> None:
        lines = [
            ":keep-alive ping",
            "event: ping",
            "data: {\"choices\":[{\"delta\":{\"content\":\"X\"}}]}",
            "",
            "data: [DONE]",
        ]
        seen: list[str] = []
        result = _parse_sse_stream(iter(lines), stream_callback=seen.append, fallback_model="m1")
        self.assertEqual(seen, ["X"])
        self.assertEqual(result.content, "X")

    def test_skips_malformed_json_frame_and_continues(self) -> None:
        # A garbled data frame must NOT abort the whole generation; the
        # parser logs and skips, then carries on with the next event.
        lines = [
            "data: {this is not json",
            "",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "",
            "data: [DONE]",
        ]
        seen: list[str] = []
        result = _parse_sse_stream(iter(lines), stream_callback=seen.append, fallback_model="m1")
        self.assertEqual(seen, ["ok"])
        self.assertEqual(result.content, "ok")

    def test_buffers_tool_call_arguments_across_chunks(self) -> None:
        # The chat-completions streaming protocol splits tool-call args
        # into many fragments tagged with the same ``index``.  We must
        # buffer and reassemble them rather than forwarding fragments to
        # the user.
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"sum","arguments":"{\\"a\\":1"}}]}}]}',
            "",
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\\"b\\":2}"}}]}}]}',
            "",
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "",
            "data: [DONE]",
        ]
        seen: list[str] = []
        result = _parse_sse_stream(iter(lines), stream_callback=seen.append, fallback_model="m1")

        # No content chunks; user shouldn't have seen any text.
        self.assertEqual(seen, [])
        self.assertEqual(result.content, "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call.id, "c1")
        self.assertEqual(call.name, "sum")
        self.assertEqual(call.arguments, {"a": 1, "b": 2})

    def test_normalises_finish_reason_when_tool_calls_present(self) -> None:
        # Some gateways send finish_reason="stop" even when tool calls are
        # present.  The parser overrides this to "tool_calls" so the
        # engine's branch decision works.
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"x","arguments":"{}"}}]}}]}',
            "",
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "",
            "data: [DONE]",
        ]
        result = _parse_sse_stream(iter(lines), stream_callback=lambda _: None, fallback_model="m1")
        self.assertEqual(result.finish_reason, "tool_calls")

    def test_captures_terminal_usage_block(self) -> None:
        # OpenAI sends usage in a final event whose choices list is empty.
        lines = [
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
            "",
            'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}',
            "",
            "data: [DONE]",
        ]
        result = _parse_sse_stream(iter(lines), stream_callback=lambda _: None, fallback_model="m1")
        self.assertEqual(result.metadata["usage"]["prompt_tokens"], 7)
        self.assertEqual(result.metadata["token_usage"]["total_tokens"], 8)


# ---------------------------------------------------------------------------
# OpenAICompatibleModel.generate — streaming dispatch & fallback
# ---------------------------------------------------------------------------


class StreamingDispatchTests(unittest.TestCase):
    """Exercises the streaming-vs-non-streaming dispatcher in generate()."""

    def setUp(self) -> None:
        self._saved_client_init = httpx.Client.__init__

    def tearDown(self) -> None:
        httpx.Client.__init__ = self._saved_client_init  # type: ignore[method-assign]

    def _install_transport(self, transport: httpx.MockTransport) -> None:
        real_init = self._saved_client_init

        def patched(self, *args, **kwargs):
            kwargs["transport"] = transport
            return real_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched  # type: ignore[method-assign]

    def test_no_callback_uses_non_streaming_path(self) -> None:
        # Without a stream_callback the dispatcher must skip the SSE path
        # entirely, even though _disable_streaming is False.
        seen_streams = {"count": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            # If this gets called, the test fails — non-streaming uses POST
            # to chat/completions with stream:false and *parses JSON*, not
            # iter_lines.  We reply with a plain JSON body to keep that
            # path happy.
            seen_streams["count"] += 1
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

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)
        provider = _make_provider()
        result = provider.generate([], [], ModelCallConfig(), _ctx(), stream_callback=None)

        self.assertEqual(result.content, "hi")
        self.assertEqual(seen_streams["count"], 1)
        # raw must be present so validate_response can introspect it.
        self.assertIn("raw", result.metadata)

    def test_callback_triggers_streaming_path_and_emits_chunks(self) -> None:
        # With stream_callback supplied we should hit the SSE branch and
        # see the raw_line iterator drained chunk by chunk.
        sse_body = (
            b'data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)
        provider = _make_provider()
        chunks: list[str] = []
        result = provider.generate([], [], ModelCallConfig(), _ctx(), stream_callback=chunks.append)

        self.assertEqual(chunks, ["He", "llo"])
        self.assertEqual(result.content, "Hello")
        self.assertTrue(result.metadata["streamed"])

    def test_disable_streaming_skips_sse_branch(self) -> None:
        # Once _disable_streaming is True we go to the non-streaming path
        # even with a stream_callback set.  This is the after-stall posture.
        json_body = {
            "model": "m1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fallback"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        used_path = {"json": 0, "stream": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            # Request body tells us which path was used.
            try:
                payload = httpx._content.encode_request_data  # noqa: F841 - sentinel access
            except Exception:
                pass
            body = req.content
            if b'"stream": true' in body or b'"stream":true' in body:
                used_path["stream"] += 1
                return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})
            used_path["json"] += 1
            return httpx.Response(200, json=json_body)

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)
        provider = _make_provider()
        provider._disable_streaming = True

        result = provider.generate([], [], ModelCallConfig(), _ctx(), stream_callback=lambda _d: None)

        self.assertEqual(result.content, "fallback")
        self.assertEqual(used_path["json"], 1)
        self.assertEqual(used_path["stream"], 0)

    def test_stall_falls_back_to_non_streaming_and_disables_streaming(self) -> None:
        # First call: stream stalls → engine falls back to non-streaming
        # transparently; the user still gets an answer.
        # Second call: streaming is now disabled, so we skip SSE entirely.
        call_log: list[str] = []
        json_body = {
            "model": "m1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "after-stall"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        def handler(req: httpx.Request) -> httpx.Response:
            body = req.content
            is_stream = b'"stream": true' in body or b'"stream":true' in body
            if is_stream:
                call_log.append("stream")
                # Simulate an immediate ReadTimeout by raising.
                raise httpx.ReadTimeout("simulated stall", request=req)
            call_log.append("json")
            return httpx.Response(200, json=json_body)

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)
        provider = _make_provider()
        provider.stream_read_timeout_sec = 0.05  # not actually used by mock

        result = provider.generate([], [], ModelCallConfig(), _ctx(), stream_callback=lambda _d: None)
        self.assertEqual(result.content, "after-stall")
        self.assertEqual(call_log, ["stream", "json"])
        self.assertTrue(provider._disable_streaming)

        # Second call: should NOT attempt streaming again.
        call_log.clear()
        result2 = provider.generate([], [], ModelCallConfig(), _ctx(), stream_callback=lambda _d: None)
        self.assertEqual(result2.content, "after-stall")
        self.assertEqual(call_log, ["json"])

    def test_stream_open_http_error_surfaces_provider_invocation_error(self) -> None:
        # A 429 returned at stream-open time must NOT trigger the
        # stall-fallback path (that's reserved for read-time stalls).
        # Instead, surface a normal ProviderInvocationError so the
        # recovery layer can apply Retry-After / backoff semantics.
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "7"}, text="rate limited")

        transport = httpx.MockTransport(handler)
        self._install_transport(transport)
        provider = _make_provider()

        with self.assertRaises(ProviderInvocationError) as cm:
            provider.generate([], [], ModelCallConfig(), _ctx(), stream_callback=lambda _d: None)

        exc = cm.exception
        self.assertNotIsInstance(exc, StreamStallError)
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.retry_after_seconds, 7.0)
        # streaming flag must NOT have flipped — a 429 is a server
        # decision about rate, not a problem with the streaming protocol.
        self.assertFalse(provider._disable_streaming)


# ---------------------------------------------------------------------------
# StreamStallError surface
# ---------------------------------------------------------------------------


class StreamStallErrorTests(unittest.TestCase):
    def test_classified_as_network_error_for_recovery(self) -> None:
        # The recovery layer retries on is_network_error=True, which is
        # how StreamStallError participates in the retry chain even
        # though it has no status_code.
        exc = StreamStallError(stalled_after_seconds=12.5, body_summary="silent for 12.5s")
        self.assertTrue(exc.is_network_error)
        self.assertIsNone(exc.status_code)
        self.assertEqual(exc.stalled_after_seconds, 12.5)
        self.assertIn("silent for 12.5s", str(exc))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
