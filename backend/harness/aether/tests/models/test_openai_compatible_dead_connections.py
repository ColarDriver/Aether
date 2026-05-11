"""Sprint 5 / PR 5.10 — OpenAI-compatible dead connection cleanup."""

from __future__ import annotations

import unittest
from typing import Any

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.provider.openai_compatible import OpenAICompatibleModel
from aether.runtime.contracts import TurnContext


def _ctx() -> TurnContext:
    return TurnContext(session_id="dead-conn", iteration=1, metadata={})


def _good_response(text: str = "ok") -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
        json={
            "model": "m1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


class _FakeClient:
    def __init__(
        self,
        outcomes: list[Any],
        *,
        stale: bool = False,
        cleanup_raises: bool = False,
    ) -> None:
        self._outcomes = outcomes
        self._stale = stale
        self.cleanup_raises = cleanup_raises
        self.closed = False
        self.posts = 0

    @property
    def is_stale(self) -> bool:
        if self.cleanup_raises:
            raise RuntimeError("cleanup state failed")
        return self._stale

    def post(self, *_args: Any, **_kwargs: Any) -> httpx.Response:
        self.posts += 1
        if not self._outcomes:
            raise AssertionError("no fake HTTP outcomes left")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


class _ClientFactoryProvider(OpenAICompatibleModel):
    def __init__(self, clients: list[_FakeClient]) -> None:
        super().__init__(
            model="m1",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            request_timeout_sec=5,
        )
        self._clients = clients
        self.rebuilds = 0

    def _build_http_client(self) -> Any:
        self.rebuilds += 1
        if not self._clients:
            raise AssertionError("no fake clients left")
        return self._clients.pop(0)


class OpenAICompatibleDeadConnectionTests(unittest.TestCase):
    def test_cleanup_rebuilds_stale_reusable_client(self) -> None:
        stale_client = _FakeClient([_good_response("must-not-run")], stale=True)
        fresh_client = _FakeClient([_good_response("fresh")])
        provider = _ClientFactoryProvider([fresh_client])
        provider._client = stale_client

        cleaned = provider.cleanup_dead_connections()
        result = provider.generate([], [], ModelCallConfig(), _ctx())

        self.assertTrue(cleaned)
        self.assertTrue(stale_client.closed)
        self.assertEqual(result.content, "fresh")
        self.assertEqual(fresh_client.posts, 1)

    def test_cleanup_exception_is_logged_and_generate_still_runs(self) -> None:
        broken_state_client = _FakeClient(
            [_good_response("ok-after-cleanup-error")],
            cleanup_raises=True,
        )
        provider = _ClientFactoryProvider([])
        provider._client = broken_state_client
        context = _ctx()

        with self.assertLogs(
            "aether.models.provider.openai_compatible",
            level="ERROR",
        ):
            result = provider.generate([], [], ModelCallConfig(), context)

        self.assertEqual(result.content, "ok-after-cleanup-error")
        self.assertNotIn("dead_connections_cleaned", context.metadata)
        self.assertEqual(broken_state_client.posts, 1)

    def test_stale_connection_error_rebuilds_and_retries_once(self) -> None:
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        stale_error = httpx.RemoteProtocolError(
            "Server disconnected without sending a response.",
            request=request,
        )
        first_client = _FakeClient([stale_error])
        second_client = _FakeClient([_good_response("recovered")])
        provider = _ClientFactoryProvider([first_client, second_client])
        context = _ctx()

        result = provider.generate([], [], ModelCallConfig(), context)

        self.assertEqual(result.content, "recovered")
        self.assertTrue(first_client.closed)
        self.assertEqual(first_client.posts, 1)
        self.assertEqual(second_client.posts, 1)
        self.assertTrue(context.metadata["dead_connections_cleaned"])
        self.assertTrue(context.metadata["dead_connection_retry"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
