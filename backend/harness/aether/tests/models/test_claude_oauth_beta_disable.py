"""Sprint 5 / PR 5.10 — Claude OAuth 1M context beta disable retry."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import anthropic
import httpx

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from aether.config.schema import ModelCallConfig
from aether.models.credential_loader import OAUTH_CONTEXT_1M_BETA
from aether.models.provider.claude import ClaudeChatModel
from aether.runtime.contracts import TurnContext


def _ctx() -> TurnContext:
    return TurnContext(session_id="oauth-beta", iteration=1, metadata={})


def _config() -> ModelCallConfig:
    return ModelCallConfig(max_tokens=128, extra={"model": "claude-sonnet-4-6"})


def _ok_response(text: str = "ok") -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _status_error(status_code: int, text: str) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, text=text, request=request)
    if status_code == 400:
        return anthropic.BadRequestError(text, response=response, body=text)
    if status_code == 401:
        return anthropic.AuthenticationError(text, response=response, body=text)
    if status_code == 403:
        return anthropic.PermissionDeniedError(text, response=response, body=text)
    return anthropic.APIStatusError(text, response=response, body=text)


class _FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def create(self, **_kwargs: Any) -> Any:
        self.calls += 1
        if not self._outcomes:
            raise AssertionError("no fake Anthropic outcomes left")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeAnthropicFactory:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.default_headers: list[dict[str, str]] = []
        self.clients: list[Any] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.default_headers.append(dict(kwargs.get("default_headers") or {}))
        client = SimpleNamespace(
            api_key=kwargs.get("api_key"),
            auth_token=kwargs.get("auth_token"),
            closed=False,
            messages=_FakeMessages(self.outcomes),
        )

        def _close() -> None:
            client.closed = True

        client.close = _close
        self.clients.append(client)
        return client


class ClaudeOAuthBetaDisableTests(unittest.TestCase):
    def test_oauth_long_context_beta_error_strips_beta_and_retries_once(self) -> None:
        factory = _FakeAnthropicFactory(
            [
                _status_error(
                    400,
                    "The long context beta is not yet available for this subscription.",
                ),
                _ok_response("recovered"),
            ]
        )

        with mock.patch(
            "aether.models.provider.claude.anthropic.Anthropic",
            new=factory,
        ):
            model = ClaudeChatModel(
                anthropic_api_key="sk-ant-oat-test-token",
                enable_prompt_caching=False,
                auto_thinking_budget=False,
                retry_max_attempts=1,
            )
            context = _ctx()
            result = model.generate([], [], _config(), context)

        self.assertEqual(result.content, "recovered")
        self.assertTrue(model._oauth_1m_beta_disabled)
        self.assertTrue(context.metadata["oauth_1m_beta_disabled"])
        self.assertEqual(len(factory.default_headers), 2)
        first_beta = factory.default_headers[0]["anthropic-beta"]
        second_beta = factory.default_headers[1]["anthropic-beta"]
        self.assertIn(OAUTH_CONTEXT_1M_BETA, first_beta)
        self.assertNotIn(OAUTH_CONTEXT_1M_BETA, second_beta)
        self.assertTrue(factory.clients[0].closed)

    def test_non_oauth_token_does_not_trigger_beta_recovery(self) -> None:
        factory = _FakeAnthropicFactory(
            [
                _status_error(
                    400,
                    "The long context beta is not yet available for this subscription.",
                ),
                _ok_response("must-not-run"),
            ]
        )

        with mock.patch(
            "aether.models.provider.claude.anthropic.Anthropic",
            new=factory,
        ):
            model = ClaudeChatModel(
                anthropic_api_key="sk-ant-api-test-key",
                enable_prompt_caching=False,
                auto_thinking_budget=False,
                retry_max_attempts=1,
            )
            with self.assertRaises(anthropic.BadRequestError):
                model.generate([], [], _config(), _ctx())

        self.assertFalse(model._oauth_1m_beta_disabled)
        self.assertEqual(len(factory.default_headers), 1)

    def test_other_auth_errors_do_not_trigger_beta_recovery(self) -> None:
        factory = _FakeAnthropicFactory(
            [
                _status_error(403, "permission denied for this workspace"),
                _ok_response("must-not-run"),
            ]
        )

        with mock.patch(
            "aether.models.provider.claude.anthropic.Anthropic",
            new=factory,
        ):
            model = ClaudeChatModel(
                anthropic_api_key="sk-ant-oat-test-token",
                enable_prompt_caching=False,
                auto_thinking_budget=False,
                retry_max_attempts=1,
            )
            with self.assertRaises(anthropic.PermissionDeniedError):
                model.generate([], [], _config(), _ctx())

        self.assertFalse(model._oauth_1m_beta_disabled)
        self.assertEqual(len(factory.default_headers), 1)

    def test_beta_recovery_only_retries_once(self) -> None:
        factory = _FakeAnthropicFactory(
            [
                _status_error(
                    400,
                    "The long context beta is not yet available for this subscription.",
                ),
                _status_error(
                    400,
                    "The long context beta is not yet available for this subscription.",
                ),
                _ok_response("must-not-run"),
            ]
        )

        with mock.patch(
            "aether.models.provider.claude.anthropic.Anthropic",
            new=factory,
        ):
            model = ClaudeChatModel(
                anthropic_api_key="sk-ant-oat-test-token",
                enable_prompt_caching=False,
                auto_thinking_budget=False,
                retry_max_attempts=1,
            )
            with self.assertRaises(anthropic.BadRequestError):
                model.generate([], [], _config(), _ctx())

        self.assertTrue(model._oauth_1m_beta_disabled)
        self.assertEqual(len(factory.default_headers), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
