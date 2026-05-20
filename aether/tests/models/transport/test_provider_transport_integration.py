from __future__ import annotations

import unittest

from aether.models.provider.claude import ClaudeChatModel
from aether.models.provider.codex import CodexChatModel
from aether.models.provider.openai_compatible import OpenAICompatibleModel
from aether.models.transport.anthropic_messages import AnthropicMessagesTransport
from aether.models.transport.codex_responses import CodexResponsesTransport
from aether.models.transport.openai_chat import OpenAIChatCompletionsTransport


class ProviderTransportIntegrationTests(unittest.TestCase):
    def test_openai_provider_exposes_transport_metadata(self) -> None:
        provider = OpenAICompatibleModel(
            model="gpt-5.4",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        )

        self.assertEqual(provider.provider_name, "openai")
        self.assertEqual(provider.api_mode, "chat")
        self.assertEqual(provider.transport_name, "openai_chat_completions")
        self.assertEqual(provider.transport_api_mode, "chat")
        self.assertIsInstance(provider._transport, OpenAIChatCompletionsTransport)

    def test_claude_provider_exposes_transport_metadata(self) -> None:
        provider = ClaudeChatModel(
            anthropic_api_key="sk-test",
            enable_prompt_caching=False,
            auto_thinking_budget=False,
            retry_max_attempts=1,
        )

        self.assertEqual(provider.provider_name, "anthropic")
        self.assertEqual(provider.api_mode, "messages")
        self.assertEqual(provider.transport_name, "anthropic_messages")
        self.assertEqual(provider.transport_api_mode, "anthropic_messages")
        self.assertIsInstance(provider._transport, AnthropicMessagesTransport)

    def test_codex_provider_exposes_transport_metadata(self) -> None:
        provider = CodexChatModel(access_token="token", account_id="acct")

        self.assertEqual(provider.provider_name, "codex")
        self.assertEqual(provider.api_mode, "responses")
        self.assertEqual(provider.transport_name, "codex_responses")
        self.assertEqual(provider.transport_api_mode, "codex_responses")
        self.assertIsInstance(provider._transport, CodexResponsesTransport)


if __name__ == "__main__":
    unittest.main()
