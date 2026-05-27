from __future__ import annotations

import unittest

from aether.config.schema import ModelCallConfig
from aether.models.transport.anthropic_messages import AnthropicMessagesTransport
from aether.runtime.core.contracts import TurnContext
from aether.tools.base import ToolDescriptor


class AnthropicMessagesTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = AnthropicMessagesTransport()

    def test_build_payload_adds_system_metadata_and_tools(self) -> None:
        context = TurnContext(session_id="s1", iteration=2)

        payload = self.transport.build_payload(
            model="claude-default",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hi"},
            ],
            tools=[
                ToolDescriptor(
                    name="read_file",
                    description="Read a file",
                    parameters={"path": {"type": "string"}},
                    required=["path"],
                )
            ],
            config=ModelCallConfig(
                temperature=0.1,
                extra={"model": "claude-override", "stop_sequences": ["stop"]},
            ),
            max_tokens=2048,
            context=context,
        )

        self.assertEqual(payload["model"], "claude-override")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["system"], [{"type": "text", "text": "system"}])
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["metadata"], {"session_id": "s1", "iteration": 2})
        self.assertEqual(payload["stop_sequences"], ["stop"])
        self.assertEqual(payload["tools"][0]["input_schema"]["type"], "object")
        self.assertEqual(payload["tools"][0]["input_schema"]["required"], ["path"])

    def test_user_multimodal_content_converts_base64_image_parts(self) -> None:
        _system, messages = self.transport.convert_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ]
        )

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "describe"})
        self.assertEqual(messages[0]["content"][1]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["source"]["media_type"], "image/png")
        self.assertEqual(messages[0]["content"][1]["source"]["data"], "abc")

    def test_convert_messages_preserves_assistant_tool_and_tool_result(self) -> None:
        system, messages = self.transport.convert_messages(
            [
                {"role": "system", "content": "rules"},
                {
                    "role": "assistant",
                    "content": "I'll read it",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "README.md"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_1",
                    "content": "contents",
                    "is_error": True,
                },
            ]
        )

        self.assertEqual(system, [{"type": "text", "text": "rules"}])
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"][1]["type"], "tool_use")
        self.assertEqual(messages[0]["content"][1]["input"], {"path": "README.md"})
        self.assertEqual(messages[1]["content"][0]["type"], "tool_result")
        self.assertTrue(messages[1]["content"][0]["is_error"])

    def test_web_search_descriptor_converts_to_anthropic_hosted_tool(self) -> None:
        self.assertEqual(
            self.transport.convert_tools([ToolDescriptor(name="web_search")]),
            [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        )

    def test_normalize_text_response_preserves_usage(self) -> None:
        raw = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
            },
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.finish_reason, "end_turn")
        self.assertEqual(result.metadata["model"], "claude-sonnet-4-6")
        self.assertEqual(result.metadata["usage"], raw["usage"])

    def test_normalize_tool_use_response(self) -> None:
        raw = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "shell",
                    "input": {"cmd": "pwd"},
                }
            ],
            "usage": {},
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "toolu_1")
        self.assertEqual(result.tool_calls[0].arguments, {"cmd": "pwd"})

    def test_hosted_web_search_metadata_and_sources(self) -> None:
        raw = {
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvu_1",
                    "name": "web_search",
                    "input": {"query": "Aether"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvu_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "Aether Docs",
                            "url": "https://docs.example/aether",
                        }
                    ],
                },
                {
                    "type": "text",
                    "text": "Found docs.",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "title": "Aether Docs",
                            "url": "https://docs.example/aether",
                        }
                    ],
                },
            ],
            "usage": {},
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertIn("Sources:", result.content or "")
        self.assertIn("[Aether Docs](https://docs.example/aether)", result.content or "")
        hosted = result.metadata["hosted_web_search"]
        self.assertEqual(hosted["provider"], "anthropic")
        self.assertEqual(hosted["source_count"], 1)
        self.assertEqual(hosted["calls"][0]["input"], {"query": "Aether"})
        self.assertEqual(hosted["results"][0]["result_count"], 1)

    def test_validate_raw_response_rejects_empty_content(self) -> None:
        ok, reasons = self.transport.validate_raw_response({"content": []})

        self.assertFalse(ok)
        self.assertEqual(reasons, ["raw.content is empty or missing"])


if __name__ == "__main__":
    unittest.main()
