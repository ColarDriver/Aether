from __future__ import annotations

import unittest

from aether.config.schema import ModelCallConfig
from aether.models.transport.codex_responses import CodexResponsesTransport
from aether.tools.base import ToolDescriptor


class CodexResponsesTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = CodexResponsesTransport()

    def test_build_payload_sets_reasoning_and_tool_choice(self) -> None:
        payload = self.transport.build_payload(
            model="gpt-default",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hi"},
            ],
            tools=[
                ToolDescriptor(
                    name="shell",
                    description="Run shell",
                    parameters={"cmd": {"type": "string"}},
                    required=["cmd"],
                )
            ],
            config=ModelCallConfig(
                temperature=0.3,
                max_tokens=256,
                extra={"model": "gpt-5.4", "reasoning_effort": "high"},
            ),
            reasoning_effort="medium",
        )

        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertEqual(payload["instructions"], "system")
        self.assertEqual(payload["input"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["reasoning"], {"effort": "high", "summary": "detailed"})
        self.assertEqual(payload["max_output_tokens"], 256)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["parameters"]["required"], ["cmd"])

    def test_user_multimodal_content_converts_to_responses_parts(self) -> None:
        _instructions, input_items = self.transport.convert_messages(
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

        self.assertEqual(input_items[0]["role"], "user")
        self.assertEqual(input_items[0]["content"][0], {"type": "input_text", "text": "describe"})
        self.assertEqual(
            input_items[0]["content"][1],
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        )

    def test_convert_messages_includes_assistant_function_call_and_tool_output(self) -> None:
        instructions, input_items = self.transport.convert_messages(
            [
                {"role": "system", "content": "rules"},
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "shell",
                                "arguments": {"cmd": "pwd"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "/workspace"},
            ]
        )

        self.assertEqual(instructions, "rules")
        self.assertEqual(input_items[0], {"role": "assistant", "content": "checking"})
        self.assertEqual(input_items[1]["type"], "function_call")
        self.assertEqual(input_items[1]["arguments"], '{"cmd": "pwd"}')
        self.assertEqual(input_items[2]["type"], "function_call_output")
        self.assertEqual(input_items[2]["output"], "/workspace")

    def test_web_search_descriptor_converts_to_codex_hosted_tool(self) -> None:
        self.assertEqual(
            self.transport.convert_tools([ToolDescriptor(name="web_search")]),
            [{"type": "web_search", "search_context_size": "medium"}],
        )

    def test_normalize_text_response_preserves_reasoning_usage_and_sources(self) -> None:
        raw = {
            "model": "gpt-5.4",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "Aether"},
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Found docs.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "Aether Docs",
                                    "url": "https://docs.example/aether",
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertIn("Found docs.", result.content or "")
        self.assertIn("Sources:", result.content or "")
        self.assertEqual(result.metadata["model"], "gpt-5.4")
        self.assertEqual(result.metadata["usage"], raw["usage"])
        self.assertEqual(result.metadata["token_usage"]["total_tokens"], 30)
        self.assertEqual(result.metadata["reasoning_content"], "thought")
        hosted = result.metadata["hosted_web_search"]
        self.assertEqual(hosted["provider"], "codex")
        self.assertEqual(hosted["source_count"], 1)
        self.assertEqual(hosted["calls"][0]["action"]["query"], "Aether")

    def test_normalize_function_call_response(self) -> None:
        raw = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": '{"cmd": "pwd"}',
                }
            ],
            "usage": {},
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertEqual(result.content, "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "shell")
        self.assertEqual(result.tool_calls[0].arguments, {"cmd": "pwd"})

    def test_invalid_function_arguments_are_reported_as_content(self) -> None:
        raw = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "not-json",
                }
            ],
            "usage": {},
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertEqual(result.tool_calls, [])
        self.assertIn("Invalid tool call arguments", result.content or "")

    def test_validate_raw_response_rejects_empty_output(self) -> None:
        ok, reasons = self.transport.validate_raw_response({"output": []})

        self.assertFalse(ok)
        self.assertEqual(reasons, ["raw.output is empty or missing"])


if __name__ == "__main__":
    unittest.main()
