from __future__ import annotations

import unittest

from aether.config.schema import ModelCallConfig
from aether.models.provider.openai_compatible import OpenAICompatibleModel
from aether.models.transport.openai_chat import OpenAIChatCompletionsTransport
from aether.runtime.core.contracts import NormalizedResponse
from aether.tools.base import ToolDescriptor


class OpenAIChatCompletionsTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = OpenAIChatCompletionsTransport()

    def test_build_payload_without_tools_preserves_config_extra(self) -> None:
        payload = self.transport.build_payload(
            model="default-model",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            tools=[],
            config=ModelCallConfig(
                temperature=0.2,
                max_tokens=128,
                extra={"model": "override-model", "top_p": 0.9, "stream": True},
            ),
        )

        self.assertEqual(payload["model"], "override-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertNotIn("tools", payload)

    def test_build_payload_with_tools_sets_tool_choice_auto(self) -> None:
        payload = self.transport.build_payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                ToolDescriptor(
                    name="read_file",
                    description="Read a file",
                    parameters={"path": {"type": "string"}},
                    required=["path"],
                ),
                ToolDescriptor(name="web_search"),
            ],
            config=ModelCallConfig(),
        )

        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["type"], "object")
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["required"], ["path"])
        self.assertEqual(payload["tools"][1]["function"]["name"], "web_search")

    def test_message_conversion_preserves_assistant_thought_signature(self) -> None:
        converted = self.transport.convert_messages(
            [
                {"role": "system", "content": "system"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "README.md"},
                                "thought_signature": "sig-1",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
                {"role": "unknown", "content": "fallback"},
            ]
        )

        self.assertEqual(converted[0], {"role": "system", "content": "system"})
        assistant_tool = converted[1]["tool_calls"][0]
        self.assertEqual(assistant_tool["function"]["arguments"], '{"path": "README.md"}')
        self.assertEqual(assistant_tool["function"]["thought_signature"], "sig-1")
        self.assertEqual(converted[2]["role"], "tool")
        self.assertEqual(converted[3], {"role": "user", "content": "fallback"})

    def test_normalize_response_with_text_usage_and_raw(self) -> None:
        callbacks: list[str] = []
        raw = {
            "model": "m1",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }

        result = self.transport.normalize_response(
            raw,
            fallback_model="fallback",
            stream_callback=callbacks.append,
        )

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(callbacks, ["hello"])
        self.assertEqual(result.metadata["model"], "m1")
        self.assertEqual(result.metadata["usage"], raw["usage"])
        self.assertEqual(result.metadata["token_usage"]["total_tokens"], 8)
        self.assertIs(result.metadata["raw"], raw)

    def test_normalize_response_with_tool_call(self) -> None:
        raw = {
            "model": "m2",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"cmd": "pwd"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        result = self.transport.normalize_response(raw, fallback_model="fallback")

        self.assertEqual(result.content, "")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "call_1")
        self.assertEqual(result.tool_calls[0].name, "shell")
        self.assertEqual(result.tool_calls[0].arguments, {"cmd": "pwd"})

    def test_malformed_empty_choices_round_trips_to_validation(self) -> None:
        raw = {"model": "bad", "choices": []}

        result = self.transport.normalize_response(raw, fallback_model="fallback")
        ok, reasons = self.transport.validate_response(result)

        self.assertEqual(result.content, "")
        self.assertEqual(result.metadata["raw"], raw)
        self.assertFalse(ok)
        self.assertIn("raw.choices is empty or missing", reasons)

    def test_validate_raw_response_reports_embedded_error(self) -> None:
        ok, reasons = self.transport.validate_raw_response(
            {"error": {"message": "credit exhausted"}, "choices": []}
        )

        self.assertFalse(ok)
        self.assertTrue(any("credit exhausted" in reason for reason in reasons))
        self.assertTrue(any("choices is empty" in reason for reason in reasons))

    def test_provider_private_helpers_delegate_to_transport(self) -> None:
        provider = OpenAICompatibleModel(
            model="m",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        )
        raw = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }

        self.assertEqual(provider._convert_messages([{"role": "user", "content": "hi"}]), self.transport.convert_messages([{"role": "user", "content": "hi"}]))
        self.assertEqual(provider._parse_response(raw, stream_callback=None).content, "ok")
        self.assertEqual(
            provider.validate_response(NormalizedResponse(content="", metadata={"raw": {"choices": []}}))[0],
            False,
        )


if __name__ == "__main__":
    unittest.main()
