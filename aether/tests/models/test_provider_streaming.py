from __future__ import annotations

import unittest

from aether.config.schema import ModelCallConfig
from aether.models.provider.codex import CodexChatModel
from aether.tools.base import ToolDescriptor


class ProviderStreamingTests(unittest.TestCase):
    def test_extract_stream_delta_prefers_direct_delta_string(self) -> None:
        event = {"type": "response.output_text.delta", "delta": "Hello"}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "Hello")

    def test_extract_stream_delta_from_delta_object(self) -> None:
        event = {"type": "response.output_text.delta", "delta": {"text": " world"}}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), " world")

    def test_extract_stream_delta_from_output_text_event_text(self) -> None:
        event = {"type": "response.output_text.delta", "text": "!"}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "!")

    def test_extract_stream_delta_from_item_text(self) -> None:
        event = {"type": "response.output_item.added", "item": {"text": "item-text"}}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "item-text")

    def test_extract_stream_delta_from_item_content_blocks(self) -> None:
        event = {
            "type": "response.output_item.done",
            "item": {
                "content": [
                    {"type": "output_text", "text": "A"},
                    {"type": "output_text", "text": "B"},
                ]
            },
        }
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "AB")

    def test_emit_stream_delta_invokes_callback_for_text(self) -> None:
        deltas: list[str] = []
        event = {"type": "response.output_text.delta", "delta": "chunk"}

        CodexChatModel._emit_stream_delta(deltas.append, event)

        self.assertEqual(deltas, ["chunk"])

    def test_emit_stream_delta_ignores_non_text_events(self) -> None:
        deltas: list[str] = []
        event = {"type": "response.completed", "response": {"id": "r_1"}}

        CodexChatModel._emit_stream_delta(deltas.append, event)

        self.assertEqual(deltas, [])

    def test_function_call_argument_delta_is_silent_progress(self) -> None:
        visible: list[str] = []
        silent: list[str] = []
        event = {"type": "response.function_call_arguments.delta", "delta": '{"cmd":'}

        CodexChatModel._emit_stream_delta(visible.append, event)
        CodexChatModel._emit_stream_silent_delta(silent.append, event)

        self.assertEqual(visible, [])
        self.assertEqual(silent, ['{"cmd":'])

    def test_build_payload_sets_tool_choice_auto_when_tools_present(self) -> None:
        provider = CodexChatModel(access_token="test-token", account_id="acct")
        payload = provider._build_payload(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                ToolDescriptor(
                    name="exec_command",
                    description="Run a shell command",
                    parameters={
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                        },
                    },
                    required=["cmd"],
                )
            ],
            config=ModelCallConfig(),
        )

        self.assertEqual(payload.get("tool_choice"), "auto")
        self.assertIn("tools", payload)


if __name__ == "__main__":
    unittest.main()
