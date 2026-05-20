from __future__ import annotations

import unittest
from typing import Any, Iterable

from aether.config.schema import ModelCallConfig
from aether.models.transport import (
    available_transports,
    clear_transports_for_tests,
    get_transport,
    register_transport,
)
from aether.runtime.core.contracts import NormalizedResponse
from aether.tools.base import ToolDescriptor


class _FakeTransport:
    api_mode = "fake"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

    def convert_tools(self, tools: Iterable[ToolDescriptor], **kwargs: Any) -> Any:
        del kwargs
        return [{"name": tool.name} for tool in tools]

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Iterable[ToolDescriptor],
        config: ModelCallConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        return {
            "model": model,
            "messages": self.convert_messages(messages),
            "tools": self.convert_tools(tools),
            "max_tokens": config.max_tokens,
        }

    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        del kwargs
        return NormalizedResponse(content=str(response.get("content", "")))

    def validate_raw_response(self, response: Any) -> tuple[bool, list[str]]:
        return (True, []) if isinstance(response, dict) else (False, ["not a dict"])


class _ReplacementTransport(_FakeTransport):
    pass


class TransportRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_transports_for_tests()

    def tearDown(self) -> None:
        clear_transports_for_tests()

    def test_register_get_and_available_transports(self) -> None:
        register_transport(" fake ", _FakeTransport)

        self.assertEqual(available_transports(), ("fake",))
        transport = get_transport("FAKE")
        self.assertIsInstance(transport, _FakeTransport)
        assert transport is not None
        payload = transport.build_payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[ToolDescriptor(name="read_file")],
            config=ModelCallConfig(max_tokens=12),
        )
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["tools"], [{"name": "read_file"}])
        self.assertEqual(payload["max_tokens"], 12)
        self.assertEqual(transport.normalize_response({"content": "ok"}).content, "ok")
        self.assertEqual(transport.validate_raw_response({"content": "ok"}), (True, []))

    def test_duplicate_same_factory_is_idempotent(self) -> None:
        register_transport("fake", _FakeTransport)
        register_transport("fake", _FakeTransport)

        self.assertEqual(available_transports(), ("fake",))

    def test_duplicate_different_factory_raises(self) -> None:
        register_transport("fake", _FakeTransport)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_transport("fake", _ReplacementTransport)

    def test_unknown_api_mode_returns_none(self) -> None:
        self.assertIsNone(get_transport("missing"))

    def test_empty_api_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "api_mode is required"):
            register_transport("", _FakeTransport)


if __name__ == "__main__":
    unittest.main()
