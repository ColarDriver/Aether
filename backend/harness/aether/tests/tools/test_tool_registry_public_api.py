from __future__ import annotations

import unittest

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, UnknownToolError
from aether.tools.registry import ToolRegistry


class _NoopTool(ToolExecutor):
    def __init__(self, name: str) -> None:
        self._descriptor = ToolDescriptor(
            name=name,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")


class ToolRegistryPublicApiTests(unittest.TestCase):
    def test_get_descriptor_returns_registered_descriptor(self) -> None:
        registry = ToolRegistry()
        registry.register(_NoopTool("read_file"))

        descriptor = registry.get_descriptor("read_file")

        self.assertEqual(descriptor.name, "read_file")

    def test_get_descriptor_raises_for_unknown_tool(self) -> None:
        registry = ToolRegistry()

        with self.assertRaises(UnknownToolError):
            registry.get_descriptor("missing")

    def test_list_names_returns_registration_order(self) -> None:
        registry = ToolRegistry()
        registry.register(_NoopTool("first"))
        registry.register(_NoopTool("second"))

        self.assertEqual(registry.list_names(), ["first", "second"])


if __name__ == "__main__":
    unittest.main()
