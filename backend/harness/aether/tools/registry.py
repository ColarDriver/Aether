"""Tool registry and dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor, UnknownToolError


@dataclass(slots=True)
class ToolRegistry:
    _tools: Dict[str, ToolExecutor] = field(default_factory=dict)

    def register(self, executor: ToolExecutor) -> None:
        name = executor.descriptor.name
        self._tools[name] = executor

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolExecutor:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]

    def list_descriptors(self) -> List[ToolDescriptor]:
        return [tool.descriptor for tool in self._tools.values()]

    def dispatch(self, call: ToolCall, context: TurnContext) -> ToolResult:
        executor = self.get(call.name)
        executor.validate(call)
        return executor.execute(call, context)
