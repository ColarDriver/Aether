"""Base contracts for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext


class UnknownToolError(KeyError):
    """Raised when a tool call references an unknown tool."""


@dataclass(slots=True)
class ToolDescriptor:
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


class ToolExecutor(ABC):
    """Tool execution interface."""

    @property
    @abstractmethod
    def descriptor(self) -> ToolDescriptor:
        raise NotImplementedError

    def validate(self, call: ToolCall) -> None:
        if call.name != self.descriptor.name:
            raise ValueError(f"Tool name mismatch: expected {self.descriptor.name}, got {call.name}")
        if not isinstance(call.arguments, dict):
            raise ValueError("Tool arguments must be a dict")

    @abstractmethod
    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        raise NotImplementedError
