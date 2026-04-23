"""Shared contracts for the Aether loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from aether.config.schema import ModelCallConfig


class LoopState(str, Enum):
    INIT = "INIT"
    PREPARE = "PREPARE"
    PRE_LLM = "PRE_LLM"
    LLM_CALL = "LLM_CALL"
    POST_LLM = "POST_LLM"
    TOOL_DISPATCH = "TOOL_DISPATCH"
    TOOL_EXECUTE = "TOOL_EXECUTE"
    CHECK_EXIT = "CHECK_EXIT"
    FINALIZE = "FINALIZE"
    DONE = "DONE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ExitReason(str, Enum):
    TEXT_RESPONSE = "TEXT_RESPONSE"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    INTERRUPTED = "INTERRUPTED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    MIDDLEWARE_ERROR = "MIDDLEWARE_ERROR"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"


class EngineStatus(str, Enum):
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    MAX_ITERATIONS = "MAX_ITERATIONS"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedResponse:
    content: str | None = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnContext:
    session_id: str
    iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineRequest:
    session_id: str
    user_message: str | None = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model_config: ModelCallConfig = field(default_factory=ModelCallConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineResult:
    session_id: str
    status: EngineStatus
    exit_reason: ExitReason
    messages: List[Dict[str, Any]]
    iterations: int
    final_response: str | None = None
    error: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
