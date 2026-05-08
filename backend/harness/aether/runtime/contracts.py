"""Shared contracts for the Aether loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List

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
    # Sprint 1 / PR 1.1: response-shape validation failed even after the
    # recovery layer's retry budget was exhausted.  Distinct from
    # PROVIDER_ERROR so observers can tell "the API itself broke" apart
    # from "the API kept returning malformed bodies".
    RESPONSE_INVALID = "RESPONSE_INVALID"
    # Sprint 1 / PR 1.2: response hit the model's output budget but we
    # successfully stitched a continuation and returned the full text.
    LENGTH_RECOVERED = "LENGTH_RECOVERED"
    # Sprint 1 / PR 1.2: response hit the output budget repeatedly and we
    # had to stop after rollback / partial return.
    LENGTH_EXHAUSTED = "LENGTH_EXHAUSTED"
    # Sprint 1 / PR 1.3: assistant tool-call payload was cut off mid-stream
    # (either because finish_reason="length" arrived alongside tool_calls
    # or because the JSON arguments did not terminate with "}" / "]").  We
    # refused to dispatch the truncated call and either retried once or
    # surfaced a partial turn.  Distinct from LENGTH_EXHAUSTED so observers
    # can branch on "the model asked for a tool but never finished writing
    # the arguments" vs "the model exhausted its output budget on prose".
    TOOL_CALL_TRUNCATED = "TOOL_CALL_TRUNCATED"
    # Sprint 1.5 / phantom-tool recovery: model wrote shell commands or
    # ``<function=NAME>`` / ``<invoke>`` markers in prose instead of
    # populating the structured ``tool_calls`` field, and the loop sent
    # corrective messages for ``max_phantom_tool_retries`` turns without
    # ever getting back a properly structured invocation.  Distinct from
    # TEXT_RESPONSE because the model *intended* to invoke a tool — the
    # turn is broken, not just complete.  Surfaced to the UI so the
    # user sees a clear "the model never got around to actually running
    # anything" footer instead of a misleading green checkmark.
    PHANTOM_TOOL_INTENT = "PHANTOM_TOOL_INTENT"




StreamDeltaCallback = Callable[[str], None]

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
    task_id: str | None = None
    turn_id: str | None = None


@dataclass(slots=True)
class EngineRequest:
    session_id: str
    user_message: str | None = None
    system_message: str | None = None
    stream_callback: StreamDeltaCallback | None = None
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
    task_id: str | None = None
    turn_id: str | None = None
    system_prompt: str | None = None
    streamed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
