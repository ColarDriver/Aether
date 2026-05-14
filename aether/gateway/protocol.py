"""Wire schema for the Aether gateway JSON-RPC protocol.

The gateway speaks newline-delimited JSON over its bound transport.
Each frame is one of:

* **Request** — has ``id`` and ``method``; the peer expects a response.
* **Notification** — has ``method`` but no ``id``; fire-and-forget.
* **Response** — has ``id`` and exactly one of ``result`` / ``error``.

All envelope models use ``extra="forbid"`` so unknown fields surface
as -32600 invalid-request errors rather than silently passing through.
Event models layered on top of the JSON-RPC envelope (delivered as
notifications) likewise forbid extras.

This module deliberately does NOT do batching.  Each frame is one
envelope; the dispatcher rejects arrays at parse time.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


JSONRPC_VERSION = "2.0"


# ── Error codes ──────────────────────────────────────────────────────
# Standard JSON-RPC 2.0 reserved range:
#   -32700  Parse error      — frame is not valid JSON
#   -32600  Invalid request  — envelope shape is wrong
#   -32601  Method not found
#   -32602  Invalid params   — handler param schema validation failed
#   -32603  Internal error   — handler raised something we did not classify
#
# Aether application range:
#   -32000  Application error — handler raised GatewayError on purpose
#   -32001  Handler timed out
#   -32002  Handler cancelled

ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603
ERROR_APPLICATION = -32000
ERROR_TIMED_OUT = -32001
ERROR_CANCELLED = -32002


# ── Application-level exception ──────────────────────────────────────


class GatewayError(Exception):
    """Raised by handlers to surface an application error to the peer.

    The dispatcher converts a :class:`GatewayError` into an
    ``RpcError`` with the requested code (default -32000) and
    optional ``data`` payload, then returns it as the request's
    response.  Other exceptions get mapped to -32603 (internal).
    """

    def __init__(
        self,
        message: str,
        *,
        code: int = ERROR_APPLICATION,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.data = data


# ── JSON-RPC envelope ────────────────────────────────────────────────


_RpcId = Union[str, int]


class RpcRequest(BaseModel):
    """Inbound RPC request frame."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: _RpcId
    method: str = Field(min_length=1)
    params: dict[str, Any] | None = None


class RpcNotification(BaseModel):
    """Inbound RPC notification frame (no response expected)."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    method: str = Field(min_length=1)
    params: dict[str, Any] | None = None


class RpcError(BaseModel):
    """JSON-RPC error object."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: dict[str, Any] | None = None


class RpcResponse(BaseModel):
    """Outbound RPC response frame.

    Exactly one of ``result`` or ``error`` must be set.  The model
    validator enforces this so the gateway never emits an ambiguous
    frame to the peer.
    """

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: Optional[_RpcId]
    result: Any = None
    error: Optional[RpcError] = None

    @model_validator(mode="after")
    def _exactly_one_of_result_or_error(self) -> "RpcResponse":
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result and has_error:
            raise ValueError("RpcResponse must have exactly one of result/error, not both")
        if not has_result and not has_error:
            raise ValueError("RpcResponse must have exactly one of result/error")
        return self


# ── Event envelope (transported as a notification) ───────────────────
#
# Notifications using the method name ``"event"`` carry a typed event
# in ``params``.  The dispatcher itself emits ``gateway.ready`` and
# ``gateway.error`` directly as notifications under those method names
# (no ``"event"`` wrapper) — those two events are part of the gateway
# lifecycle, not the agent event stream.


class EventBase(BaseModel):
    """Base for typed events.

    Subclasses must declare a ``type: Literal[...]`` field to act as
    the discriminator for any future tagged-union routing.  We
    deliberately do NOT put ``type: str`` on the base because that
    would make every subclass override variance-unsafe under strict
    type checkers (a mutable field cannot be narrowed in subclasses).
    """

    model_config = ConfigDict(extra="forbid")


class GatewayReady(EventBase):
    """Sent once, immediately after the gateway finishes booting."""

    type: Literal["gateway.ready"] = "gateway.ready"
    version: str
    capabilities: list[str] = Field(default_factory=list)


class GatewayErrorEvent(EventBase):
    """Asynchronous gateway-level error notification.

    Distinct from ``RpcError`` (which is a response envelope) — this
    event lets the gateway report internal anomalies that are not tied
    to a specific request, e.g. background-task crashes.
    """

    type: Literal["gateway.error"] = "gateway.error"
    message: str
    where: str | None = None


class AgentEventBase(EventBase):
    """Base for events emitted by ``agent.run``.

    ``session_id`` and ``run_id`` are intentionally present on every
    agent-stream event so clients can route concurrent sessions without
    relying on transport ordering.
    """

    session_id: str
    run_id: str


class TextDelta(AgentEventBase):
    type: Literal["text.delta"] = "text.delta"
    text: str
    sequence: int


class Reasoning(AgentEventBase):
    type: Literal["reasoning.delta"] = "reasoning.delta"
    text: str
    sequence: int


class ToolCall(AgentEventBase):
    type: Literal["tool.call"] = "tool.call"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    iteration: int


class ToolResult(AgentEventBase):
    type: Literal["tool.result"] = "tool.result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    iteration: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class IterationStart(AgentEventBase):
    type: Literal["iteration.start"] = "iteration.start"
    iteration: int


class IterationEnd(AgentEventBase):
    type: Literal["iteration.end"] = "iteration.end"
    iteration: int


class LoopStateChanged(AgentEventBase):
    type: Literal["loop.state"] = "loop.state"
    state: str


class Status(AgentEventBase):
    type: Literal["status"] = "status"
    kind: Literal["thinking", "responding", "tool_use", "idle"]
    detail: str | None = None


class TokenUsage(AgentEventBase):
    type: Literal["usage"] = "usage"
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Done(AgentEventBase):
    type: Literal["done"] = "done"
    final_text: str = ""
    exit_reason: str = "done"


class Cancelled(AgentEventBase):
    type: Literal["cancelled"] = "cancelled"
    reason: str | None = None
    partial_text: str = ""


class Error(AgentEventBase):
    type: Literal["error"] = "error"
    message: str


class ApprovalQuestion(BaseModel):
    """Question item sent by ``approval.request`` reverse RPC."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    kind: Literal["open", "select"] = "open"
    options: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    """Params for server-initiated ``approval.request``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["plan", "questions"]
    session_id: str
    run_id: str
    tool_call_id: str | None = None
    plan_text: str | None = None
    questions: list[ApprovalQuestion] = Field(default_factory=list)
    deadline_ms: int


class PermissionPreview(BaseModel):
    """Wire mirror of ``ToolPermissionPreview``."""

    model_config = ConfigDict(extra="forbid")

    title: str
    subtitle: str | None = None
    body: str | None = None
    diff: str | None = None
    path: str | None = None
    command: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PermissionToolRequest(BaseModel):
    """Wire mirror of ``ToolPermissionRequest``."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    category: str
    risk: str
    preview: PermissionPreview | None = None
    reason: str | None = None
    allow_session: bool = True


class PermissionRequest(BaseModel):
    """Params for server-initiated ``permission.request``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    run_id: str
    request: PermissionToolRequest
    deadline_ms: int


# ── Helpers used by the dispatcher ───────────────────────────────────


def make_error_response(
    request_id: _RpcId | None,
    code: int,
    message: str,
    *,
    data: dict[str, Any] | None = None,
) -> RpcResponse:
    """Build an error response for ``request_id`` with the given code/message."""

    return RpcResponse(id=request_id, error=RpcError(code=code, message=message, data=data))


def make_success_response(request_id: _RpcId, result: Any) -> RpcResponse:
    """Build a success response carrying ``result``."""

    return RpcResponse(id=request_id, result=result)


__all__ = [
    "AgentEventBase",
    "ApprovalQuestion",
    "ApprovalRequest",
    "Cancelled",
    "Done",
    "ERROR_APPLICATION",
    "ERROR_CANCELLED",
    "ERROR_INTERNAL",
    "ERROR_INVALID_PARAMS",
    "ERROR_INVALID_REQUEST",
    "ERROR_METHOD_NOT_FOUND",
    "ERROR_PARSE",
    "ERROR_TIMED_OUT",
    "EventBase",
    "Error",
    "GatewayError",
    "GatewayErrorEvent",
    "GatewayReady",
    "IterationEnd",
    "IterationStart",
    "JSONRPC_VERSION",
    "LoopStateChanged",
    "PermissionPreview",
    "PermissionRequest",
    "PermissionToolRequest",
    "Reasoning",
    "RpcError",
    "RpcNotification",
    "RpcRequest",
    "RpcResponse",
    "Status",
    "TextDelta",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "make_error_response",
    "make_success_response",
]
