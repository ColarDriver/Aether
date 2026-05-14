"""Wire schemas shared by the gateway handlers.

These mirror the gateway handler contract. All models
use ``extra="forbid"`` so unknown fields surface as validation errors
rather than silently passing through.  The handler functions
serialise instances via ``model_dump(mode="json", exclude_none=True)``
before handing them to the RPC dispatcher.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionInfo(BaseModel):
    """Per-session metadata returned by ``session.*`` methods.

    ``created_at`` / ``updated_at`` are Unix timestamps (seconds since
    epoch, UTC).  The on-disk :class:`SessionRecord` stores ISO-8601
    strings; the handler converts on the way out so wire consumers
    don't need a date parser.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: float
    updated_at: float
    provider: str
    model: str
    base_url: str | None = None
    system_prompt: str | None = None
    message_count: int = 0
    summary: str | None = None


class TranscriptToolCall(BaseModel):
    """Flattened OpenAI-style function call shipped on assistant turns.

    The on-disk session stores tool invocations under the OpenAI shape
    ``{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}``.
    The wire model collapses that into ``{id, name, arguments}`` so the
    TUI does not need to know about the function-calling envelope.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TranscriptMessage(BaseModel):
    """One conversation message, as transmitted by ``session.resume``."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system", "tool"]
    text: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    # Populated for assistant turns whose only action was tool-calling
    # (content="" + tool_calls=[...]). Empty list for everyone else so
    # the TUI can iterate uniformly.
    tool_calls: list[TranscriptToolCall] = Field(default_factory=list)
    # Populated for tool turns so the TUI can render `(failed)` markers
    # without re-parsing the content body.
    is_error: bool = False
    metadata: dict[str, Any] | None = None


class ProviderInfo(BaseModel):
    """Display + credential metadata for a known provider."""

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    requires_api_key: bool
    default_base_url: str | None = None


class ModelInfo(BaseModel):
    """A model offered by a provider."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    context_window: int | None = None


class SlashCommandInfo(BaseModel):
    """One entry from the slash-command catalog returned by ``commands.catalog``.

    ``category`` is a hint for the TS UI to group commands.  Stable
    values: ``local`` (no RPC), ``session``, ``control``, ``remote``.
    Future categories may be added; clients should treat unknown
    categories as ``local``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    category: str | None = None


__all__ = [
    "ModelInfo",
    "ProviderInfo",
    "SessionInfo",
    "SlashCommandInfo",
    "TranscriptMessage",
    "TranscriptToolCall",
]
