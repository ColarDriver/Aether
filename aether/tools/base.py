"""Base contracts for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Protocol

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext


class UnknownToolError(KeyError):
    """Raised when a tool call references an unknown tool."""


class TaskResourceAware(Protocol):
    """Optional protocol for tools that own task-scoped resources."""

    def acquire_task_resource(self, task_id: str) -> None:
        ...

    def release_task_resource(self, task_id: str) -> None:
        ...


@dataclass(slots=True)
class ToolDescriptor:
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


class ToolExecutor(ABC):
    """Tool execution interface."""

    interrupt_behavior: Literal["cancel", "block"] = "block"

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


# ---------------------------------------------------------------------------
# Shared spill helper
# ---------------------------------------------------------------------------
#
# Standalone module function (rather than a ToolExecutor mixin) because the
# base class is an ABC whose subclasses must keep their __init__ shape
# unchanged for backwards compatibility.  Tools call this directly:
#
#     content = maybe_spill_for_tool(
#         full_output, call=call, context=context,
#         max_chars=self.MAX_RESULT_CHARS,
#         extension="txt",
#         full_lines=full_output.count("\n") + 1,
#     )
#
# The helper reads the optional ``EngineConfig`` injected at
# ``context.metadata["_engine_config"]`` (set by
# ``AgentEngine._prepare_turn_entry``).  When config is missing
# (e.g. unit tests that hand-roll a TurnContext), we default to spill
# enabled with the canonical ``~/.aether/tool_results`` root.


def maybe_spill_for_tool(
    full_output: str,
    *,
    call: ToolCall,
    context: TurnContext,
    max_chars: int,
    extension: str = "txt",
    full_lines: int | None = None,
    full_bytes: int | None = None,
) -> str:
    """Either return ``full_output`` unchanged, or spill + return preview.

    Parameters
    ----------
    full_output:
        The complete tool output the model would have seen pre-spill.
    call, context:
        Plumbing for naming the spill file and bumping the
        ``tier1_spilled_count`` counter on ``context.metadata`` (which
        :func:`AgentEngine._build_result` then mirrors into
        ``EngineResult.metadata['compaction']['tier1_spilled_count']``).
    max_chars:
        Threshold for the per-tool decision.  Tool authors set this to
        whatever balance feels right for their data shape (40k for
        shell, 60k for read_file, 30k for grep, etc.).
    extension:
        Spill file suffix \u2014 informational only; never re-parsed.
    full_lines / full_bytes:
        Optional metrics fed to ``build_truncation_notice`` so the model
        can see ``"40000 chars / 1234 lines"`` instead of just chars.

    Returns
    -------
    str
        Either ``full_output`` itself (when below threshold or spill is
        disabled), the spilled-preview-plus-notice, or the
        plain-truncated fallback when the disk write itself failed.
    """
    config = context.metadata.get("_engine_config") if context.metadata else None
    spill_enabled = bool(getattr(config, "tool_result_spill_enabled", True))
    spill_dir = getattr(config, "tool_result_spill_dir", None)

    if not spill_enabled or len(full_output) <= max_chars:
        return full_output

    # Local imports keep the storage module's import cost off the cold
    # path for tools that never trigger spill.
    from aether.runtime.tools.tool_result_storage import (
        build_truncation_notice,
        spill_to_disk,
    )

    preview = full_output[:max_chars]
    try:
        receipt = spill_to_disk(
            full_output,
            session_id=context.session_id,
            call_id=call.id,
            extension=extension,
            config_dir=spill_dir,
            preview_chars=len(preview),
        )
        notice = build_truncation_notice(
            receipt, full_lines=full_lines, full_bytes=full_bytes
        )
        # Bump the per-turn spill counter; ``AgentEngine._build_result``
        # mirrors this into
        # ``EngineResult.metadata['compaction']['tier1_spilled_count']``.
        context.metadata["tier1_spilled_count"] = (
            int(context.metadata.get("tier1_spilled_count", 0)) + 1
        )
        return preview + notice
    except OSError as exc:
        # Disk-write failed (full disk / read-only / permission).  Fall
        # back to plain truncation so the tool still returns *something*
        # the model can act on, with the failure cause inlined.
        return preview + (
            f"\n\n... [output truncated: {len(full_output)} chars total, "
            f"could not spill to disk: {exc}] ..."
        )
