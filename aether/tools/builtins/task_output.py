"""Built-in ``task_output`` tool — PR 10.6.

Reads a subagent task's persisted state (``TaskRecord`` + ``output.log``)
from the on-disk :class:`TaskStore` introduced by PR 10.4.  Works for
both async tasks (PR 10.5) currently in flight and historical tasks
left over from prior gateway sessions.

Modes:

- ``block=False``: return the current snapshot immediately.
- ``block=True`` (default): wait until the task reaches a terminal
  status or ``timeout_ms`` elapses, then return whatever's there.
  A timeout returns ``is_error=False`` with a ``timed_out=True`` flag
  in the metadata so the model can choose to poll again.

Catalog resolution mirrors :class:`SkillTool`: constructor injection
wins, then ``context.metadata['_task_store']`` (set by
:meth:`AgentEngine._prepare_turn_entry`).  Without a store the tool
returns a clear "not configured" error rather than silently degrading.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_TIMEOUT_MS = 30_000
_MAX_TIMEOUT_MS = 600_000
_POLL_INTERVAL_S = 0.2
_OUTPUT_TAIL_BYTES = 20_000


class TaskOutputTool(ToolExecutor):
    """Read a subagent task's status + tailing output."""

    NAME = "task_output"
    MAX_RESULT_CHARS = 60_000

    def __init__(self, task_store: TaskStore | None = None) -> None:
        self._task_store = task_store
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Retrieve the current status, progress, and output of a "
                "subagent task by task_id.  Set ``block=true`` (default) "
                "to wait until the task reaches a terminal state (or "
                "``timeout_ms`` elapses); ``block=false`` returns the "
                "current snapshot immediately.  Works for completed "
                "historical tasks too (reads from "
                "~/.aether/tasks/{task_id}/).  task_ids come from prior "
                "calls to `task(run_in_background=true)`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": (
                            "Identifier returned by `task` when "
                            "``run_in_background=true``."
                        ),
                    },
                    "block": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "If true, wait for the task to finish "
                            "before returning; if false, return the "
                            "current snapshot."
                        ),
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": _MAX_TIMEOUT_MS,
                        "default": _DEFAULT_TIMEOUT_MS,
                        "description": (
                            "Maximum time to block in milliseconds when "
                            "``block=true``.  Capped at 600000 (10 min)."
                        ),
                    },
                },
                "required": ["task_id"],
            },
            required=["task_id"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        task_id = args.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            return _error(call, "'task_id' is required and must be a non-empty string")
        task_id = task_id.strip()

        block_arg = args.get("block", True)
        if not isinstance(block_arg, bool):
            return _error(call, "'block' must be a boolean when provided")
        block = bool(block_arg)

        timeout_arg = args.get("timeout_ms", _DEFAULT_TIMEOUT_MS)
        if isinstance(timeout_arg, bool) or not isinstance(timeout_arg, int):
            return _error(call, "'timeout_ms' must be an integer when provided")
        timeout_ms = max(0, min(int(timeout_arg), _MAX_TIMEOUT_MS))

        store = self._resolve_store(context)
        if store is None:
            return _error(
                call,
                "task_output unavailable: no TaskStore configured "
                "(check EngineConfig.task_store_enabled).",
            )

        record = store.read(task_id)
        if record is None:
            return _error(call, f"unknown task_id: {task_id!r}")

        timed_out = False
        if block and not record.status.is_terminal and timeout_ms > 0:
            record, timed_out = self._wait_terminal(store, task_id, timeout_ms)
            if record is None:
                # Race: the task got deleted while we were waiting.
                return _error(call, f"task {task_id!r} disappeared while polling")

        return self._render(call, context, store, record, timed_out=timed_out)

    # ------------------------------------------------------------ helpers

    def _resolve_store(self, context: TurnContext) -> TaskStore | None:
        if self._task_store is not None:
            return self._task_store
        injected = context.metadata.get("_task_store") if context.metadata else None
        return injected if isinstance(injected, TaskStore) else None

    def _wait_terminal(
        self,
        store: TaskStore,
        task_id: str,
        timeout_ms: int,
    ) -> tuple[TaskRecord | None, bool]:
        """Poll ``task.json`` until the status is terminal or we time out.

        Returns ``(record, timed_out)``.  Returns ``(None, ...)`` only if
        the record disappears mid-poll (unlikely but possible if the
        store root was wiped).
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            record = store.read(task_id)
            if record is None:
                return None, False
            if record.status.is_terminal:
                return record, False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return record, True
            time.sleep(min(_POLL_INTERVAL_S, remaining))

    def _render(
        self,
        call: ToolCall,
        context: TurnContext,
        store: TaskStore,
        record: TaskRecord,
        *,
        timed_out: bool,
    ) -> ToolResult:
        status = record.status.value
        duration: float | None = None
        if record.finished_at is not None:
            duration = max(0.0, record.finished_at - record.started_at)

        lines: list[str] = [
            f"# Task {record.task_id} — {status}",
            f"- subagent_type: {record.subagent_type}",
            f"- started_at: {_iso(record.started_at)}",
        ]
        if record.finished_at is not None:
            lines.append(f"- finished_at: {_iso(record.finished_at)}")
            assert duration is not None
            lines.append(f"- duration: {duration:.1f}s")
        if record.background:
            lines.append("- background: true")
        if record.model:
            lines.append(f"- model: {record.model}")
        if record.worktree_path:
            lines.append(f"- worktree: {record.worktree_path}")
        if record.parent_task_id:
            lines.append(f"- parent_task_id: {record.parent_task_id}")
        if timed_out:
            lines.append(
                f"- _timed_out: true (block timed out before terminal status)"
            )

        lines.append("")
        lines.append("## Progress")
        lines.append(f"- iterations: {record.iterations}")
        lines.append(f"- tool_use_count: {record.tool_use_count}")
        lines.append(
            f"- tokens: in={record.input_tokens} out={record.output_tokens}"
        )

        if record.summary:
            lines.append("")
            lines.append("## Summary")
            lines.append(record.summary)

        if record.error:
            lines.append("")
            lines.append("## Error")
            lines.append(record.error)

        tail = store.read_output_tail(record.task_id, max_bytes=_OUTPUT_TAIL_BYTES)
        if tail.strip():
            lines.append("")
            lines.append("## Output (tail)")
            lines.append(tail.rstrip())

        body = "\n".join(lines) + "\n"
        content = maybe_spill_for_tool(
            body,
            call=call,
            context=context,
            max_chars=self.MAX_RESULT_CHARS,
            extension="md",
            full_lines=body.count("\n") + 1,
        )

        progress = {
            "tool_use_count": record.tool_use_count,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "iterations": record.iterations,
        }
        metadata: Dict[str, Any] = {
            "status": status,
            "task_id": record.task_id,
            "duration_seconds": duration,
            "progress": progress,
            "summary": record.summary,
            "error": record.error,
            "result_path": record.result_path,
            "background": record.background,
            "timed_out": timed_out,
        }
        # is_error reflects whether the TASK failed, not whether the
        # tool call itself succeeded; FAILED is the only status that
        # surfaces as an error to the parent's loop.
        is_error = record.status == TaskStatus.FAILED
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=is_error,
            metadata=metadata,
        )


def _iso(ts: float) -> str:
    if ts <= 0:
        return "n/a"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["TaskOutputTool"]
