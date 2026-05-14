# PR 10.6 — TaskOutput Tool

## 目标 / Goal

把 `aether/tools/builtins/task_output.py`（当前 sprint 3.5 占位）改写为完整工具：

- 输入：`task_id`、`block: bool`（默认 `true`）、`timeout_ms`（默认 30_000，最大 600_000）。
- 行为：从 `TaskStore` 读取目标任务的 `task.json` + `output.log`；可阻塞等待终态。
- 输出：人类可读 content + 结构化 metadata（status / progress / summary / 失败原因）。

参考：`open-claude-code/src/tools/TaskOutputTool/TaskOutputTool.ts`（block/non-block 两种模式 + 文件路径返回）。

## 当前问题 / Current Problem

现存 `aether/tools/builtins/task_output.py`（sprint 3.5）返回 "not supported in sync mode" 字符串。Async lifecycle（PR 10.5）落地后，这条占位毫无用处，必须重写。

## 改动 / Changes

### 1. 重写 `aether/tools/builtins/task_output.py`

```python
"""TaskOutput — read a subagent task's progress + final result.

Backed by ``TaskStore`` (PR 10.4).  Both sync and async subagents
finalize through the store, so this tool works uniformly for both.

Modes:
- ``block=False``  (instant): return current snapshot of task.json + output.log tail.
- ``block=True``   (default): wait until task reaches terminal status or timeout.
                              Falls back to snapshot on timeout (is_error=False).

Output schema (in result.metadata):
  status:    "running" | "completed" | "failed" | "interrupted" | "killed"
  task_id:   str
  duration_seconds: float | None
  progress:  {tool_use_count, input_tokens, output_tokens, iterations}
  summary:   str | None   (if completed)
  error:     str | None   (if not completed)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tasks import TaskStatus, TaskStore
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


_DEFAULT_TIMEOUT_MS = 30_000
_MAX_TIMEOUT_MS = 600_000
_POLL_INTERVAL_S = 0.2


class TaskOutputTool(ToolExecutor):
    NAME = "task_output"
    MAX_RESULT_CHARS = 60_000

    def __init__(self, task_store: TaskStore | None = None) -> None:
        self._task_store = task_store
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Retrieve output and final result of a subagent task (sync "
                "or async) by task_id.  Set block=true (default) to wait "
                "for the task to finish; block=false returns the current "
                "snapshot immediately.  Works for completed historical "
                "tasks too (reads from ~/.aether/tasks/{task_id}/)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned by `task` when run_in_background=true.",
                    },
                    "block": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, wait until task finishes; else return current snapshot.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "default": _DEFAULT_TIMEOUT_MS,
                        "minimum": 0,
                        "maximum": _MAX_TIMEOUT_MS,
                        "description": "Maximum wait time in ms when block=true.",
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

        block = bool(args.get("block", True))
        try:
            timeout_ms = int(args.get("timeout_ms", _DEFAULT_TIMEOUT_MS))
        except (TypeError, ValueError):
            return _error(call, "'timeout_ms' must be an integer")
        timeout_ms = max(0, min(timeout_ms, _MAX_TIMEOUT_MS))

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

        if block and not record.status.is_terminal and timeout_ms > 0:
            record = self._wait_terminal(store, task_id, timeout_ms)

        return self._render(call, context, store, record)

    # ------------------------------------------------------------------ helpers

    def _resolve_store(self, context: TurnContext) -> TaskStore | None:
        if self._task_store is not None:
            return self._task_store
        injected = context.metadata.get("_task_store") if context.metadata else None
        return injected if isinstance(injected, TaskStore) else None

    def _wait_terminal(self, store: TaskStore, task_id: str, timeout_ms: int):
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            record = store.read(task_id)
            if record is None or record.status.is_terminal:
                return record
            time.sleep(_POLL_INTERVAL_S)
        return store.read(task_id)

    def _render(self, call, context, store: TaskStore, record) -> ToolResult:
        status = record.status.value
        # Build human-readable content.
        lines: list[str] = [
            f"# Task {record.task_id} — {status}",
            f"- subagent_type: {record.subagent_type}",
            f"- started_at: {_iso(record.started_at)}",
        ]
        if record.finished_at is not None:
            duration = record.finished_at - record.started_at
            lines.append(f"- duration: {duration:.1f}s")
        if record.background:
            lines.append("- background: true")
        if record.worktree_path:
            lines.append(f"- worktree: {record.worktree_path}")
        lines.append("")
        lines.append("## Progress")
        lines.append(f"- iterations: {record.iterations}")
        lines.append(f"- tool_use_count: {record.tool_use_count}")
        lines.append(f"- tokens: in={record.input_tokens} out={record.output_tokens}")

        if record.summary:
            lines.append("")
            lines.append("## Summary")
            lines.append(record.summary)

        if record.error:
            lines.append("")
            lines.append("## Error")
            lines.append(record.error)

        tail = store.read_output_tail(record.task_id, max_bytes=20_000)
        if tail.strip():
            lines.append("")
            lines.append("## Output (last 20KB)")
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

        metadata: dict[str, Any] = {
            "status": status,
            "task_id": record.task_id,
            "duration_seconds": (
                record.finished_at - record.started_at if record.finished_at else None
            ),
            "progress": {
                "tool_use_count": record.tool_use_count,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "iterations": record.iterations,
            },
            "summary": record.summary,
            "error": record.error,
            "result_path": record.result_path,
        }
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=record.status == TaskStatus.FAILED,
            metadata=metadata,
        )


def _iso(ts: float) -> str:
    if ts <= 0:
        return "n/a"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _error(call: ToolCall, message: str, *, metadata=None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["TaskOutputTool"]
```

### 2. 重新注册

**改 `aether/tools/builtins/__init__.py`** —— `build_default_tool_registry` 中确保用本 PR 的 `TaskOutputTool`（替换占位）。

### 3. AgentEngine 把 task_store 塞进 metadata

PR 10.4 已经把 store 放进 `context.metadata["_task_store"]`；本 PR 直接读。

## 测试 / Tests

### Python

新建 `aether/tests/tools/test_task_output.py`：

- `test_unknown_task_id_errors` —— 未知 task_id → is_error。
- `test_completed_task_returns_immediately` —— 写一个 COMPLETED 记录 + result.json → block=True 立即返回，metadata.status=="completed"。
- `test_block_false_returns_snapshot` —— RUNNING 记录 + 100 字节 output.log → block=False 立即返回，content 含 "Output" 节。
- `test_block_true_unblocks_on_status_change` —— 起一个线程 500ms 后 update_status(COMPLETED)；调 block=True 应在 500-700ms 间返回。
- `test_block_true_timeout_falls_back_to_snapshot` —— RUNNING 记录，timeout_ms=200 → 阻塞约 200ms 后返回当前快照（非 error）。
- `test_failed_task_is_error_true` —— FAILED 状态 → is_error=True，但 content 仍含 progress / error sections。
- `test_long_output_spills_to_file` —— output.log 含 200KB → content 被 spill，metadata 含 spill 路径。
- `test_killed_task_after_recovery` —— 模拟 recovery 把 RUNNING→KILLED；block=True → 立即返回 status=="killed"。

新建 `aether/tests/tools/test_task_output_e2e.py`（与 PR 10.5 配合）：

- spawn async child（scripted provider），父立刻调 `task_output(block=True)`；child 完成后父收到 summary。

### 验收 / Acceptance

- `uv run pytest aether/tests/tools/test_task_output.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**：
  1. 先 `task(prompt="...", run_in_background=true)` → 拿到 task_id。
  2. `task_output(task_id="...", block=false)` → 立即看到 progress。
  3. `task_output(task_id="...", block=true)` → 阻塞，子完成后立即返回 summary。
  4. 杀 gateway → 重启 → 同样的 task_id 调 `task_output` 得到 `killed` 状态。
- timeout 精度：`block=True, timeout_ms=200` 实际阻塞 200-400ms（含 200ms poll 间隔的抖动）。

## 不在本 PR / Deferred

- **Tail 流式 streaming（每条新行立刻 emit）** —— 留给后续，本 PR 仅快照式 tail。
- **多 task_id 批量查询** —— 单个 ID 已够用，不引入复杂 API。
- **TUI 端 inline tail panel** —— 等 task dashboard 上线。
