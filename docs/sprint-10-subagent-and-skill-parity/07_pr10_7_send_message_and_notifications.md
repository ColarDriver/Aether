# PR 10.7 — SendMessage Tool + Task Notifications

## 目标 / Goal

两条独立但相关的链路：

1. **SendMessage**：父 agent（或任何同进程的 agent）可以向运行中的 async 子 agent 投递追加 user 消息；子 agent 在 iteration boundary drain 并把消息当 user turn 插入上下文。
2. **Task notifications**：async child 进入终态（completed / failed / interrupted / killed）时，自动把"完成事件"封装为 `<task-notification>` 标记的 user message 推回父 agent 的下一轮 messages。

参考：
- `open-claude-code/src/tools/SendMessageTool/SendMessageTool.ts:67-200+`（工具实现）
- `open-claude-code/src/tasks/LocalAgentTask/LocalAgentTask.tsx:162-192`（pending message queue）
- `open-claude-code/src/coordinator/coordinatorMode.ts`（`<task-notification>` 描述与格式）

## 当前问题 / Current Problem

- 没有 SendMessage 工具。
- 父 agent 启 async child 后**无法得知 child 完成事件**——除非主动轮询 `task_output`。这与 `open-claude-code` 的"完成自动推 notification"行为不一致，体感很差。

## 改动 / Changes

### 1. 新文件 `aether/tools/builtins/send_message.py`

```python
"""SendMessage — queue an additional user message for a running subagent.

Mirrors `open-claude-code/src/tools/SendMessageTool/SendMessageTool.ts`.
The message is appended to the target task's `pending.jsonl`; the child
drains it at the next iteration boundary (PR 10.7 hooks the drain into
`AgentEngine._prepare_turn_entry`).

Constraints:
- Target must be in RUNNING status (not yet terminal).
- Only same-process tasks; cross-process / remote routing is not in scope.
"""
from __future__ import annotations

from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tasks import TaskStatus, TaskStore
from aether.tools.base import ToolDescriptor, ToolExecutor


class SendMessageTool(ToolExecutor):
    NAME = "send_message"

    def __init__(self, task_store: TaskStore | None = None) -> None:
        self._task_store = task_store
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Queue an additional user message for a running async "
                "subagent.  The child sees it at its next iteration "
                "boundary as a user-role turn.  Use this to add "
                "clarifications, redirect work, or share results from "
                "peer tasks.  Returns immediately; does not wait for "
                "the child to read."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Target task_id (returned by `task` when run_in_background=true).",
                    },
                    "message": {
                        "type": "string",
                        "description": "Content to deliver as a user-role turn to the subagent.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional 5-10 word preview (currently telemetry-only).",
                    },
                },
                "required": ["to", "message"],
            },
            required=["to", "message"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        to = args.get("to")
        message = args.get("message")
        if not isinstance(to, str) or not to.strip():
            return _error(call, "'to' is required and must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            return _error(call, "'message' is required and must be a non-empty string")
        to = to.strip()
        message = message.strip()

        store = self._resolve_store(context)
        if store is None:
            return _error(call, "send_message unavailable: no TaskStore configured")

        record = store.read(to)
        if record is None:
            return _error(call, f"unknown task_id: {to!r}")
        if record.status != TaskStatus.RUNNING:
            return _error(
                call,
                f"cannot send to {to!r}: status is {record.status.value!r} (must be running)",
            )

        store.enqueue_pending_message(to, message)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=(
                f"Queued. Subagent {to} will see the message at its next "
                f"iteration boundary."
            ),
            is_error=False,
            metadata={
                "to": to,
                "summary": args.get("summary"),
                "queued_chars": len(message),
            },
        )

    def _resolve_store(self, context: TurnContext) -> TaskStore | None:
        if self._task_store is not None:
            return self._task_store
        injected = context.metadata.get("_task_store") if context.metadata else None
        return injected if isinstance(injected, TaskStore) else None


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
    )


__all__ = ["SendMessageTool"]
```

### 2. Pending message drain（child 侧）

**改 `aether/agents/core/agent.py`** —— 在主循环 PRE_LLM 进入时（line 1574 附近）新增方法 + 调用：

```python
def _drain_pending_messages(
    self,
    messages: list[dict[str, Any]],
    context: TurnContext,
) -> None:
    """Pull SendMessage / notification messages from TaskStore and inject
    them as user-role turns before the next LLM call.

    Two source streams are merged:
    1. TaskStore.drain_pending_messages — peer SendMessage / parent->child
       deliveries (only meaningful for async subagents that have a task_id).
    2. self._external_event_queue — in-memory queue for the root engine
       (parent receives <task-notification> from finished async children
       this way, since the root has no task_id).
    """
    drained: list[str] = []
    if self._task_store is not None and context.task_id:
        drained.extend(self._task_store.drain_pending_messages(context.task_id))
    if self._external_event_queue:
        drained.extend(self._external_event_queue)
        self._external_event_queue.clear()

    for msg in drained:
        if not msg:
            continue
        messages.append({
            "role": "user",
            "content": msg,
            "metadata": {"source": "send_message" if "<task-notification>" not in msg else "task_notification"},
        })
```

在主循环每次 PRE_LLM 前调一次（与 `_maybe_inject_skill_nudge` 同处）。

`AgentEngine.__init__` 里建：

```python
self._external_event_queue: list[str] = []

def enqueue_external_event(self, message: str) -> None:
    self._external_event_queue.append(message)
```

`context.task_id` 已经存在于 `TurnContext`（`aether/runtime/core/contracts.py:166-173`）；async lifecycle 在创建 child 时把 `task.task_id` 设给 child 的 `_current_task_id`，确保 child 进入 PRE_LLM 时 `context.task_id` 有值。

### 3. `<task-notification>` 注入路径

**改 `aether/subagents/manager.py`** —— 在 `_on_async_done`（PR 10.5 的 callback）里：

```python
def _on_async_done(self, task_id: str, future) -> None:
    try:
        result = future.result()
    except Exception:
        self.logger.exception("Async task %s callback raised", task_id)
        return

    notification = _build_task_notification(task_id, result)
    # Route to parent: prefer task store if parent is itself a task;
    # otherwise push to parent engine's in-memory queue.
    parent_task_id = result.metadata.get("parent_task_id")
    if parent_task_id and self._task_store is not None:
        self._task_store.enqueue_pending_message(parent_task_id, notification)
        return

    parent_session_id = result.metadata.get("parent_session_id")
    parent_engine = self._lookup_parent_engine(parent_session_id)
    if parent_engine is not None:
        try:
            parent_engine.enqueue_external_event(notification)
        except Exception:
            self.logger.exception("failed to enqueue notification on parent %s", parent_session_id)


def _lookup_parent_engine(self, session_id: str | None) -> "AgentEngine | None":
    # SubagentManager already tracks active children by task_id;  for
    # parent lookup we keep a weakref map session_id -> root engine,
    # populated by AgentEngine.__init__ when delegate_depth==0.
    return self._root_engines.get(session_id)
```

新增 `self._root_engines: weakref.WeakValueDictionary[str, AgentEngine]`（manager 构造时建）；`AgentEngine.__init__` 末尾若 `delegate_depth == 0` 时：

```python
if self.subagent_manager is not None:
    self.subagent_manager._root_engines[self._current_session_id or "default"] = self  # noqa: SLF001
```

Notification XML：

```python
def _build_task_notification(task_id: str, result: SubagentResult) -> str:
    status = result.status.value if hasattr(result.status, "value") else str(result.status)
    summary = (result.summary or "").strip()
    error = (result.error or "").strip()
    parts = [
        "<task-notification>",
        f"  <task_id>{_xml_escape(task_id)}</task_id>",
        f"  <subagent_type>{_xml_escape(result.metadata.get('subagent_type', ''))}</subagent_type>",
        f"  <status>{_xml_escape(status)}</status>",
        f"  <duration_seconds>{result.duration_seconds:.1f}</duration_seconds>",
    ]
    if summary:
        parts.append(f"  <summary>{_xml_escape(summary)}</summary>")
    if error:
        parts.append(f"  <error>{_xml_escape(error)}</error>")
    parts.append("</task-notification>")
    return "\n".join(parts)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
```

### 4. 注册 SendMessageTool + 加入 META_TOOLS

**改 `aether/tools/builtins/__init__.py`** —— `build_default_tool_registry` 中注册 `SendMessageTool(task_store=...)`。

**改 `aether/tools/filter.py`**（PR 10.3 创建）—— 确保 `META_TOOLS_ALWAYS_ALLOWED` 含 `send_message`（已在 PR 10.3 加，本 PR 验证即可）。

### 5. Wire 事件

**改 `aether/gateway/protocol.py`** —— 新增：

```python
class TaskNotificationEvent(AgentEventBase):
    type: Literal["task.notification"] = "task.notification"
    task_id: str
    parent_task_id: str | None
    parent_session_id: str | None
    subagent_type: str
    status: str
    summary: str | None
    error: str | None
    duration_seconds: float
```

`_on_async_done` 同步 emit 一次（gateway sink 在 manager 注入时拿到）。TS 端补类型。

## 测试 / Tests

### Python

新建 `aether/tests/tools/test_send_message.py`：

- `test_unknown_task_errors`
- `test_completed_task_errors_with_status_hint`
- `test_running_task_succeeds_and_enqueues` —— enqueue 后 store.read 出 pending.jsonl 有一行
- `test_long_message_preserved` —— 50KB message 完整入队，drain 后等值
- `test_summary_is_telemetry_only` —— summary 参数不进 pending 内容

新建 `aether/tests/agents/test_pending_message_drain.py`：

- `test_drain_empties_queue` —— store 中 enqueue 2 条 → engine 进 PRE_LLM 一次 → messages 末尾多 2 条 user turn；store 中 pending.jsonl 空
- `test_drain_marks_metadata_source` —— drained 消息的 `metadata.source == "send_message"` 或 `"task_notification"`
- `test_drain_on_root_engine_uses_external_queue` —— 根 engine 调 `enqueue_external_event(text)` → 下一次 PRE_LLM messages 末尾出现 text

新建 `aether/tests/subagents/test_task_notification.py`：

- `test_notification_xml_format` —— `_build_task_notification` 输出含 `<task_id>`, `<status>`, `<duration_seconds>`, optional `<summary>`/`<error>`
- `test_notification_routes_to_parent_root_engine` —— async child 完成 → parent root engine 的 `_external_event_queue` 多一条
- `test_nested_notification_routes_via_task_store` —— 父也是 async task（嵌套）→ store enqueue_pending_message(parent_task_id, ...) 被调
- `test_notification_xml_escapes_special_chars` —— summary 含 `<>&` 被正确转义

### 验收 / Acceptance

- `uv run pytest aether/tests/tools/test_send_message.py aether/tests/agents/test_pending_message_drain.py aether/tests/subagents/test_task_notification.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**（场景：父启 async child，发 send_message，等通知）：
  1. `task(prompt="iterate 5 times, print iteration number, then ask if you should stop", run_in_background=true)` → 收到 tid。
  2. `send_message(to="<tid>", message="please also count to 3 in chinese before stopping")` → 立即 ack。
  3. 等几秒；child 应该在 iteration 边界看到追加 user message，并在 summary 中提到 "一 二 三"。
  4. child 终态 → 父下一次 turn 收到 `<task-notification>` 用户消息（在 messages 流里可见）。

## 不在本 PR / Deferred

- **Name-based routing** —— 现在只用 task_id；未来若引入 "agent name registry" 可扩展。
- **Broadcast (`to="*"`)** —— 不实现。
- **Cross-process notification** —— 同进程为限，不支持远端 child 通知本地 parent。
