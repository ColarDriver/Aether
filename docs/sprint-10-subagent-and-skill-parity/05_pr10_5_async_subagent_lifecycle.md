# PR 10.5 — Async Subagent Lifecycle

## 目标 / Goal

新增 `AgentTool` 的 `run_in_background=true` 路径：调用立即返回 `task_id`，子 agent 在后台 `ThreadPoolExecutor` 上跑完整 lifecycle；所有 message / progress / 最终 result 通过 `TaskStore`（PR 10.4）写盘。

参考：
- `open-claude-code/src/tools/AgentTool/AgentTool.tsx:686-764`（async branch）
- `open-claude-code/src/tools/AgentTool/agentToolUtils.ts:508-686`（`runAsyncAgentLifecycle`）
- `open-claude-code/src/tasks/LocalAgentTask/LocalAgentTask.tsx:466-515`（`registerAsyncAgent`）

## 当前问题 / Current Problem

`aether/tools/builtins/agent_tool.py:155`：

```python
metadata={
    "subagent_type": subagent_type,
    "expected_output": expected_output,
    "run_in_background": False,                # hardcoded
},
```

`aether/subagents/manager.py:44`：

```python
def run_task(self, parent, task: SubagentTask) -> SubagentResult:
    return self.run_tasks(parent=parent, tasks=[task])[0]
```

—— `run_tasks` 内用 `ThreadPoolExecutor` + `as_completed` 等到全部完成才返回。父循环被完全阻塞。

## 改动 / Changes

### 1. `SubagentManager` 新方法

**改 `aether/subagents/manager.py`**：

```python
from concurrent.futures import ThreadPoolExecutor
from aether.runtime.tasks import TaskStore, TaskStatus, TaskRecord
import time

class SubagentManager:
    def __init__(
        self,
        *,
        builder: SubagentBuilder | None = None,
        max_concurrent_children: int = 3,
        max_spawn_depth: int = 2,
        max_concurrent_background: int = 8,
        task_store: TaskStore | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        ...
        self._task_store = task_store
        # Dedicated executor for long-running async tasks.  Bigger than
        # max_concurrent_children because async tasks can outlive the
        # parent's tool dispatch.
        self._async_executor = ThreadPoolExecutor(
            max_workers=max(1, max_concurrent_background),
            thread_name_prefix="aether-async-subagent",
        )

    def run_task_async(self, *, parent, task: SubagentTask) -> str:
        """Spawn the child on a background thread, return task_id immediately."""
        if self._task_store is None:
            raise RuntimeError("run_task_async requires a TaskStore on the manager")

        if parent.delegate_depth >= self.max_spawn_depth:
            raise RuntimeError(
                f"Delegation depth limit reached: depth={parent.delegate_depth}"
            )

        # Write initial RUNNING record before submitting; this guarantees
        # external observers can see the task even if the executor is busy.
        record = self._build_initial_record(parent=parent, task=task)
        self._task_store.create(record)

        future = self._async_executor.submit(
            self._execute_one_async,
            parent=parent,
            task=task,
            child_depth=parent.delegate_depth + 1,
        )
        future.add_done_callback(
            lambda f, tid=task.task_id: self._on_async_done(tid, f)
        )
        return task.task_id

    def _build_initial_record(self, *, parent, task) -> TaskRecord:
        definition = task.metadata.get("_agent_type_def")
        return TaskRecord(
            task_id=task.task_id,
            parent_session_id=task.request.metadata.get("parent_session_id", ""),
            subagent_type=task.metadata.get("subagent_type", "general-purpose"),
            prompt=task.request.user_message or "",
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            last_heartbeat=time.time(),
            agent_type_def_snapshot=definition.to_snapshot() if definition else {},
            model=task.request.model_config.model if task.request.model_config else None,
            isolation=task.metadata.get("isolation"),
            parent_task_id=task.metadata.get("parent_task_id"),
            child_depth=parent.delegate_depth + 1,
            background=True,
        )

    def _execute_one_async(self, *, parent, task: SubagentTask, child_depth: int) -> SubagentResult:
        """Runs in the async executor.  Mirrors _execute_one but writes to TaskStore."""
        started_at = time.monotonic()
        child = self.builder.build_child(parent=parent, task=task, child_depth=child_depth)
        # Inject hooks that fan out to TaskStore.
        self._install_task_store_hooks(child, task.task_id)
        parent._register_child(child)

        stop_event = threading.Event()
        with self._stop_events_lock:
            self._stop_events[task.task_id] = stop_event
            self._active_children[task.task_id] = child
        setattr(child, "_stop_event", stop_event)

        try:
            child_result = child.run_loop(task.request)
            status = _engine_to_subagent_status(child_result.status)
            summary = child_result.final_response
            result = SubagentResult(
                task_id=task.task_id,
                status=status,
                summary=summary,
                engine_result=child_result,
                error=child_result.error,
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                    "background": True,
                },
            )
            self._finalize_to_store(task.task_id, result)
            return result
        except Exception as exc:
            self.logger.exception("Async subagent task %s crashed: %s", task.task_id, exc)
            result = SubagentResult(
                task_id=task.task_id,
                status=SubagentStatus.FAILED,
                summary=None,
                engine_result=None,
                error=str(exc),
                duration_seconds=time.monotonic() - started_at,
                metadata={"goal": task.goal, "child_depth": child_depth, "background": True},
            )
            self._finalize_to_store(task.task_id, result)
            return result
        finally:
            parent._unregister_child(child)
            with self._stop_events_lock:
                self._stop_events.pop(task.task_id, None)
                self._active_children.pop(task.task_id, None)

    def _install_task_store_hooks(self, child, task_id: str) -> None:
        """Wrap child's hooks pipeline so each after_tool / after_llm writes to store."""
        store = self._task_store
        if store is None:
            return
        original = child._hooks

        class _StoreFanoutHooks:
            def after_llm(self_inner, response, context):
                store.append_message(task_id, {
                    "role": "assistant",
                    "content": getattr(response, "text", "") or "",
                    "iteration": getattr(context, "iteration", 0),
                })
                store.record_progress(
                    task_id,
                    input_tokens=getattr(response, "input_tokens", 0) or 0,
                    output_tokens=getattr(response, "output_tokens", 0) or 0,
                    iterations=getattr(context, "iteration", 0),
                )
                if original is not None:
                    original.after_llm(response, context)

            def after_tool(self_inner, result, context):
                store.append_message(task_id, {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "name": result.name,
                    "content": result.content[:2000],
                    "is_error": result.is_error,
                })
                store.append_output(task_id, f"\n[{result.name}] {result.content[:500]}\n")
                store.record_progress(task_id, tool_use_count_delta=1)
                if original is not None:
                    original.after_tool(result, context)

            def on_iteration(self_inner, context):
                store.record_heartbeat(task_id)
                if original is not None and hasattr(original, "on_iteration"):
                    original.on_iteration(context)

        child._hooks = _StoreFanoutHooks()

    def _finalize_to_store(self, task_id: str, result: SubagentResult) -> None:
        if self._task_store is None:
            return
        terminal_map = {
            SubagentStatus.COMPLETED: TaskStatus.COMPLETED,
            SubagentStatus.FAILED: TaskStatus.FAILED,
            SubagentStatus.INTERRUPTED: TaskStatus.INTERRUPTED,
        }
        status = terminal_map.get(result.status, TaskStatus.FAILED)
        self._task_store.update_status(
            task_id,
            status,
            summary=result.summary,
            error=result.error,
        )
        self._task_store.write_result(task_id, {
            "task_id": result.task_id,
            "status": status.value,
            "summary": result.summary,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
            "metadata": result.metadata,
        })

    def _on_async_done(self, task_id: str, future) -> None:
        try:
            future.result()
        except Exception:
            self.logger.exception("Async task %s done callback raised", task_id)
```

### 2. `AgentTool` 增加 async 分支

**改 `aether/tools/builtins/agent_tool.py`**：

#### a. Schema 加 `run_in_background`

```python
"run_in_background": {
    "type": "boolean",
    "default": False,
    "description": (
        "Set to true to run this agent in the background.  You will be "
        "notified when it completes via a <task-notification> message."
    ),
},
```

#### b. `execute` 分支

```python
def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
    args = call.arguments or {}
    ...

    run_in_background = bool(args.get("run_in_background", False))
    # Type definition can force async
    if definition is not None and definition.background:
        run_in_background = True

    ...
    task = SubagentTask(
        task_id=task_id,
        goal=goal,
        request=request,
        metadata={
            "subagent_type": subagent_type,
            "expected_output": expected_output,
            "run_in_background": run_in_background,
            "_agent_type_def": definition,
            "model_override": args.get("model") or None,
            "isolation": args.get("isolation") or (definition.isolation if definition else None),
        },
    )

    if run_in_background:
        if not getattr(context.metadata.get("_engine_config"), "subagent_async_enabled", True):
            return _error(call, "async subagent dispatch is disabled by configuration")
        try:
            tid = manager.run_task_async(parent=parent, task=task)
        except RuntimeError as exc:
            return _error(call, f"async dispatch failed: {exc}", metadata={"task_id": task_id})
        store_dir = manager._task_store.root / tid  # noqa: SLF001
        body = (
            f"Subagent launched in background.\n"
            f"- task_id: {tid}\n"
            f"- subagent_type: {subagent_type}\n"
            f"- output_path: {store_dir / 'output.log'}\n\n"
            f"Use `task_output(task_id=\"{tid}\")` to poll, "
            f"`send_message(to=\"{tid}\", message=\"...\")` to message, "
            f"`task_stop(task_id=\"{tid}\")` to cancel."
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=body,
            is_error=False,
            metadata={
                "status": "async_launched",
                "task_id": tid,
                "subagent_type": subagent_type,
                "output_path": str(store_dir / "output.log"),
                "result_path": str(store_dir / "result.json"),
            },
        )

    # Sync path — unchanged below.
    try:
        result = manager.run_task(parent=parent, task=task)
    ...
```

#### c. 同步路径也写盘

为了 `TaskOutput` 工具能查到历史同步任务，在 `_execute_one`（同步路径）末尾也调一次 `self._finalize_to_store`。但同步路径开头不一定有 TaskRecord，所以在 `_execute_one` 一进来若 `self._task_store` 在且 record 不存在，create 一条 RUNNING；正常 finally 转 terminal。

具体改动：把 `_install_task_store_hooks` / `_finalize_to_store` 提到公共位置，让同步 `_execute_one` 也调一次。

### 3. Hooks 接口扩展

**改 `aether/runtime/core/hooks.py`** —— 添加（如不存在）：

```python
class EngineHooks:
    def after_llm(self, response, context): pass
    def after_tool(self, result, context): pass
    def on_iteration(self, context): pass            # NEW — heartbeat
    def on_external_event(self, event_type, payload): pass  # NEW — used by PR 10.7
```

### 4. Config

`aether/config/schema.py`：

```python
max_concurrent_background: int = 8
subagent_async_enabled: bool = True
```

### 5. Engine 把 TaskStore 传给 Manager

**改 `aether/agents/core/agent.py`** 在 `AgentEngine.__init__` 末尾，若 `self.subagent_manager._task_store is None` 且 `self._task_store is not None`，把 store 设上：

```python
if self.subagent_manager is not None and getattr(self.subagent_manager, "_task_store", None) is None:
    self.subagent_manager._task_store = self._task_store  # noqa: SLF001
```

或更优雅地，在 `SubagentManager.__init__` 接收 `task_store=`，由 `AgentEngine` 构造时传入。

### 6. Wire 事件（gateway → TUI）

**改 `aether/gateway/protocol.py`** —— 新增事件：

```python
class TaskProgressEvent(AgentEventBase):
    type: Literal["task.progress"] = "task.progress"
    task_id: str
    status: str           # "running" / "completed" / ...
    tool_use_count: int
    input_tokens: int
    output_tokens: int
    iterations: int


class TaskLaunchedEvent(AgentEventBase):
    type: Literal["task.launched"] = "task.launched"
    task_id: str
    subagent_type: str
    parent_task_id: str | None
    background: bool
```

把这两个事件加进 union；TS 端 `tui/src/gatewayTypes.ts` 补类型定义。本 sprint TUI 不渲染，但 schema 必须先定下来。

### 7. `_GatewayAgentHooks` 桥接

**改 `aether/gateway/handlers/agent_methods.py`** —— 给 hooks 桥加 `on_iteration` / progress emit：

```python
def on_iteration(self, context):
    self._sink.emit(TaskProgressEvent(
        session_id=self._sink.session_id,
        run_id=self._sink.run_id,
        task_id=context.task_id or "",
        status="running",
        ...
    ))
```

## 测试 / Tests

### Python

新建 `aether/tests/subagents/test_async_lifecycle.py`：

- `test_run_task_async_returns_immediately` —— scripted provider 跑一个会"睡 2s"的脚本；`run_task_async` 返回时间 < 100ms。
- `test_async_task_writes_running_record_immediately` —— `run_task_async` 返回后立即 `store.read(tid)` → status==RUNNING, last_heartbeat>0。
- `test_async_task_completes_and_writes_result` —— 轮询 store 直到 terminal；result.json 存在；summary 等值。
- `test_async_concurrent_tasks` —— spawn 5 个；都跑完后 store 有 5 个 COMPLETED 记录。
- `test_async_task_failure_recorded` —— scripted 抛异常；status==FAILED，error 字段非空。
- `test_async_iteration_heartbeat_advances` —— 跑 3 个 iteration；last_heartbeat 随时间增长。

新建 `aether/tests/tools/test_agent_tool_async.py`：

- `test_async_branch_returns_async_launched_metadata` —— `run_in_background=True` → metadata.status=="async_launched", task_id, output_path 都有。
- `test_async_disabled_via_config` —— `subagent_async_enabled=False` → is_error=True 含 "disabled"。
- `test_type_definition_background_forces_async` —— 类型 definition.background=True 时即便参数没传 run_in_background 也走 async。

### 验收 / Acceptance

- `uv run pytest aether/tests/subagents/test_async_lifecycle.py aether/tests/tools/test_agent_tool_async.py` 全绿。
- `uv run pyright` 无新告警。
- **手测**：
  1. 跑 `uv run aether`，prompt `task(prompt="count to 100 slowly", run_in_background=true)` → 立即收到 task_id。
  2. `ls ~/.aether/tasks/{tid}/` → 看到 task.json、output.log、messages.jsonl。
  3. `tail -f ~/.aether/tasks/{tid}/output.log` 在另一个终端 → 看到 child 流式输出。
  4. 等子完成；`cat ~/.aether/tasks/{tid}/result.json` 显示 summary。
- **性能**：`run_task_async` 调用 → 控制流返回耗时 < 50ms（即便 child 真要跑几分钟）。

## 不在本 PR / Deferred

- **`TaskOutput` 工具实现** —— 在 PR 10.6。
- **`SendMessage` 工具实现 + 通知冒泡** —— 在 PR 10.7。
- **Worktree 隔离** —— 在 PR 10.8（本 PR 仅写 isolation 字段到 record，不真的创建 worktree）。
- **TUI 端进度可视化** —— wire 事件本 sprint 定义，TUI 渲染下个 sprint。
