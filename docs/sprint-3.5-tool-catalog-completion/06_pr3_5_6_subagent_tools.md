# PR 3.5.6 — `AgentTool` + `TaskOutputTool` + `TaskStopTool`

> **角色**：把 Aether 已有的 `SubagentManager` **暴露给 LLM**。
> 模型可以在主线程做规划，把"读 50 个文件"这样的 I/O-heavy 任务派发给 subagent，
> 主线程上下文不污染。

## 一、目标

1. `AgentTool` — 派发一个 subagent 任务（同步，等结果返回）。
2. `TaskOutputTool` — 取一个**正在运行**的 subagent 的实时输出（v1 留 hook，按现状 SubagentManager 是 sync 的，这个工具主要用于 v2）。
3. `TaskStopTool` — 中止运行中的 subagent。
4. 与现有 `SubagentManager.run_task` / `run_tasks` 集成；**不重新实现派发**。

## 二、为什么要做

### 2.1 现状

Aether 已经实现了 SubagentManager：

```python
# subagents/manager.py - 已存在
class SubagentManager:
    def run_task(self, parent, task: SubagentTask) -> SubagentResult:
        ...
    def run_tasks(self, *, parent, tasks: List[SubagentTask], ...) -> List[SubagentResult]:
        ...
```

但**没有工具入口**。LLM 看不到，无法在 turn 中触发 subagent 派发。
这是当前最大的 capability gap：基础设施齐了，能力没暴露。

### 2.2 典型场景

* 用户：「review 这 30 个 Python 文件，找出所有 `print` 调用」
* 主 agent 不应该在自己上下文里读 30 个文件（污染）
* **应该**派 30 个 subagent 各读一个，每个返回简短结果，主 agent 汇总

claude-code 实测：subagent fan-out 是处理大型 codebase 任务的核心机制。

## 三、设计

### 3.1 `AgentTool`

#### 3.1.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "subagent_type": {
            "type": "string",
            "description": (
                "Which subagent persona to use. Defaults to 'general-purpose'. "
                "Other types depend on registered builders."
            ),
            "default": "general-purpose",
        },
        "prompt": {
            "type": "string",
            "description": (
                "The task description for the subagent. Be specific and "
                "self-contained — the subagent doesn't see the parent's history."
            ),
        },
        "expected_output": {
            "type": "string",
            "description": (
                "Brief description of what the subagent should return — "
                "e.g. 'a list of files matching pattern X' or 'a JSON object "
                "with fields foo, bar, baz'."
            ),
        },
    },
    "required": ["prompt"],
}
```

#### 3.1.2 算法（v1：同步派发）

```python
class AgentTool(ToolExecutor):
    NAME = "task"  # claude-code 同名
    MAX_RESULT_CHARS = 60_000

    def __init__(self, subagent_manager):
        self.subagent_manager = subagent_manager

    def execute(self, call, context):
        prompt = call.arguments["prompt"]
        subagent_type = call.arguments.get("subagent_type", "general-purpose")
        expected_output = call.arguments.get("expected_output")

        # 取 parent 引用（agent.py 在 _prepare_turn_entry 注入）
        parent_agent = context.metadata.get("_parent_agent")
        if parent_agent is None:
            return ToolResult(
                content="AgentTool not wired: _parent_agent missing from context",
                is_error=True,
            )

        task = self._build_task(
            prompt=prompt,
            subagent_type=subagent_type,
            expected_output=expected_output,
            session_id=context.session_id,
        )

        try:
            result = self.subagent_manager.run_task(parent=parent_agent, task=task)
        except RuntimeError as exc:  # depth limit
            return ToolResult(content=f"subagent dispatch failed: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"subagent crashed: {exc}", is_error=True)

        body = self._format_result(result, expected_output=expected_output)
        content = self._maybe_spill(body, call=call, context=context, extension="md")
        return ToolResult(call_id=call.id, content=content, is_error=False)

    def _format_result(self, result, *, expected_output):
        lines = [
            f"# Subagent task complete",
            f"- task_id: {result.task_id}",
            f"- subagent_id: {result.metadata.get('subagent_id')}",
            f"- status: {result.status}",
            f"- duration: {result.duration_seconds:.1f}s",
        ]
        if expected_output:
            lines.append(f"- expected: {expected_output}")
        if result.error:
            lines.append(f"\n## Error\n{result.error}")
        if result.summary:
            lines.append(f"\n## Summary\n{result.summary}")
        return "\n".join(lines)
```

#### 3.1.3 同步 vs 异步

claude-code 的 `AgentTool` 是异步的（subagent 在后台跑，主 agent 继续 turn，用 `TaskOutputTool` 拉结果）。
**v1 我们只做同步**：
* SubagentManager 已经是 sync API（等子 agent 跑完返回）
* 改成 async 需要事件循环 + state machine，是大工程
* 同步版已经覆盖 90% 用例

异步版留给 v2（可能放 Sprint 4 做）。

### 3.2 `TaskOutputTool`

#### 3.2.1 v1 行为

由于 v1 是同步派发，`TaskOutput` 在 sync 模式下**不可能拿到正在运行的输出**（subagent 跑完才返回）。

v1 实现：返回明确的 "not supported in sync mode" 错误，**给将来 async 版留入口**。

```python
class TaskOutputTool(ToolExecutor):
    NAME = "task_output"

    def execute(self, call, context):
        return ToolResult(
            content=(
                "TaskOutput is not supported in synchronous subagent mode (v1). "
                "All subagent tasks complete before returning to the parent. "
                "Read the task result directly from the AgentTool output."
            ),
            is_error=True,
        )
```

为什么还实现：claude-code prompt 里会引导模型用 TaskOutput，工具不存在会显示 "tool not found"，
而**显式说明"不可用"**比"不存在"更友好；后期升级到 async 时模型 prompt 不变。

#### 3.2.2 v2 升级方向

Async 版：
* `task_output(task_id, since_offset=0)` 返回 task 的 stdout/stderr/log 自 offset 起的内容
* SubagentManager 需要返回 future-like 句柄；状态写到 `~/.aether/subagents/<task_id>/output.log`
* PR 3.5.6 不实现，记入 follow-up

### 3.3 `TaskStopTool`

#### 3.3.1 v1 行为

同步模式下，task 在 tool execute 期间已经跑完，stop 永远来不及。

但**一种例外**：subagent 自己也可能 fan-out 派发更深的任务；上层模型决定中止某个 task_id 时，
应该能中断（哪怕 v1 是 best-effort）。

v1 实现：
* SubagentManager 维护一份 `running_tasks: dict[task_id, threading.Event]`
* `TaskStop(task_id)` 把对应 event set，subagent 的 run_loop 在 iteration 边界检查 event，命中则 raise InterruptException
* 如果 task_id 已经结束，返回 "task already completed" 错误

```python
class TaskStopTool(ToolExecutor):
    NAME = "task_stop"

    def execute(self, call, context):
        task_id = call.arguments["task_id"]
        manager = context.metadata.get("_subagent_manager")
        if manager is None or not hasattr(manager, "stop_task"):
            return ToolResult(
                content="TaskStop not available (SubagentManager.stop_task missing)",
                is_error=True,
            )
        ok = manager.stop_task(task_id)
        if ok:
            return ToolResult(content=f"stop signal sent to task {task_id}")
        return ToolResult(
            content=f"task {task_id} not found or already complete",
            is_error=True,
        )
```

需要在 SubagentManager 加 `stop_task(task_id) -> bool` 方法（**本 PR 范围内**）。

### 3.4 SubagentManager 的小升级

```python
# subagents/manager.py — Sprint 3.5 / PR 3.5.6 增量
class SubagentManager:
    def __init__(...):
        ...
        self._stop_events: dict[str, threading.Event] = {}

    def _execute_one(self, ...):
        stop_event = threading.Event()
        self._stop_events[task.task_id] = stop_event
        try:
            child._stop_event = stop_event  # child agent 在 run_loop 里检查
            ...
        finally:
            self._stop_events.pop(task.task_id, None)

    def stop_task(self, task_id: str) -> bool:
        event = self._stop_events.get(task_id)
        if event is None:
            return False
        event.set()
        return True
```

`agent.py::run_loop` 在 iteration 边界增加：

```python
stop_event = getattr(self, "_stop_event", None)
if stop_event is not None and stop_event.is_set():
    raise InterruptException("subagent stopped by parent")
```

### 3.5 注入引用到 context.metadata

`_prepare_turn_entry` 加：

```python
context.metadata["_parent_agent"] = self
context.metadata["_subagent_manager"] = self.services.subagent_manager
```

加 `_METADATA_INTERNAL_KEYS`：

```python
_METADATA_INTERNAL_KEYS = frozenset({
    ...,
    "_parent_agent",
    "_subagent_manager",
})
```

## 四、文件改动清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `backend/harness/aether/tools/builtins/agent_tool.py` | **新文件** | ~200 |
| `backend/harness/aether/tools/builtins/task_output.py` | **新文件** | ~50 |
| `backend/harness/aether/tools/builtins/task_stop.py` | **新文件** | ~80 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | ~10 |
| `backend/harness/aether/subagents/manager.py` | 修改 | 加 `stop_task` + `_stop_events` | ~30 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 注入 `_parent_agent` / `_subagent_manager`；run_loop 检查 stop_event | ~15 |
| `backend/harness/aether/tests/test_agent_tool.py` | **新文件** | ~250 |
| `backend/harness/aether/tests/test_task_stop.py` | **新文件** | ~150 |

## 五、测试用例

### 5.1 测试组 A：AgentTool 派发

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 派一个 task；mock SubagentManager 返回 success | content 含 "status: COMPLETED" + summary |
| **T-A2** | SubagentManager.run_task 抛 RuntimeError（depth limit） | `is_error=True`；提示 "depth limit" |
| **T-A3** | SubagentManager 抛通用 Exception | `is_error=True` |
| **T-A4** | `_parent_agent` 缺失（mock context） | `is_error=True`；提示 "not wired" |
| **T-A5** | summary > 60k | spill 触发 |
| **T-A6** | expected_output 在 prompt 里 | format 输出包含该字段 |

### 5.2 测试组 B：TaskOutput v1

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 任意调用 | `is_error=True`；明确 "not supported in sync mode" |

### 5.3 测试组 C：TaskStop

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | task_id 在 _stop_events 中 | 返回 success；event 被 set |
| **T-C2** | task_id 不存在 | `is_error=True` |
| **T-C3** | manager 没有 stop_task 方法 | `is_error=True`；提示 "not available" |

### 5.4 测试组 D：SubagentManager.stop_task 集成

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | 起一个长跑 subagent，主线程 stop_task | subagent 在下个 iteration 抛 InterruptException |
| **T-D2** | stop_task 后 task 仍能从 _stop_events 清理 | 不泄漏 |

## 六、验收门

* [ ] 15+ case 全绿
* [ ] 真实跑：模型用 `task` 工具派发"读这个目录所有 Python 文件并报告 print 数"，得到正确摘要
* [ ] 派发 3 个 task 并发跑（依赖 max_concurrent_children）

## 七、回滚开关

* `EngineConfig.allow_subagent_dispatch=False`（新加） → 工具返回 "subagent dispatch disabled"
* 完全 revert：删工具 + manager 改动

## 八、实施顺序（建议 2.5 天）

| 步骤 | 时长 |
|---|---|
| 1. agent_tool.py + 测试 | 4h |
| 2. task_stop + manager 改动 | 3h |
| 3. task_output 简单实现 | 30min |
| 4. agent.py 注入 + stop_event 检查 | 1h |
| 5. 测试 | 4h |
| 6. 真实 subagent 派发 smoke | 2h |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| 模型滥用导致 subagent 雪崩 | SubagentManager 已有 max_concurrent_children=3 / max_spawn_depth=2 限制 |
| stop_event 竞态（subagent 已结束才被 stop） | `_stop_events.pop` 在 finally；stop_task 找不到返回 False |
| 异步版需求紧迫 | v1 同步够用；v2 提前的话需大改 |
| AgentTool 不在 cheap_tool_names | 故意；subagent 派发是真实"做事"，应消耗 budget |

## 十、与后续 PR 的接合

* **后续异步化**（v2 / Sprint 5）：TaskOutput 真正实现；AgentTool 改 `dispatch_async`
* **PR 3.5.7 AskUserQuestion**：subagent 内部不能问用户（应转给 parent），由 agent.py 检查 `delegate_depth > 0` 拦截
