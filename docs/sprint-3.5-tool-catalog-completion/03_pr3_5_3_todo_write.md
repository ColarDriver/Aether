# PR 3.5.3 — `TodoWriteTool`：闭环 PR 3.2 cheap_tool 路径

> **角色**：补齐 PR 3.2 默认配置里引用却不存在的 `todo_write` / `update_todo` 工具，
> 让 `IterationBudget.refund()` 路径在默认配置下真正可用。

## 一、目标

1. 实现 `TodoWriteTool` — 让模型管理一份 session 级别的 todo list。
2. 状态绑定 session_id，turn 间持久（不是单 turn 内有效）。
3. 输出**短 ack message**（"Todos updated, N items"），不浪费上下文。
4. 注册名为 `todo_write`，自动落入 PR 3.2 的 `cheap_tool_names` 默认名单 → cheap-tool refund 立即生效。

## 二、为什么要做

### 2.1 PR 3.2 的 dangling 引用

```python
# config/schema.py - PR 3.2 落地的默认值
cheap_tool_names: tuple[str, ...] = (
    "update_todo",
    "todo_write",       # ← 不存在
    ...
)
```

cheap-tool refund 机制：如果一整轮 LLM 调用**只**用了 cheap tool（不消耗实际进展），
不计入 IterationBudget。`todo_write` 完美符合定义（更新 todo 不算"做事"），
但工具不存在，机制无法触发。

### 2.2 长任务推进的工程价值

claude-code 的实测：1000 token 复杂任务平均产生 4-7 次 `TodoWrite` 调用。模型用它：
* turn 1: 规划任务，写 5 项 todo（all `pending`）
* turn 2-5: 完成各项，逐个标 `completed`
* turn 6: `[]`（all done）触发 cleanup

没有这个机制，模型容易：
* 忘记任务（"我刚才做完了什么？"）
* 跳步骤（user 让做 5 件事，模型只做 3 件就 finalize）
* 不必要地重复说"接下来我要做..."（占 token）

## 三、设计

### 3.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": "The complete updated todo list (replaces the old list).",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable identifier."},
                    "content": {"type": "string", "description": "What this task involves."},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "cancelled"],
                    },
                },
                "required": ["id", "content", "status"],
            },
        },
    },
    "required": ["todos"],
}
```

**完整替换**语义（claude-code 同款）：每次调用传**整个**列表，不是 patch / append。
模型省心，工具实现简单。

### 3.2 状态存储

存放在 session 级 in-memory map：

```python
# tools/builtins/todo_write.py 内
_TODO_STORE: dict[str, list[dict]] = {}  # session_id → todos


def get_session_todos(session_id: str) -> list[dict]:
    return list(_TODO_STORE.get(session_id, []))


def set_session_todos(session_id: str, todos: list[dict]) -> None:
    _TODO_STORE[session_id] = todos
```

**为什么是模块级 dict**：
* turn 间存活（不能放 `TurnContext`，每 turn 重建）
* 不能放 `EngineConfig`（应该是数据，不是配置）
* 我们当前没有 session-store 抽象。模块级 dict 是 v1 的最小可工作方案。
* v2 可以替换成持久化（写盘 / SessionStore）

**清理策略**（v1 不做）：进程长期运行时 dict 会增长。
v2 加 weak-ref / TTL，或集成进未来的 SessionStore。

### 3.3 工具实现

```python
class TodoWriteTool(ToolExecutor):
    NAME = "todo_write"

    def execute(self, call, context):
        todos = call.arguments.get("todos", [])
        if not isinstance(todos, list):
            return ToolResult(content="todos must be a list", is_error=True)

        # 校验 schema
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                return ToolResult(
                    content=f"todos[{i}] must be an object",
                    is_error=True,
                )
            for required in ("id", "content", "status"):
                if required not in item:
                    return ToolResult(
                        content=f"todos[{i}] missing required field: {required}",
                        is_error=True,
                    )
            if item["status"] not in {"pending", "in_progress", "completed", "cancelled"}:
                return ToolResult(
                    content=f"todos[{i}].status invalid: {item['status']}",
                    is_error=True,
                )

        # all-done 触发 cleanup
        all_done = todos and all(t["status"] in {"completed", "cancelled"} for t in todos)
        new_todos = [] if all_done else todos
        set_session_todos(context.session_id, new_todos)

        # ack message — 短，因为模型已经知道刚发的内容
        pending = sum(1 for t in todos if t["status"] == "pending")
        in_progress = sum(1 for t in todos if t["status"] == "in_progress")
        completed = sum(1 for t in todos if t["status"] == "completed")
        msg = (
            f"todos updated: {len(todos)} total "
            f"({pending} pending, {in_progress} in progress, {completed} completed)"
        )
        if all_done:
            msg += "\n[cleared — all tasks complete]"
        return ToolResult(call_id=call.id, content=msg, is_error=False)
```

### 3.4 与 PR 3.2 cheap_tool 的接合

`config/schema.py` 已经把 `"todo_write"` 写入 `cheap_tool_names` 默认值。
本 PR 落地后，**无需配置改动**，cheap-tool refund 自动生效。

验证方式：
1. 跑一轮模型只调用 `todo_write` 的 turn
2. `result.metadata["iteration_budget"]["used"]` 不增加
3. `result.metadata["iteration_budget"]["refund_count"] += 1`

### 3.5 spill 决策

**不 spill**。输出是几十字符的 ack message。

### 3.6 CLI 渲染（可选 v2 增强）

PR 3.5.3 v1：仅工具能力，CLI 不显示 todo list。
v2：CLI 可订阅 `_TODO_STORE` 在 footer 显示当前 todos，类似 claude-code 的 `[ ] / [x]` UI。
**v2 不在本 PR 范围**，记入 follow-up。

## 四、文件改动清单

| 文件 | 类型 | 内容 | 行数 |
|---|---|---|---|
| `backend/harness/aether/tools/builtins/todo_write.py` | **新文件** | 工具 + 模块级 store | ~120 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | 注册 `TodoWriteTool` | ~3 |
| `backend/harness/aether/tests/test_todo_write_tool.py` | **新文件** | 见 § 五 | ~250 |
| `backend/harness/aether/tests/test_cheap_tool_refund.py` | 修改 | 加 1 个 case 验证 `todo_write` 也走 refund | ~30 |

## 五、测试用例

### 5.1 测试组 A：状态管理

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 写 3 项 todo | `get_session_todos` 返回这 3 项 |
| **T-A2** | 同一 session 第二次写 | 第二次内容**完全替换**第一次 |
| **T-A3** | 不同 session_id 互不影响 | session A 的 todos 不出现在 session B |
| **T-A4** | 空数组 | store 设为 `[]`；ack 含 "0 total" |
| **T-A5** | 全部 status=completed | store 自动清空（all-done cleanup） |
| **T-A6** | 全部 status=completed/cancelled 混合 | 同样触发 cleanup |
| **T-A7** | 部分 cancelled，其他 pending | 不 cleanup |

### 5.2 测试组 B：schema 校验

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | `todos` 不是 list | `is_error=True` |
| **T-B2** | item 缺 `id` | `is_error=True`；提示具体 index |
| **T-B3** | item.status 无效值（如 "wip"） | `is_error=True`；提示有效值 |
| **T-B4** | 缺 `todos` 字段 | 默认 `[]`，不报错（视作 cleanup） |

### 5.3 测试组 C：与 cheap_tool refund 集成

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1**（在 `test_cheap_tool_refund.py`） | 一轮 turn 只调 `todo_write` | `iteration_budget.used` 不增；`refund_count += 1` |
| **T-C2** | 同 turn 调 `todo_write` + `read_file` | 不 refund（混合调用） |

### 5.4 测试组 D：ack message

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | 5 个 pending | ack 含 "5 pending, 0 in progress, 0 completed" |
| **T-D2** | 3 in progress + 2 completed | ack 计数正确 |
| **T-D3** | all-done | ack 含 "[cleared — all tasks complete]" |

## 六、验收门

* [ ] 16+ 个测试全绿
* [ ] 真实跑：模型用 `todo_write` 多轮维护任务列表，cheap-tool refund 触发
* [ ] `result.metadata["iteration_budget"]` 显示 `used` 增长 < `consume_count`（refund 生效）

## 七、回滚开关

工具新增；不注册即回滚。

## 八、实施顺序（建议 1 天）

| 步骤 | 时长 |
|---|---|
| 1. `todo_write.py` 实现 | 2h |
| 2. 注册 + 测试 | 3h |
| 3. cheap_tool 集成测试 | 1h |
| 4. 真实模型 smoke | 1h |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| 模块级 dict 在长进程中增长 | v1 接受；v2 加 TTL / SessionStore |
| 多线程并发写不安全 | Aether 当前 single-threaded REPL；v2 如多线程加 lock |
| 模型滥用（每 turn 都重写整个列表，没意义） | description 引导使用场景；不是工具能强制的 |

## 十、与后续 PR 的接合

* **CLI 渲染（v2 / Sprint 4）**：footer 显示当前 todos
* **PR 3.5.6 AgentTool**：Subagent 也可调用 `todo_write`，store key 用 `subagent_id` 而非 `session_id`，互不污染
