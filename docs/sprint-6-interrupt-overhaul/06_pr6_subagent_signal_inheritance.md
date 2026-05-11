# PR 6.6 — Subagent 共享父 InterruptSignal

## 目标

让父 turn 按 ESC 时，**任何正在跑的 subagent / Task tool 也立刻停下**。当前
`_interrupt_active_children` 只在父 controller 设标志，但子 engine 持有独立的
`InterruptController` —— flag 设了等于没设，子 turn 还得跑完自己的全部 iteration。

## 当前问题

```
父 engine                                子 engine（subagent）
─────────────                            ──────────────
InterruptController (parent)             InterruptController (child) ← 独立实例！
  └─ flag dict                              └─ flag dict

用户按 ESC
  └─ parent.request(parent_session)
  └─ _interrupt_active_children()
       └─ ??? — 当前实现可能调了 parent.request(child_session)
                但 child engine 的 controller 不是同一个！
                child 仍然完全感知不到。
```

实测：subagent 跑 30s 的"研究"任务，按 ESC，subagent 继续跑完才停（**完全失效**）。

## 设计

参考 `open-claude-code/src/tools/AgentTool/runAgent.ts` 行 520-528：

```typescript
const agentAbortController = override?.abortController
    ? override.abortController            // 显式覆盖优先
    : isAsync
      ? new AbortController()             // 异步 subagent: 独立
      : toolUseContext.abortController    // 同步 subagent: 共享父
```

claude-code 默认 sync subagent **直接复用父 controller**，事件自动传播。我们用
PR 6.3 引入的 `InterruptSignal(parent=...)` 做同款效果。

### 核心改动：子 engine 共用父 signal

```python
# agents/entry/subagent_base.py（或 Task tool 的入口处）
class SubagentExecutor:
    def execute(
        self,
        *,
        parent_signal: InterruptSignal,
        parent_session_id: str,
        ...,
        is_async: bool = False,
    ) -> EngineResult:
        # 1. 派生 child signal
        if is_async:
            # 异步 subagent（background Task）：完全独立
            child_signal = InterruptSignal(name=f"subagent:{child_session_id}")
        else:
            # 同步 subagent：父 abort 自动传给子
            child_signal = InterruptSignal(
                parent=parent_signal,
                name=f"subagent:{child_session_id}",
            )

        # 2. 子 engine 构造时把 child_signal 注入
        child_request = EngineRequest(
            session_id=child_session_id,
            user_message=task_description,
            interrupt_signal=child_signal,  # PR 6.3 新加的字段
            ...,
        )

        # 3. 子 engine 内部 run_loop 用 request.interrupt_signal 而不是
        #    自己 services.interrupt_controller.signal(sid)
        return child_engine.run_turn(child_request)
```

### `EngineRequest.interrupt_signal` 的优先级

PR 6.3 已在 `EngineRequest` 加了 `interrupt_signal` 字段。run_loop 取 signal 时：

```python
# agents/core/agent.py run_loop 开头
signal = (
    request.interrupt_signal                                     # subagent 路径传入
    or self.services.interrupt_controller.signal(request.session_id)  # 默认顶层 session signal
)
```

这样：
- **顶层 REPL turn**：`request.interrupt_signal=None` → 用 session-scoped signal。
- **Subagent turn**：`request.interrupt_signal=child_signal` → 用父继承来的 child。

调用方决定 signal，engine 不关心来源。

### 为什么异步 subagent 要独立

claude-code 把"用户提交新消息中途的 abort"（reason="interrupt"）跟"用户主动取消"
（reason="user-cancel"）区分得很细。异步 subagent（用户用 `Run in background`
显式启动）是一个**独立生命周期**的任务 —— 用户按 ESC 取消当前 foreground turn
不应该顺手把后台研究也干掉。

所以同步 subagent → 共享父；异步 subagent → 独立 signal（用户要单独操作的话有
`/task kill <id>` 等命令）。

### 嵌套 subagent

```
Root signal
  └─ Child A signal
       └─ Grandchild signal  ← 嵌套也对，PR 6.3 的 InterruptSignal 支持任意深度
```

PR 6.3 的 `InterruptSignal(parent=...)` listener 链就是为了这个 —— 顶层 abort
经过 A → Grandchild 一路同步传到。延迟仍然是 0（listener 在 abort 调用方线程
同步 fire）。

### 老 `_interrupt_active_children` 的处理

现有 `AgentEngine._interrupt_active_children` 是为了"父 engine 知道哪些子在跑"。
PR 6.6 后这个方法**变成 no-op**（或彻底删除）—— 父 signal abort 时 child signal
通过 PR 6.3 的 listener 链自动 fire，不需要父显式遍历。

如果还有 telemetry / log 需要"枚举 active children"，可以保留 `_active_children`
集合但不在 interrupt 路径上调它，只用作 debug 输出。

## 文件改动

```
agents/entry/subagent_base.py
  ~ 创建子 EngineRequest 时根据 is_async 派生 child_signal 并塞进 request

tools/builtins/agent_tool.py（或 Task tool 入口）
  ~ 调用 SubagentExecutor 时传 parent_signal=ctx.interrupt_signal

agents/core/agent.py
  ~ run_loop 开头：signal = request.interrupt_signal or controller.signal(session_id)
  ~ _interrupt_active_children: 改为 no-op 或删除（保留 telemetry hook）
  ~ tool dispatch 注入 ctx.interrupt_signal = 上面那个 signal（PR 6.3 已做）

runtime/contracts.py
  ~ EngineRequest.interrupt_signal 字段 docstring 补全（subagent 用法）

tests/runtime/test_subagent_interrupt.py  [新文件]
```

## 实现细节

### Task tool 是 sync 还是 async 的判定

```python
# 视项目现状，可能在 Task tool 的 input schema 有 background: bool 字段
def execute(self, call, ctx):
    is_async = bool(call.arguments.get("background", False))
    return SubagentExecutor().execute(
        parent_signal=ctx.interrupt_signal,
        is_async=is_async,
        ...,
    )
```

如果 Task tool 当前没有这个字段，先全部按 sync 处理（共享父 signal），等
async 路径成熟再加。

### Subagent 的 session_id 跟 controller 的关系

```
controller.signal("root-session") → root_signal
controller.signal("subagent-abc") → 全新独立 signal （不跟 root 联动！）
```

注意：**不要**通过 `controller.signal(child_session_id)` 来拿子的 signal —— 那
会得到一个跟 root 不联动的全新 signal。子的 signal 必须用
`InterruptSignal(parent=root_signal)` 显式派生。

或者：让 controller 支持"派生 child signal"的工厂：

```python
class InterruptController:
    def fork_signal(self, parent_session_id: str, child_session_id: str) -> InterruptSignal:
        """为 child_session_id 派生一个继承 parent_session_id 的 signal。"""
        parent = self.signal(parent_session_id)
        child = InterruptSignal(parent=parent, name=f"session:{child_session_id}")
        with self._lock:
            # 注意：child 不进 self._signals 字典 — 它的生命周期跟着 subagent
            # turn 走，不是 session-scoped。
            pass
        return child
```

实现可以更简单：`InterruptSignal(parent=...)` 直接给 caller 用，不进 controller dict。

## 测试

`tests/runtime/test_subagent_interrupt.py`

```python
class SubagentInterruptInheritanceTests(unittest.TestCase):
    def test_sync_subagent_inherits_parent_signal(self):
        parent_sig = InterruptSignal(name="parent")
        child_sig = InterruptSignal(parent=parent_sig, name="child")

        # Subagent 跑一个 100ms 的"任务" — 模拟流式
        # 主线程 50ms 后 abort 父
        ...
        self.assertTrue(child_sig.is_aborted())
        self.assertEqual(child_sig.reason, "user-interrupt")

    def test_async_subagent_isolated_from_parent_interrupt(self):
        parent_sig = InterruptSignal(name="parent")
        async_child_sig = InterruptSignal(name="async-child")  # no parent

        parent_sig.abort("user-interrupt")
        self.assertFalse(async_child_sig.is_aborted())

    def test_nested_subagents_all_inherit_root_abort(self):
        root = InterruptSignal(name="root")
        mid = InterruptSignal(parent=root)
        leaf = InterruptSignal(parent=mid)

        root.abort("user-interrupt")

        self.assertTrue(mid.is_aborted())
        self.assertTrue(leaf.is_aborted())

    def test_subagent_signal_dropped_does_not_leak_in_parent(self):
        root = InterruptSignal(name="root")
        child = InterruptSignal(parent=root)
        weak = weakref.ref(child)
        del child
        gc.collect()
        # parent 的 listener 列表里不能留 dead listener
        self.assertIsNone(weak())
        root.abort("user-interrupt")  # 不抛即过


class SubagentInterruptEndToEndTests(unittest.TestCase):
    def test_parent_interrupt_stops_subagent_stream_within_500ms(self):
        # 1. Engine + AgentTool 配置 subagent
        # 2. subagent 跑一个会持续 5s 流式输出的 stub provider
        # 3. parent 100ms 后 engine.interrupt(parent_session)
        # 4. assert 父 + 子 EngineResult 都 INTERRUPTED
        # 5. assert 总耗时 < 500ms
        ...

    def test_parent_interrupt_kills_subagent_shell_subprocess(self):
        # 1. subagent 内跑 sleep 30
        # 2. parent 200ms 后 abort
        # 3. assert subprocess 实际被 SIGTERM/SIGKILL
        # 4. assert 总耗时 < 700ms
        ...

    def test_subagent_interrupted_metadata_reaches_parent_result(self):
        # parent 的 EngineResult.metadata 应该能看到"子任务因父中断被取消"
        # 具体 shape 待定 — 至少父 result.metadata["interrupt"]["had_active_subagents"] 之类
        ...

    def test_async_subagent_keeps_running_when_parent_interrupted(self):
        # 1. 启动一个 async (background=True) subagent
        # 2. parent 立即 interrupt
        # 3. async subagent 应该继续跑（独立生命周期）
        ...
```

## 验收门

- `test_subagent_interrupt.py` 全过（10+ 个测试）。
- 手测：用 Task tool 跑长任务（"读 30 个文件并总结"），按 ESC，subagent < 500ms 停。
- 手测：用 `Run in background` 启动 async subagent，按 ESC 当前 turn，确认 async subagent 继续跑。
- 现有 subagent / Task tool 测试不回归。

## 不在本 PR 内（后续 polish）

- async subagent 的独立 cancel UI（让用户能单独 kill 一个 background subagent）—— 留给 sprint 7 跟 Task tool UI 改造一起。
- 多层 subagent 的 cancel chain 在 UI 上的可视化（比如显示"取消了 3 个子任务"）—— UI polish。
- subagent 内部对父 abort 原因的解释性 marker（"我是因为父被打断而中止"）—— 可选。
