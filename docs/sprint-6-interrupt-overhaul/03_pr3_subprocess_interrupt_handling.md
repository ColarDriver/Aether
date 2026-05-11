# PR 6.3 — InterruptSignal 基础设施 + 事件驱动 subprocess 中断

> **架构重要 PR**：本 PR 是 Sprint 6 后半段所有 PR（6.4 / 6.5 / 6.6 / 6.7）的基础。
> 引入 `InterruptSignal` 事件驱动原语，把 polling-only 模型升级成
> "polling + event listener" 的 hybrid 模型。

## 目标

两件事：

1. **Phase A — 基础设施**：建立 `runtime/interrupt_signal.py`，提供
   `AbortController` / `AbortSignal` 在 Python 上的最小等价物。`InterruptController`
   内部改成 `InterruptSignal` 仓库，旧 API 通过 wrapper 保持兼容。
2. **Phase B — 落地 shell tool**：让 shell tool 在用户按打断键后**立即终止子进程**
   （走 `InterruptSignal.add_listener` 事件回调 + `os.killpg` 杀整个进程组）。
   不再写 polling 循环。

为什么把两件事合并到同一个 PR：Phase A 单独 merge 没有 user-visible 行为变化，
review 难评估；Phase B 是 Phase A 的最直观应用，合并能让 reviewer 一次性看清
"事件驱动到底怎么用"。整个 PR 的 diff 控制在 ~600 行，比原计划（纯 polling）
多 ~200 行但换来后续 PR 都能省事。

## 当前问题（PR 6.1 / 6.2 之后剩下的）

`tools/builtins/shell.py` 用的是 `subprocess.run(command, shell=True, ..., timeout=timeout, check=False)`：

- `subprocess.run` 是阻塞调用，子进程不结束、不超时就一直 hang。
- 调用线程被卡，`InterruptController` 即使在 main thread 设了 flag，shell tool 看不到。
- 等 `timeout` 触发 `TimeoutExpired` 后才能退出 —— 默认 60s，最长 600s。
- 用户按 ESC → 主线程 `_on_interrupt` 调 `engine.interrupt()` → flag 设上 → 但 shell tool worker thread 完全不感知。

参考 `open-claude-code/src/utils/Shell.ts:265`：

```typescript
abortSignal.addEventListener('abort', this.#boundAbortHandler, { once: true })
// abortHandler → treeKill(pid, 'SIGKILL')
```

`spawn` 之后**挂监听器**，零 polling，事件触发立即 kill。这是我们要复制的模式。

## Phase A — `InterruptSignal` 设计

### 目标 API

```python
# runtime/interrupt_signal.py
class InterruptSignal:
    """Python 上 AbortSignal 的最小等价物。

    线程安全的事件信号 + listener 列表 + 可选的父子继承。

    用法分三种：
      1. polling: signal.is_aborted() —— stream callback per-chunk
      2. event-driven: signal.add_listener(cb) —— subprocess / httpx 阻塞 IO
      3. interruptible wait: signal.wait(timeout) —— recovery backoff
    """

    def __init__(
        self,
        parent: "InterruptSignal | None" = None,
        *,
        name: str = "",
    ) -> None: ...

    # ---- core ----
    def abort(self, reason: str | None = "user-interrupt") -> None:
        """Idempotent. 触发 listeners + 标记 aborted。"""

    def is_aborted(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...

    # ---- event ----
    def add_listener(
        self,
        callback: Callable[[str | None], None],
        *,
        once: bool = True,
    ) -> Callable[[], None]:
        """注册回调，返回 unregister 函数。

        once=True（默认）= fire 后自动 remove —— 跟 claude-code 行为一致。

        如果 signal 已 aborted 了，callback 立即在当前线程同步调用一次。
        """

    def remove_listener(self, callback: Callable[[str | None], None]) -> None: ...

    # ---- blocking ----
    def wait(self, timeout: float | None = None) -> bool:
        """阻塞直到 aborted 或 timeout。返回 True = aborted。

        给 recovery wait / sleep 等场景用：可被打断的 sleep。
        """

    # ---- lifecycle ----
    def reset(self) -> None:
        """清掉 aborted 状态，可以再次 abort。"""
```

### 父子继承（claude-code 同款思想）

```python
# 实现要点
def __init__(self, parent=None, *, name=""):
    self._event = threading.Event()
    self._reason: str | None = None
    self._listeners: list[Callable] = []
    self._lock = threading.RLock()
    self._parent = parent
    self._name = name
    if parent is not None:
        # 父 abort 时自动传给 child。
        # 注意：用 weak ref 避免 child 被 GC 后还 leak listener slot。
        weak_self = weakref.ref(self)
        def _on_parent_abort(reason):
            child = weak_self()
            if child is not None:
                child.abort(reason)
        parent.add_listener(_on_parent_abort, once=True)
```

为什么用 `weakref.ref`：如果 child 被丢弃但父还活着，没 weakref 的话父的
listener 列表会留住 child 不让 GC 回收。这是 claude-code `WeakRef` 包装
的 Python 等价。

### Listener 执行规则（重要）

- `abort()` 在调用方线程**同步 fire** 所有 listeners。
- Listeners **必须 non-blocking**（设标志、close socket、send signal、kill PID）。
- 如果 listener 抛异常，**捕获并 log**，不影响后续 listener。
- `once=True` 时 fire 后从列表移除（默认行为）。

```python
def abort(self, reason="user-interrupt"):
    with self._lock:
        if self._event.is_set():
            return  # idempotent
        self._reason = reason
        self._event.set()
        listeners_to_fire = list(self._listeners)
        # `once=True` listeners 都 fire 一次就移除
        self._listeners = [
            (cb, once) for cb, once in self._listeners if not once
        ]
    # 锁外 fire — 避免 listener 里再调 abort 死锁
    for cb, _once in listeners_to_fire:
        try:
            cb(reason)
        except Exception:
            logger.exception("InterruptSignal listener failed: %s", cb)
```

### `InterruptController` 改造（向后兼容）

```python
# runtime/interrupts.py 改后
class InterruptController:
    """Session-scoped interrupt signals.

    Sprint 6 / PR 6.3 之后内部改成 dict[session_id, InterruptSignal]。
    request / is_interrupted / clear 三个旧 API 保持原语义（PR 6.1 / 6.2
    调用点不变），新增 signal(session_id) 给后续 PR 用。
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._signals: dict[str, InterruptSignal] = {}

    def request(self, session_id: str, reason: str | None = None) -> None:
        self.signal(session_id).abort(reason or "user-interrupt")

    def is_interrupted(self, session_id: str) -> bool:
        with self._lock:
            sig = self._signals.get(session_id)
        return sig is not None and sig.is_aborted()

    def reason(self, session_id: str) -> str | None:
        with self._lock:
            sig = self._signals.get(session_id)
        return sig.reason if sig else None

    def clear(self, session_id: str) -> None:
        """重置 signal，下一轮 turn 干净。"""
        with self._lock:
            sig = self._signals.get(session_id)
        if sig is not None:
            sig.reset()

    # ---- NEW: signal accessor ----
    def signal(self, session_id: str) -> InterruptSignal:
        """拿当前 session 的 InterruptSignal。不存在则创建。"""
        with self._lock:
            sig = self._signals.get(session_id)
            if sig is None:
                sig = InterruptSignal(name=f"session:{session_id}")
                self._signals[session_id] = sig
            return sig

    def discard(self, session_id: str) -> None:
        """彻底删除 signal —— REPL 退出时调，释放 listener。"""
        with self._lock:
            self._signals.pop(session_id, None)
```

### EngineRequest 携带 signal（为 PR 6.6 subagent 铺路）

```python
# runtime/contracts.py
@dataclass(slots=True)
class EngineRequest:
    ...
    # NEW: 调用方可以传入一个 InterruptSignal（child of parent's signal）。
    # 不传 → engine 从 services.interrupt_controller.signal(session_id) 拿默认。
    # 这是 PR 6.6 subagent 共享父 signal 的接入点。
    interrupt_signal: InterruptSignal | None = None
```

## Phase B — shell tool 改造

### Popen + 事件驱动 kill（参考 `open-claude-code/src/utils/Shell.ts`）

```python
# tools/builtins/shell.py
import os
import signal as _signal
import subprocess
import threading
import time

POLL_FOR_OUTPUT_INTERVAL_SEC = 0.2  # 单纯让 communicate 不要无脑阻塞
GRACE_AFTER_SIGTERM_SEC = 2.0

class ShellTool(ToolExecutor):
    interrupt_behavior = "cancel"  # PR 6.5 定义这个类属性

    def run(self, call: ToolCall, ctx: ToolDispatchContext) -> ToolResult:
        sig: InterruptSignal | None = getattr(ctx, "interrupt_signal", None)

        # 已经 aborted？立即返回 — 跟 claude-code `if (abortSignal.aborted) return createAbortedCommand()` 一致
        if sig is not None and sig.is_aborted():
            return _make_aborted_result(call, reason=sig.reason or "user-interrupt")

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if cwd else None,
            start_new_session=(os.name != "nt"),  # POSIX 进程组
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )

        # NEW: 事件驱动 kill — 主线程按 ESC → signal.abort() → 这里 listener 同步 fire。
        # listener 只做"发 SIGTERM、起 2s SIGKILL timer"两件事，不阻塞。
        unregister: Callable[[], None] | None = None
        kill_state = {"interrupted": False}

        if sig is not None:
            def _on_abort(reason):
                kill_state["interrupted"] = True
                _terminate_process_group(proc)
                # 起一个 grace timer，2s 还没死就 SIGKILL
                threading.Timer(
                    GRACE_AFTER_SIGTERM_SEC,
                    lambda: _kill_process_group(proc) if proc.poll() is None else None,
                ).start()
            unregister = sig.add_listener(_on_abort, once=True)

        try:
            # communicate 自然走完即可；不再 polling sig.is_aborted —— listener 已经接管。
            # 给 timeout 留一个上限做超时兜底（旧逻辑保留）。
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_process_group(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=GRACE_AFTER_SIGTERM_SEC)
                except subprocess.TimeoutExpired:
                    _kill_process_group(proc)
                    stdout, stderr = proc.communicate(timeout=1.0)
                raise  # 让上层 timeout 路径处理
        finally:
            if unregister is not None:
                unregister()
            if proc.poll() is None:
                _kill_process_group(proc)

        if kill_state["interrupted"]:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    f"[command interrupted by user after {duration_ms}ms]\n"
                    f"--- partial stdout ---\n{stdout[:1024]}"
                ),
                is_error=True,
                metadata={
                    "exit_code": -_signal.SIGTERM,
                    "interrupted": True,
                    "duration_ms": duration_ms,
                    "command": command,
                },
            )

        # 正常完成路径，原逻辑不动
        return ToolResult(...)
```

### Process group helpers

```python
def _terminate_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM 整个 process group（POSIX）或进程树（Windows）。"""
    try:
        if os.name == "nt":
            proc.send_signal(_signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass  # 已退出


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL fallback。"""
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
```

### ToolDispatchContext 加 `interrupt_signal`

```python
# tools/base.py / runtime/contracts.py
@dataclass
class ToolDispatchContext:
    ...
    interrupt_signal: InterruptSignal | None = None
```

### run_loop 注入 signal

```python
# agents/core/agent.py 内部，tool dispatch 之前
ctx = ToolDispatchContext(
    ...,
    interrupt_signal=request.interrupt_signal or self.services.interrupt_controller.signal(request.session_id),
)
```

### run_loop 检查 tool 的 interrupted metadata

```python
# 在 after_tool 之后
if result.metadata.get("interrupted"):
    context.metadata["interrupt"] = context.metadata.get("interrupt") or {
        "reason": "user-interrupt",
        "partial_text": "",  # tool 路径下没有 stream partial
        "was_in_tool_call": True,  # ← 给 PR 6.4 选 marker 用
        "triggered_at": time.time(),
    }
    state_machine.transition(LoopState.INTERRUPTED)
    exit_reason = ExitReason.INTERRUPTED
    break
```

## 为什么用 listener 而不是继续 polling

| 维度 | 老方案（polling 200ms） | 新方案（event listener） |
|---|---|---|
| 响应延迟 | ≤ 200ms + SIGTERM grace | **0ms** (listener 同步触发) + SIGTERM grace |
| CPU 开销 | 每 200ms 一次 syscall + dict lookup | 0（listener 只在事件发生时 fire 一次） |
| 跟 subagent 协同 | 每个 subagent 自己 poll，不联动 | sync subagent 共享父 signal，listener 自动传播 |
| 代码量 | 每个 long-running tool 都要写 poll 循环 | tool 只调 `add_listener` 一行 |
| 跟 timeout 共存 | 自己维护 deadline | `proc.communicate(timeout=N)` 不变 |

唯一例外：stream callback **保留 polling** —— provider 在 chunk 间不回到 Python
主循环，没有 event 可挂的点（PR 6.2 已实施，不动）。

## 文件改动

```
runtime/interrupt_signal.py  [新文件]
  + class InterruptSignal: abort / is_aborted / add_listener /
    remove_listener / wait / reset
  + 父子继承（weakref）
  + once=True listener 自动移除

runtime/interrupts.py
  ~ InterruptController 内部从 dict[sid, flag] 改为 dict[sid, InterruptSignal]
  ~ request / is_interrupted / clear 改成 wrapper 调底层 signal
  + signal(session_id) -> InterruptSignal （新方法）
  + discard(session_id)（彻底删 signal）

runtime/__init__.py
  + 导出 InterruptSignal

runtime/contracts.py
  ~ EngineRequest: + interrupt_signal: InterruptSignal | None = None
  ~ ToolDispatchContext: + interrupt_signal: InterruptSignal | None = None

agents/core/agent.py
  ~ run_loop: tool dispatch 之前注入 ctx.interrupt_signal
  ~ tool dispatch loop: after_tool 之后检查 result.metadata["interrupted"]
  ~ stream callback 入口的 _is_interrupted 调用 → 内部走 signal.is_aborted()
    （wrapper 透明，调用点不动）

tools/builtins/shell.py
  ~ run(): 用 Popen + sig.add_listener 替代 subprocess.run
  + _terminate_process_group / _kill_process_group helpers
  + start_new_session=True (POSIX) / CREATE_NEW_PROCESS_GROUP (Windows)
  ~ 中断时返回带 metadata["interrupted"]=True 的 ToolResult

tests/runtime/test_interrupt_signal.py  [新文件]
  + InterruptSignal 单元测试（abort / listener / 父子 / weakref / 并发）

tests/runtime/test_interrupt_controller_compat.py  [新文件]
  + 验证旧 API（request / is_interrupted / clear）跟原 flag-dict 行为一致

tests/tools/test_shell_interrupt.py  [新文件]
  + POSIX 4 个 + run_loop 1 个集成
```

## 实现细节

### POSIX vs Windows process group

- POSIX：`os.setsid()`（通过 `start_new_session=True`）让子进程成为新 process group leader，`os.killpg(pid, sig)` 一次 kill 整组。
- Windows：`CREATE_NEW_PROCESS_GROUP` flag 让子进程能接收 `CTRL_BREAK_EVENT`。`proc.send_signal(signal.CTRL_BREAK_EVENT)` 是 Windows 上的 SIGTERM 等价。`proc.kill()` 直接走 TerminateProcess 是 SIGKILL 等价。
- 测试需要分别覆盖（Windows runner 用 skip-on-nt 标记，至少在 CI 跑 Linux 路径）。

### Listener 注册时机

务必在 `Popen` **之后**、`communicate` **之前**注册。如果先 Popen 后没注册，
ESC 来了 listener 不存在；如果先注册后 Popen，ESC 来了 listener fire 时
进程还没起，`proc.pid` 是 None。正确顺序：

```python
proc = subprocess.Popen(...)         # 1. spawn
unregister = sig.add_listener(...)    # 2. 注册
try:
    stdout, stderr = proc.communicate(timeout=...)
finally:
    unregister()                      # 3. 卸载（避免 listener 累积）
```

### `add_listener(once=True)` 的死锁防护

如果 listener 在 fire 过程中调 `signal.add_listener` / `signal.remove_listener`
会怎样？实现里把 listener 列表 snapshot 出来再锁外 fire，所以 fire 内部
再操作 listener 列表是安全的（操作的是新一轮的，不影响这一轮 fire）。

### EngineRequest.interrupt_signal 的注入路径

```
REPL 启动
  └─ services.interrupt_controller.signal(session_id) → root_signal
  └─ engine.run_turn(request)
       ├─ request.interrupt_signal 为 None → engine 自己用 root_signal
       └─ ctx.interrupt_signal = root_signal
            ├─ ShellTool 拿到 root_signal
            └─ ESC 按下 → engine.interrupt(session_id) → root_signal.abort()
                 └─ ShellTool listener 立刻 fire → killpg → 200ms 内返回
```

Subagent 路径（PR 6.6 实现）：

```
父 run_loop 在 AgentTool dispatch 时
  └─ child_signal = InterruptSignal(parent=root_signal)
  └─ sub_request = EngineRequest(..., interrupt_signal=child_signal)
  └─ sub_engine.run_turn(sub_request)
       └─ ctx.interrupt_signal = child_signal
            └─ ESC → root_signal.abort() → 自动传给 child_signal → subagent 立即停
```

## 参考实现

- `open-claude-code/src/utils/abortController.ts` 行 68-99：`createChildAbortController` 父子传播 + WeakRef。我们用 `weakref.ref`。
- `open-claude-code/src/utils/Shell.ts` 行 181 + 行 339-345：`exec(cmd, abortSignal, ...)` —— `spawn` 之后挂 listener，触发时 `treeKill(pid, 'SIGKILL')`。我们用 `os.killpg`。
- `open-claude-code/src/utils/ShellCommand.ts` 行 186-193：`#abortHandler` —— `if reason === 'interrupt' return` 这种"不同 reason 不同行为"的灵活性。我们也可以让 listener 检查 reason。
- `open-claude-code/src/tools/BashTool/BashTool.tsx` 行 283：`interrupted: z.boolean()` 字段。我们用 `metadata["interrupted"] = True`。

## 测试

### `tests/runtime/test_interrupt_signal.py`

```python
class InterruptSignalCoreTests(unittest.TestCase):
    def test_abort_sets_aborted_and_reason(self):
        sig = InterruptSignal()
        self.assertFalse(sig.is_aborted())
        sig.abort("user-interrupt")
        self.assertTrue(sig.is_aborted())
        self.assertEqual(sig.reason, "user-interrupt")

    def test_abort_is_idempotent(self):
        sig = InterruptSignal()
        sig.abort("first")
        sig.abort("second")  # no-op
        self.assertEqual(sig.reason, "first")

    def test_listener_fires_on_abort(self):
        sig = InterruptSignal()
        calls = []
        sig.add_listener(lambda r: calls.append(r))
        sig.abort("user-interrupt")
        self.assertEqual(calls, ["user-interrupt"])

    def test_listener_fires_immediately_if_already_aborted(self):
        sig = InterruptSignal()
        sig.abort("first")
        calls = []
        sig.add_listener(lambda r: calls.append(r))
        # 添加时 signal 已 aborted → 立即同步调一次
        self.assertEqual(calls, ["first"])

    def test_once_listener_fires_only_once(self):
        sig = InterruptSignal()
        calls = []
        sig.add_listener(lambda r: calls.append(r), once=True)
        sig.abort("a")
        sig.reset()
        sig.abort("b")
        self.assertEqual(calls, ["a"])  # 第二次 abort 不再 fire

    def test_unregister_function_removes_listener(self):
        sig = InterruptSignal()
        calls = []
        unregister = sig.add_listener(lambda r: calls.append(r), once=False)
        unregister()
        sig.abort("user-interrupt")
        self.assertEqual(calls, [])

    def test_listener_exception_does_not_block_others(self):
        sig = InterruptSignal()
        calls = []
        sig.add_listener(lambda r: calls.append("first"))
        sig.add_listener(lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        sig.add_listener(lambda r: calls.append("third"))
        sig.abort("user-interrupt")
        self.assertEqual(calls, ["first", "third"])

    def test_wait_returns_immediately_when_already_aborted(self):
        sig = InterruptSignal()
        sig.abort("user-interrupt")
        start = time.monotonic()
        result = sig.wait(timeout=5.0)
        self.assertLess(time.monotonic() - start, 0.05)
        self.assertTrue(result)

    def test_wait_blocks_until_abort(self):
        sig = InterruptSignal()
        result_box = []
        def waiter():
            result_box.append(sig.wait(timeout=2.0))
        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        sig.abort("user-interrupt")
        t.join(timeout=1.0)
        self.assertEqual(result_box, [True])

    def test_reset_clears_aborted(self):
        sig = InterruptSignal()
        sig.abort("first")
        sig.reset()
        self.assertFalse(sig.is_aborted())
        self.assertIsNone(sig.reason)


class InterruptSignalParentChildTests(unittest.TestCase):
    def test_parent_abort_propagates_to_child(self):
        parent = InterruptSignal()
        child = InterruptSignal(parent=parent)
        parent.abort("user-interrupt")
        self.assertTrue(child.is_aborted())
        self.assertEqual(child.reason, "user-interrupt")

    def test_child_abort_does_not_affect_parent(self):
        parent = InterruptSignal()
        child = InterruptSignal(parent=parent)
        child.abort("user-interrupt")
        self.assertFalse(parent.is_aborted())

    def test_child_attached_after_parent_aborted_inherits(self):
        parent = InterruptSignal()
        parent.abort("user-interrupt")
        child = InterruptSignal(parent=parent)
        self.assertTrue(child.is_aborted())

    def test_grandchild_inherits_through_chain(self):
        parent = InterruptSignal()
        child = InterruptSignal(parent=parent)
        grandchild = InterruptSignal(parent=child)
        parent.abort("user-interrupt")
        self.assertTrue(grandchild.is_aborted())

    def test_dropped_child_does_not_leak(self):
        # GC 之后父的 listener 列表里不能留死引用
        parent = InterruptSignal()
        child = InterruptSignal(parent=parent)
        weak_child = weakref.ref(child)
        del child
        gc.collect()
        self.assertIsNone(weak_child())
        # parent abort 不应该 fire 已死 child 的 listener
        parent.abort("user-interrupt")  # 不抛异常即过


class InterruptSignalConcurrencyTests(unittest.TestCase):
    def test_concurrent_abort_and_add_listener_safe(self):
        sig = InterruptSignal()
        listener_count = 64
        for _ in range(listener_count):
            sig.add_listener(lambda r: None, once=False)

        threads = []
        for _ in range(32):
            threads.append(threading.Thread(target=sig.abort))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 不死锁 / 不抛即过
        self.assertTrue(sig.is_aborted())
```

### `tests/runtime/test_interrupt_controller_compat.py`

```python
class InterruptControllerBackwardCompatTests(unittest.TestCase):
    def test_request_then_is_interrupted_returns_true(self):
        ic = InterruptController()
        self.assertFalse(ic.is_interrupted("s"))
        ic.request("s", "user-interrupt")
        self.assertTrue(ic.is_interrupted("s"))

    def test_reason_is_preserved(self):
        ic = InterruptController()
        ic.request("s", "user-cancel")
        self.assertEqual(ic.reason("s"), "user-cancel")

    def test_clear_resets_flag(self):
        ic = InterruptController()
        ic.request("s")
        ic.clear("s")
        self.assertFalse(ic.is_interrupted("s"))

    def test_different_sessions_isolated(self):
        ic = InterruptController()
        ic.request("a")
        self.assertFalse(ic.is_interrupted("b"))

    def test_signal_accessor_returns_same_instance(self):
        # NEW API — pin signal identity
        ic = InterruptController()
        sig1 = ic.signal("s")
        sig2 = ic.signal("s")
        self.assertIs(sig1, sig2)

    def test_request_aborts_underlying_signal(self):
        ic = InterruptController()
        sig = ic.signal("s")
        ic.request("s")
        self.assertTrue(sig.is_aborted())
```

### `tests/tools/test_shell_interrupt.py`

```python
import os
import threading
import time
import unittest

import pytest

from aether.runtime.interrupt_signal import InterruptSignal
from aether.tools.builtins.shell import ShellTool
# ... 

@pytest.mark.skipif(os.name == "nt", reason="POSIX-only process group test")
class ShellInterruptPosixTests(unittest.TestCase):
    def test_long_sleep_interrupted_within_500ms(self):
        sig = InterruptSignal()
        ctx = _make_ctx(interrupt_signal=sig)
        call = _make_call(command="sleep 30")

        threading.Timer(0.2, lambda: sig.abort("user-interrupt")).start()

        start = time.monotonic()
        result = ShellTool().run(call, ctx)
        elapsed = time.monotonic() - start

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["interrupted"])
        self.assertLess(elapsed, 0.5)  # 200ms 触发 + listener 同步 fire + 进程 die

    def test_normal_completion_returns_no_interrupted_flag(self):
        sig = InterruptSignal()
        ctx = _make_ctx(interrupt_signal=sig)
        call = _make_call(command="echo hello")
        result = ShellTool().run(call, ctx)
        self.assertFalse(result.is_error)
        self.assertNotEqual(result.metadata.get("interrupted"), True)
        self.assertIn("hello", result.content)

    def test_partial_stdout_preserved_on_interrupt(self):
        sig = InterruptSignal()
        threading.Timer(0.5, lambda: sig.abort("user-interrupt")).start()
        ctx = _make_ctx(interrupt_signal=sig)
        call = _make_call(command="for i in 1 2 3 4 5; do echo line$i; sleep 0.2; done")
        result = ShellTool().run(call, ctx)
        self.assertTrue(result.metadata["interrupted"])
        self.assertIn("line1", result.content)
        self.assertIn("line2", result.content)

    def test_subshell_grandchildren_killed_too(self):
        sig = InterruptSignal()
        threading.Timer(0.2, lambda: sig.abort("user-interrupt")).start()
        ctx = _make_ctx(interrupt_signal=sig)
        call = _make_call(command="sleep 30 & echo $! ; wait")
        result = ShellTool().run(call, ctx)

        import re
        m = re.search(r"\d+", result.content)
        self.assertIsNotNone(m)
        gc_pid = int(m.group())
        time.sleep(0.5)
        with self.assertRaises(ProcessLookupError):
            os.kill(gc_pid, 0)

    def test_listener_unregistered_after_normal_completion(self):
        sig = InterruptSignal()
        ctx = _make_ctx(interrupt_signal=sig)
        call = _make_call(command="echo hello")
        _ = ShellTool().run(call, ctx)
        # 完成后 listener 已卸载 — 后续 abort 不该影响什么
        sig.abort("user-interrupt")
        # 不抛即过


class ShellInterruptIntegrationTests(unittest.TestCase):
    def test_run_loop_marks_interrupted_when_tool_returns_interrupted_metadata(self):
        # 1. Engine with stub provider that asks shell to run sleep 30
        # 2. Spawn engine.run_loop in worker thread
        # 3. After 300ms, engine.interrupt(...)
        # 4. result.exit_reason == ExitReason.INTERRUPTED
        # 5. result.metadata["interrupt"]["was_in_tool_call"] == True
        ...
```

## 验收门

- `tests/runtime/test_interrupt_signal.py` 全过（22+ 个测试）。
- `tests/runtime/test_interrupt_controller_compat.py` 全过 —— 老调用点不回归。
- `tests/tools/test_shell_interrupt.py` POSIX 路径全过；Windows runner 跳 grandchild 测试。
- 现有 `test_shell.py` / `test_engine.py` 等已有测试不回归。
- 手测：跑 `pytest tests/`、`npm install` 等长命令，按 ESC 立刻停（<500ms）。
- 整体打断响应延迟（按下 ESC → shell tool 返回）≤ 500ms（比原 PR 6.3 polling 方案的 700ms 更紧，因为去掉了 200ms poll 间隔）。

## 不在本 PR 内（留给后续 PR）

- subagent 共享父 signal → PR 6.6
- WebFetch / HTTP MCP 用 signal listener 关 httpx client → PR 6.7
- tool 定义 `interrupt_behavior: "cancel" | "block"` 类属性的引擎层 dispatch（"block" tool 中断时不 kill）→ PR 6.5
- `INTERRUPT_MESSAGE_FOR_TOOL_USE` 注入对话历史 → PR 6.4
