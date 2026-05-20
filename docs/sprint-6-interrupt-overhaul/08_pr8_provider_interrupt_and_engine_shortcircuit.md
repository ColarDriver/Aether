# PR 6.8 — Provider HTTP 中断 + 引擎短路 + UI 去重

> **实施状态**：已完成，分支 `fix/interrupt-responsiveness`。
> 本 PR 解决的是"按 Ctrl+C 后 2-3 秒才真正停下来"的根本原因。

## 背景

PR 6.3 / 6.7 建立了 `InterruptSignal` 事件驱动基础设施和 tool 级的中断传播，
但有三个关键环节被遗漏，导致中断响应仍然不可靠：

1. **LLM provider 流式请求没有中断监听** — `openai_compatible.py` / `codex.py` /
   `claude.py` 的 streaming HTTP 连接没有挂 `InterruptSignal` 的 listener。
   当模型在 "thinking"（time-to-first-token 延迟）或 token 产出缓慢时，
   stream callback 的 per-delta polling 无法触发（因为没有 delta 到达），
   中断信号被阻塞在 HTTP 读等待中。

2. **引擎错误处理不检查中断状态** — 当 listener 关闭 HTTP 连接后，
   产生的 `ProviderInvocationError`（或原始异常）进入引擎的 retry/recovery
   路径（`agent.py` 的 `except ProviderInvocationError`），recovery strategy
   可能决定重试并带退避延迟（backoff），白白等 2+ 秒。

3. **Shell tool `_cancel` listener 阻塞** — `_cancel` 在 SIGTERM 后用
   `time.sleep(0.05)` 循环忙等 2 秒再 SIGKILL，违反 listener "non-blocking only" 的规则。

4. **`InterruptSignal` 缺少 `reset()`** — `InterruptController.clear()` 用
   `pop()` + `close()` 销毁 signal，导致 `RunHandle` 等持有旧引用的对象失效。

5. **UI 冗余** — 中断时 ActivityBar（消息区）和 Composer（输入框下方）同时显示
   "interrupting" 文字，视觉干扰。

## 改动总览

### P0: Provider HTTP 流式中断监听

**核心思路**：在 provider 创建 httpx 客户端 / Anthropic 流时，注册一个
`InterruptSignal` listener，收到中断信号立即关闭连接/流。

#### `openai_compatible.py`

```python
# _streaming_generate() 内部
with httpx.Client(timeout=timeout) as client:
    _unregister = None
    if interrupt_signal is not None:
        _unregister = _register_interrupt_listener(interrupt_signal, client)
    try:
        with client.stream("POST", url, ...) as resp:
            ...
    finally:
        if _unregister is not None:
            _unregister()
```

模块级 helper：

```python
def _register_interrupt_listener(
    signal: InterruptSignal,
    client: httpx.Client,
) -> Callable[[], None]:
    def _on_abort(_reason: str | None) -> None:
        try:
            client.close()
        except Exception:
            pass
    signal.add_listener(_on_abort)
    def _unregister() -> None:
        signal.remove_listener(_on_abort)
    return _unregister
```

`generate()` 从 `context.interrupt_signal` 提取信号并传给 `_streaming_generate()`。

#### `codex.py`

同 `openai_compatible.py` 的 pattern。`_stream_response()` 接受
`interrupt_signal` 参数，在 `with httpx.Client(...)` 内部注册 listener，
`try/finally` 确保 unregister。

#### `claude.py`

Anthropic SDK 的 `messages.stream()` 返回 `MessageStream`，有 `close()` 方法。
listener 直接调 `stream.close()` 中断流迭代：

```python
with self._client.messages.stream(**request_payload) as stream:
    if interrupt_signal is not None:
        def _on_abort(_reason):
            try:
                stream.close()
            except Exception:
                pass
        interrupt_signal.add_listener(_on_abort)
        _unregister = lambda: interrupt_signal.remove_listener(_on_abort)
    try:
        for chunk in stream.text_stream:
            ...
    finally:
        if _unregister is not None:
            _unregister()
```

### P0+: 引擎中断短路（关键补丁）

**这是解决 2-3 秒延迟的根本修复**。

当 listener 关闭 HTTP 连接后，provider 抛出异常（`ProviderInvocationError`
或原始 `httpx.TransportError`），进入引擎的异常处理。原来的代码直接走
retry/recovery 路径，没有检查中断状态。

修复：在 `except ProviderInvocationError` 和 `except Exception` 两个处理分支的
**最前面**加中断检查——如果 `interrupt_signal.is_aborted()`，立即返回
`interrupted=True`，跳过整个重试流程。

```python
# agents/core/agent.py — _invoke_provider_with_recovery 方法
except ProviderInvocationError as exc:
    if self._is_interrupted(request.session_id, context):
        self._record_interrupt_metadata(context, ...)
        return AgentEngine._ProviderInvocationOutcome(interrupted=True)
    # ... 原有的 recovery 逻辑 ...

except Exception as exc:
    if self._is_interrupted(request.session_id, context):
        self._record_interrupt_metadata(context, ...)
        return AgentEngine._ProviderInvocationOutcome(interrupted=True)
    # ... 原有的 non-structured error 逻辑 ...
```

### P1: Shell tool 非阻塞 listener

将 `_cancel` 中 SIGTERM → busy-wait → SIGKILL 的 2 秒阻塞循环改为
`threading.Timer` 异步调度：

```python
def _cancel(_reason: str | None) -> None:
    nonlocal interrupted
    if process.poll() is not None:
        return
    interrupted = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    def _escalate() -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    threading.Timer(_INTERRUPT_GRACE_SEC, _escalate).start()
```

Listener 立即返回（non-blocking），SIGKILL 在 2 秒后异步触发。

### P2: InterruptSignal `reset()` + InterruptController 修复

`InterruptSignal` 新增 `reset()` 方法：

```python
def reset(self) -> None:
    with self._lock:
        self._event.clear()
        self._reason = None
```

`InterruptController.clear()` 改为调 `reset()` 而非 `pop()` + `close()`：

```python
def clear(self, session_id: str) -> None:
    with self._lock:
        signal = self._signals.get(session_id)
    if signal is not None:
        signal.reset()
```

这样 `RunHandle` 等持有旧 signal 引用的对象在下一轮 turn 仍然有效。

### UI: 去除冗余 "interrupting" 提示

`ActivityBar.tsx` 中 `interruptPending` 状态下原来显示 "… interrupting" 文字，
改为返回空片段 `<></>`（隐藏 activity bar）。只保留 `Composer.tsx` 底部的
"interrupting…" 作为唯一中断指示器。

## 完整中断链路（修复后）

```
1. 用户按 Ctrl+C / ESC
   └─ TUI: handleCancel()
        ├─ activityActions.markInterruptPending()     ← 即时视觉反馈
        ├─ composerActions.clearQueued()
        └─ await client.request('agent.cancel', ...)  ← RPC 发到 gateway

2. Gateway 收到 agent.cancel
   └─ running_runs.cancel(session_id)
        └─ handle.cancel(reason)
             ├─ cancel_event.set()
             └─ interrupt_signal.abort(reason)         ← 触发所有 listener

3. Listener 同步 fire（0ms 延迟）
   ├─ Provider listener: client.close() / stream.close()
   │    └─ httpx 正在读的 iter_lines() 立即抛 TransportError
   ├─ Shell listener: os.killpg(SIGTERM) + Timer(2s, SIGKILL)
   └─ WebFetch listener: client.close()（已有实现）

4. 异常冒泡到引擎
   └─ except ProviderInvocationError / except Exception
        ├─ _is_interrupted() → True                    ← 新增短路检查
        └─ return interrupted=True                     ← 跳过 retry/recovery

5. run_loop 看到 interrupted=True
   └─ ExitReason.INTERRUPTED → 结束 turn → TUI 恢复 idle
```

预期端到端延迟：**< 200ms**（RPC round-trip + listener fire + 异常冒泡）。

## 文件改动

```
aether/models/provider/openai_compatible.py
  + import InterruptSignal
  ~ generate(): 提取 interrupt_signal，传给 _streaming_generate()
  ~ _streaming_generate(): 接受 interrupt_signal，注册/注销 listener
  + _register_interrupt_listener() 模块级 helper

aether/models/provider/codex.py
  + import InterruptSignal
  ~ generate(): 提取 interrupt_signal，传给 _call_codex_api()
  ~ _call_codex_api(): 传给 _stream_response()
  ~ _stream_response(): 修复缩进，添加 try/finally + listener 注册
  + _register_interrupt_listener() 模块级 helper

aether/models/provider/claude.py
  + import InterruptSignal, Callable
  ~ generate(): 提取 interrupt_signal，传给 _create_streaming()
  ~ _create_streaming(): 在 stream context 内注册 _on_abort listener

aether/agents/core/agent.py
  ~ except ProviderInvocationError: 前置 _is_interrupted() 短路检查
  ~ except Exception: 前置 _is_interrupted() 短路检查

aether/runtime/control/interrupt_signal.py
  + reset(): 清除 aborted 状态和 reason

aether/runtime/control/interrupts.py
  ~ clear(): 改用 signal.reset() 替代 pop() + close()

aether/tools/builtins/shell.py
  + import threading
  ~ _cancel(): 用 threading.Timer 替代 busy-wait loop

tui/src/components/ActivityBar.tsx
  ~ interruptPending: 返回 <></> 替代 "… interrupting" 文字
```

## 与原设计文档的差异

| 原设计 | 实际实现 | 原因 |
|---|---|---|
| PR 6.7 只覆盖 tool 级 HTTP | 扩展到 provider 级 HTTP streaming | Provider 是中断延迟的主要来源（time-to-first-token 期间完全无法 polling） |
| 未提及引擎 error handler 短路 | 新增 `except` 块的前置中断检查 | HTTP 连接关闭产生的异常进入 retry 路径是 2-3 秒延迟的根本原因 |
| PR 6.3 设计中 `reset()` 有定义 | 实际代码未实现，本 PR 补上 | `InterruptController.clear()` 用 `pop()+close()` 会导致引用失效 |
| Shell listener 设计为 non-blocking | 实际代码用 busy-wait，本 PR 修复 | 违反 listener "non-blocking only" 规则 |
| 未提及 UI 去重 | 移除 ActivityBar 冗余提示 | 两处 "interrupting" 视觉干扰 |

## 验收

- [x] `openai_compatible.py` / `codex.py` / `claude.py` 语法正确（`ast.parse` 通过）
- [x] `interrupt_signal.py` / `interrupts.py` 语法正确
- [x] `shell.py` 语法正确
- [x] `agent.py` 语法正确，无新 type error
- [x] TUI `tsc --noEmit` 通过
- [x] `InterruptSignal.reset()` 功能测试通过
- [x] `InterruptController.clear()` 重置后 signal 可复用，引用不变
- [ ] 手测：模型 streaming 中按 Ctrl+C，< 500ms 内停止
- [ ] 手测：模型 thinking（等待首 token）时按 Ctrl+C，< 500ms 内停止
- [ ] 手测：shell tool 跑长命令时按 Ctrl+C，< 500ms 内停止
