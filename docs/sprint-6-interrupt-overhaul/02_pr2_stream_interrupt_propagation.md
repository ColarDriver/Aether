# PR 6.2 — Stream callback 打断传播

## 目标

让模型**正在流式输出**时按下打断键能在 < 200ms 内停下，不需要等流自然完成。这是用户感知最强的"打断有没有用"指标 —— 长回复（5 ~ 60s）期间打断必须立即生效。

## 当前问题

`agents/core/agent.py` `_build_stream_callback._wrapped` 只做两件事：

1. 累计 partial 文本到 `context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT]`
2. 转发 delta 给 UI callback

**完全没检查 `InterruptController` 标志位**。所以 provider 一旦开始 stream，要么自己产完所有 chunks，要么 HTTP 链接超时；用户按 Ctrl-C / ESC 只是在 `InterruptController` 里设置了一个 flag，但 stream loop 看不到它，flag 要等到下一个迭代边界（即 stream 结束 + tool 调用完 + 下一轮 `before_llm`）才被检查。

实测：用 GPT-4o 让它写 1000 字回复，按 Ctrl-C 后等了 23 秒才真正停下。

## 设计

### 中断信号传播路径

```
User presses ESC
  └→ AetherApp._handle_esc → on_interrupt callback
       └→ repl._on_interrupt → engine.interrupt(session_id)
            └→ InterruptController.request(session_id)  [flag set]

—— 同一时刻，engine worker thread 在 provider.generate() 里 ——

provider yields delta_N
  └→ stream_callback_wrapped(delta_N)
       └→ if interrupt_controller.is_interrupted(session_id):
            raise EngineInterrupted("user-interrupt")
       └→ append delta to metadata + forward to UI

——

provider 抛 EngineInterrupted 上来
  └→ provider wrapper 必须 re-raise（不吞）
  └→ run_loop catch → state_machine.transition(INTERRUPTED)
  └→ 走正常 INTERRUPTED 退出路径
```

### `EngineInterrupted` 异常

新增 `runtime/exceptions.py`：

```python
class EngineInterrupted(RuntimeError):
    """Raised by stream callback / tool when user-interrupt is detected.

    Crossing this exception boundary means we stop processing the
    current LLM call / tool execution immediately and unwind to the
    run loop's INTERRUPTED branch.  Carries the partial text already
    streamed (if any) so the run loop can preserve it on the way out.
    """

    def __init__(
        self,
        reason: str = "user-interrupt",
        *,
        partial_text: str = "",
        was_in_tool_call: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial_text = partial_text
        self.was_in_tool_call = was_in_tool_call
```

### Stream callback 加 polling

```python
# agents/core/agent.py _build_stream_callback._wrapped
def _wrapped(delta: str) -> None:
    if not isinstance(delta, str) or not delta:
        return

    # NEW: poll interrupt before any work — abort fast.
    if self._is_interrupted(request.session_id):
        partial = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
        raise EngineInterrupted(
            reason="user-interrupt",
            partial_text=partial,
            was_in_tool_call=False,
        )

    # 原有逻辑
    if getattr(self.config, "empty_response_partial_stream_recovery_enabled", True):
        current = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
        context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = current + delta
    if callback is None:
        return
    try:
        callback(delta)
        ...
```

中断检查放在 callback 入口最前面：每个 delta 都 poll 一次，cost 是 1 次字典查询 + 1 次锁，~µs 级别。

### Stream callback 异常的传播路径

provider 内部通常长这样：

```python
for chunk in http_response.iter_lines():
    if stream_callback:
        stream_callback(chunk.delta)   # ← 我们这里抛 EngineInterrupted
    accumulator.append(chunk.delta)
return accumulator.finalise()
```

如果 provider 里有 `try: ... except Exception:` 吞了所有异常，`EngineInterrupted` 也会被吞。我们必须确保 provider wrapper **明确 re-raise** `EngineInterrupted`。

实施方案：检查所有 provider，确认 `EngineInterrupted` 不被 broad-except 吞。安全做法是在 `runtime/exceptions.py` 让 `EngineInterrupted` 继承 `BaseException` 而不是 `Exception`，这样 `except Exception` 会自动放行：

```python
class EngineInterrupted(BaseException):
    """Inherits BaseException so 'except Exception' wrappers don't swallow it.

    Same rationale as KeyboardInterrupt / SystemExit being BaseException
    subclasses — interrupt is a control-flow signal, not a regular error.
    """
```

trade-off：调用栈中所有 `except Exception` 都不能捕获它。run_loop 里要明确 `except EngineInterrupted` 才能处理。

### run_loop 接住 EngineInterrupted

```python
# agents/core/agent.py run_loop main loop
try:
    invoke_outcome = self._invoke_provider_with_recovery(...)
except EngineInterrupted as exc:
    state_machine.transition(LoopState.INTERRUPTED)
    exit_reason = ExitReason.INTERRUPTED
    # 把 partial 塞进 context metadata，给 _build_result 用
    context.metadata["interrupt"] = {
        "reason": exc.reason,
        "partial_text": exc.partial_text,
        "was_in_tool_call": exc.was_in_tool_call,
        "triggered_at": time.time(),
    }
    break
```

### 性能：polling cost

`is_interrupted(session_id)` 是 dict lookup + RLock acquire/release。锁的 hot path（无竞争）实测 ~50ns + dict lookup ~50ns ≈ 100ns/call。即使 stream 1000 chunks/s 也只占 0.01% CPU。无需优化。

## 文件改动

```
runtime/exceptions.py
  + EngineInterrupted(BaseException)  [新文件]

runtime/__init__.py
  + 导出 EngineInterrupted

agents/core/agent.py
  ~ _build_stream_callback._wrapped: 入口加 is_interrupted poll
  ~ run_loop main loop: 加 except EngineInterrupted 分支
  ~ _build_result: 读 context.metadata["interrupt"] 写到 EngineResult.metadata

models/* (各 provider)
  ~ 检查所有 except Exception, 确认 EngineInterrupted (现在是 BaseException)
    自然不会被吞；如果有 except BaseException, 加白名单 re-raise

tests/runtime/test_interrupt_propagation.py  [新文件]
```

## 实现细节

### Provider 兼容性 audit

需要审查的 provider 文件：

```
models/openai_provider.py
models/anthropic_provider.py
models/openai_compat_provider.py
models/local_*.py
... etc
```

每个文件搜 `except Exception` 和 `except BaseException`。前者 OK（不会捕获 BaseException 子类），后者必须改成显式列表 + EngineInterrupted re-raise：

```python
# 不行 ❌
try:
    chunk = next(stream)
except BaseException as exc:
    logger.exception(...)
    return None

# 改成 ✅
try:
    chunk = next(stream)
except EngineInterrupted:
    raise
except Exception as exc:
    logger.exception(...)
    return None
```

### context metadata schema

新增字段：

```python
context.metadata["interrupt"] = {
    "reason": "user-interrupt",        # str
    "partial_text": "I think the answer is...",  # str, 可能为空
    "was_in_tool_call": False,         # bool
    "triggered_at": 1715423456.789,    # float, time.time()
}
```

`EngineResult.metadata["interrupt"]` 同 shape；`_build_result` 直接 copy。

### 不在 PR 6.2 范围内的事

- partial_text 写回 `state.messages` —— 留给 PR 6.4
- 注入 `[Request interrupted by user]` marker —— 留给 PR 6.4
- subprocess 中断 —— 留给 PR 6.3

PR 6.2 只保证：**stream 能立即停下，partial 信息透传到 metadata**。下游使用方法在 PR 6.4 解决。

## 参考实现

- `open-claude-code/src/QueryEngine.ts`：stream loop 在每个 chunk 检查 `abortController.signal.aborted`，aborted 则 break。
- `open-claude-code/src/screens/REPL.tsx` 行 2125-2130：`if (streamingText?.trim()) setMessages(prev => [...prev, createAssistantMessage({...})])` 在 abort 之前先存 partial。

## 测试

`tests/runtime/test_interrupt_propagation.py`

```python
class StreamInterruptPropagationTests(unittest.TestCase):
    def test_stream_callback_raises_engine_interrupted_when_flag_set(self):
        # 1. Build a fake EngineRequest with stream_callback
        # 2. Pre-set interrupt flag
        # 3. Call wrapped callback("hello")
        # 4. Assert EngineInterrupted raised, partial_text == ""
        ...

    def test_stream_callback_accumulates_partial_before_interrupt(self):
        # 1. Build engine, NO interrupt flag yet
        # 2. Call wrapped("hello"), wrapped(" world") — ok, no raise
        # 3. Set interrupt flag
        # 4. Call wrapped("!") → EngineInterrupted raised
        # 5. Assert exc.partial_text == "hello world"
        ...

    def test_engine_interrupted_propagates_through_provider_wrapper(self):
        # Use a fake provider that re-yields whatever stream_callback raises
        # Assert run_loop catches EngineInterrupted and sets ExitReason.INTERRUPTED
        ...

    def test_engine_interrupted_metadata_in_result(self):
        # After interrupt, EngineResult.metadata["interrupt"] has all 4 fields
        ...

    def test_engine_interrupted_does_not_get_swallowed_by_except_exception(self):
        # Construct a wrapper that does:
        #   try: stream_callback(...) except Exception: pass
        # Set interrupt flag, call wrapper.
        # Assert EngineInterrupted PROPAGATES out of wrapper
        # (because it's BaseException, not Exception subclass).
        ...

    def test_interrupt_response_latency_under_200ms(self):
        # 1. Spawn a thread that runs run_loop with a synthetic provider
        #    yielding 1 chunk every 50ms, total 100 chunks (5s).
        # 2. After 500ms, set interrupt flag.
        # 3. Wait for thread to finish; measure elapsed.
        # 4. Assert elapsed < 700ms (500 + 200ms grace).
        ...
```

## 验收门

- 所有测试通过 (`pytest tests/runtime/test_interrupt_propagation.py`).
- 实测打断响应延迟 < 500ms（用 GPT-4o 长回复场景手测）。
- 现有 `test_engine.py` / `test_streaming.py` 等 stream 相关测试不回归。
- `EngineResult.metadata["interrupt"]` 在 INTERRUPTED 退出时必须存在，shape 跟设计一致。
- 所有 provider 文件审查完毕，没有 `except BaseException` 吞掉 `EngineInterrupted`。

## 不在本 PR 内（留给后续 PR）

- subprocess / tool 内部的中断响应 → PR 6.3
- partial_text 写回 `state.messages` 让模型看见 → PR 6.4
- 注入 `[Request interrupted by user]` marker → PR 6.4
        #   try: stream_callback(...