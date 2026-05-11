# PR 6.5 — 收尾：双击 Ctrl-C / `/refresh` / interrupt_behavior 声明 / 调试日志

> **架构补强（参考 open-claude-code）**：本 PR 增加 `interrupt_behavior` tool 类属性
> （对应 `open-claude-code/src/Tool.ts:416` 的 `interruptBehavior()`），让每个 tool
> 声明被打断时的行为（`'cancel'` kill / `'block'` 放过）。引擎在 signal abort
> 触发时按声明派发。这样 ms 级的 `Read` / `Edit` 不会被 ESC 干扰，长跑的
> `Shell` / `WebFetch` 会被立刻 kill。

## 目标

把 sprint 6 收尾，做完之前几个 PR 留下的小尾巴 + 加一些质量提升 hook：

- **双击 Ctrl-C 退出**完整化（PR 6.1 已起头）
- **`/refresh` slash 命令**手动清屏 + 重画（顺带解决一些 prompt_toolkit resize 之后的偶发残影）
- **`interrupt_behavior` tool 类属性**：tool 自己声明 cancel vs block
- **打断 debug 日志**：把每次 interrupt 的来源、上下文记录到 logger，便于事后排查
- **footer queued badge 在打断时正确清空**

每一项独立小，全部加起来 < 250 行 diff。可以单独 cherry-pick。

## 当前问题（PR 6.1-6.4 完成后剩下的）

1. **PR 6.1 留了"双击 ESC 触发 /clear"的 stub**，但 `engine.clear_history(session_id)` 这个方法当前不存在 —— stub 调的是 noop。需要补上 actual implementation。
2. **PR 6.1 的双击 Ctrl-C 退出**也只在 cli/app.py 层做了窗口判断，但 idle 时第一次 Ctrl-C 不显示 toast 提示，用户不知道还要再按一次。
3. **没有 `/refresh` 命令**：偶尔 prompt_toolkit 在长时间渲染后会留少量残影；之前 sprint 讨论过加这个 escape hatch 一直没做。
4. **打断 debug 日志缺失**：现在 `_on_interrupt` 只 `ui.warn("interrupted")`，logger 没记录"什么时候被打断、当前在哪个 iteration、上一个 tool 是什么"，事后排查费劲。
5. **Footer queued badge** (`▸ queued (n)`) 在打断后理论上 `_pending` 已清空，但 `app._busy` flip 到 False 之前的窗口里 footer 会渲染 `queued (0)` —— 边界情况，看着别扭。

## 设计

### 1. `engine.clear_history` + `/clear` slash 命令

新增 `AgentEngine.clear_history(session_id)`：

```python
# agents/core/agent.py
def clear_history(self, session_id: str | None = None) -> None:
    """Clear in-memory conversation history for a session.

    Does not touch persisted SessionStore (slash command ``/clear`` is
    expected to be a transient affordance — use ``/new`` to also wipe
    the persisted session).  Safe to call even when no session is
    active.
    """
    effective = session_id or self._current_session_id
    if not effective:
        return
    self.services.session_store.clear_in_memory(effective)
    self.services.steer_inbox.clear(effective)
    self.services.interrupt_controller.clear(effective)
```

slash 命令（如果还没 wired）：

```python
# cli/commands.py
def _cmd_clear(state: "ReplState", _args: list[str]) -> CommandResult:
    state.engine.clear_history(state.session_id)
    state.messages = []
    state.ui.success("conversation cleared (session preserved, use /new to fully reset)")
    return CommandResult()

SLASH_COMMANDS["/clear"] = SlashCommand(
    "/clear", "Clear conversation history (keep session id)", _cmd_clear,
)
```

PR 6.1 stub 的 `_on_clear_history` 改成调这个：

```python
# cli/repl.py
on_clear_history=lambda: slash.dispatch(state, "/clear"),
```

### 2. Ctrl-C 第一次按时显示 toast

PR 6.1 实现的"双击 Ctrl-C 退出 idle 模式"逻辑里加 toast：

```python
def _handle_ctrl_c(self, event):
    if self._busy:
        self._on_interrupt()
        ...
        return

    if self._buffer.text:
        self._buffer.reset()
        self._last_ctrl_c_at = None
        return

    if self._is_double_press(self._last_ctrl_c_at):
        self._exit_requested = True
        event.app.exit()
        return

    self._last_ctrl_c_at = time.monotonic()
    # NEW: visible feedback
    self._show_ephemeral_hint(
        "Press Ctrl-C again to exit",
        timeout_sec=DOUBLE_PRESS_WINDOW_SEC,
    )
```

`_show_ephemeral_hint` 是新的 helper：在 footer 旁边或 activity bar 上方放一行 dim hint，N 秒后自动消失。

### 3. `/refresh` slash 命令

```python
# cli/commands.py
def _cmd_refresh(state: "ReplState", _args: list[str]) -> CommandResult:
    """Force a clean redraw of the prompt + bottom region.

    Useful after a terminal resize that left ghost frames (some IDE
    terminals don't always send SIGWINCH cleanly), or after a noisy
    tool produced cells the renderer's diff missed.
    """
    try:
        state.app._app.renderer.clear()  # erases visible viewport
    except Exception:
        pass
    try:
        state.app._app.invalidate()
    except Exception:
        pass
    state.ui.info("screen refreshed")
    return CommandResult()

SLASH_COMMANDS["/refresh"] = SlashCommand(
    "/refresh", "Clear visible viewport and redraw (no data loss)", _cmd_refresh,
)
```

注意：`renderer.clear()` 调用 `output.erase_screen()` 即 `\x1b[2J`，wipe 可见区域**保留终端 scroll buffer**。用户能 mouse-scroll 找回之前的内容。

### 4. Tool `interrupt_behavior` 类属性

参考 `open-claude-code/src/Tool.ts:416`：

```typescript
interruptBehavior?(): 'cancel' | 'block'
```

我们的 Python 等价：

```python
# tools/base.py
from typing import Literal

class ToolExecutor:
    """Base class for all tools.

    Subclasses can override ``interrupt_behavior`` to declare how the
    engine should treat them on user-interrupt.  Default is "block"
    — finish the call naturally, don't kill — because most tools are
    sub-second pure-Python (Read / Edit / Grep) and killing them
    leaves the codebase in a weird state.
    """

    interrupt_behavior: Literal["cancel", "block"] = "block"
    ...


# tools/builtins/shell.py
class ShellTool(ToolExecutor):
    interrupt_behavior = "cancel"  # subprocess that may run minutes


# tools/builtins/web_fetch.py
class WebFetchTool(ToolExecutor):
    interrupt_behavior = "cancel"  # network IO


# tools/builtins/read.py — 默认 "block" 即可（sub-second，不需要改）
```

#### 引擎层的派发

PR 6.3 的 `InterruptSignal.add_listener` 由 tool 在 `run()` 入口注册；
本 PR 在 listener 注册点统一加一层 `interrupt_behavior` 检查：

```python
# tools/base.py 提供 helper
def install_interrupt_listener(
    sig: InterruptSignal | None,
    tool: ToolExecutor,
    on_abort: Callable[[str | None], None],
) -> Callable[[], None]:
    """Tool 在 run() 入口调，按 interrupt_behavior 决定是否真挂 listener。"""
    if sig is None or tool.interrupt_behavior == "block":
        return lambda: None  # no-op unregister
    return sig.add_listener(on_abort, once=True)
```

shell tool 改成：

```python
class ShellTool(ToolExecutor):
    interrupt_behavior = "cancel"

    def run(self, call, ctx):
        proc = subprocess.Popen(...)

        def _on_abort(reason):
            _terminate_process_group(proc)
            ...

        unregister = install_interrupt_listener(
            ctx.interrupt_signal, self, _on_abort,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        finally:
            unregister()
```

#### 为什么这层抽象有用

1. **"block" tool 中断时不被干扰**：用户按 ESC 时 ms 级 tool 可能已经快跑完，
   强制 kill 反而让 partial 数据丢失（写到一半的文件、半个 git diff）。
2. **新加 tool 时心智清晰**：作者只需要回答一个问题 —— "我这个 tool 长跑吗？
   如果是 → `'cancel'`；否则保持默认 `'block'`"。
3. **跟 claude-code 的 `'interrupt'` reason 语义对齐**：未来如果引入"用户输入新消息
   触发的隐性 abort"（不是显式 ESC），可以让 `'block'` tool 跑完再继续。

#### 不破坏 PR 6.3 现有 shell tool

PR 6.3 的 shell tool 已经无条件挂 listener。本 PR 把无条件改成"按 interrupt_behavior"。
对 shell tool 没行为变化（依旧是 `"cancel"`），但 review 时要 pin 测试：

```python
def test_block_behavior_tool_not_killed_on_interrupt(self):
    class StubBlockTool(ToolExecutor):
        interrupt_behavior = "block"
        def run(self, call, ctx):
            time.sleep(0.5)
            return ToolResult(content="finished", ...)

    sig = InterruptSignal()
    threading.Timer(0.1, sig.abort).start()
    start = time.monotonic()
    result = StubBlockTool().run(call, _make_ctx(interrupt_signal=sig))
    elapsed = time.monotonic() - start
    # 没被 kill，完整跑完 500ms
    self.assertGreaterEqual(elapsed, 0.45)
    self.assertEqual(result.content, "finished")
```

### 5. 打断 debug 日志

每次进入打断路径时记录关键上下文：

```python
# agents/core/agent.py interrupt() 入口
def interrupt(self, session_id: str | None = None, reason: str | None = None) -> None:
    effective_session = session_id or self._current_session_id
    if effective_session:
        # NEW: structured debug log
        self.services.logger.info(
            "engine.interrupt",
            extra={
                "session_id": effective_session,
                "reason": reason or "unspecified",
                "current_iteration": self._current_iteration_for_log(effective_session),
                "in_flight_tools": self._in_flight_tool_names_for_log(effective_session),
                "stream_chars_so_far": self._stream_chars_for_log(effective_session),
            },
        )
        self.services.interrupt_controller.request(effective_session, reason)
        self.services.steer_inbox.clear(effective_session)
    self._interrupt_active_children(reason=reason)
```

helper 实现可以读 `context.metadata`（如果当前有 active turn）或返回 0。允许失败（日志路径绝对不能让中断失败）：

```python
def _stream_chars_for_log(self, session_id: str) -> int:
    try:
        ctx = self._active_contexts.get(session_id)  # 假设 engine 维护这个
        if ctx is None:
            return 0
        return len(str(ctx.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "")))
    except Exception:
        return 0
```

如果 `_active_contexts` 不存在（engine 没有这种 plumbing），可以简化为 best-effort：从最近一次 stream callback 的 metadata 拿；拿不到就记 unknown。

### 6. Footer queued badge 边界

bug 复现：

1. `_busy=True`, `_pending=["msg1", "msg2"]` → footer 显示 `queued (2)`
2. 用户按 ESC → `_pending.clear()` 立刻发生，但 `_busy` 还是 True（engine 还没退出）
3. 下一个 render frame footer 渲染 `queued (0)` —— 数字是 0 但 badge 还在
4. ~50ms 后 engine 真正退出，`_busy=False`，badge 消失

修复：footer 渲染加条件：

```python
if self._busy and self._pending:  # NEW: also gate on non-empty
    fragments.append(sep)
    fragments.append(("class:footer.queued", f"▸ queued ({len(self._pending)})"))
```

实际上现有代码看起来已经是这样了（`if self._busy and self._pending`），但 PR 6.1 重构 footer 时可能会引入回归 —— 在测试里 pin 一下：

```python
def test_footer_busy_with_empty_queue_omits_badge(self):
    app = self._build_app()
    app._busy = True
    app._pending = []
    text = _plain(app._get_footer_text())
    self.assertNotIn("queued", text)
```

## 文件改动

```
agents/core/agent.py
  + clear_history(session_id) method
  ~ interrupt(): 加 logger.info 调用
  + _stream_chars_for_log / _in_flight_tool_names_for_log / _current_iteration_for_log
    （或简化为读 context.metadata 的 try/except helper）

cli/commands.py
  + _cmd_clear, SLASH_COMMANDS["/clear"]
  + _cmd_refresh, SLASH_COMMANDS["/refresh"]

cli/repl.py
  ~ AetherApp 构造: on_clear_history=lambda: slash.dispatch(state, "/clear")

cli/app.py
  + _show_ephemeral_hint(message, timeout_sec) helper
  ~ _handle_ctrl_c idle 第一次按时显示 toast
  + ephemeral hint 渲染（在 activity bar 之上 1 行，conditional container）

tests/cli/test_cli_commands.py
  + /clear 测试
  + /refresh 测试

tests/cli/test_cli_app.py
  + Ctrl-C 第一次按显示 toast 测试
  + footer empty queue 不显示 badge 测试

tests/engine/test_clear_history.py  [新文件]
  + 测试 clear_history 清 messages / steer / interrupt
  + 测试 next turn 状态 OK
```

## 实现细节

### `_show_ephemeral_hint` 实现

加一个 `_ephemeral_hint_text` 和 `_ephemeral_hint_until` 字段，footer 渲染时检查：

```python
def _show_ephemeral_hint(self, message: str, *, timeout_sec: float) -> None:
    self._ephemeral_hint_text = message
    self._ephemeral_hint_until = time.monotonic() + timeout_sec
    # 不需要显式 timer — refresh_interval=0.2 会自然让它过期
    try:
        self._app.invalidate()
    except Exception:
        pass

def _get_ephemeral_hint(self) -> str:
    if not self._ephemeral_hint_text:
        return ""
    if time.monotonic() > self._ephemeral_hint_until:
        self._ephemeral_hint_text = ""
        return ""
    return self._ephemeral_hint_text
```

放进 footer 的方式：

```python
hint = self._get_ephemeral_hint()
if hint:
    fragments.append(sep)
    fragments.append(("class:footer.busy", hint))
```

### `/refresh` 跟之前 sprint resize 修复的关系

我们之前已经实现了 `before_render` polling 检测 terminal size 变化并触发 `renderer.clear()`。`/refresh` 是用户手动触发的兜底：

- 如果 polling 因为某种原因没检测到（比如 terminal stays 同 size 但 cell content 错位），用户能手动救场。
- 不重复 polling 的逻辑，命令层就一句 `renderer.clear() + invalidate()`。

## 参考实现

- `open-claude-code/src/hooks/useCancelRequest.ts` 行 84+：Ctrl-C 双击窗口（虽然 claude-code 用的是不同的 trigger，但模式一致）。
- `open-claude-code/src/components/PromptInput/PromptInputFooterLeftSide.tsx` 行 506-507：footer hint "esc to interrupt" 渲染。

## 测试

`tests/engine/test_clear_history.py`

```python
class ClearHistoryTests(unittest.TestCase):
    def test_clear_history_wipes_in_memory_messages(self):
        # 1. Build engine, run a turn that produces messages
        # 2. clear_history(session_id)
        # 3. Next turn starts with messages == []
        ...

    def test_clear_history_clears_steer_inbox(self):
        # 1. send_steer(session_id, "hint")
        # 2. clear_history(session_id)
        # 3. steer_inbox.peek(session_id) is empty
        ...

    def test_clear_history_clears_interrupt_flag(self):
        # 1. interrupt(session_id)
        # 2. clear_history(session_id)
        # 3. is_interrupted(session_id) is False
        ...

    def test_clear_history_safe_when_no_session(self):
        # No active session → no-op, no exception
        ...
```

`tests/cli/test_cli_commands.py`

```python
class SlashClearTests(unittest.TestCase):
    def test_slash_clear_calls_engine_and_resets_messages(self):
        ...

class SlashRefreshTests(unittest.TestCase):
    def test_slash_refresh_calls_renderer_clear(self):
        ...
    def test_slash_refresh_swallows_renderer_exception(self):
        # if renderer.clear() raises, command still returns success
        ...
```

`tests/cli/test_cli_app.py`

```python
class AetherAppCtrlCToastTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_ctrl_c_shows_ephemeral_hint(self):
        ...
    async def test_second_ctrl_c_within_window_exits(self):
        ...
    async def test_ephemeral_hint_expires_after_timeout(self):
        ...

class AetherAppFooterQueuedBadgeTests(unittest.TestCase):
    def test_busy_with_empty_queue_omits_queued_badge(self):
        ...
    def test_busy_with_pending_shows_queued_badge(self):
        ...
```

## 验收门

- 所有新测试通过。
- 手测：
  - 按 ESC 两次（第二次在 800ms 内）触发 `/clear`，会话历史确实清空。
  - 按 Ctrl-C 一次显示 "Press Ctrl-C again to exit" toast，2s 内再按真的退出。
  - 拖拽 terminal 边界后输入 `/refresh`，残影消失。
  - 打断长 turn 后看 logger 输出，能找到 `engine.interrupt` 记录 + reason + iteration。
- 不破坏既有 `/help` / `/exit` / `/model` / `/system` 等 slash 命令。
- 不影响 PR 6.1-6.4 的核心打断流程。

## Sprint 6 完成

合并 PR 6.5 后，整个 Sprint 6 视为完成。`run-loop-roadmap` 的"打断系统"项可以勾掉。建议下一个 sprint 关注：

- subagent / Task tool 的中断（当前 `_interrupt_active_children` 只设标志，不传播到 subagent 的 stream / subprocess）
- `INTERRUPT_MESSAGE_FOR_TOOL_USE` 的 UI 卡片化（区分于普通 user message）
- 把 `/clear` / `/refresh` / `/system` 等 slash 命令文档化到 `docs/run-loop-roadmap/`
