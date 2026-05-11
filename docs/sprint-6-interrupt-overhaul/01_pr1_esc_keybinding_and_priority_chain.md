# PR 6.1 — ESC 单按打断 + 优先级链

## 目标

让用户按 **ESC** 即可打断当前任务（claude-code / codex 一致的肌肉记忆），并实现按场景分发的优先级链：当前没有任务在跑时 ESC 做"上下文相关动作"（清输入框、关弹窗、双击清对话），不会粗暴 fallthrough 到中断逻辑。Ctrl-C 保留但行为对齐（idle 时改为双击退出）。

本 PR **只改 CLI 层**，不动 engine 内部，是整个 sprint 风险最低、收益最直观的一步。

## 当前问题

- `cli/app.py` 只 bind 了 `escape,enter`（换行）和 `c-c`（中断），**没有单按 ESC 的处理**。
- Ctrl-C idle 时直接退出，没有"再按一次确认"的二次保护。
- 当 `_busy=True` + 输入框有文字时按 Ctrl-C，会同时触发 interrupt + reset buffer + clear queue，副作用过多没分层。
- 没有 footer "esc to interrupt" 的视觉提示，用户根本不知道有这个键。

## 设计

### ESC 优先级链（参考 `open-claude-code/src/components/PromptInput/PromptInput.tsx` 1922-1957）

按下 ESC 时按顺序匹配，第一个命中的执行：

| 优先级 | 条件 | 行为 |
|---|---|---|
| 1 | 当前 turn 在跑（`self._busy` 或有 in-flight tool） | 触发 `_on_interrupt`；清 `_pending`；invalidate 重绘 |
| 2 | 输入框非空 | `buf.reset()` 清空输入；记录 `_last_esc_at` 时间戳 |
| 3 | 输入框空 + `_pending` 队列非空 | 弹出最近一条 queued 消息回到 buffer 编辑 |
| 4 | 输入框空 + 距上次 ESC < 800ms（双击）+ 有对话历史 | 触发 `/clear`（清当前 session 历史，保留 session id） |
| 5 | 默认 | no-op，但短 toast hint："press ESC again to clear" |

**理由**：每一层都是"用户最可能想做的事"。1 → 2 → 3 → 4 是一个**逐级回退**的反应链，跟 claude-code 的 ESC 分发一致。

### Ctrl-C 行为重写

| 场景 | 旧行为 | 新行为 |
|---|---|---|
| `_busy=True` | interrupt + 清 queue | 同 ESC 优先级 1 |
| 输入框非空 | reset buffer | 同 ESC 优先级 2 |
| idle + 输入框空 | **直接退出** | 双击：第一次"press Ctrl-C again to exit"（2s 窗口），第二次才真正退出 |

claude-code 用的也是双击退出（`useDoublePress` hook），避免误退出 30 分钟会话的悲剧。

### footer hint

任务在跑时（`_busy=True`），footer 在 `running…` 或 `queued (n)` 旁边追加 `· esc to interrupt`。idle 时不显示，避免视觉噪音。

## 文件改动

```
cli/app.py
  + AetherApp._build_keybindings: 新增 ESC 单按 binding（eager=True）
  + AetherApp._handle_esc(): 优先级链分发
  + AetherApp._handle_ctrl_c(): 重写为双击退出
  + AetherApp._last_esc_at: float | None
  + AetherApp._last_ctrl_c_at: float | None
  + AetherApp._on_clear_history: Optional[Callable[[], None]]  (新增构造参数)
  ~ AetherApp._get_footer_text: busy 时追加 esc-to-interrupt hint

cli/repl.py
  ~ 构造 AetherApp 时传入 on_clear_history=lambda: state.engine.clear_history(state.session_id)
  + 如果 engine.clear_history 不存在，先在 ReplState 里加一个简易实现（清 state.messages，保留 session_id）

tests/cli/test_cli_app.py
  + ESC 优先级链分支测试（5 个 case 各一个）
  + Ctrl-C 双击退出测试（窗口内 / 窗口外两个 case）
  + footer busy hint 测试
```

## 实现细节

### ESC binding（处理 prompt_toolkit 的 ESC-序列坑）

prompt_toolkit 默认会等 `Application.timeoutlen`（25ms）来区分 `ESC` 单按 vs `ESC <key>` 序列。直接 bind `("escape",)` 不带 `eager=True` 会引入 25ms 延迟。

```python
@bindings.add("escape", eager=True)
def _esc(event):  # noqa: ANN001
    self._handle_esc(event)
```

`eager=True` 会让 ESC 立刻触发，不等续键。**副作用**：之前的 `bindings.add("escape", "enter")` 会被覆盖掉。我们仍需要 `Esc+Enter` 换行，所以让 `_handle_esc` 内部检测 `event.key_sequence` 长度 —— 但其实 `eager=True` + `escape,enter` 同时存在是合法的（prompt_toolkit 会优先匹配长序列）。需要在测试里验证两个 binding 不冲突。

### 双击窗口

```python
import time

DOUBLE_PRESS_WINDOW_SEC = 0.8

def _is_double_press(self, last_at: float | None) -> bool:
    if last_at is None:
        return False
    return (time.monotonic() - last_at) < DOUBLE_PRESS_WINDOW_SEC
```

### footer hint

```python
def _get_footer_text(self) -> FormattedText:
    fragments = [...]  # 现有 fragments
    if self._busy:
        fragments.append(sep)
        fragments.append(("class:footer.kbd", "esc"))
        fragments.append(("class:footer", " to interrupt"))
    return FormattedText(fragments)
```

## 参考实现

- `open-claude-code/src/keybindings/defaultBindings.ts` 行 66：`escape: 'chat:cancel'`
- `open-claude-code/src/hooks/useCancelRequest.ts` 行 87-115：`handleCancel` 优先级
- `open-claude-code/src/components/PromptInput/PromptInput.tsx` 行 1922-1957：ESC 上下文分发
- `open-claude-code/src/components/Spinner/SpinnerAnimationRow.tsx` 行 216：`(esc to interrupt)` footer hint

## 测试

文件：`tests/cli/test_cli_app.py`

```python
class AetherAppEscPriorityChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_esc_with_busy_triggers_interrupt(self):
        # _busy=True → interrupt callback fires
        ...

    async def test_esc_with_text_clears_buffer(self):
        # _busy=False, buffer="hello" → buffer.reset(), no interrupt
        ...

    async def test_esc_with_pending_queue_pops_last(self):
        # _busy=False, buffer="", _pending=["a", "b"] → buffer="b", _pending=["a"]
        ...

    async def test_double_esc_clears_history(self):
        # _busy=False, buffer="", _pending=[]
        # First ESC: no-op + toast
        # Second ESC within 800ms: triggers on_clear_history
        ...

    async def test_esc_outside_double_window_only_toasts(self):
        # First ESC, sleep 1.0s, second ESC → still no-op
        ...


class AetherAppCtrlCDoublePressExitTests(unittest.IsolatedAsyncioTestCase):
    async def test_ctrl_c_idle_first_press_warns(self):
        # _busy=False, buffer="" → no exit, _last_ctrl_c_at set
        ...

    async def test_ctrl_c_idle_double_press_exits(self):
        # First Ctrl-C, second within 2s → exit_requested=True
        ...

    async def test_ctrl_c_busy_interrupts(self):
        # _busy=True → on_interrupt called regardless of double-press window
        ...
```

footer hint：

```python
class AetherAppFooterEscHintTests(unittest.TestCase):
    def test_busy_footer_includes_esc_hint(self):
        # _busy=True → footer text contains "esc to interrupt"
        ...
    def test_idle_footer_omits_esc_hint(self):
        # _busy=False → footer text does NOT contain "esc"
        ...
```

## 验收门

- 所有 ESC / Ctrl-C 测试通过。
- 不破坏现有 `Esc+Enter` 换行行为。
- 不破坏 `c-j` (`^J` 备用换行) 行为。
- footer 在 idle 时**不**显示 `esc` hint（避免噪音）；任务跑起来才显示。
- 不引入 engine 改动；只动 `cli/`。

## 不在本 PR 内（留给后续 PR）

- 真正的"流式打断响应延迟 < 500ms" → PR 6.2
- subprocess 中断 → PR 6.3
- partial 文本保留 + interrupt marker 注入 → PR 6.4
- 双击 ESC 真的清 conversation history（需要 `clear_history` 在 engine 侧实现）→ 如果 engine 已有 `/clear` slash 命令，可以直接复用；否则 stub 一个 no-op，等 PR 6.4 一并完善
