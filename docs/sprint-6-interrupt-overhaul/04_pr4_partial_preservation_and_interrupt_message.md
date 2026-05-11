# PR 6.4 — Partial 保留 + INTERRUPT_MESSAGE 注入

> **架构调整（参考 open-claude-code）**：在 PR 6.3 引入 `InterruptSignal` 之后，
> 本 PR 的 `context.metadata["interrupt"]` 写入点从"polling 检查点"改成
> "signal listener 回调"。好处：abort 触发的瞬间 listener 同步 fire，
> 把"当时在干什么（stream / tool / idle）"准确记下来，不需要在 run_loop
> 的三处 INTERRUPTED 分支各写一遍。
>
> **与 claude-code 的对应**：
> - `open-claude-code/src/screens/REPL.tsx` 行 2120-2130 的"先存 partial 再 abort"
>   顺序在我们这里变成"abort 触发 listener → listener 把 partial 从
>   stream buffer snapshot 到 context.metadata"。
> - 两个 marker 常量直接照搬 `src/utils/messages.ts` 行 207-213。

## 目标

打断之后让模型**知道用户打断了**，并且能**看到自己刚才说到一半的内容**。这是让"打断"从"硬切流"升级成"自然对话操作"的关键 PR —— 没有它，打断只是停止流，下一轮模型不知道发生了什么，可能直接重复刚才的尝试。

需要 PR 6.1 / 6.2 / 6.3 完成（打断要能立刻触发，partial 要能透传出来），本 PR 完成消费侧。

## 当前问题

打断流程现在（假设 PR 6.2 / 6.3 已合）：

```
User presses ESC
  └→ engine.interrupt() → flag set
  └→ stream callback / shell tool 抛出（或返回带 interrupted=True 的 ToolResult）
  └→ run_loop 捕获，state = INTERRUPTED, exit_reason = INTERRUPTED
  └→ run_loop 返回 EngineResult(status=INTERRUPTED, messages=...)
  └→ repl.py 收到结果
        - status_value = "INTERRUPTED"
        - exit_reason_value = "user_interrupt"
        - ui.warn("interrupted")
        - state.messages = result.messages   # ← 这里是关键
```

问题：

1. **`result.messages` 没有 partial assistant**。stream 中断时，partial 文本只在 `context.metadata["interrupt"]["partial_text"]`（PR 6.2 透传），run_loop 返回时不会自动塞回 messages。
2. **`result.messages` 没有 INTERRUPT_MESSAGE marker**。下一轮模型收到的对话历史里看不到"用户打断了你"，会直接续上一轮 thinking 或 tool 失败的轨迹。
3. **`was_in_tool_call` 区分没用上**。即使 PR 6.2 设了这个字段，没人去读 → 没人去选 marker。

## 设计

### Marker 常量

新增 `runtime/interrupt_messages.py`（参考 `open-claude-code/src/utils/messages.ts` 207-213）：

```python
"""Marker strings appended to conversation history when a turn is interrupted.

Mirrors claude-code's ``INTERRUPT_MESSAGE`` / ``INTERRUPT_MESSAGE_FOR_TOOL_USE``
so the model treats interruption as a first-class event in the next turn.
"""

# 模型在思考 / 流响应时被打断
INTERRUPT_MESSAGE = "[Request interrupted by user]"

# 模型在执行工具时被打断 — 提示模型工具的副作用是不完整的
INTERRUPT_MESSAGE_FOR_TOOL_USE = "[Request interrupted by user for tool use]"


def select_interrupt_marker(*, was_in_tool_call: bool) -> str:
    return INTERRUPT_MESSAGE_FOR_TOOL_USE if was_in_tool_call else INTERRUPT_MESSAGE
```

两个 marker 的语义差异：

- `INTERRUPT_MESSAGE`：模型只是说话被打断；副作用为零。下一轮模型可以继续之前的思路或重新询问用户。
- `INTERRUPT_MESSAGE_FOR_TOOL_USE`：模型在跑 tool 时被打断；tool 的副作用**可能不完整**（写一半的文件、跑一半的 git commit）。模型应该先确认状态，不要 blindly 重试。

### 注入位置：engine vs repl

两种实现路径：

**Option A：engine 在 `_build_result` 里直接写到 `result.messages`**

```python
# agents/core/agent.py _build_result
def _build_result(self, ..., messages, exit_reason, ...) -> EngineResult:
    if exit_reason == ExitReason.INTERRUPTED:
        interrupt_meta = context.metadata.get("interrupt", {})
        partial = interrupt_meta.get("partial_text", "")
        was_in_tool = interrupt_meta.get("was_in_tool_call", False)

        # 1. partial assistant
        if partial.strip():
            messages.append({"role": "assistant", "content": partial})
        # 2. interrupt marker (作为 user message,模型下一轮看到)
        marker = select_interrupt_marker(was_in_tool_call=was_in_tool)
        messages.append({"role": "user", "content": marker})

    return EngineResult(messages=messages, ...)
```

**Option B：repl 在收到 result 后自己拼**

```python
# cli/repl.py _run_turn_blocking
if status_value == "INTERRUPTED":
    interrupt_meta = result.metadata.get("interrupt", {})
    partial = interrupt_meta.get("partial_text", "")
    was_in_tool = interrupt_meta.get("was_in_tool_call", False)

    if partial.strip():
        state.messages.append({"role": "assistant", "content": partial})
    marker = select_interrupt_marker(was_in_tool_call=was_in_tool)
    state.messages.append({"role": "user", "content": marker})
```

**选 A**。理由：

- 让 `EngineResult.messages` 始终自包含 ——任何 caller（CLI / 未来 SDK / 测试）都直接拿到完整对话状态，不需要复制相同逻辑。
- engine 已经有访问 `context.metadata["interrupt"]` 的权限，repl 拿不到 metadata 全貌。
- 减少 plumbing：repl 只需要照常 `state.messages = result.messages`。

副作用：现有 `engine.run_loop` 测试可能预期 INTERRUPTED 的 messages 不变，需要更新。

### `was_in_tool_call` 的判定（PR 6.3 InterruptSignal 之后的写法）

**新做法（推荐）**：注册一个**单一 listener** 在 `InterruptSignal` 上，由 listener
统一把当下的 `was_in_tool_call` / `partial_text` 写到 `context.metadata["interrupt"]`。
不需要在 run_loop 的三处分支各写一遍。

```python
# agents/core/agent.py 在 run_loop 进入时
context.metadata["_in_tool_call"] = False  # 状态机标志，下面切换

signal = request.interrupt_signal or self.services.interrupt_controller.signal(request.session_id)

def _record_interrupt_state(reason):
    """signal 触发的瞬间 fire — 同步在按 ESC 的线程跑。

    只做轻量 snapshot，不阻塞、不调用 engine 重活。
    """
    if "interrupt" in context.metadata:
        return  # 已被记录（双源中断），first-write-wins
    context.metadata["interrupt"] = {
        "reason": reason or "user-interrupt",
        "partial_text": str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""),
        "was_in_tool_call": bool(context.metadata.get("_in_tool_call", False)),
        "triggered_at": time.time(),
    }

unregister = signal.add_listener(_record_interrupt_state, once=True)

# 整个 turn 期间用 _in_tool_call 标记当前阶段
# - PRE_LLM / LLM_CALL / POST_LLM: False
# - TOOL_DISPATCH / TOOL_EXECUTE: True
# 切换在 state_machine.transition() 旁边一行：
state_machine.transition(LoopState.TOOL_EXECUTE)
context.metadata["_in_tool_call"] = True
# ... tool 跑完之后 ...
context.metadata["_in_tool_call"] = False

# turn 退出时
unregister()
```

**老做法（PR 6.2 已部分实施）**：在三个中断检测点（stream callback raise / tool result metadata / iteration boundary）各自写 `context.metadata["interrupt"]`。

旧做法的问题是**重复**且**易错** —— 加一个新的中断检测点就要记得也写 metadata。
新做法只有 **一个写入点**（listener），run_loop 各分支只需要标记 `_in_tool_call` 状态。

**保留**老做法作为 fallback：PR 6.2 已经在 `_invoke_provider_with_recovery` 里
catch `EngineInterrupted` 并写过 metadata；listener 检查"已存在则跳过"（first-write-wins）。
两种机制共存不冲突。

### 三个中断检测点（向后兼容老逻辑）

run_loop 在抛 / 检测中断的位置就要记下当时是不是在 tool 中：

```python
# Stream callback path (PR 6.2)
raise EngineInterrupted(
    reason="user-interrupt",
    partial_text=partial,
    was_in_tool_call=False,  # ← stream callback 时一定不在 tool 中
)

# Tool dispatch path (PR 6.3)
if result.metadata.get("interrupted"):
    context.metadata["interrupt"] = {
        "reason": "user-interrupt",
        "partial_text": "",
        "was_in_tool_call": True,  # ← 来自 tool result metadata
        "triggered_at": time.time(),
    }
    state_machine.transition(LoopState.INTERRUPTED)
    break

# Iteration boundary path (旧逻辑)
if self._is_interrupted(request.session_id):
    state_machine.transition(LoopState.INTERRUPTED)
    exit_reason = ExitReason.INTERRUPTED
    # 边界检查时不在任何具体操作里
    context.metadata["interrupt"] = {
        "reason": "user-interrupt",
        "partial_text": str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "")),
        "was_in_tool_call": False,
        "triggered_at": time.time(),
    }
    break
```

三处分别设 `was_in_tool_call`，最终通过 metadata 透传给 `_build_result`。

### Partial 的 cleaning

`partial_text` 可能含 `<thinking>` 块、不闭合的 tool tag、未完成的 markdown。注入 messages 之前要做基础清理（复用现有 `cli/ui.py` 的 `strip_tool_blocks`）：

```python
# 但 strip_tool_blocks 在 CLI 层，engine 不应该 import CLI
# 解决：把 strip_tool_blocks 提升到 runtime/text_utils.py
# 或者 engine 这边只做最小处理（trim、丢弃明显残缺）：

def _clean_partial_assistant(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    # 截掉末尾不闭合的 ``` 代码块
    if cleaned.count("```") % 2 == 1:
        cleaned = cleaned.rsplit("```", 1)[0].rstrip()
    # 截掉末尾不闭合的 <tool_call> 等 XML
    last_open = cleaned.rfind("<")
    if last_open >= 0 and ">" not in cleaned[last_open:]:
        cleaned = cleaned[:last_open].rstrip()
    return cleaned
```

激进点：直接调 `cli.ui.strip_tool_blocks`，但这就要把 `strip_tool_blocks` 抽到 runtime 共享模块。本 PR 用最小清理，后续可以做共享化重构。

### Empty partial 怎么办

如果 stream 一字符没产出就被打断（极快的 ESC），`partial.strip() == ""`，那只 append marker，不 append assistant message。模型看到的是 `[..., user_msg, [Request interrupted by user]]`，自然反应是"OK 你打断了，我没说什么"。

### 跨 turn 清理

InterruptController 的 flag 必须在下一轮开始前清掉，否则下一轮一开始 `_is_interrupted` 就是 True，立刻又退出。

现在的 `engine.clear_interrupt(session_id)` 只在显式调用时才清。需要确认：

- `_run_turn_blocking` 在 INTERRUPTED 路径之后是否调了 `clear_interrupt`？
  - 看代码：repl.py `_run_turn_blocking` 的 `except KeyboardInterrupt` 分支调了 `interrupt(...)`（设标志），但没看到调 `clear_interrupt(...)`。
- 应该在 `_run_turn_blocking` 的 `finally` 里：

```python
finally:
    try:
        state.engine.clear_interrupt(state.session_id)
    except Exception:
        pass
```

下一轮起调时 flag 是干净的。

## 文件改动

```
runtime/interrupt_messages.py  [新文件]
  + INTERRUPT_MESSAGE / INTERRUPT_MESSAGE_FOR_TOOL_USE
  + select_interrupt_marker(*, was_in_tool_call) -> str

runtime/__init__.py
  + 导出 INTERRUPT_MESSAGE / INTERRUPT_MESSAGE_FOR_TOOL_USE / select_interrupt_marker

agents/core/agent.py
  + _clean_partial_assistant(text) helper
  ~ _build_result: INTERRUPTED 分支注入 partial assistant + marker
  ~ 三处中断检测点（stream callback raise / tool result metadata / iteration boundary）
    都把 partial_text + was_in_tool_call 写到 context.metadata["interrupt"]

cli/repl.py
  ~ _run_turn_blocking finally: 加 engine.clear_interrupt(session_id)
  - 移除现有"INTERRUPTED 时 ui.warn('interrupted')" 之前的 partial 处理代码
    （改为 result.messages 已自带 partial + marker，repl 无脑用即可）

cli/ui.py
  ~ end_turn INTERRUPTED 路径：footer 上加一个 dim 提示
    "interrupted — context preserved"，告诉用户"已经把你刚才看到的内容
    存进对话了，下一轮继续就行"

tests/engine/test_interrupt_message_injection.py  [新文件]
tests/engine/test_interrupt_partial_preservation.py  [新文件]
```

## 实现细节

### `_build_result` 的修改（核心）

```python
# agents/core/agent.py
from aether.runtime.interrupt_messages import select_interrupt_marker

def _build_result(
    self,
    *,
    request: EngineRequest,
    context: TurnContext,
    messages: list[dict],
    exit_reason: ExitReason,
    iterations: int,
    final_response: NormalizedResponse | None,
) -> EngineResult:
    # ... 现有 metadata 装配 ...

    if exit_reason == ExitReason.INTERRUPTED:
        interrupt_meta = context.metadata.get("interrupt", {}) or {}
        partial = self._clean_partial_assistant(
            str(interrupt_meta.get("partial_text", "") or "")
        )
        was_in_tool = bool(interrupt_meta.get("was_in_tool_call", False))

        if partial:
            messages.append({"role": "assistant", "content": partial})

        marker = select_interrupt_marker(was_in_tool_call=was_in_tool)
        messages.append({"role": "user", "content": marker})

        # surface to result.metadata for caller
        result_metadata.setdefault("interrupt", {}).update({
            "marker": marker,
            "partial_assistant_chars": len(partial),
            "was_in_tool_call": was_in_tool,
        })

    return EngineResult(
        messages=messages,
        metadata=result_metadata,
        ...
    )
```

### Edge case: 同时多源中断

可能场景：stream 抛了 EngineInterrupted（partial="hello"），run_loop 准备走 INTERRUPTED 退出，**正在退出的过程中**又有一个 tool result 带 `interrupted=True` 进来。这种情况下 `context.metadata["interrupt"]` 会被覆盖。

策略：第一个写入的胜出（"已经被打断了，后面的事不算数"）：

```python
def _record_interrupt(context, *, partial_text, was_in_tool_call):
    if "interrupt" in context.metadata:
        return  # 已经记录过，保留首次
    context.metadata["interrupt"] = {
        "reason": "user-interrupt",
        "partial_text": partial_text,
        "was_in_tool_call": was_in_tool_call,
        "triggered_at": time.time(),
    }
```

三个中断检测点统一调 `_record_interrupt`，避免乱覆盖。

### Footer 提示

PR 6.1 的 ESC hint 配套：

```python
# cli/ui.py end_turn INTERRUPTED 分支
if status == "INTERRUPTED":
    line.append(f"{icon('interrupt')} interrupted", style=f"bold {AETHER_WARNING}")

    # NEW: tell user partial was preserved
    interrupt_meta = (...)  # 从 result 拿
    partial_chars = interrupt_meta.get("partial_assistant_chars", 0)
    if partial_chars > 0:
        line.append(*sep)
        line.append(
            f"context preserved ({partial_chars} chars)",
            style=f"dim {AETHER_DIM}",
        )
```

让用户知道"我刚才看到的输出没有被丢掉，模型下一轮还能用"。

## 参考实现

- `open-claude-code/src/utils/messages.ts` 行 207-209：marker 常量定义。
- `open-claude-code/src/utils/messages.ts` 行 545-560：`createUserInterruptionMessage({toolUse})` 选择 marker。
- `open-claude-code/src/screens/REPL.tsx` 行 2120-2130：partial-then-marker 注入顺序。
- `open-claude-code/src/components/messages/UserToolResultMessage/UserToolErrorMessage.tsx` 行 33：`if param.content.includes(INTERRUPT_MESSAGE_FOR_TOOL_USE)` —— UI 把 marker 渲染成专门的"interrupted"卡片，而不是普通 user message。本 PR 不做 UI 卡片化（保持和普通 user message 一样渲染就行）。

## 测试

`tests/engine/test_interrupt_message_injection.py`

```python
class InterruptMessageInjectionTests(unittest.TestCase):
    def test_marker_is_for_tool_use_when_interrupted_in_tool(self):
        # Build engine where tool returns metadata["interrupted"]=True
        # Run loop → result.messages last item is INTERRUPT_MESSAGE_FOR_TOOL_USE
        ...

    def test_marker_is_plain_when_interrupted_in_stream(self):
        # Build engine where stream callback raises EngineInterrupted
        # (was_in_tool_call=False)
        # Run loop → result.messages last item is INTERRUPT_MESSAGE
        ...

    def test_marker_at_iteration_boundary_is_plain(self):
        # Set interrupt flag between iterations (no stream/tool active)
        # Result.messages last item is INTERRUPT_MESSAGE (not _FOR_TOOL_USE)
        ...

    def test_no_partial_assistant_when_partial_text_empty(self):
        # Interrupt before any stream chunk
        # Result.messages does NOT contain an empty assistant message
        # Only the interrupt marker
        ...

    def test_double_interrupt_preserves_first(self):
        # 1. Trigger stream interrupt (partial="hello")
        # 2. Before run_loop exits, simulate tool also returning interrupted=True
        # Result: partial="hello", was_in_tool_call=False (first write wins)
        ...


class InterruptPartialPreservationTests(unittest.TestCase):
    def test_partial_assistant_appended_before_marker(self):
        # Stream "I think the answer", interrupt
        # result.messages tail: [..., {"role": "assistant", "content": "I think the answer"},
        #                       {"role": "user", "content": INTERRUPT_MESSAGE}]
        ...

    def test_partial_with_unclosed_codeblock_is_trimmed(self):
        # Stream "Here's the code:\n```python\ndef foo():" → interrupt
        # Cleaned partial does not contain ```python (trim unclosed fence)
        ...

    def test_partial_with_partial_xml_tag_is_trimmed(self):
        # Stream "Let me try <tool_call" → interrupt
        # Cleaned partial does not contain "<tool_call"
        ...


class InterruptClearedAcrossTurnsTests(unittest.TestCase):
    def test_next_turn_starts_clean_after_interrupt(self):
        # 1. Run turn 1, interrupt mid-stream
        # 2. Start turn 2 (don't manually clear flag)
        # 3. Turn 2 should NOT immediately exit with INTERRUPTED
        # repl._run_turn_blocking finally block must call clear_interrupt
        ...
```

## 验收门

- 所有新测试通过。
- 手测：跑长回复 → ESC → 下一轮发"你刚才说到一半，继续"，模型能基于 partial 内容回应。
- 手测：跑 `pytest` → ESC → 下一轮发"上一个测试结果如何"，模型回应类似"我跑到一半被打断，可能不准"，而不是说"我没跑过 pytest"。
- 现有 `test_engine.py` 等 INTERRUPTED 路径相关测试更新但功能不回归。
- `EngineResult.metadata["interrupt"]` shape 跟 PR 6.2 设计一致（marker 字段是 PR 6.4 新增）。

## 不在本 PR 内（留给后续）

- `INTERRUPT_MESSAGE_FOR_TOOL_USE` 在 UI 上渲染成专门的卡片（与普通 user message 区分）—— 当前用通用 user echo 即可，未来可加 styled rendering。
- 双击 ESC 真正触发 `/clear` 命令的 wiring（PR 6.1 留了 stub，本 PR 也不依赖）—— 由 PR 6.5 收尾。
- 把"刚才被打断的 turn"标到 `EngineResult.metadata` 的 telemetry / debug log —— 可选 polish。
