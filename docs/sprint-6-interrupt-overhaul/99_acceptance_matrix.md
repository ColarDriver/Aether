# Sprint 6 Interrupt Overhaul — Acceptance Matrix

## PR 验收矩阵

| PR | 必须通过的专项测试 | 必须确认的回归点 |
|---|---|---|
| 6.1 ESC binding + priority chain | ESC 5 个优先级分支、Ctrl-C 双击退出窗口、busy footer hint | `Esc+Enter` 换行、`c-j` 备用换行、 `/help`/`/model` 等 slash 命令照常 |
| 6.2 Stream interrupt propagation | stream callback 抛 EngineInterrupted、partial 透传、provider wrapper 不吞、< 200ms latency、metadata shape | 现有 `test_engine.py` / `test_streaming*.py` stream 路径不回归；`EngineResult.exit_reason == INTERRUPTED` |
| 6.3 Subprocess interrupt | sleep 30 中断 < 700ms、partial stdout 保留、grandchild 一并 kill、normal completion 不打 `interrupted` flag | shell tool 现有 timeout / cwd / FileNotFoundError 路径不回归；其他 tool 不受影响 |
| 6.4 Partial preservation + INTERRUPT_MESSAGE | marker 选择（tool/non-tool）、partial 注入顺序、空 partial 跳过、unclosed 代码块清洗、跨 turn flag 清理 | `state.messages` shape 调用方兼容；现有 INTERRUPTED 路径单测更新但语义不变 |
| 6.5 Polish (clear / refresh / hint / log) | `clear_history` 三处状态清理、Ctrl-C toast、`/refresh`、footer queued empty 不显示 badge | 既有 slash 命令、resize 残影修复 |

## 推荐测试命令

从 `backend/harness/aether` 执行：

```bash
# 单 PR 跑：
python -m pytest tests/cli/test_cli_app.py -v               # PR 6.1 + 6.5 部分
python -m pytest tests/runtime/test_interrupt_propagation.py -v  # PR 6.2
python -m pytest tests/tools/test_shell_interrupt.py -v     # PR 6.3
python -m pytest tests/engine/test_interrupt_message_injection.py \
                tests/engine/test_interrupt_partial_preservation.py -v  # PR 6.4
python -m pytest tests/engine/test_clear_history.py \
                tests/cli/test_cli_commands.py -v           # PR 6.5

# 全量回归（每个 PR 合并前都跑）：
python -m pytest tests/cli tests/runtime tests/engine tests/tools
```

## 端到端验收场景

按"用户视角的打断体验"逐项验证：

### 流式输出中按 ESC

1. 让模型写一篇 1000 字回复（`请你详细解释 Aether 的架构和核心设计`）。
2. 流到 1/3 时按 ESC。
3. **预期**：
   - < 500ms 内流停下。
   - footer 出现 `⏹ interrupted · context preserved (~XXX chars)`。
   - 输入框可立即接受新输入。
4. 输入"刚才说到 middleware 那部分，继续"。
5. **预期**：模型不会从头说起，而是基于打断时已经写出的内容继续。

### Tool 执行中按 ESC

1. 让模型跑长命令：`帮我用 pytest 跑一下 backend/harness/aether/tests/`。
2. pytest 跑了一半时按 ESC。
3. **预期**：
   - < 700ms 内 pytest 进程被 SIGTERM（grandchild 也死）。
   - footer 出现 `⏹ interrupted`。
   - 对话历史里能看到 partial pytest 输出 + `[Request interrupted by user for tool use]` marker。
4. 输入"刚才打断了，那些已经跑过的测试结果是什么样"。
5. **预期**：模型基于看到的 partial output 回应，且承认"我中途被打断、不完整"。

### Idle 时按 ESC（输入框非空）

1. 没有 turn 在跑，输入框敲了"asdf"但没回车。
2. 按 ESC。
3. **预期**：输入框被清空，没有任何 turn 触发。

### Idle 时按 ESC（有 queued 命令）

1. 一个 turn 在跑，回车提交"再看一眼"（被加进 queue 显示 `▸ queued (1)`）。
2. ESC（触发优先级 1：interrupt 当前 turn + 清 queue）。
3. **预期**：queue 数清 0，turn 中断。
4. （新场景）：连续提交两条 queued，**只 ESC 一次**让当前 turn 完成。
5. **预期**：完成后剩下两条 queued 自动跑（这跟 6.1 优先级链有关 —— ESC 在 busy 时不应该清后面 queued）。
   - **或者**：根据 PR 6.1 的设计，ESC busy 时也清 queue（claude-code 风格）。具体行为以 PR 6.1 实现为准，这里测试要 pin 住选择。

### Idle 双击 ESC 清对话

1. 跑过几轮对话，scroll 上去能看到历史。
2. 输入框空，按 ESC。
3. **预期**：toast "Press ESC again to clear conversation"。
4. 800ms 内再按 ESC。
5. **预期**：执行 `/clear`，scrollback 还在但 `state.messages` 清空 + session id 保留。

### 双击 Ctrl-C 退出

1. Idle，输入框空，按 Ctrl-C。
2. **预期**：toast "Press Ctrl-C again to exit"。
3. 2s 内再按 Ctrl-C。
4. **预期**：进程退出，"Bye." 出现。
5. 反向：按一次 Ctrl-C 然后 sleep 3s 再按一次。
6. **预期**：第二次按属于"新的第一次"，又显示 toast，不退出。

### `/refresh` 命令

1. 拖拽 terminal 边界产生残影。
2. 输入 `/refresh`。
3. **预期**：可见区清屏并重画，scrollback 上的内容仍可滚回。

### 中断后下一轮干净

1. 触发任意中断。
2. 立刻输入"hello" 回车。
3. **预期**：新 turn 正常启动（`is_interrupted` flag 已被 `clear_interrupt` 清，不会立刻又退出）。

### 打断响应延迟测量

`tests/runtime/test_interrupt_propagation.py::test_interrupt_response_latency_under_200ms` 自动测；手测可以在 footer 加临时 timing 输出验证。

### Engine result metadata 检查

打断后用 `/stats` 或调试查看 `EngineResult.metadata["interrupt"]`：

```python
{
    "reason": "user-interrupt",
    "partial_text": "...",       # PR 6.2 透传，PR 6.4 后还有
    "was_in_tool_call": True,    # 视场景
    "triggered_at": 1715423456.789,
    "marker": "[Request interrupted by user for tool use]",  # PR 6.4 新加
    "partial_assistant_chars": 234,                          # PR 6.4 新加
}
```

## Sprint 完成定义

- 5 个 PR 全部 merge，对应文档实施完毕。
- 每个 PR 至少 1 个新增测试文件 + 必要的现有测试更新。
- 全量 `tests/cli` + `tests/runtime` + `tests/engine` + `tests/tools` 通过。
- 手测端到端 9 个场景全部 OK。
- `docs/run-loop-roadmap/` 中"打断 / cancel"相关条目可以勾掉。

## 后续 Sprint 建议（不在本 sprint 内）

- **Subagent 中断传播**：当前 `_interrupt_active_children` 只设标志，不传播到 subagent 的 stream callback / subprocess。等本 sprint 这套 pattern 在主 engine 跑稳之后，把同样的 PR 6.2 + 6.3 改造做到 subagent 上。
- **`INTERRUPT_MESSAGE_FOR_TOOL_USE` UI 卡片**：当前会以普通 user message 渲染（`> [Request interrupted by user for tool use]`），可以做成 styled "⏹ interrupted" 卡片，跟 codex 视觉一致。
- **打断 telemetry**：把 PR 6.5 的 debug log 升级成结构化 trajectory 记录（依赖 sprint 5 的 trajectory 能力），便于事后 replay。
- **Permission denial / tool reject 的 marker 系列**：claude-code 还有 `REJECT_MESSAGE` / `PLAN_REJECTION_PREFIX` 等专门 marker。我们当前没有 plan mode / permission UI，等那些功能引入后一并对齐。
