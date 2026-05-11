# Sprint 6 Interrupt Overhaul — Acceptance Matrix

> **架构修订**：Sprint 6 中段引入 `InterruptSignal`（参考 open-claude-code
> AbortSignal 思想）后，原 5 个 PR 扩展为 7 个。PR 6.1 / 6.2 保留旧实现（通过
> wrapper 向后兼容），PR 6.3 起改用事件驱动模型。详见 `00_overview.md` 的
> "架构演进"小节。

## PR 验收矩阵

| PR | 状态 | 必须通过的专项测试 | 必须确认的回归点 |
|---|---|---|---|
| 6.1 ESC binding + priority chain | ✅ 已合 | ESC 5 个优先级分支、Ctrl-C 双击退出窗口、busy footer hint | `/help`/`/model` 等 slash 命令照常；Shift+Enter / Ctrl-J 换行 |
| 6.2 Stream interrupt propagation | ✅ 已合 | stream callback 抛 EngineInterrupted、partial 透传、provider wrapper 不吞、< 500ms latency、metadata shape | 现有 `test_engine.py` / `test_streaming*.py` 不回归；`EngineResult.exit_reason == INTERRUPTED` |
| 6.3 InterruptSignal infra + 事件驱动 subprocess | 待做 | `InterruptSignal` 单元 22+、`InterruptController` 兼容性、shell tool POSIX 4+ + 集成 1 | shell tool 现有 timeout / cwd / FileNotFoundError 路径不回归；PR 6.1 / 6.2 的 `_is_interrupted` 调用点透明 |
| 6.4 Partial preservation + INTERRUPT_MESSAGE | 待做 | marker 选择（tool/non-tool）、partial 注入顺序、空 partial 跳过、unclosed 代码块清洗、listener 写 metadata、跨 turn flag 清理 | `state.messages` shape 调用方兼容；现有 INTERRUPTED 路径单测更新但语义不变 |
| 6.5 Polish (clear / refresh / hint / interrupt_behavior / log) | 待做 | `clear_history` 三处状态清理、Ctrl-C toast、`/refresh`、footer queued empty 不显示 badge、tool `interrupt_behavior` block vs cancel | 既有 slash 命令、resize 残影修复；默认 `"block"` 不破坏 ms 级 tool |
| 6.6 Subagent 共享父 InterruptSignal | 待做（新） | 父子 / 嵌套 / async 隔离单元 4+、parent abort → subagent stream stop < 500ms、parent abort → subagent subprocess kill < 700ms | 现有 Task tool / AgentTool 测试不回归；async background subagent 不被父中断 |
| 6.7 HTTP / WebFetch 事件驱动取消 | 待做（新） | `httpx_client_aware_of` 单元 5+、WebFetch 集成中断 < 300ms | 现有 WebFetch / WebSearch / MCP 测试不回归 |

## 推荐测试命令

从 `backend/harness/aether` 执行：

```bash
# 单 PR 跑：
python -m pytest tests/cli/test_cli_app.py -v                          # PR 6.1 + 6.5 部分
python -m pytest tests/runtime/test_interrupt_propagation.py -v        # PR 6.2
python -m pytest tests/runtime/test_interrupt_signal.py \
                tests/runtime/test_interrupt_controller_compat.py \
                tests/tools/test_shell_interrupt.py -v                 # PR 6.3
python -m pytest tests/engine/test_interrupt_message_injection.py \
                tests/engine/test_interrupt_partial_preservation.py -v # PR 6.4
python -m pytest tests/engine/test_clear_history.py \
                tests/cli/test_cli_commands.py -v                      # PR 6.5
python -m pytest tests/runtime/test_subagent_interrupt.py -v           # PR 6.6
python -m pytest tests/runtime/test_interrupt_helpers.py \
                tests/tools/test_web_fetch_interrupt.py -v             # PR 6.7

# 全量回归（每个 PR 合并前都跑）：
python -m pytest tests/cli tests/runtime tests/engine tests/tools
```

## 端到端验收场景

按"用户视角的打断体验"逐项验证：

### 流式输出中按 ESC（PR 6.1 + 6.2）

1. 让模型写一篇 1000 字回复（`请你详细解释 Aether 的架构和核心设计`）。
2. 流到 1/3 时按 ESC。
3. **预期**：
   - < 500ms 内流停下。
   - footer 出现 `⏹ interrupted · context preserved (~XXX chars)`（PR 6.4 后）。
   - 输入框可立即接受新输入。
4. 输入"刚才说到 middleware 那部分，继续"。
5. **预期**：模型不会从头说起，而是基于打断时已经写出的内容继续。

### Tool 执行中按 ESC（PR 6.3 + 6.5）

1. 让模型跑长命令：`帮我用 pytest 跑一下 backend/harness/aether/tests/`。
2. pytest 跑了一半时按 ESC。
3. **预期**：
   - < 500ms 内 pytest 进程被 SIGTERM（grandchild 也死）。listener 同步触发，无 200ms polling 间隔。
   - footer 出现 `⏹ interrupted`。
   - 对话历史里能看到 partial pytest 输出 + `[Request interrupted by user for tool use]` marker。
4. 输入"刚才打断了，那些已经跑过的测试结果是什么样"。
5. **预期**：模型基于看到的 partial output 回应，且承认"我中途被打断、不完整"。

### Subagent / Task tool 中按 ESC（PR 6.6 — 新）

1. 让模型用 Task tool 跑长任务：`用 Explore agent 读一遍 backend/harness/aether 下所有 .py 文件并总结架构`。
2. subagent 跑了几秒后按 ESC。
3. **预期**：
   - < 500ms 内 subagent stream 停下。
   - 如果 subagent 正在跑 shell tool（比如 `ripgrep`），subprocess 也 < 700ms 内被 kill。
   - 父 turn 也立即停下，回到 prompt。
4. 输入"刚才让你研究的事，你看到哪了"。
5. **预期**：模型回应"我开始读 X / Y / Z，被打断时进展到 Z"。

### WebFetch / 慢响应 endpoint 按 ESC（PR 6.7 — 新）

1. 让模型 `WebFetch` 一个故意慢的 URL（或一个实际很慢的 endpoint）。
2. 2-3s 内按 ESC。
3. **预期**：
   - < 300ms 内 httpx client 被 close，请求立即返回。
   - 对话历史里看到 `[fetch interrupted by user before response]`。

### Idle 时按 ESC（输入框非空）— PR 6.1

1. 没有 turn 在跑，输入框敲了"asdf"但没回车。
2. 按 ESC。
3. **预期**：输入框被清空，没有任何 turn 触发。

### Idle 时按 ESC（有 queued 命令）— PR 6.1

1. 一个 turn 在跑，回车提交"再看一眼"（被加进 queue 显示 `▸ queued (1)`）。
2. ESC（触发优先级 1：interrupt 当前 turn + 清 queue）。
3. **预期**：queue 数清 0，turn 中断。

### Idle 双击 ESC 清对话 — PR 6.1 + 6.5

1. 跑过几轮对话，scroll 上去能看到历史。
2. 输入框空，按 ESC。
3. **预期**：toast "Press ESC again to clear conversation"。
4. 800ms 内再按 ESC。
5. **预期**：执行 `/clear`，scrollback 还在但 `state.messages` 清空 + session id 保留。

### 双击 Ctrl-C 退出 — PR 6.1 + 6.5

1. Idle，输入框空，按 Ctrl-C。
2. **预期**：toast "Press Ctrl-C again to exit"。
3. 2s 内再按 Ctrl-C。
4. **预期**：进程退出，"Bye." 出现。
5. 反向：按一次 Ctrl-C 然后 sleep 3s 再按一次。
6. **预期**：第二次按属于"新的第一次"，又显示 toast，不退出。

### `/refresh` 命令 — PR 6.5

1. 拖拽 terminal 边界产生残影。
2. 输入 `/refresh`。
3. **预期**：可见区清屏并重画，scrollback 上的内容仍可滚回。

### Async subagent 不被父中断 — PR 6.6

1. 用 Task tool 启动 background subagent（`Run in background`）。
2. 父 turn 还在跑，立即按 ESC 取消父。
3. **预期**：父 turn 停下，但 background subagent 继续跑（独立生命周期）。
4. 用 `/tasks` 或 `TaskOutput` 查看 background subagent，应该还在 running 或自然完成。

### "Block" tool 不被 ESC 中断 — PR 6.5

1. 让模型用 `Read` 读一个稍大的文件（几 MB）。
2. 即使读到一半按 ESC，`Read` 也应该跑完返回（因为 `interrupt_behavior = "block"`）。
3. **预期**：tool result 正常返回内容，no `metadata["interrupted"]` flag。
4. 但 **下一轮 turn 不开始**（顶层 turn 已 INTERRUPTED 路径退出）。

### 中断后下一轮干净

1. 触发任意中断。
2. 立刻输入"hello" 回车。
3. **预期**：新 turn 正常启动（`is_interrupted` flag 已被 `clear_interrupt` 清，不会立刻又退出）。

### 打断响应延迟测量

| 场景 | 测试位置 | 目标 |
|---|---|---|
| Stream callback 打断 | `tests/runtime/test_interrupt_propagation.py::test_interrupt_latency_under_500ms` | < 500ms |
| Shell subprocess 打断 | `tests/tools/test_shell_interrupt.py::test_long_sleep_interrupted_within_500ms` | < 500ms |
| Subagent 全链打断 | `tests/runtime/test_subagent_interrupt.py::test_parent_interrupt_stops_subagent_stream_within_500ms` | < 500ms |
| WebFetch 打断 | `tests/tools/test_web_fetch_interrupt.py` | < 300ms |

### Engine result metadata 检查

打断后用 `/stats` 或调试查看 `EngineResult.metadata["interrupt"]`：

```python
{
    "reason": "user-interrupt",
    "partial_text": "...",       # PR 6.2 透传，PR 6.4 后还有
    "was_in_tool_call": True,    # 视场景；PR 6.4 listener 准确写入
    "triggered_at": 1715423456.789,
    "marker": "[Request interrupted by user for tool use]",  # PR 6.4 新加
    "partial_assistant_chars": 234,                          # PR 6.4 新加
}
```

## Sprint 完成定义

- 7 个 PR 全部 merge（6.1 / 6.2 已合，6.3 / 6.4 / 6.5 / 6.6 / 6.7 待做），对应文档实施完毕。
- 每个 PR 至少 1 个新增测试文件 + 必要的现有测试更新。
- 全量 `tests/cli` + `tests/runtime` + `tests/engine` + `tests/tools` 通过。
- 手测端到端 12 个场景全部 OK（原 9 个 + subagent + async + block-tool）。
- `docs/run-loop-roadmap/` 中"打断 / cancel"相关条目可以勾掉。
- **架构里程碑**：`InterruptSignal` 成为 Aether 的标准 cancellation 原语；新 long-running tool 作者按 `interrupt_behavior = "cancel"` + `signal.add_listener(close_resource)` 模板照着写即可。

## 后续 Sprint 建议（不在本 sprint 内）

- **MCP 应用层 `notifications/cancelled`**：PR 6.7 只做了 transport 层 close client；告诉远端 server "我不要这个 request 的结果了"留给后续 PR（看 MCP server 生态支持率）。
- **`INTERRUPT_MESSAGE_FOR_TOOL_USE` UI 卡片**：当前会以普通 user message 渲染（`> [Request interrupted by user for tool use]`），可以做成 styled "⏹ interrupted" 卡片，跟 codex 视觉一致。
- **打断 telemetry**：把 PR 6.5 的 debug log 升级成结构化 trajectory 记录（依赖 sprint 5 的 trajectory 能力），便于事后 replay。
- **Permission denial / tool reject 的 marker 系列**：claude-code 还有 `REJECT_MESSAGE` / `PLAN_REJECTION_PREFIX` 等专门 marker。我们当前没有 plan mode / permission UI，等那些功能引入后一并对齐。
- **Async subagent 单独 cancel UI**：让用户能单独 kill 一个 background subagent（命令、UI 入口），跟 Task tool UI 改造一起做。
- **`asyncio.AsyncClient` 路径**：如果未来把 tool 改成 async（比如同时跑多个 HTTP），现在的同步 `client.close()` 替换为 `signal.add_listener(lambda r: task.cancel())`。
