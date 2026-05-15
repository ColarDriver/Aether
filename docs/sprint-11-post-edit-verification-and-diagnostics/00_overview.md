# Sprint 11 — Post-Edit Verification & Diagnostics Loop (Overview)

## 背景 / Motivation

用户反馈：Aether 自研 agent 写完代码后，常出现 *undefined-variable / unused-import / 类型错误*，而它**完全没有"写完之后再检查一遍"的步骤**。同样的 prompt 在 `open-claude-code` 那边几乎不会留下编译期错误，因为它在每次 `file_edit` / `write_file` 后会自动把 LSP 诊断结果灌回下一轮，模型必须把诊断清零才能宣布"完成"。

参考实现：`/workspace/open-claude-code`。本 sprint 对齐它的 **post-edit verification loop** —— 一条贯穿 *工具 → LSP → 诊断 tracker → attachment → system prompt* 的闭环。

### Gap Matrix

| # | 现象 | 根因（带文件:行号） |
|---|---|---|
| **G1** | 写完代码后模型看不到 LSP 报错 | `aether/tools/builtins/file_edit.py` / `write_file.py` 完全不调用任何 LSP/诊断接口；返回值只有 diff 文本 |
| **G2** | `LSPManager` 已存在但只能被模型当成工具显式调（要求模型主动 `lsp` ） | `aether/tools/builtins/lsp.py:38,94`；`runtime/resources/lsp_manager.py:42` 仅作 lazy LSPClient pool，写操作触发不了 `didChange` / `didSave` |
| **G3** | `EngineHooks` 不暴露 tool 边界 | `aether/runtime/core/hooks.py` 仅有 `pre_llm_call` / `post_llm_call` / `pre_api_request` / `post_api_request` / `on_session_*` —— 没有 `post_tool_use` / `post_tool_use_failure` |
| **G4** | `middleware.after_tool` 只能改 ToolResult，无法把后续轮次的 prompt 增强 | `aether/agents/middlewares/base.py:24`；仅 `tool_error_handling.py` 用到，仅为格式化错误 |
| **G5** | System prompt 没有"完成前必须验证"的硬约束 | `aether/agents/core/system_prompt.py` 64 行总长，只注入 `<tool_use_contract>`，无 verification 段落 |
| **G6** | 非平凡变更也不需独立 verifier subagent PASS | `aether/agents/types/`（Sprint 10 落地）暂无 `Verifier` 类型；`AgentTool` 不强制 spawn |
| **G7** | `DefaultSubagentBuilder` 不携带 diagnostic tracker，子 agent 看不到自己改动产生的报错 | `aether/subagents/default_builder.py:53-69` 没有把 tracker 透传 |

### 对照参考 / Reference Behavior

`open-claude-code` 的 closed-loop 长这样：

| 步骤 | 文件:行 |
|---|---|
| 1. 编辑前快照诊断基线 | `src/tools/FileEditTool/FileEditTool.ts:425`；`src/tools/FileWriteTool/FileWriteTool.ts:247` → `diagnosticTracker.beforeFileEdited()` |
| 2. 通知 LSP 新内容 + 触发再分析 | `FileEditTool.ts:508`；`FileWriteTool.ts:313,320` → `lspManager.changeFile()` / `lspManager.saveFile()` |
| 3. 用 mtime + hash 跟踪 file state，防 stale write | 同上文件，`readFileState` map |
| 4. 拉新诊断、与基线 diff，只留**新增**条目 | `src/services/diagnosticTracking.ts:188-283` → `getNewDiagnostics()` |
| 5. 下一轮 LLM 调用前把新诊断作为 attachment 注入 | `src/utils/attachments.ts:956` → `getDiagnosticAttachments(toolUseContext)` |
| 6. `PostToolUse` / `PostToolUseFailure` 生命周期 hook | `src/types/hooks.ts:101,109`；`src/Tool.ts:476` |
| 7. System prompt 硬约束 "完成前必须验证" | `src/constants/prompts.ts:211` |
| 8. 非平凡变更必须经独立 verifier subagent PASS | `src/constants/prompts.ts:394` |

## 设计原则 / Design Principles

1. **复用 Sprint 9 / 10 的骨架** —— `LSPManager` / `LSPClient` / `SubagentBuilder` / `EngineHooks` / `TurnContext` 全部不动 schema；只补缺的回路。
2. **不阻塞模型主流程** —— LSP 调用全部 fire-and-forget（带超时回退），永远不能让一次诊断采集把 turn 卡死；参考 `FileEditTool.ts:493-514` 的 `.catch(err => …)` 写法。
3. **死代码不留** —— `aether/runtime/resources/lsp_manager.py` 此前几乎只在 `LSPTool` 内被消费，本 sprint 必须让"自动诊断采集"成为它的真正主路径，否则两条路径并存会很快烂。
4. **wire schema 只加不改** —— `tool.result` 仍是 string；diagnostics 走 attachment（user-role message）通道，与现有 `<system-reminder>` 同款，前端不需要新事件。
5. **subagent 一视同仁** —— `DefaultSubagentBuilder` 必须继承 tracker，否则 Sprint 10 落地的 Explore / general-purpose / verifier 子 agent 会出现"父见诊断子不见"的诡异不对称。
6. **`<diagnostics>` 块是 system-reminder 的同胞** —— 在 prompt cache prefix 之外、用户 turn 之前的位置注入，与 `open-claude-code/src/utils/messages.ts:3728-3738` 的 `wrapMessagesInSystemReminder` 行为对齐。
7. **没有 LSP 不报错** —— 当工程不带语言服务器（如纯 markdown 仓库）时整条 path 退化为 no-op，绝不能阻塞写操作。
8. **prompt 比代码先行** —— PR 11.1 是纯 prompt 变更，零代码风险且能带来肉眼可见的行为变化，应该最先合入并跑一周 A/B 观察。

## Sub-PR 路线图 / Roadmap

按依赖顺序：

```
PR 11.1 (system prompt directive) ── 独立，最小，可第一个合入

PR 11.2 (post_tool_use hook)       ──┐
                                       ├─→ PR 11.4 (edit tools wire LSP + tracker) ──┐
PR 11.3 (DiagnosticTracker)        ──┘                                                 │
                                                                                        ├─→ PR 11.5 (attachment injection)
                                                                                        │
PR 11.6 (verifier subagent)        ── 消费 Sprint 10 类型注册表 + 本 sprint 11.1 / 11.5
```

| PR | 标题 | 主要文件（新建/修改） | Gap |
|---|---|---|---|
| 11.1 | System Prompt: Verification Directive | `aether/agents/core/system_prompt.py`, `aether/config/schema.py` | G5 |
| 11.2 | `post_tool_use` Hook 上 Engine | `aether/runtime/core/hooks.py`, `aether/agents/core/agent.py` | G3, G4 |
| 11.3 | DiagnosticTracker Service | `aether/runtime/diagnostics/`(NEW dir, 3 files), `runtime/resources/lsp_manager.py`(extend) | G2 |
| 11.4 | Edit Tools 接入 LSP + Tracker | `tools/builtins/file_edit.py`, `tools/builtins/write_file.py`, `tools/builtins/notebook_edit.py` | G1, G2 |
| 11.5 | Diagnostic Attachments 注入下一轮 | `aether/runtime/diagnostics/attachments.py`(NEW), `aether/agents/core/agent.py`, `subagents/default_builder.py` | G1, G7 |
| 11.6 | Verifier Subagent Type | `aether/agents/types/builtin/verifier.py`(NEW), `agent_tool.py`, `constants/prompts.py` | G6 |

每个 PR 都对应本目录中一个独立文档，包含 **目标 / 当前问题 / 改动 / 测试 / 验收** 五节。

## Out of Scope（延后到 Sprint 12+）

- **Pyright / mypy 多进程 worker pool** —— 本 sprint 只对接每语言一个 `pyright-langserver` / `tsserver` 进程；并发 worker 留给后续性能优化。
- **第三方 linter（ruff / eslint）作为独立 channel** —— 现阶段诊断只走 LSP；如要把 ruff 单独拉成 channel，独立立项。
- **诊断 → 自动修复（auto-fix）** —— 只把诊断 *暴露给模型*，让模型自己改；不做 quick-fix patch 自动应用（与 `open-claude-code` 一致）。
- **TUI 端的 "Diagnostics panel"** —— 诊断流仅在上下文里，TUI 不显式渲染；如要做面板留给 Sprint 12+。
- **跨 worktree 的诊断** —— PR 11.5 仅在 child engine 自己的 working dir 内采集；当 child 跑在 worktree 里（Sprint 10.8）时，tracker 也只看 worktree 内的文件，主 tree 的诊断不冒泡到父。

## 验收 / Acceptance（汇总见 `99_acceptance_matrix.md`）

- 端到端：12 个场景全部通过（覆盖 prompt directive、LSP 触发、诊断采集、attachment 注入、verifier 子 agent gate、subagent 透传）
- 单元测试：`uv run pytest aether/tests/` 全绿，新增模块 `aether/tests/runtime/diagnostics/` 覆盖 ≥ 90%
- 类型检查：`uv run pyright` 零新增告警
- 手测：在 Python 仓库中 prompt "rename `foo` to `bar` in this file"，制造一个 NameError → 下一轮 LLM 调用前的 messages 末尾必须含 `<diagnostics>` 块，且模型主动发出修复
- 手测：在 TypeScript 仓库中 prompt "delete the import of `useState`"，模型必须能看到 *未使用 / 未定义* 的 tsserver 诊断
- 性能：`beforeFileEdited` + 一次 `changeFile` + `saveFile` + `getNewDiagnostics` 一轮 round-trip ≤ 500ms（在 50KB 文件上）
