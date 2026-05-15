# Sprint 11 — Post-Edit Verification & Diagnostics Loop

把 Aether 的"写完代码后是否检查"这条链路对齐到 `open-claude-code` 的成熟实现：让 `file_edit` / `write_file` 在工具成功返回之后**自动**完成 *LSP didChange → didSave → getDiagnostics → diff baseline → 注入下一轮 prompt* 的闭环；并通过 system prompt 与 verifier subagent 双重 gate，让模型在非平凡变更上无法绕过验证。

参考实现：`/workspace/open-claude-code`（`src/tools/FileEditTool/`, `src/tools/FileWriteTool/`, `src/services/diagnosticTracking.ts`, `src/services/lsp/`, `src/utils/attachments.ts:956`, `src/constants/prompts.ts:211,394`）。

## 用户痛点 / Symptom

> "我们的 agent 写完代码后经常出现 undefined-variable / unused-import / 类型错误，看起来它没有写完之后再验证一遍的步骤。"

—— 复现路径：在任意 Python 工程里 `task "rename foo to bar in module X"`，agent 完成后 `pyright` / 解释器立即报 `NameError: foo is not defined`。Aether 工具链此时**没有**任何回路把这个错误送回模型。

## 文档索引 / Table of Contents

| # | 文档 | 内容 |
|---|---|---|
| 0 | [`00_overview.md`](./00_overview.md) | 背景、Gap matrix、对照参考、设计原则、Sub-PR 路线图 |
| 1 | [`01_pr11_1_system_prompt_verification_directive.md`](./01_pr11_1_system_prompt_verification_directive.md) | system prompt 增加 verification + faithful-reporting 段（G5） |
| 2 | [`02_pr11_2_post_tool_use_hook.md`](./02_pr11_2_post_tool_use_hook.md) | `EngineHooks` 暴露 `post_tool_use` / `post_tool_use_failure`（G3, G4） |
| 3 | [`03_pr11_3_diagnostic_tracker.md`](./03_pr11_3_diagnostic_tracker.md) | `DiagnosticTracker`：baseline → diff → new diagnostics（G2） |
| 4 | [`04_pr11_4_edit_tools_lsp_wire.md`](./04_pr11_4_edit_tools_lsp_wire.md) | `file_edit` / `write_file` / `notebook_edit` 自动触发 LSP + tracker（G1, G2） |
| 5 | [`05_pr11_5_diagnostic_attachments.md`](./05_pr11_5_diagnostic_attachments.md) | 下一轮 LLM 调用前把新诊断注入为 `<diagnostics>` user-role 消息（G1, G7） |
| 6 | [`06_pr11_6_verifier_subagent_type.md`](./06_pr11_6_verifier_subagent_type.md) | 内置 `Verifier` subagent type + 父 prompt gate（G6） |
| 99 | [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 端到端 / 单元 / 类型 / 遥测验收矩阵 |

## PR 依赖图 / Dependency Graph

```
PR 11.1 (system prompt directive) ── 独立，最小，可第一个合入

PR 11.2 (post_tool_use hook)       ──┐
                                       ├─→ PR 11.4 (edit tools wire LSP + tracker) ──┐
PR 11.3 (DiagnosticTracker)        ──┘                                                 │
                                                                                        ├─→ PR 11.5 (attachment injection)
                                                                                        │
PR 11.6 (verifier subagent)        ── 消费 Sprint 10 类型注册表 + 本 sprint 11.1 / 11.5
```

合入策略：
1. PR 11.1 先合并并观察一周——纯 prompt 变更，零代码风险，但可能立刻让模型多花一些 token 自查；如果用户体感变好，后续 PR 才有意义。
2. PR 11.2 + 11.3 并行 review，互不依赖。
3. PR 11.4 落地后即可手测端到端基本能力。
4. PR 11.5 是"模型真正看到诊断"的里程碑——之前都是后台采集，到此为止才暴露给上下文。
5. PR 11.6 收尾，把多文件改动放进强 gate。

## 与其他 Sprint 的关系 / Cross-Sprint

- **Sprint 9（Coding UX Parity）**：`EditSummary` 已经能在 TUI 上显示 `+N −M`；本 sprint 不动 UI，所有诊断在上下文里 invisible 给用户。
- **Sprint 10（Subagent + Skill Parity）**：依赖其 `AgentTypeRegistry` 与 `DefaultSubagentBuilder` 类型化扩展点；PR 11.6 直接复用。
- **Sprint 12+（候选）**：把诊断面板搬到 TUI；ruff / eslint 作为独立 channel；auto-fix patch。

## 不在本 sprint / Deferred

见 [`00_overview.md` 的 Out of Scope 节](./00_overview.md#out-of-scope延后到-sprint-12)。
