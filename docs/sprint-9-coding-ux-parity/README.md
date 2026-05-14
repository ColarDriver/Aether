# Sprint 9 — Coding-UX Parity with Claude Code

把 Aether TUI 在 **文件编辑 / 命令执行** 两个核心 coding 流程上的展示效果，对齐 Claude Code 的呈现层。

参考截图：`/workspace/Aether/tmp/issues4/image copy {9,11,13,14,16,18}.png`。

## 文档索引 / Table of Contents

| # | 文档 | 内容 |
|---|---|---|
| 0 | [`00_overview.md`](./00_overview.md) | 背景、截图导览、Gap matrix、设计原则、Sub-PR 路线图 |
| 1 | [`01_pr9_1_tool_result_metadata_wire.md`](./01_pr9_1_tool_result_metadata_wire.md) | 让 `ToolResult.metadata` 沿 wire 送达 TUI（G3） |
| 2 | [`02_pr9_2_chat_edit_summary.md`](./02_pr9_2_chat_edit_summary.md) | Chat 中显示 `● Edited X (+N −M)` + 内联 diff（G2） |
| 3 | [`03_pr9_3_permission_modal_fallbacks.md`](./03_pr9_3_permission_modal_fallbacks.md) | 修复 `write_file` 弹框退化到 JSON dump 的 bug（G1） |
| 4 | [`04_pr9_4_shell_command_rendering.md`](./04_pr9_4_shell_command_rendering.md) | shell 多命令拆行 + `[exit · ms]` footer（G4） |
| 99 | [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 端到端 / 单元 / 类型 / 遥测验收矩阵 |

## PR 依赖图 / Dependency Graph

```
PR 9.1 (wire metadata)  ──┬─→ PR 9.2 (chat edit summary)
                          └─→ PR 9.4 (shell footer)

PR 9.3 (modal fallback)  ── 独立，可与 9.1 并行
```

建议顺序：

1. **PR 9.1** 先合 —— 是其他两个 PR 的前置条件
2. **PR 9.3** 并行合 —— 不依赖 9.1，bug fix 优先级最高
3. **PR 9.2** 合 —— 需要 9.1 提供的 `result.metadata`
4. **PR 9.4** 合 —— 同样需要 9.1，弹框部分独立但 footer 部分依赖

## 不在本 sprint / Out of Scope

- Notebook 编辑 diff（cell-aware，需独立设计）
- Diff 行内语法高亮（polish）
- 多文件 batch summary
- 完整 POSIX shell 解析

## 相关历史文档 / Related Past Docs

- [`../sprint-7-tool-permission-confirmation/`](../sprint-7-tool-permission-confirmation/) —— 当前权限合约的设计来源；本 sprint 复用该合约不做修改
- [`../sprint-3.5-tool-catalog-completion/`](../sprint-3.5-tool-catalog-completion/) —— `file_edit` / `write_file` / `shell` 工具规范来源
- [`../tui-split/05_pr5_approval_permission_bridge.md`](../tui-split/05_pr5_approval_permission_bridge.md) —— Reverse-RPC `permission.request` 链路
