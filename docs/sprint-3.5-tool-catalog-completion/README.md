# Sprint 3.5 — Tool Catalog Completion + Tier 1 Persistence

> **范围扩展**：本 sprint 在 Sprint 3 既定的"压缩流水线"工作流之外，
> **同步补齐 Aether 的工具集**。把现状 6 个内置工具升级到 18 个，
> 与 claude-code 默认开启的工具对齐（剔除产品特定的远程基础设施）。
>
> **执行节奏**：10 个 PR，预计 4 周，每个 PR 独立 review 独立合并。

## 一、为什么要做这个 sprint

PR 3.2 在落地 `cheap_tool_names` 默认名单时暴露了一个**当前可见的内部不一致**：

```python
cheap_tool_names: tuple[str, ...] = (
    "update_todo",
    "todo_write",       # ← 这个工具在 Aether 还不存在！
    "memory",
    "memory_write",
    "memory_read",
    "skill_manage",     # ← 同样不存在
    "session_search",   # ← 同样不存在
)
```

PR 3.2 的 cheap-tool refund 机制依赖这些工具被 LLM 调用 — 但 LLM 根本没法调用不存在的工具。
这是补齐工具集的最直接动力。

更进一步，对比 claude-code 默认开启的 17 个工具，Aether 现状只有 6 个，**缺失了**：

* 局部文件编辑（`FileEditTool`） — 没有它，模型只能用 `write_file` 整文件覆盖，
  上下文消耗、版本控制噪声、误改风险全部放大。
* 任务清单（`TodoWriteTool`） — 长任务分步推进的核心机制。
* Web 抓取/搜索（`WebFetchTool` / `WebSearchTool`） — 任何"查文档"任务的前提。
* Subagent 派发（`AgentTool` 等） — 我们已经有 `SubagentManager` 基础设施，但没暴露给模型。
* 计划/交互/技能/LSP/浏览器 — 高级工作流必需。

补齐这些工具的同时，PR 3.3（原计划的"per-tool 持久化"）合并进来，让所有工具
**一上来就具备 spill 能力**，而不是先做一遍再回头改。

## 二、Sprint 范围（10 个 PR）

| PR | 标题 | 文档 | 工具变化 | 预计 |
|---|---|---|---|---|
| 3.5.1 | Spill 基础 + 6 现有工具升级 | [01](01_pr3_5_1_spill_foundation.md) | 6 升级 | 2d |
| 3.5.2 | FileEditTool（局部 search/replace 编辑） | [02](02_pr3_5_2_file_edit.md) | +1 | 1.5d |
| 3.5.3 | TodoWriteTool（闭环 PR 3.2 cheap_tool） | [03](03_pr3_5_3_todo_write.md) | +1 | 1d |
| 3.5.4 | NotebookEditTool（Jupyter 单元格） | [04](04_pr3_5_4_notebook_edit.md) | +1 | 1d |
| 3.5.5 | WebFetchTool + WebSearchTool | [05](05_pr3_5_5_web_tools.md) | +2 | 2.5d |
| 3.5.6 | AgentTool + TaskOutputTool + TaskStopTool | [06](06_pr3_5_6_subagent_tools.md) | +3 | 2.5d |
| 3.5.7 | EnterPlanMode + ExitPlanMode + AskUserQuestion | [07](07_pr3_5_7_interaction_tools.md) | +3 | 2d |
| 3.5.8 | SkillTool + skill catalog 基础设施 | [08](08_pr3_5_8_skill_tool.md) | +1 | 3d |
| 3.5.9 | LSPTool + LSP server 集成 | [09](09_pr3_5_9_lsp_tool.md) | +1 | 4d |
| 3.5.10 | WebBrowserTool + Playwright 集成 | [10](10_pr3_5_10_web_browser.md) | +1 | 3d |
| — | 跨 PR 验收矩阵 | [99](99_acceptance_matrix.md) | — | — |

**最终工具总数**：6（升级） + 14（新增） = **20** 个。
（claude-code 默认 17 + 我们独有 `list_dir` + `TaskOutputTool/TaskStopTool` 算 3 个 = 20）

## 三、与已合并 PR 的接合

* **PR 3.1（token usage / metadata 基础）**：已经在 `EngineResult.metadata['compaction']` 中
  预留了 `tier1_spilled_count` 字段，PR 3.5.1 只需要在工具 execute 时累加。
* **PR 3.2（IterationBudget + cheap_tool）**：PR 3.5.3 落地 `TodoWriteTool` 后，
  cheap_tool refund 路径才真正可端到端验证。
* **Sprint 4（Snip / Microcompact / Compact 等 Tier 2-5）**：会消费本 sprint 落地的
  spill 计数（`tier1_spilled_count`）做触发判断。

## 四、阅读顺序建议

1. [`00_overview.md`](00_overview.md) — 完整背景、与 claude-code 工具集逐项对比、PR 依赖图。
2. 按 PR 编号阅读 01-10 — 每篇文档都是 self-contained 的设计 + 测试规划。
3. [`99_acceptance_matrix.md`](99_acceptance_matrix.md) — sprint 完成时的验收清单。
