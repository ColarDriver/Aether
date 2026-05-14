# Sprint 10 — Subagent Async/Background + Skill Listing Parity

把 Aether 的 **subagent dispatch** 与 **skill discovery** 两条链路对齐到 `open-claude-code` 的成熟实现：让 `subagent_type` 真的影响子 agent 行为、让 skill 对模型可见、让后台任务可持久化和恢复。

参考实现：`/workspace/open-claude-code`（`src/tools/AgentTool/`, `src/skills/`, `src/tasks/`）。

## 文档索引 / Table of Contents

| # | 文档 | 内容 |
|---|---|---|
| 0 | [`00_overview.md`](./00_overview.md) | 背景、Gap matrix、设计原则、Sub-PR 路线图 |
| 1 | [`01_pr10_1_skill_listing_injection.md`](./01_pr10_1_skill_listing_injection.md) | 把 `SkillCatalog` 转成 `<system-reminder>` 注入到模型（G2） |
| 2 | [`02_pr10_2_agent_type_registry.md`](./02_pr10_2_agent_type_registry.md) | `AgentTypeRegistry`：内置类型 + `.claude/agents/*.md` 加载（G1.A） |
| 3 | [`03_pr10_3_per_type_child_config.md`](./03_pr10_3_per_type_child_config.md) | `DefaultSubagentBuilder` 消费类型定义，过滤 tool / override system_prompt / model（G1.B） |
| 4 | [`04_pr10_4_on_disk_task_store.md`](./04_pr10_4_on_disk_task_store.md) | `~/.aether/tasks/{id}/` 持久化层 + 启动时恢复（G1.C） |
| 5 | [`05_pr10_5_async_subagent_lifecycle.md`](./05_pr10_5_async_subagent_lifecycle.md) | `run_in_background=true` + 后台 lifecycle（G1.D） |
| 6 | [`06_pr10_6_task_output_tool.md`](./06_pr10_6_task_output_tool.md) | `TaskOutput` 工具：阻塞 / 非阻塞两种模式（G1.E） |
| 7 | [`07_pr10_7_send_message_and_notifications.md`](./07_pr10_7_send_message_and_notifications.md) | `SendMessage` 工具 + 完成通知 `<task-notification>` 回注（G1.F） |
| 8 | [`08_pr10_8_worktree_isolation.md`](./08_pr10_8_worktree_isolation.md) | `git worktree` 隔离（G1.G） |
| 99 | [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 端到端 / 单元 / 类型 / 遥测验收矩阵 |

## PR 依赖图 / Dependency Graph

```
PR 10.1 (skill listing)        ── 独立，最小，可第一个合入

PR 10.2 (agent type registry) ──┐
                                 ├─→ PR 10.3 (per-type child config) ──┐
                                 │                                      │
PR 10.4 (on-disk TaskStore)    ──┴───────────────────────────────────── ┴─→ PR 10.5 (async lifecycle)
                                                                              ├─→ PR 10.6 (TaskOutput)
                                                                              └─→ PR 10.7 (SendMessage + notifications)

PR 10.8 (worktree)             ── 消费 10.3 的 isolation 字段，但实现独立
```

## 推荐合入顺序 / Recommended Merge Order

1. **PR 10.1** 先合 —— 与其它完全解耦；落地后模型立即看到 skill 菜单。
2. **PR 10.2** 合 —— foundation，10.3/10.5 依赖。
3. **PR 10.3** 与 **PR 10.4** 并行合 —— 两条互不依赖的支线。
4. **PR 10.5** 合 —— async/background 上线，依赖 10.3 + 10.4。
5. **PR 10.6** 与 **PR 10.7** 并行合 —— 都消费 10.5。
6. **PR 10.8** 最后合 —— worktree 隔离锦上添花。

## 不在本 sprint / Out of Scope

- **TUI 端的 task dashboard / progress bar** —— wire 事件本 sprint 定义，但 TUI 渲染留到 Sprint 11+。
- **跨进程 SendMessage** —— 本 sprint 只支持同一个 gateway 进程内父子互发；远端 agent（CCR / `isolation=remote`）不实现。
- **Agent swarm / 多 worker 编排** —— `coordinator mode` 不实现，等真有这类场景再做。
- **`/<agent-type>` slash UX** —— 与 skill 的 `/<skill>` 一样，得 TUI 配合，留给后续 sprint。

## 相关历史文档 / Related Past Docs

- [`../sprint-3.5-tool-catalog-completion/`](../sprint-3.5-tool-catalog-completion/) —— `task` / `task_stop` / `task_output` 占位实现来源
- [`../sprint-8-memory-reimplementation/`](../sprint-8-memory-reimplementation/) —— `SkillCatalog` 与 `should_review_skills` 死代码路径的引入处
- [`../sprint-9-coding-ux-parity/`](../sprint-9-coding-ux-parity/) —— `ToolResult.metadata` wire 形式参考；本 sprint 的 `task.progress` / `task.notification` 事件沿用 additive schema 原则
