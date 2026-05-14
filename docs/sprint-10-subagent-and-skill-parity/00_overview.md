# Sprint 10 — Subagent + Skill Parity (Overview)

## 背景 / Motivation

Aether 当前的 `aether/` runtime 已经具备：

- **Subagent skeleton**：`AgentTool`（`tools/builtins/agent_tool.py`）、`SubagentManager`（`subagents/manager.py`）、`DefaultSubagentBuilder`（`subagents/default_builder.py`）、`TaskStopTool`（`tools/builtins/task_stop.py`），含深度 / 并发上限、interrupt signal 透传。
- **Skill skeleton**：`SkillCatalog`（`runtime/tools/skill_catalog.py`）发现 27 个 SKILL.md，`SkillTool`（`tools/builtins/skill.py`）按需读 body + `$ARGUMENTS` / `${AETHER_SESSION_ID}` 替换。

但与 `open-claude-code`（`/workspace/open-claude-code`）的成熟实现对比，端到端层面有两条核心链路**实际上不工作**——参考其 `src/tools/AgentTool/AgentTool.tsx`、`src/tools/AgentTool/runAgent.ts`、`src/tools/AgentTool/loadAgentsDir.ts`、`src/skills/loadSkillsDir.ts`、`src/utils/attachments.ts`、`src/tasks/LocalAgentTask/LocalAgentTask.tsx`。

### Gap Matrix

| # | 现象 | 根因（带文件:行号） |
|---|---|---|
| **G1.A** | `task(subagent_type="Explore")` 与 `general-purpose` 行为完全一样 | `agent_tool.py:104,142,153` 把 `subagent_type` 写进 metadata 后**无人读取**；`default_builder.py:21-72` 完全继承父配置 |
| **G1.B** | 子 agent 拿到父的整套工具，没有"只读 Explore"概念 | `default_builder.py:51` 直接 `tool_registry=parent.services.tool_registry`；无 allow / deny 列表机制 |
| **G1.C** | gateway 重启即丢子 agent 状态；无法事后查任务结果 | `SubagentManager._stop_events` / `_active_children` 全 in-memory；无磁盘持久化 |
| **G1.D** | `task` 工具强制同步：长任务阻塞父循环 | `agent_tool.py:155` 写死 `"run_in_background": False`；`manager.run_task` 同步返回 |
| **G1.E** | 现有 `TaskOutputTool` 是占位，返回 "not supported in sync mode" | `tools/builtins/task_output.py` 按 Sprint 3.5 计划留空 |
| **G1.F** | 无法向运行中的子 agent 投递追加指令；子完成后父不知道 | 没有 SendMessage 工具；没有 `<task-notification>` 注入路径 |
| **G1.G** | 子 agent 修改文件直接落到父的 working tree | 没有 worktree 隔离 |
| **G2** | 模型在 `task(prompt="use skill X")` 之前不知道有哪些 skill | `should_review_skills` flag 是死代码（`agent.py:1574-1599` 写，无人读）；`system_prompt.py:43-61` 只注入 tool contract |

### 对照参考 / Reference Behavior

`open-claude-code` 的对应实现一目了然：

| 能力 | 文件:行 |
|---|---|
| `AgentTool` 解析 `subagent_type` 并加载定义 | `src/tools/AgentTool/AgentTool.tsx:318-356` |
| 内置 agent 类型 | `src/tools/AgentTool/builtInAgents.ts:22-72` |
| Markdown agent 加载器 | `src/tools/AgentTool/loadAgentsDir.ts:296-393, 541-600+` |
| Worker tool pool 装配（过滤） | `src/tools/AgentTool/AgentTool.tsx:573-577` |
| Worktree 隔离 | `src/tools/AgentTool/AgentTool.tsx:590-603` |
| Async lifecycle 注册 | `src/tasks/LocalAgentTask/LocalAgentTask.tsx:466-515` |
| Background lifecycle 主循环 | `src/tools/AgentTool/agentToolUtils.ts:508-686` |
| `SendMessage` 工具 | `src/tools/SendMessageTool/SendMessageTool.ts:67-200+` |
| Pending message queue | `src/tasks/LocalAgentTask/LocalAgentTask.tsx:162-192` |
| Skill 清单 attachment | `src/utils/attachments.ts:2661-2751` |
| 清单 → system-reminder 包装 | `src/utils/messages.ts:3728-3738` |
| Skill 按需加载（lazy body） | `src/skills/loadSkillsDir.ts:344-399` |

## 设计原则 / Design Principles

1. **不重造已有骨架** —— `SubagentManager` / `SubagentBuilder` / `SubagentTask` / `SubagentResult` 全部复用；新代码只补当前缺失的支线（async lifecycle、TaskStore、类型注册表、tool 过滤）。
2. **wire schema 只加不改** —— 沿用 Sprint 9 的 additive 原则；新事件 `task.progress` / `task.notification` 用 Pydantic 默认值，旧客户端零变化。
3. **Lazy loading 纪律不破** —— 与 `open-claude-code` 一致：skill 清单（frontmatter）放进系统提示，body 永远 on-demand；agent 类型定义同理（仅元信息进 enum，system_prompt 仅在 spawn 时拼装）。
4. **死代码不留** —— `should_review_skills` / `skill_nudge_interval` 在本 sprint 真正接上消费方；否则要么删，要么补，不能再留半成品。
5. **类型定义集中 + 外部可扩展** —— 内置 4 个类型（参考 `builtInAgents.ts`），加 markdown 加载器允许用户/项目添加自定义类型；解析复用 `SkillCatalog._split_frontmatter` 的同款 frontmatter 风格。
6. **元工具 (`task_stop` / `task_output` / `send_message`) 永不在 deny 列表中失踪** —— 否则 child 会"看见任务但够不到"，调试体验崩盘。
7. **TaskStore = 单一事实源** —— 任何"任务状态"问题（运行中？完成了？出错了？）都从磁盘读，不从内存读；in-memory 缓存仅作 fast path。
8. **同步路径不变** —— 默认 `run_in_background=false` 时行为与今天完全一致；async 是新增路径。

## Sub-PR 路线图 / Roadmap

按依赖顺序：

```
PR 10.1 (skill listing)        ── 独立

PR 10.2 (agent type registry) ──┐
                                 ├─→ PR 10.3 (per-type child config) ──┐
                                 │                                      │
PR 10.4 (on-disk TaskStore)    ──┴───────────────────────────────────── ┴─→ PR 10.5 (async lifecycle)
                                                                              ├─→ PR 10.6 (TaskOutput)
                                                                              └─→ PR 10.7 (SendMessage + notifications)

PR 10.8 (worktree)             ── 消费 10.3 + 10.4
```

| PR | 标题 | 主要文件（新建/修改） | Gap |
|---|---|---|---|
| 10.1 | Skill Listing System-Reminder Injection | `aether/runtime/tools/skill_listing.py`(NEW), `agent.py`, `system_prompt.py`, `config/schema.py` | G2 |
| 10.2 | Agent Type Registry | `aether/agents/types/`(NEW dir, 5 files), `agent_tool.py`, `agent.py`, `config/schema.py` | G1.A |
| 10.3 | Per-Type Child Configuration | `default_builder.py`, `agent_tool.py`, `tools/registry.py` (helper) | G1.B |
| 10.4 | On-Disk Task Store | `aether/runtime/tasks/`(NEW dir, 4 files), `agent.py`, `config/schema.py` | G1.C |
| 10.5 | Async Subagent Lifecycle | `subagents/manager.py`, `agent_tool.py`, `runtime/core/hooks.py`, `agent.py` | G1.D |
| 10.6 | TaskOutput Tool | `tools/builtins/task_output.py` (重写) | G1.E |
| 10.7 | SendMessage + Task Notifications | `tools/builtins/send_message.py`(NEW), `agent.py`, `subagents/manager.py`, `runtime/core/hooks.py` | G1.F |
| 10.8 | Worktree Isolation | `runtime/tasks/worktree.py`(NEW), `default_builder.py`, `agent_tool.py`, `config/schema.py` | G1.G |

每个 PR 都对应本目录中一个独立文档，包含 **目标 / 当前问题 / 改动 / 测试 / 验收** 五节。

## Out of Scope（延后到 Sprint 11+）

- **TUI Task Dashboard** —— 现在只把 `task.progress` / `task.notification` 上 wire，TUI 端的可视化不做。
- **Cross-process SendMessage** —— 只支持同一个 gateway 进程内的父子；远端 / CCR / `isolation=remote` 留给后续。
- **Coordinator mode（多 worker 编排）** —— `open-claude-code/src/coordinator/coordinatorMode.ts` 描述的协调器人格不实现。
- **`/<agent-type>` slash 直接 spawn** —— 与 skill 的 `/<skill>` 一样得 TUI 配合，本 sprint 仅做后端 API。
- **Auto-fork / prompt cache 优化的 subagent** —— `open-claude-code` 的 fork experiment 不做。

## 验收 / Acceptance（汇总见 `99_acceptance_matrix.md`）

- 端到端：13 个场景全部通过（覆盖 skill 清单、类型过滤、async lifecycle、send_message、notification、recovery、worktree）
- 单元测试：`uv run pytest aether/tests/` 全绿
- 类型检查：`uv run pyright` 与 `npm run typecheck --prefix tui` 零新增告警
- 手测：跑 `uv run aether`，prompt "find usages of `ToolRegistry`, use the Explore subagent" → 子 agent 拿不到 `write_file` / `file_edit` / `shell` 工具且能正确完成
