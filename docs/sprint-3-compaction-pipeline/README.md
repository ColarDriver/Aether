# Sprint 3 — 五级上下文压缩流水线（详细设计文档）

> 这是 Sprint 3 的工程详细设计目录。Sprint 级别的总规划见
> [`docs/run-loop-roadmap/07_sprint_execution_plan.md`](../run-loop-roadmap/07_sprint_execution_plan.md)。
> 这里把 Sprint 3 的 7 个 PR 一一拆开，每个 PR 一份独立详细方案，
> 含数据结构、文件改动清单、测试用例、验收门、回滚开关、风险与缓解。

---

## 一、本目录的作用

- **单一事实来源**：实施期间所有疑问都在这里查；与代码同步演进。
- **PR 边界**：7 个 PR 各自独立可合入主分支，不引入未来 PR 才能修复的回归。
- **可追溯**：每条改动都对应一条文档原句，方便 review 和回滚定位。

## 二、文件导航

| 文档 | 内容 | 角色 |
|---|---|---|
| [`00_overview.md`](./00_overview.md) | Sprint 3 整体设计、五级流水线哲学、与 claude-code 对照、数据流图 | 入口 |
| [`01_pr3_1_token_metadata.md`](./01_pr3_1_token_metadata.md) | PR 3.1 — Token / Result 基础（P1-4 + P1-11） | 地基 |
| [`02_pr3_2_iteration_budget.md`](./02_pr3_2_iteration_budget.md) | PR 3.2 — IterationBudget + max_iterations summary（P1-2 + P1-3） | 地基 |
| [`03_pr3_3_tier1_tool_persistence.md`](./03_pr3_3_tier1_tool_persistence.md) | PR 3.3 — Tier 1 per-tool 结果持久化 | 流水线第 1 层 |
| [`04_pr3_4_tier5_autocompact.md`](./04_pr3_4_tier5_autocompact.md) | PR 3.4 — services/compact/ 骨架 + Tier 5 Autocompact | 流水线第 5 层 + 骨架 |
| [`05_pr3_5_tier3_microcompact.md`](./05_pr3_5_tier3_microcompact.md) | PR 3.5 — Tier 3 Microcompact 时间触发分支 | 流水线第 3 层 |
| [`06_pr3_6_tier2_snip.md`](./06_pr3_6_tier2_snip.md) | PR 3.6 — Tier 2 Snip 冗余消息修剪 | 流水线第 2 层 |
| [`07_pr3_7_tier4_collapse.md`](./07_pr3_7_tier4_collapse.md) | PR 3.7 — Tier 4 Context Collapse 投影式折叠 | 流水线第 4 层 |
| [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 端到端验收场景、回归测试矩阵、性能基准、监控指标 | 收尾 |

## 三、阅读顺序

1. **先读 [`00_overview.md`](./00_overview.md)** 拿到整体心智模型。
2. 实施时按 **PR 编号顺序**读对应 PR 文档（前序 PR 是后序 PR 的依赖）。
3. 每个 PR 完成时回读对应 PR 文档的"验收门"小节确认通过。
4. Sprint 收尾时跑 [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) 的端到端场景。

## 四、改动约定

- 文档落地后不再做大幅结构调整，只在 PR 实施完成后追加"实施记录"段（含实际偏差、追加的测试、踩坑记录）。
- 任何超出本目录设计的工程决策（例如换 provider 抽象、引入新中间件类型）必须先在本目录追加 ADR-style 子文档，再回到对应 PR 文档更新。
- 文档与代码的对应关系：每个 PR 文档的"文件改动清单"是该 PR commit 的代码 review checklist。

## 五、与其他文档的关系

- **总规划** [`docs/run-loop-roadmap/07_sprint_execution_plan.md`](../run-loop-roadmap/07_sprint_execution_plan.md)：Sprint 3 实施前同步把"## Sprint 3"小节内容指向本目录的 README。
- **缺口跟踪** [`docs/run-loop-roadmap/03_p1_robustness_gaps.md`](../run-loop-roadmap/03_p1_robustness_gaps.md)：Sprint 3 完成后把 P1-1/2/3/4/11 状态全部从 ❌ 改 ✅。
- **引擎结构** [`docs/agent-engine/`](../agent-engine/)：Sprint 3 引入的 `services/compact/` 是新增模块，落地后追加一份 `08_compaction_pipeline.md` 到 `agent-engine/` 描述其与 `runtime/` 的协作。

---

下一步：阅读 [`00_overview.md`](./00_overview.md) 了解整体设计。
