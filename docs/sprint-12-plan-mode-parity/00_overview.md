# Sprint 12 — Plan Mode Parity (Overview)

## 背景 / Motivation

本 sprint 目标是把 Aether 的本地 `/plan` 模式补齐到可用、可恢复、可观测、可验收的完整链路，而不是只增加一个 slash command。完成后，用户可以手动进入 plan mode，让模型在受限环境中探索、把计划写入当前 session 的 plan artifact，通过 `exit_plan_mode` 请求批准；批准后回到 agent mode 执行，拒绝后继续停留在 plan mode 修订计划。

参考实现是 `open-claude-code` 的 `/plan` 语义，但本 sprint 只对齐 Aether 本地产品需要的部分：slash command、session mode 状态、plan-mode prompt、plan artifact、plan-file write exception、ExitPlanMode approval、resume/clear/new 生命周期。Claude Code 专属的 remote session、ultraplan、teammate mailbox、Opus plan routing 不进入本 sprint。

## 当前状态 / Current State

Aether 已经具备几块底座：

- `enter_plan_mode` / `exit_plan_mode` 工具已经存在，模型可以通过工具进入或请求退出 plan mode。
- plan approval bridge 已存在，`exit_plan_mode` 可以触发用户确认。
- 写工具阻断已经接入 registry / engine gate，plan mode 下的写入不应继续走普通 permission prompt。

当前已开发实现覆盖了大量骨架：`/plan` catalog、`plan.mode_get` / `plan.mode_set` / `plan.current` / `plan.clear`、TUI `/plan` command、session store `mode`、basic plan artifact、approval markdown、approve/reject mode transition、resume / clear 的基础集成。

复核 open-claude-code 后仍存在几个必须补齐的核心差异：

- plan file 在 plan mode 中不是普通 artifact，而是唯一允许写的文件。Aether 不能全阻断 `write_file` / `file_edit`，否则模型无法像 OCC 一样边探索边维护 plan 文件。
- engine reminder 必须带 plan file path、当前是否已有 plan，以及只能编辑这个 plan file。
- `exit_plan_mode` 应优先从 plan artifact 读取计划；`plan` 参数只能作为兼容 / UI 编辑回传，不应是唯一数据源。
- `/plan` 展示应包含 `Current Plan`、plan path、`/plan open` hint；`/plan open` 的 mode 行为需要明确按 OCC 对齐或声明 Aether deliberate divergence。
- approval UI 基础 approve/reject 已有，但若要接近 OCC，还需要 plan path、外部编辑、编辑后回传 updated plan 等能力。

## Reference Model

`open-claude-code` 的 `/plan` 由以下能力组成：

| 能力 | 语义 | Aether 对齐方式 |
|---|---|---|
| Slash command | 用户输入 `/plan` 或 `/plan <description>` 进入规划流程 | TS TUI 注册 `/plan`，必要时继续提交一轮 `agent.run` |
| Permission mode | session 处于 plan / agent 等模式 | gateway RPC 读写 `aether.runtime.session.session_state` |
| Plan file | 当前 session 的计划有稳定文件承载，plan mode 下只能写这个文件 | `${AETHER_HOME}/plans/<session-prefix>.md`，允许 `write_file` / `file_edit` 仅写当前 session plan path |
| Prompt attachment | 每轮说明 plan path、plan exists、唯一可编辑文件、最终必须 ExitPlanMode | Engine 注入 artifact-aware reminder |
| ExitPlanMode approval | 默认从 plan file 读 plan，用户 approve/reject | 复用 Aether approval bridge，支持 artifact-first 和 markdown approval |

## Sprint Goals

1. 用户入口完整：`/plan`、`/plan <description>`、`/plan open`、help/completion 都可用。
2. session mode 一致：gateway、engine、TUI store、resume/new/clear 都能观察并维护 `agent` / `plan`。
3. plan mode prompt 明确：每轮提醒模型当前处于 plan mode、plan path、是否已有 plan、唯一可编辑文件、允许只读探索、禁止 shell/commit/subagent/memory mutation、必须用 `exit_plan_mode` 请求批准。
4. plan artifact 稳定：每个 session 一个 Markdown 计划文件，可读、可清理、可恢复，并作为 `exit_plan_mode` 的主数据源。
5. approval parity：`exit_plan_mode` 弹出 plan approval，approve 切回 agent mode，reject 留在 plan mode。
6. plan-file write exception：plan mode 下 `write_file` / `file_edit` 仅允许写当前 session plan file，其他写路径仍被 blocker 拒绝。
7. 测试闭环：Python 和 TS 单测覆盖 RPC、engine prompt、blocker、artifact、slash command、approval modal、session lifecycle。

## Non-Goals

以下能力明确排除，后续如有 Aether 产品需求再单独开 sprint：

- Claude Code web remote session。
- `ultraplan` subscription flow。
- team mailbox / teammate approval。
- Opus-plan model routing 或 plan 专用模型切换。
- CCR remote file snapshot / pod recovery。
- teammate / team mailbox 的 approve/reject 协作流。
- 复刻 open-claude-code 的 Claude 专属文案、订阅提示、远程协作控制面。

以下能力可以作为 Aether 后续增强，但不是本 sprint 完成条件：

- readable random word slug 替代 `<session_id[:8]>.md`。
- plan history / versioning / diff。
- approval 后 clear-context / keep-context 多选策略。
- plan-specific verifier workflow。

## Roadmap

| # | 文档 | 内容 |
|---|---|---|
| 1 | [`01_pr12_1_plan_mode_control_rpc_and_catalog.md`](./01_pr12_1_plan_mode_control_rpc_and_catalog.md) | 后端 slash catalog、gateway plan RPC、session mode persistence、wire schema |
| 2 | [`02_pr12_2_tui_plan_slash_entry.md`](./02_pr12_2_tui_plan_slash_entry.md) | TUI `/plan` 入口、query continuation、mode store、plan display、`/plan open` |
| 3 | [`03_pr12_3_engine_plan_mode_prompt_and_guardrails.md`](./03_pr12_3_engine_plan_mode_prompt_and_guardrails.md) | engine plan-mode reminder、plan path 注入、只写 plan file 的 guardrails |
| 4 | [`04_pr12_4_plan_artifact_storage_and_tools.md`](./04_pr12_4_plan_artifact_storage_and_tools.md) | `${AETHER_HOME}/plans` artifact、plan-file write exception、ExitPlanMode artifact-first |
| 5 | [`05_pr12_5_exit_plan_approval_ui_parity.md`](./05_pr12_5_exit_plan_approval_ui_parity.md) | approval modal markdown、path、external edit、approve/reject mode transition |
| 6 | [`06_pr12_6_resume_clear_session_integration.md`](./06_pr12_6_resume_clear_session_integration.md) | `/new`、`/clear`、`/resume` 与 mode/artifact 生命周期 |
| 7 | [`07_pr12_7_tests_observability_and_acceptance.md`](./07_pr12_7_tests_observability_and_acceptance.md) | 总体验收、观测字段、回归清单 |

## Dependency Graph

```
PR 12.1 (RPC + catalog + mode schema)
  ├─→ PR 12.2 (TUI /plan)
  ├─→ PR 12.3 (engine prompt)
  └─→ PR 12.4 (artifact + plan-file write exception)
        ├─→ PR 12.5 (approval UI + edited plan round-trip)
        └─→ PR 12.6 (resume/clear/new lifecycle)

PR 12.7 runs across all merged pieces.
```

推荐合入顺序：

1. PR 12.1 先落地 RPC、catalog、mode persistence 和 wire contract。
2. PR 12.4 尽早落地 artifact 和 plan-file write exception，因为 PR 12.3 / PR 12.5 都依赖 plan path。
3. PR 12.3 接入 artifact-aware reminder 和 blocker metadata。
4. PR 12.2 接完整 TUI `/plan` 行为。
5. PR 12.5 补 approval UI parity 和 edited plan round-trip。
6. PR 12.6 收尾 session lifecycle。
7. PR 12.7 做端到端验收与回归确认。

## Acceptance Summary

- `/plan add auth flow` 会进入 plan mode，并继续发起一轮模型规划。
- plan mode 下 provider 每轮收到 plan reminder，里面包含当前 plan path 和是否已有 plan。
- plan mode 下 `write_file` / `file_edit` 仅允许目标等于当前 session plan path；其他写文件、shell、commit、subagent dispatch、memory mutation、`todo_write` 在 permission prompt 前被拒绝。
- 模型可以边探索边通过 `write_file` / `file_edit` 更新 plan artifact。
- 模型最终调用 `exit_plan_mode()` 或兼容形式 `exit_plan_mode(plan="...")` 请求批准；普通文本询问不能替代 approval。
- `exit_plan_mode()` 无 `plan` 参数时从 artifact 读计划；如果 artifact 不存在或为空，返回明确错误。
- approval approve 后 mode 切回 `agent`；reject 后保持 `plan`。
- approval tool result 包含 plan path 和 approved plan 内容，便于模型批准后继续引用。
- `/plan` 能查看最近 plan、plan path 和 editor hint；`/plan open` 能打开 plan artifact 或给出清晰失败提示。
- `/new` 不继承旧 plan mode；`/clear` 不显示旧 plan；`/resume` 能恢复对应 session 的 mode 和 plan artifact。

## Parity Gaps To Track

以下是本 sprint 必须显式解决或明确排除的差异：

| Gap | Sprint decision |
|---|---|
| OCC 允许写 plan file；Aether 不能全阻断 write tools | 必须补：只允许当前 session plan file |
| OCC ExitPlanMode 默认读 plan file；Aether 若只接 `plan` 参数不完整 | 必须补：artifact-first，`plan` 参数兼容 |
| OCC prompt 包含 plan path / exists / workflow；Aether 短 reminder 不够 | 必须补：path-aware reminder，短但不能缺核心语义 |
| OCC `/plan open` 只在已处于 plan mode 时打开 | 推荐对齐；如 Aether 保留 agent mode 打开能力，必须作为 deliberate divergence 写入 PR 12.2 |
| OCC approval 可 Ctrl+G 编辑 plan | 本地 parity 必做 external edit；remote / ultraplan variants 排除 |
| OCC word slug / resume recovery / fork copy | 第一版排除，使用 session prefix；PR 12.6 记录后续迁移点 |
