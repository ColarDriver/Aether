# Sprint 12 — Plan Mode Parity (Overview)

## 背景 / Motivation

本 sprint 目标是把 Aether 的本地 `/plan` 模式补齐到可用、可恢复、可验收的完整链路，而不是只增加一个 slash command。完成后，用户可以手动进入 plan mode，让模型只读探索并产出计划，通过 `exit_plan_mode` 请求批准；批准后回到 agent mode 执行，拒绝后继续停留在 plan mode 修订计划。

参考实现是 `open-claude-code` 的 `/plan` 语义，但本 sprint 只对齐 Aether 本地产品需要的部分：slash command、session mode 状态、plan artifact、ExitPlanMode approval。Claude Code 专属的 remote session、ultraplan、teammate mailbox、Opus plan routing 不进入本 sprint。

## 当前状态 / Current State

Aether 已经具备几块底座：

- `enter_plan_mode` / `exit_plan_mode` 工具已经存在，模型可以通过工具进入或请求退出 plan mode。
- plan approval bridge 已存在，`exit_plan_mode` 可以触发用户确认。
- 写工具阻断已经接入 registry / engine gate，plan mode 下的写入不应继续走普通 permission prompt。

但用户侧链路还缺关键能力：

- 用户无法通过 `/plan` 手动进入 plan mode。
- TUI 没有当前 session mode 的可见状态，也没有 `/plan` 的 slash 入口。
- engine 没有在 plan mode 的每一轮注入稳定 system reminder，模型容易把 plan mode 当成普通 agent mode。
- plan 内容没有 artifact 文件承载，`/plan` 无法查看最近计划，resume / clear 也无法一致处理。

## Reference Model

`open-claude-code` 的 `/plan` 由四块组成：

| 能力 | 语义 | Aether 对齐方式 |
|---|---|---|
| Slash command | 用户输入 `/plan` 或 `/plan <description>` 进入规划流程 | TS TUI 注册 `/plan`，必要时继续提交一轮 `agent.run` |
| Permission mode | session 处于 plan / agent 等模式 | gateway RPC 读写 `aether.runtime.session.session_state` |
| Plan file | 当前 session 的计划有稳定文件承载 | `${AETHER_HOME}/plans/<session-prefix>.md` |
| ExitPlanMode approval | 模型必须用 tool 请求批准，用户 approve/reject | 复用 Aether approval bridge，并让 TUI 渲染 plan markdown |

## Sprint Goals

1. 用户入口完整：`/plan`、`/plan <description>`、`/plan open`、help/completion 都可用。
2. session mode 一致：gateway、engine、TUI store、resume/new/clear 都能观察并维护 `agent` / `plan`。
3. plan mode prompt 明确：每轮提醒模型当前只能读、搜索、询问，最后必须调用 `exit_plan_mode(plan="...")`。
4. plan artifact 稳定：每个 session 一个 Markdown 计划文件，可读、可清理、可恢复。
5. approval parity：`exit_plan_mode` 弹出 plan approval，approve 切回 agent mode，reject 留在 plan mode。
6. 测试闭环：Python 和 TS 单测覆盖 RPC、engine prompt、blocker、artifact、slash command、approval modal、session lifecycle。

## Non-Goals

以下能力明确排除，后续如有 Aether 产品需求再单独开 sprint：

- Claude Code web remote session。
- `ultraplan` subscription flow。
- team mailbox / teammate approval。
- Opus-plan model routing 或 plan 专用模型切换。
- 复刻 open-claude-code 的 Claude 专属文案、订阅提示、远程协作控制面。

## Roadmap

| # | 文档 | 内容 |
|---|---|---|
| 1 | [`01_pr12_1_plan_mode_control_rpc_and_catalog.md`](./01_pr12_1_plan_mode_control_rpc_and_catalog.md) | 后端 slash catalog、gateway plan RPC、session state wire |
| 2 | [`02_pr12_2_tui_plan_slash_entry.md`](./02_pr12_2_tui_plan_slash_entry.md) | TUI `/plan` 入口、query continuation、mode store、`/plan open` |
| 3 | [`03_pr12_3_engine_plan_mode_prompt_and_guardrails.md`](./03_pr12_3_engine_plan_mode_prompt_and_guardrails.md) | engine plan-mode system reminder、写工具阻断回归 |
| 4 | [`04_pr12_4_plan_artifact_storage_and_tools.md`](./04_pr12_4_plan_artifact_storage_and_tools.md) | `${AETHER_HOME}/plans` artifact、`exit_plan_mode` 写入、`plan.current` 读取 |
| 5 | [`05_pr12_5_exit_plan_approval_ui_parity.md`](./05_pr12_5_exit_plan_approval_ui_parity.md) | approval modal markdown、approve/reject mode transition |
| 6 | [`06_pr12_6_resume_clear_session_integration.md`](./06_pr12_6_resume_clear_session_integration.md) | `/new`、`/clear`、`/resume` 与 mode/artifact 生命周期 |
| 7 | [`07_pr12_7_tests_observability_and_acceptance.md`](./07_pr12_7_tests_observability_and_acceptance.md) | 总体验收、观测字段、回归清单 |

## Dependency Graph

```
PR 12.1 (RPC + catalog) ──┬─→ PR 12.2 (TUI /plan)
                           ├─→ PR 12.3 (engine prompt)
                           └─→ PR 12.4 (plan artifact) ──┐
                                                          ├─→ PR 12.5 (approval UI)
PR 12.2 + PR 12.4 ───────────────────────────────────────┤
                                                          └─→ PR 12.6 (resume/clear)

PR 12.7 runs across all merged pieces.
```

推荐合入顺序：

1. PR 12.1 先落地 RPC 和 catalog，给 TUI / engine 后续工作一个稳定 wire contract。
2. PR 12.2 和 PR 12.3 可并行，分别补用户入口和模型行为约束。
3. PR 12.4 接 plan artifact，让 `/plan` 和 `exit_plan_mode` 有共享数据源。
4. PR 12.5 补 approval UI parity，完成 approve/reject 的 session mode transition。
5. PR 12.6 收尾 session lifecycle。
6. PR 12.7 做端到端验收与回归确认。

## Acceptance Summary

- `/plan add auth flow` 会进入 plan mode，并继续发起一轮模型规划。
- plan mode 下 provider 每轮收到 plan reminder。
- plan mode 下写工具、shell、commit、subagent dispatch、memory mutation 在 permission prompt 前被拒绝。
- 模型必须调用 `exit_plan_mode(plan=...)` 请求批准，普通文本询问不能替代 approval。
- approval approve 后 mode 切回 `agent`；reject 后保持 `plan`。
- `/plan` 能查看最近 plan，`/plan open` 能打开 plan artifact 或给出清晰失败提示。
- `/new` 不继承旧 plan mode；`/clear` 不显示旧 plan；`/resume` 能恢复对应 session 的 mode 和 plan artifact。

