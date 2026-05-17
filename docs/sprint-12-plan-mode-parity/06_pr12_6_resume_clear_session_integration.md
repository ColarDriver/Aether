# PR 12.6 — Resume, Clear, and Session Lifecycle Integration

## 目标 / Goal

把 plan mode 和 plan artifact 接入 session 生命周期：`/new` 不继承旧状态，`/clear` 不显示旧计划，`/resume` 能恢复对应 session 的 mode 和 plan artifact。

本 PR 依赖 PR 12.1 的 mode RPC、PR 12.2 的 TUI store、PR 12.4 的 plan artifact。

## 当前问题 / Current Problem

如果只实现 `/plan` 和 `exit_plan_mode`，session 生命周期会出现状态漂移：

- `/new` 后可能沿用旧 session 的 plan mode。
- `/clear` 清理 transcript 后，`/plan` 仍显示旧计划，用户会误以为新上下文还有旧 plan。
- `/resume` 后如果只恢复 transcript，不恢复 mode，engine / TUI / artifact 可能互相矛盾。
- plan 文件缺失时 `/plan` 不能报错崩溃。

## 改动 / Changes

### 1. `/new`

创建新 session 后：

- mode 默认为 `agent`。
- TUI session store 设置为 `agent`。
- 新 session 的 plan artifact 为空。

要求：

- 不从上一 session 继承 `plan` mode。
- 不复制上一 session 的 plan file。
- 如果 session id 生成后已有同名前缀 plan file，按 session id prefix 策略这应极低概率；测试中可用临时 home 避免污染。

### 2. `/clear`

清理当前 transcript 时，同时清理当前 session plan artifact。

推荐行为：

```python
clear_plan(session_id)
set_mode(session_id, "agent")
```

理由：

- `/clear` 表示当前对话上下文重置，旧 plan 不应继续影响用户视图。
- 切回 `agent` 避免用户清空后仍卡在只读 plan mode。

如果产品希望 `/clear` 只清 transcript 不改 mode，需要在文档和 UI 中非常明确。第一版推荐清 plan 且回 agent。

### 3. `/resume`

resume 后：

- 读取 session mode；缺失时默认 `agent`。
- TUI store 同步 mode。
- 如果 plan artifact 存在，`/plan` 可以展示。
- 如果 plan artifact 缺失，`/plan` 显示 no-plan 提示，不报错。

如果 Aether 当前 session store 尚未持久化 mode：

- 第一版可以在 resume 时 fallback 到 `agent`。
- wire schema 仍保留 mode 字段，后续持久化时只增加能力，不破坏消费者。

### 4. `session.current` / `plan.current` metadata

确保至少一个 RPC 能稳定返回：

```json
{
  "mode": "plan",
  "plan_path": "/home/user/.aether/plans/7f8d4f67.md",
  "has_plan": true,
  "plan_content": "..."
}
```

推荐：

- `plan.current` 返回完整 plan metadata。
- `session.current` 返回轻量 `mode`，可选返回 `has_plan`。

这样 TUI 初始加载可以快速显示 mode，用户输入 `/plan` 时再读取完整 plan content。

### 5. 后向兼容

如果未来 Aether session store 持久化 mode：

- 旧 session 缺 mode 时默认 `agent`。
- wire 字段只加不改。
- plan file 缺失不等于 session 损坏。
- `plan.current` 不应因为 artifact 缺失返回 RPC error。

## 测试 / Tests

Python 单测建议：

- `test_new_session_defaults_to_agent_mode`：新 session mode 为 `agent`。
- `test_new_session_does_not_inherit_plan_mode`：旧 session 为 `plan`，新 session 仍是 `agent`。
- `test_clear_removes_plan_artifact`：写入 plan 后 clear，`read_plan(session_id) is None`。
- `test_clear_resets_mode_to_agent`：plan mode 下 clear 后 mode 为 `agent`。
- `test_resume_defaults_missing_mode_to_agent`：旧 session 没 mode 字段时 resume 不报错，mode 为 `agent`。
- `test_resume_keeps_existing_plan_artifact`：resume 后 `plan.current` 能读到对应 session plan。
- `test_missing_plan_file_does_not_break_plan_current`：artifact 缺失返回 `has_plan=false`。

TS 单测建议：

- `/new updates session store mode to agent`。
- `/clear clears plan view state`。
- `/resume syncs mode from session.current or plan.mode_get`。
- `/plan after clear shows no plan`。
- `/plan after resume shows resumed session plan`。

## 验收 / Acceptance

- `/new` 后状态区域不显示 `plan`，`/plan` 不展示旧 session plan。
- `/clear` 后 `/plan` 显示 `Already in plan mode. No plan written yet.` 或普通 no-plan 提示，按当前 mode 行为决定，但不能显示旧 plan。
- `/resume <session>` 后，如果该 session 有 plan artifact，`/plan` 能展示。
- resume 一个没有 plan file 的 session 不报错。
- 普通 agent mode session lifecycle 不受影响。

## 不在本 PR / Deferred

- pending approval 的持久化和 resume：如果当前 approval bridge 不支持，留给后续。
- plan artifact history / archive：本 sprint 只维护每 session 最新计划。

