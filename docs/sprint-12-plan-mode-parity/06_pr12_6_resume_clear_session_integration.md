# PR 12.6 — Resume, Clear, and Session Lifecycle Integration

## 目标 / Goal

把 plan mode 和 plan artifact 接入 session 生命周期：`/new` 不继承旧状态，`/clear` 不显示旧计划，`/resume` 能恢复对应 session 的 mode 和 plan artifact，并让缺失 / 空 artifact 成为可读状态而不是错误。

本 PR 依赖 PR 12.1 的 mode RPC、PR 12.2 的 TUI store、PR 12.4 的 plan artifact。

## 当前问题 / Current Problem

如果只实现 `/plan` 和 `exit_plan_mode`，session 生命周期会出现状态漂移：

- `/new` 后可能沿用旧 session 的 plan mode。
- `/clear` 清理 transcript 后，`/plan` 仍显示旧计划，用户会误以为新上下文还有旧 plan。
- `/resume` 后如果只恢复 transcript，不恢复 mode，engine / TUI / artifact 可能互相矛盾。
- plan 文件缺失时 `/plan` 不能报错崩溃。
- 如果 plan path 基于 session id prefix，测试或极端情况下可能出现 prefix 碰撞，需要明确 first version 的约束和后续迁移策略。

## 改动 / Changes

### 1. `/new`

创建新 session 后：

- mode 默认为 `agent`。
- TUI session store 设置为 `agent`。
- 新 session 的 plan artifact 为空。

要求：

- 不从上一 session 继承 `plan` mode。
- 不复制上一 session 的 plan file。
- 旧 session 的 plan file 不删除。
- 如果 session id 生成后已有同名前缀 plan file，按 session id prefix 策略这应极低概率；测试中可用临时 home 避免污染。
- 后续若改 readable slug，`/new` 应生成新 slug，不复用旧 slug。

### 2. `/clear`

清理当前 transcript 时，同时清理当前 session plan artifact。

推荐行为：

```python
clear_plan(session_id)
set_mode(session_id, "agent")
persist_session_mode(session_id, "agent")
```

理由：

- `/clear` 表示当前对话上下文重置，旧 plan 不应继续影响用户视图。
- 切回 `agent` 避免用户清空后仍卡在只读 plan mode。

如果产品希望 `/clear` 只清 transcript 不改 mode，需要在文档和 UI 中非常明确。第一版推荐清 plan 且回 agent。

TUI 行为：

- 调用 `plan.clear`。
- 本地 session store 设置 `mode="agent"`。
- 清空 transcript。
- 不显示旧 plan note。

### 3. `/resume`

resume 后：

- 读取 session mode；缺失时默认 `agent`。
- TUI store 同步 mode。
- 如果 plan artifact 存在，`/plan` 可以展示。
- 如果 plan artifact 缺失，`/plan` 显示 no-plan 提示，不报错。
- 如果 session record mode 为 `plan`，engine 下一轮继续注入 plan reminder，并使用该 session 的 plan path。

如果 Aether 当前 session store 尚未持久化 mode：

- 第一版可以在 resume 时 fallback 到 `agent`。
- wire schema 仍保留 mode 字段，后续持久化时只增加能力，不破坏消费者。

### 4. session record schema

`SessionRecord` 应包含：

```python
mode: str = "agent"
```

要求：

- `from_json` 对旧记录缺失 `mode` fallback 到 `agent`。
- `from_json` 对未知 mode fallback 到 `agent`。
- `save_session` 不丢弃 mode。
- `session.resume` 把 persisted mode 写入 in-process `session_state`。
- `session.create` / `/new` 清理 in-process mode，并保存 `agent`。

### 5. `session.current` / `plan.current` metadata

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
- `plan.current` 即使文件不存在也返回稳定 `plan_path`。
- `session.current` 返回轻量 `mode`，可选返回 `has_plan`。

这样 TUI 初始加载可以快速显示 mode，用户输入 `/plan` 时再读取完整 plan content。

### 6. 后向兼容

如果未来 Aether session store 持久化 mode：

- 旧 session 缺 mode 时默认 `agent`。
- wire 字段只加不改。
- plan file 缺失不等于 session 损坏。
- `plan.current` 不应因为 artifact 缺失返回 RPC error。

### 7. Open-Claude-Code Parity Notes

OCC 更复杂：

- 使用 word slug，并把 slug 写入 session log。
- resume 时从 log 恢复 slug。
- remote 场景下如果 plan file 缺失，会尝试从 file snapshot 或 message history 恢复。
- fork session 时生成新 slug 并复制旧 plan。

Aether 第一版明确不实现上述恢复策略。Aether 的可验收语义是：同一个 session id 映射同一个 `${AETHER_HOME}/plans/<session-prefix>.md`；resume 只按这个路径读取，文件缺失就显示 no-plan。

## 测试 / Tests

Python 单测建议：

- `test_new_session_defaults_to_agent_mode`：新 session mode 为 `agent`。
- `test_new_session_does_not_inherit_plan_mode`：旧 session 为 `plan`，新 session 仍是 `agent`。
- `test_clear_removes_plan_artifact`：写入 plan 后 clear，`read_plan(session_id) is None`。
- `test_clear_resets_mode_to_agent`：plan mode 下 clear 后 mode 为 `agent`。
- `test_resume_defaults_missing_mode_to_agent`：旧 session 没 mode 字段时 resume 不报错，mode 为 `agent`。
- `test_resume_restores_persisted_plan_mode`：record mode 为 `plan` 时，resume 后 `get_mode(session_id) == "plan"`。
- `test_resume_keeps_existing_plan_artifact`：resume 后 `plan.current` 能读到对应 session plan。
- `test_missing_plan_file_does_not_break_plan_current`：artifact 缺失返回 `has_plan=false`。
- `test_plan_current_missing_file_still_returns_path`。
- `test_clear_does_not_delete_other_session_plan_artifact`。
- `test_invalid_mode_in_old_record_defaults_agent`。

TS 单测建议：

- `/new updates session store mode to agent`。
- `/clear clears plan view state`。
- `/resume syncs mode from session.current or plan.mode_get`。
- `/plan after clear shows no plan`。
- `/plan after resume shows resumed session plan`。
- banner after resume reflects `plan` if resumed session is in plan mode。

## 验收 / Acceptance

- `/new` 后状态区域不显示 `plan`，`/plan` 不展示旧 session plan。
- `/clear` 后 `/plan` 显示 `Already in plan mode. No plan written yet.` 或普通 no-plan 提示，按当前 mode 行为决定，但不能显示旧 plan。
- `/clear` 后 mode 应为 `agent`；如果用户再次输入裸 `/plan`，它会重新进入 plan mode。
- `/resume <session>` 后，如果该 session 有 plan artifact，`/plan` 能展示。
- resume 一个没有 plan file 的 session 不报错。
- resume 一个 mode 为 `plan` 的 session 后，状态区域显示 `plan`，下一轮 engine 注入 plan reminder。
- 普通 agent mode session lifecycle 不受影响。

## 不在本 PR / Deferred

- pending approval 的持久化和 resume：如果当前 approval bridge 不支持，留给后续。
- plan artifact history / archive：本 sprint 只维护每 session 最新计划。
- OCC word slug recovery / fork copy：后续如需要再做迁移设计。
