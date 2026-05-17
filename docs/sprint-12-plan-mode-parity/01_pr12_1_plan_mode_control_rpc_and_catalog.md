# PR 12.1 — Plan Mode Control RPC and Command Catalog

## 目标 / Goal

建立 plan mode 的后端控制面：让 CLI / gateway 能暴露 `/plan`，让 TUI 可以通过稳定 RPC 查询、切换、清理当前 session 的 `agent` / `plan` mode，并能读取当前 session 的 plan metadata。

本 PR 不实现 TUI 行为、不实现 engine prompt、不实现 tool blocker 细节。它只定义 wire contract，并复用已有 session state / session record，给后续 PR 提供稳定接口。

## 当前问题 / Current Problem

当前 Aether 已经有 `enter_plan_mode` / `exit_plan_mode` tool 和 plan approval bridge，但它们主要服务模型侧。用户从 TUI 输入 `/plan` 时，没有后端 command catalog 项，也没有 gateway RPC 可以让前端显式切换 session mode。

缺口：

- `/help` / slash completion 无法发现 `/plan`。
- TUI 无法读当前 session 是否已经处于 plan mode。
- TUI 无法以用户动作把 session 切到 plan mode。
- `session.current` 如果不带 mode，前端只能维护易漂移的本地状态。

## 改动 / Changes

### 1. 注册 `/plan` command catalog

修改 `aether/cli/commands.py`，增加：

```python
Command(
    name="/plan",
    description="Enable plan mode or view the current session plan",
    ...
)
```

要求：

- command name 使用 `/plan`，与其他 slash command 一致。
- description 使用固定文案：`Enable plan mode or view the current session plan`。
- 不在 CLI command 层直接修改 session state；catalog 只负责发现和帮助文本。

修改 `aether/gateway/handlers/commands_methods.py`，把 `/plan` 归类为 `session`。`control` 也可解释，但推荐 `session`，因为 `/plan` 操作当前会话 mode 和当前会话 artifact，不是全局控制。

### 2. 新增 plan gateway RPC

在 gateway command/session handler 附近新增 `plan.*` RPC。具体文件以现有 method dispatcher 为准，保持命名风格与 `session.current`、`commands.list` 一致。

建议新增：

```text
aether/gateway/handlers/plan_methods.py
```

并在 gateway handler registration 中注册：

```python
method("plan.mode_get", long=False)(plan_mode_get)
method("plan.mode_set", long=False)(plan_mode_set)
method("plan.current", long=False)(plan_current)
method("plan.clear", long=False)(plan_clear)
```

#### `plan.mode_get`

请求：

```json
{
  "session_id": "..."
}
```

响应：

```json
{
  "session_id": "...",
  "mode": "agent"
}
```

#### `plan.mode_set`

请求：

```json
{
  "session_id": "...",
  "mode": "plan"
}
```

响应：

```json
{
  "session_id": "...",
  "mode": "plan"
}
```

支持 mode：

- `agent`
- `plan`

未知 mode 返回明确 RPC error，不 silently fallback。

#### `plan.current`

第一版返回 mode 和 plan metadata。PR 12.4 落地后，`plan_path` 必须来自 `plan_artifact.get_plan_path(session_id)`。

响应 schema：

```json
{
  "session_id": "...",
  "mode": "plan",
  "plan_path": "/home/user/.aether/plans/7f8d4f67.md",
  "has_plan": false,
  "plan_content": null
}
```

要求：

- `plan_path` 是 string 或 null。只要 session valid，就应尽量返回稳定路径，即使文件尚不存在；这样 engine prompt 和 `/plan open` 可以给用户明确目标。
- `has_plan` 是 boolean，表示文件是否存在。
- `plan_content` 是 string 或 null；不存在时为 null，存在但为空时为 `""`。
- 如果未来做大文件截断，必须另加 `truncated` 字段，不要改变字段含义。

#### `plan.clear`

请求：

```json
{
  "session_id": "..."
}
```

响应：

```json
{
  "session_id": "...",
  "mode": "agent",
  "plan_path": "/home/user/.aether/plans/7f8d4f67.md",
  "has_plan": false,
  "plan_content": null
}
```

语义：

- 清理当前 session plan artifact。
- 将 session mode 重置为 `agent`。
- 持久化 session record mode。
- 主要给 `/clear` 调用，不用于普通 `/plan` 展示。

### 3. 复用 session_state

RPC 必须调用：

- `aether.runtime.session.session_state.get_mode(session_id)`
- `aether.runtime.session.session_state.set_mode(session_id, mode)`

不要在 gateway handler 内维护第二份 mode map，也不要把 mode 塞进 TUI-only state。engine、tool blocker、approval flow 都应该观察同一份 session state。

### 4. 持久化 session mode

为了 `/resume` 能恢复 mode，`SessionRecord` 增加可选字段：

```python
mode: str = "agent"
```

规则：

- 新 session 默认 `agent`。
- `SessionRecord.from_json` 对缺失或未知 mode fallback 到 `agent`。
- `plan.mode_set` 成功后同步写入 session record。
- `exit_plan_mode` approve/reject 后同步写入 session record。
- `/clear` / `plan.clear` 写回 `agent`。

### 5. 可选扩展 `session.current`

`session.current` 可以追加返回：

```json
{
  "mode": "agent"
}
```

约束：

- 只加字段，不改已有字段名称和类型。
- 旧的 `SessionInfo` 消费者不能因为缺少新字段而报错。
- TS 侧类型应把 mode 当作可选字段，直到所有 gateway 版本都支持。

### 6. 错误处理

以下场景必须返回明确 RPC error：

- 缺少 `session_id`。
- `session_id` 对应的 session 不存在或当前 gateway 无法解析。
- `mode` 不是 `agent` / `plan`。
- plan artifact path 无法生成，说明 session id 非法。
- 请求体类型错误，例如 `mode` 不是 string。

错误文案应短而可操作，例如：

- `session_id is required`
- `unknown session: <id>`
- `unsupported plan mode: <mode>`

### 7. 明确不在 RPC 重复 blocker

`plan.mode_set(plan)` 只改变 session mode。plan mode 下写工具阻断仍由 PR 12.3 / PR 12.4 的 registry / engine pre-permission gate 负责。不要在 RPC 里实现工具白名单，否则会出现两个策略源，后续难以保持一致。

## Open-Claude-Code Parity Notes

- OCC 的 permission mode 是 richer context，会保存 `prePlanMode`，退出后恢复。Aether 当前只有 `agent` / `plan`，本 PR 只需要回 `agent`；若未来引入 `auto` / `bypass` 等 mode，应在 session state 中增加 `pre_plan_mode`。
- OCC plan slug 在 session log 中恢复；Aether 第一版使用 session id prefix，不在本 PR 引入 slug cache。
- OCC command catalog 是本地 command registry；Aether 通过 gateway catalog 提供给 TUI，语义等价。

## 测试 / Tests

Python 单测建议：

- `test_command_catalog_includes_plan`：`commands.list` 或对应 catalog API 包含 `/plan`，description 为固定文案，category 为 `session`。
- `test_plan_mode_set_updates_session_state`：调用 `plan.mode_set` 为 `plan` 后，`get_mode(session_id) == "plan"`。
- `test_plan_mode_set_persists_session_record`：session record `mode == "plan"`。
- `test_plan_mode_get_returns_current_state`：默认 mode 为 `agent`，设置后读回 `plan`。
- `test_plan_current_returns_mode_and_plan_metadata`：返回 `mode`、`plan_path`、`has_plan`、`plan_content`。
- `test_plan_current_returns_path_even_when_missing`：artifact 缺失时仍返回稳定 `plan_path`。
- `test_plan_clear_clears_artifact_and_resets_mode`：`plan.clear` 清文件并回 `agent`。
- `test_plan_mode_set_rejects_unknown_mode`：未知 mode 返回 RPC error。
- `test_plan_rpc_requires_session_id`：缺 session id 返回明确 error。
- `test_plan_rpc_rejects_unknown_session`：session 不存在返回明确 error。
- `test_session_current_includes_optional_mode`。
- `test_old_session_without_mode_defaults_agent`。

## 验收 / Acceptance

- `/help` 和 gateway command catalog 都能看到 `/plan`。
- TUI 可以只通过 `plan.mode_get` / `plan.mode_set` 判断和切换 mode。
- `plan.current` 能返回当前 session 的 mode、plan path、has_plan、content。
- `plan.clear` 可供 `/clear` 调用并回到 agent mode。
- `session.current` 的旧消费者不需要改动即可继续工作。
- 计划 mode 的唯一状态源仍是 `session_state`。

## 不在本 PR / Deferred

- TUI `/plan` dispatcher 行为：PR 12.2。
- plan-mode system reminder：PR 12.3。
- plan artifact 文件读写：PR 12.4。
- approval modal UI parity：PR 12.5。
