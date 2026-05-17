# PR 12.1 — Plan Mode Control RPC and Command Catalog

## 目标 / Goal

建立 plan mode 的后端控制面：让 CLI / gateway 能暴露 `/plan`，让 TUI 可以通过稳定 RPC 查询和切换当前 session 的 `agent` / `plan` mode，并能读取当前 session 的 plan metadata。

本 PR 不实现 TUI 行为、不写 plan artifact 文件、不重复实现写工具阻断。它只定义 wire contract，并复用已有 session state。

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

第一版返回 mode 和 plan metadata。PR 12.4 落地前可以先返回空 artifact 字段；PR 12.4 再接入真实 plan 文件。

响应 schema：

```json
{
  "session_id": "...",
  "mode": "plan",
  "plan_path": null,
  "has_plan": false,
  "plan_content": null
}
```

要求：

- `plan_path` 是 string 或 null。
- `has_plan` 是 boolean。
- `plan_content` 是 string 或 null；如果未来做大文件截断，必须另加 `truncated` 字段，不要改变字段含义。

### 3. 复用 session_state

RPC 必须调用：

- `aether.runtime.session.session_state.get_mode(session_id)`
- `aether.runtime.session.session_state.set_mode(session_id, mode)`

不要在 gateway handler 内维护第二份 mode map，也不要把 mode 塞进 TUI-only state。engine、tool blocker、approval flow 都应该观察同一份 session state。

### 4. 可选扩展 `session.current`

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

### 5. 错误处理

以下场景必须返回明确 RPC error：

- 缺少 `session_id`。
- `session_id` 对应的 session 不存在或当前 gateway 无法解析。
- `mode` 不是 `agent` / `plan`。
- 请求体类型错误，例如 `mode` 不是 string。

错误文案应短而可操作，例如：

- `session_id is required`
- `unknown session: <id>`
- `unsupported plan mode: <mode>`

### 6. 明确不在 RPC 重复 blocker

`plan.mode_set(plan)` 只改变 session mode。plan mode 下写工具阻断仍由 registry / engine pre-permission gate 负责。不要在 RPC 里实现工具白名单，否则会出现两个策略源，后续难以保持一致。

## 测试 / Tests

Python 单测建议：

- `test_command_catalog_includes_plan`：`commands.list` 或对应 catalog API 包含 `/plan`，description 为固定文案，category 为 `session`。
- `test_plan_mode_set_updates_session_state`：调用 `plan.mode_set` 为 `plan` 后，`get_mode(session_id) == "plan"`。
- `test_plan_mode_get_returns_current_state`：默认 mode 为 `agent`，设置后读回 `plan`。
- `test_plan_current_returns_mode_and_empty_metadata_before_artifact`：artifact 未接入时返回 `has_plan=false`、`plan_content=null`。
- `test_plan_mode_set_rejects_unknown_mode`：未知 mode 返回 RPC error。
- `test_plan_rpc_requires_session_id`：缺 session id 返回明确 error。
- `test_plan_rpc_rejects_unknown_session`：session 不存在返回明确 error。
- `test_plan_rpc_does_not_bypass_write_blocker`：确认 blocker 测试仍在 engine/registry 层，不需要 RPC mock 工具权限。

## 验收 / Acceptance

- `/help` 和 gateway command catalog 都能看到 `/plan`。
- TUI 可以只通过 `plan.mode_get` / `plan.mode_set` 判断和切换 mode。
- `session.current` 的旧消费者不需要改动即可继续工作。
- 计划 mode 的唯一状态源仍是 `session_state`。

## 不在本 PR / Deferred

- TUI `/plan` dispatcher 行为：PR 12.2。
- plan-mode system reminder：PR 12.3。
- plan artifact 文件读写：PR 12.4。
- approval modal UI parity：PR 12.5。

