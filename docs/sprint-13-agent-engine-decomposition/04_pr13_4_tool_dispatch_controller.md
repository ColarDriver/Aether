# PR 13.4 — Tool Dispatch Controller

## 目标

把 tool call normalization、permission gate、plan-mode blocker、dedup、delegate cap、registry dispatch、after_tool middleware、post_tool hooks 从 `AgentEngine` 中抽出，形成可单测的 `ToolDispatchController`。

## 当前问题

Tool dispatch 是 Agent runtime 中最容易引入回归的部分，因为它同时影响：

- 模型输出的 tool call 修复与 canonicalization。
- plan mode 写工具阻断。
- dangerous tool permission prompt。
- allow/reject/session rule 处理。
- duplicate tool call 去重。
- delegate-class tool cap。
- registry dispatch。
- middleware `after_tool`。
- post_tool_use / post_tool_use_failure hooks。
- tool result metadata 和 UI preview。

这些逻辑现在由 `AgentEngine` 多个私有方法承载。后续任何 plan mode、permission、subagent 或 diagnostics 改动都可能影响 tool dispatch。

## 实现改动

### 新增 controller

新增 `aether/agents/runtime/tool_dispatch.py`。

核心类型：

- `ToolDispatchRequest`
  - `tool_calls: list[ToolCall]`
  - `messages: list[dict[str, Any]]`
  - `context: TurnContext`
  - `request: EngineRequest`
  - `iteration: int`

- `ToolDispatchResult`
  - `tool_results: list[ToolResult]`
  - `messages: list[dict[str, Any]]`
  - `should_continue: bool`
  - `exit_reason: ExitReason | None`
  - `error_text: str | None`
  - `all_tools_cheap: bool`
  - `dispatched_count: int`

- `ToolDispatchController`
  - constructed with `services`, `hooks`, `config`, and helper adapter。
  - method `dispatch(request: ToolDispatchRequest) -> ToolDispatchResult`

### Controller responsibilities

Controller 负责：

- 调用 `prepare_tool_calls(...)` 和 phantom / unknown tool repair 后的 dispatch plan。
- plan mode blocker：确保 blocked write tools 在 permission prompt 前返回 synthetic tool error。
- plan artifact exception：保持当前只允许 plan file write 的行为。
- permission prompt gate：构造 `ToolPermissionRequest`、处理 approve/reject/session rule。
- duplicate tool call dedup。
- delegate-class cap。
- 调用 `ToolRegistry.dispatch(...)`。
- 运行 `middleware_pipeline.run_after_tool(...)`。
- 触发 `hooks.post_tool_use(...)` / `hooks.post_tool_use_failure(...)`。
- 写入 tool result metadata，例如 permission、blocked reason、edited_paths、diagnostics hooks 所需字段。

### AgentEngine 迁移

- `run_loop` 中收到 `response.tool_calls` 后调用 controller。
- `AgentEngine` 保留 high-level branch：有 tool calls 则 dispatch，dispatch 后继续下一 iteration 或 finalize。
- 原私有方法可保留为 adapter，逐步迁移，避免单 PR 改动过大。

### 必须不变的行为

- plan mode 下非 plan 文件写入仍在 permission prompt 前被拒绝。
- `exit_plan_mode` 仍可用，不被 write-tool blocker 拦截。
- dangerous shell / file edit 仍触发 permission UI。
- 用户 reject permission 后 tool result 是 error，主 loop 继续让模型修正。
- `fail_on_tool_error=True` 时仍按原路径失败。
- tool result message shape 不变。

## 测试

新增：

- `aether/tests/agents/runtime/test_tool_dispatch_controller.py`

覆盖：

- read-only tool dispatch 成功。
- dangerous tool 触发 permission prompter。
- permission approve / reject / session rule 行为不变。
- plan mode blocker 在 permission 前触发。
- `exit_plan_mode` 不被 blocker 拦截。
- duplicate tool calls dedup 行为不变。
- delegate cap 超限返回 synthetic error。
- registry dispatch 抛异常时触发 `post_tool_use_failure`。
- after_tool middleware 仍在 post_tool_use 前生效。

回归：

- permission modal gateway / TUI tests。
- plan mode artifact write exception tests。
- diagnostics edit hook tests。
- subagent task tool tests。

## 验收

- tool dispatch 相关逻辑从 `AgentEngine` 主类显著减少。
- Controller 可用 fake registry / fake permission prompter 独立测试。
- Tool public contract 与 wire schema 不变。
- 所有现有 tool tests 全绿。
