# PR 7.2 - Engine Permission Gate

## 目标

在 `AgentEngine` 工具 dispatch 前插入权限 gate。gate 必须发生在工具真实执行前，拒绝时返回 synthetic `ToolResult`，并且所有后续 middleware、消息追加、metadata 统计都保持一致。

## 插入点

主路径在 `backend/harness/aether/agents/core/agent.py`：

```python
pre_tool = self.services.middleware_pipeline.run_before_tool(call, context)
...
result = self.services.tool_registry.dispatch(tool_call, context)
```

需要在 `run_before_tool()` 之后、`tool_registry.dispatch()` 之前调用 permission gate。

同时要覆盖 `_dispatch_synthesized_tool_calls(...)` 中同样的 dispatch 路径，否则 phantom tool recovery 生成的工具调用可以绕过权限系统。

## Gate 语义

流程：

| 步骤 | 行为 |
|---|---|
| 1 | 如果 `pre_tool` 已经是 `ToolResult`，说明 middleware 已短路，跳过 permission gate |
| 2 | 根据 tool name、arguments、session rules、config 计算 policy |
| 3 | `allow` 直接 dispatch |
| 4 | `deny` 生成 synthetic denied result，不 dispatch |
| 5 | `ask` 构建 `ToolPermissionRequest`，调用 `request.tool_permission_prompter.request_tool_permission(...)` |
| 6 | `allow_once` dispatch 一次 |
| 7 | `allow_session` 写入 session rule，然后 dispatch |
| 8 | `reject` / `abort` 生成 synthetic denied result，不 dispatch |

Synthetic result 建议：

```python
ToolResult(
    tool_call_id=tool_call.id,
    name=tool_call.name,
    content="permission denied by user before executing tool",
    is_error=True,
    metadata={
        "permission_denied": True,
        "permission_decision": "reject",
        "tool_executed": False,
    },
)
```

用户反馈要进入 content，便于模型修正：

```python
content = "permission denied by user before executing tool"
if decision.feedback:
    content += f": {decision.feedback}"
```

## Middleware 顺序

`before_tool` 仍先运行，因为 guardrail middleware 可能重写 tool args 或短路。但 UI middleware 的行为需要 PR 7.3 改：它不能在 permission 未批准前把真实工具 call 渲染成已执行状态。

`after_tool` 必须对 synthetic denied result 继续运行，原因：

| 原因 | 说明 |
|---|---|
| redaction | 拒绝反馈可能含敏感路径或文本 |
| stats | 需要统计 permission denial |
| UI | UI 需要收到结束事件来清 activity 状态 |
| transcript consistency | 模型必须收到 tool result，而不是丢失 tool call |

## Engine helper

建议在 `AgentEngine` 中新增私有 helper，避免两个 dispatch 分支复制逻辑：

```python
def _maybe_request_tool_permission(
    self,
    tool_call: ToolCall,
    *,
    request: EngineRequest,
    context: TurnContext,
) -> ToolPermissionDecision: ...

def _build_permission_denied_result(
    self,
    tool_call: ToolCall,
    decision: ToolPermissionDecision,
) -> ToolResult: ...
```

更好的实现是新增 `ToolDispatchGate` runtime helper，由 engine 注入 config、session rules 和 preview builder。这样 engine 主循环不会继续膨胀。

## Metadata

`TurnContext.metadata` 建议累计：

```python
{
    "tool_permissions": {
        "asked": 1,
        "allowed_once": 1,
        "allowed_session": 0,
        "denied": 0,
        "non_interactive_denied": 0,
    }
}
```

`EngineResult.metadata` 应镜像这个字段，方便 CLI footer、trajectory、测试断言读取。

## 中断交互

如果用户在确认 overlay 打断：

| 场景 | 行为 |
|---|---|
| ESC / Ctrl-C 取消当前确认 | decision 为 `ABORT`，生成 denied synthetic result，继续让模型看到拒绝 |
| 全局 turn interrupt 已触发 | engine 优先走 Sprint 6 interrupt 逻辑，退出为 `ExitReason.INTERRUPTED` |
| prompter timeout | 非交互等价 deny，并标记 `permission_timeout=True` |

注意：权限确认本身不是 LLM streaming。它只是 engine worker thread 等待 UI decision，不涉及服务端模型继续推理。

## 测试

新增 `tests/engine/test_tool_permission_gate.py`：

```python
def test_permission_allow_once_dispatches_tool(): ...
def test_permission_reject_does_not_dispatch_tool(): ...
def test_permission_reject_appends_tool_result_message(): ...
def test_permission_allow_session_skips_second_prompt_for_same_rule(): ...
def test_non_interactive_dangerous_tool_denied_without_prompter(): ...
def test_before_tool_short_circuit_skips_permission_prompt(): ...
def test_after_tool_runs_for_permission_denied_result(): ...
def test_synthesized_tool_calls_also_go_through_permission_gate(): ...
```

Fake tool 需要记录 `execute_calls`，确保拒绝路径没有执行。

## 验收门

- 所有危险工具 dispatch 前都经过 gate。
- 拒绝路径没有任何磁盘、subprocess、副作用。
- `after_tool` 对真实结果和 synthetic denied result 都运行。
- 非交互默认拒绝被 engine 测试覆盖。
- 现有 tool error、unknown tool、schema injection 路径不回归。

