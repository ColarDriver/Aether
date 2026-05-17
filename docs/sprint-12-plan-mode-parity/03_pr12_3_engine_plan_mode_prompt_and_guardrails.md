# PR 12.3 — Engine Plan-Mode Prompt and Guardrails

## 目标 / Goal

让 engine 在当前 session 处于 plan mode 时，每一轮 LLM 调用都收到短而稳定的 system reminder，明确当前只能探索和规划，最终必须调用 `exit_plan_mode(plan="...")` 请求用户批准。

同时回归确认现有写工具阻断仍在 permission prompt 之前生效，并且 `exit_plan_mode` 不被 blocker 拦截。

## 当前问题 / Current Problem

Aether 已经有 plan mode 状态和部分 blocker，但模型侧缺少每轮稳定提醒。只靠用户最初输入 `/plan` 或工具结果文案，长对话中容易漂移：

- 模型可能在 plan mode 下尝试写文件或跑 shell。
- 模型可能用普通文本问 "这个计划可以吗？"，而不是调用 `exit_plan_mode`。
- 模型可能误以为 plan mode 只是低权限 agent mode，而不是 approval 前的只读规划阶段。

## 改动 / Changes

### 1. 注入点

在 `AgentEngine._prepare_session_and_system_prompt` 或等价系统 prompt 拼装点检测：

```python
from aether.runtime.session.session_state import get_mode

if request.session_id and get_mode(request.session_id) == "plan":
    active_system = append_plan_mode_reminder(active_system)
```

要求：

- 使用 PR 12.1 的同一 session state，不维护第二份 mode。
- 注入发生在每轮 provider request 前。
- agent mode 下完全不注入。
- reminder 文本固定，避免每轮因为动态信息破坏 prompt cache。

### 2. Reminder 文案

建议第一版：

```text
<plan_mode_reminder>
You are currently in plan mode. Do not make changes: do not write files, run shell commands, commit, dispatch subagents, or create/update/delete memory. You may read, search, list files, fetch web/context, and ask the user clarifying questions. When you have a concrete plan, you must call exit_plan_mode(plan="...") to request approval. Do not ask for plan approval in ordinary text.
</plan_mode_reminder>
```

语义要求：

- 明确当前处于 plan mode。
- 明确禁止变更动作：写入、shell、commit、subagent dispatch、memory write/update/forget。
- 明确允许只读动作：read/search/list/web fetch/ask_user_question。
- 明确最终必须调用 `exit_plan_mode(plan="...")`。
- 明确不能用普通文本问 "计划是否可以"。

文本需要短，不要把完整 policy 表塞进 prompt。真正阻断仍靠 engine gate。

### 3. 与 `enter_plan_mode` 工具结果保持一致

如果 `enter_plan_mode` 工具当前返回的文案和 reminder 不一致，应调整工具结果文案，保证模型不会收到冲突指令。

例如工具结果可以简化为：

```text
Plan mode enabled. Read/search only; call exit_plan_mode(plan="...") when ready for approval.
```

不要在工具结果里暗示可以写 plan file、执行 shell、或直接用文本等待用户批准。

### 4. 写工具阻断保持现有路径

保持现有：

- `WRITE_TOOLS_BLOCKED_IN_PLAN`
- engine pre-permission gate
- registry / tool policy gate

要求：

- plan mode 下 shell / file edit / write file / commit / memory mutation / subagent dispatch 在 permission prompt 之前被拒绝。
- 拒绝结果带上可观察 metadata，例如 PR 12.7 中的 `plan_mode_blocked`。
- 不要让这些请求进入普通 permission modal。
- `exit_plan_mode` 必须可用，不能因为 blocker 策略过宽被拦截。

### 5. Engine metadata

在 provider request 或 run metadata 中增加可观察字段：

```json
{
  "plan_mode_active": true
}
```

约束：

- agent mode 下字段为 false 或缺省，按现有 metadata 风格决定。
- 该字段用于调试和测试，不作为权限源。

## 测试 / Tests

Python 单测建议：

- `test_plan_mode_injects_system_reminder`：设置 session mode 为 `plan`，scripted provider 收到的 system prompt 包含 `<plan_mode_reminder>`。
- `test_agent_mode_does_not_inject_plan_reminder`：默认 `agent` mode 下 system prompt 不含 reminder。
- `test_plan_reminder_is_stable`：同 session 多轮请求 reminder 文本一致。
- `test_plan_mode_blocks_file_edit_before_permission_prompt`：plan mode 下 `file_edit` 被 engine gate 拒绝，不触发 permission bridge。
- `test_plan_mode_blocks_shell_before_permission_prompt`：shell 同理。
- `test_exit_plan_mode_is_not_blocked`：plan mode 下 `exit_plan_mode` 可以触发 approval flow。
- `test_enter_plan_mode_tool_message_matches_policy`：工具结果文案包含 read/search only 和 `exit_plan_mode` 指引，不含冲突语义。

## 验收 / Acceptance

- plan mode 下每轮 provider request 都包含 plan reminder。
- agent mode 下 provider request 不包含 plan reminder。
- plan mode 下写工具不会弹普通 permission prompt。
- `exit_plan_mode(plan=...)` 仍可触发 plan approval。
- 模型收到的 plan mode 指令只来自同一套短 reminder / tool result 语义，没有互相矛盾的要求。

## 不在本 PR / Deferred

- `/plan` slash entry：PR 12.2。
- plan artifact 写入：PR 12.4。
- approval modal markdown 和 approve/reject mode transition：PR 12.5。

