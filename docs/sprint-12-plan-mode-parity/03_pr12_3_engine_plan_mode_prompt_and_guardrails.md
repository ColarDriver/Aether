# PR 12.3 — Engine Plan-Mode Prompt and Guardrails

## 目标 / Goal

让 engine 在当前 session 处于 plan mode 时，每一轮 LLM 调用都收到 artifact-aware system reminder，明确当前只能探索和规划、唯一可编辑文件是当前 session plan file，最终必须调用 `exit_plan_mode` 请求用户批准。

同时回归确认现有写工具阻断仍在 permission prompt 之前生效，并且 `exit_plan_mode` 不被 blocker 拦截。

## 当前问题 / Current Problem

Aether 已经有 plan mode 状态和部分 blocker，但模型侧缺少每轮稳定提醒。只靠用户最初输入 `/plan` 或工具结果文案，长对话中容易漂移：

- 模型可能在 plan mode 下尝试写业务文件或跑 shell。
- 模型不知道 plan artifact path，因此无法像 open-claude-code 一样边探索边维护 plan 文件。
- 模型可能用普通文本问 "这个计划可以吗？"，而不是调用 `exit_plan_mode`。
- 模型可能误以为 plan mode 只是低权限 agent mode，而不是 approval 前的只读规划阶段。

## 改动 / Changes

### 1. 注入点

在 `AgentEngine._prepare_session_and_system_prompt` 或等价系统 prompt 拼装点检测：

```python
from aether.runtime.session.session_state import get_mode

if request.session_id and get_mode(request.session_id) == "plan":
    active_system = append_plan_mode_reminder(
        active_system,
        session_id=request.session_id,
    )
```

要求：

- 使用 PR 12.1 的同一 session state，不维护第二份 mode。
- 注入发生在每轮 provider request 前。
- agent mode 下完全不注入。
- reminder 文本模板稳定；允许插入当前 session 的 plan path 和 has_plan 状态，因为这是模型正确写 plan file 的必要信息。

### 2. Reminder 文案

建议第一版：

```text
<plan_mode_reminder>
Plan mode is active. Do not execute implementation work yet.

Plan file: /home/user/.aether/plans/7f8d4f67.md
Plan file status: no plan file exists yet.

You may read, search, list files, fetch context, and ask clarifying questions.
The only file you may write or edit is the plan file above. Use write_file to create it or file_edit to revise it.
Do not write other files, run shell commands, commit, dispatch subagents, update todos, or create/update/delete memory.
When the plan is ready, call exit_plan_mode to request approval. Do not ask for plan approval in ordinary text or ask_user_question.
</plan_mode_reminder>
```

语义要求：

- 明确当前处于 plan mode。
- 明确当前 plan file path。
- 明确 plan file 是否已存在。
- 明确唯一允许编辑的文件是当前 session plan file。
- 明确禁止变更动作：写其他文件、shell、commit、subagent dispatch、`todo_write`、memory write/update/forget。
- 明确允许只读动作：read/search/list/web fetch/ask_user_question。
- 明确最终必须调用 `exit_plan_mode`。Aether 兼容 `exit_plan_mode(plan="...")`，但首选从 artifact 读取 plan。
- 明确不能用普通文本或 `ask_user_question` 问 "计划是否可以"。

文本需要短，不要把完整 policy 表塞进 prompt。真正阻断仍靠 engine gate。与 OCC 的完整五阶段工作流相比，Aether 可以保留更短模板，但不能遗漏 plan path、only editable plan file、ExitPlanMode 这三个核心点。

### 3. 与 `enter_plan_mode` 工具结果保持一致

如果 `enter_plan_mode` 工具当前返回的文案和 reminder 不一致，应调整工具结果文案，保证模型不会收到冲突指令。

例如工具结果可以简化为：

```text
Plan mode enabled. Read/search freely. You may only write the session plan file. Call exit_plan_mode when ready for approval.
```

不要在工具结果里暗示可以执行 shell、写业务文件、或直接用文本等待用户批准。

### 4. 写工具阻断保持现有路径

保持现有：

- `WRITE_TOOLS_BLOCKED_IN_PLAN`
- engine pre-permission gate
- registry / tool policy gate

- plan mode 下 shell / commit / memory mutation / subagent dispatch / `todo_write` 在 permission prompt 之前被拒绝。
- plan mode 下 `write_file` / `file_edit` 只有目标 path 等于当前 session plan file 时允许继续；其他路径在 permission prompt 之前拒绝。
- 拒绝结果带上可观察 metadata，例如 PR 12.7 中的 `plan_mode_blocked`。
- 不要让这些请求进入普通 permission modal。
- `exit_plan_mode` 必须可用，不能因为 blocker 策略过宽被拦截。

### 5. ask_user_question 约束

`ask_user_question` 工具描述或 plan-mode reminder 中必须包含 OCC 对齐语义：

- plan mode 下只能用它澄清需求或让用户在实现方向之间选择。
- 不能用它问 "这个 plan 可以吗" / "是否开始实现"。
- 不能在用户还看不到 approval modal 前要求用户审阅隐藏 plan。
- 需要批准时必须调用 `exit_plan_mode`。

### 6. Engine metadata

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
- `test_plan_reminder_contains_plan_path`：reminder 包含 `get_plan_path(session_id)`。
- `test_plan_reminder_mentions_plan_exists_state`：有/无 artifact 时状态正确。
- `test_plan_reminder_mentions_only_editable_plan_file`。
- `test_plan_mode_blocks_file_edit_before_permission_prompt`：plan mode 下 `file_edit` 被 engine gate 拒绝，不触发 permission bridge。
- `test_plan_mode_allows_file_edit_for_current_plan_file`：目标为当前 plan path 时允许继续。
- `test_plan_mode_blocks_shell_before_permission_prompt`：shell 同理。
- `test_exit_plan_mode_is_not_blocked`：plan mode 下 `exit_plan_mode` 可以触发 approval flow。
- `test_enter_plan_mode_tool_message_matches_policy`：工具结果文案包含 read/search、只允许写 plan file 和 `exit_plan_mode` 指引，不含冲突语义。
- `test_ask_user_question_prompt_disallows_plan_approval`。

## 验收 / Acceptance

- plan mode 下每轮 provider request 都包含 plan reminder、plan path 和 has_plan 状态。
- agent mode 下 provider request 不包含 plan reminder。
- plan mode 下非 plan-file 写工具不会弹普通 permission prompt。
- plan mode 下当前 session plan file 可以通过 `write_file` / `file_edit` 更新。
- `exit_plan_mode()` 和兼容的 `exit_plan_mode(plan=...)` 仍可触发 plan approval。
- 模型收到的 plan mode 指令只来自同一套短 reminder / tool result 语义，没有互相矛盾的要求。

## 不在本 PR / Deferred

- `/plan` slash entry：PR 12.2。
- plan artifact 写入：PR 12.4。
- approval modal markdown 和 approve/reject mode transition：PR 12.5。
