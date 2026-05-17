# PR 12.4 — Plan Artifact Storage and Tools

## 目标 / Goal

为每个 session 提供一个稳定的 Markdown plan artifact，让 `/plan` 能查看最近计划，`/plan open` 能打开文件，模型在 plan mode 中能只写这个 plan 文件，`exit_plan_mode` 能从 artifact 读取最终计划并提交 approval。

第一版使用 `${AETHER_HOME}/plans/<session-prefix>.md`，不引入随机 word slug 或额外配置面。该文件是 plan mode 的共享数据源，不只是 `exit_plan_mode(plan=...)` 的备份。

## 当前问题 / Current Problem

`exit_plan_mode` 可以携带 plan 文本触发 approval，但如果 plan 内容没有稳定文件承载，会和 open-claude-code 的核心语义不一致：

- `/plan` 无法展示最近计划。
- `/plan open` 没有可打开目标。
- resume 后无法按 session 查看历史计划。
- clear/new 生命周期无法定义 "旧 plan 是否还在"。
- 模型无法边探索边更新 plan file。
- `exit_plan_mode` 如果只依赖 `plan` 参数，approval UI / hook / resume 都没有 artifact-first 数据源。

## 改动 / Changes

### 1. 新增 plan artifact 模块

新增：

```text
aether/runtime/session/plan_artifact.py
```

API：

```python
from pathlib import Path

def get_plan_path(session_id: str) -> Path:
    ...

def read_plan(session_id: str) -> str | None:
    ...

def write_plan(session_id: str, content: str) -> Path:
    ...

def clear_plan(session_id: str) -> None:
    ...
```

默认目录：

```text
${AETHER_HOME}/plans
```

如果 `AETHER_HOME` 未设置，沿用 Aether 现有 home resolution helper，不在本模块重新发明 home 解析规则。

### 2. 文件命名

第一版：

```text
<session_id[:8]>.md
```

要求：

- 同一个 session id 多次调用 `get_plan_path` 返回同一路径。
- `session_id` 必须做基本校验，避免空字符串和 path traversal。
- 目录不存在时 `write_plan` 自动创建。
- `read_plan` 在文件不存在时返回 `None`，不抛异常。
- `clear_plan` 在文件不存在时 no-op。

后续如果需要更像 open-claude-code 的 readable slug，可单独迁移，不在本 PR 扩大范围。

### 3. plan-file write exception

这是本 PR 的核心 parity 点：plan mode 不能全阻断 `write_file` / `file_edit`。必须允许模型写当前 session 的 plan file，且只能写这一个路径。

允许：

```json
{
  "name": "write_file",
  "arguments": {
    "path": "/home/user/.aether/plans/7f8d4f67.md",
    "content": "# Plan\n..."
  }
}
```

允许：

```json
{
  "name": "file_edit",
  "arguments": {
    "path": "/home/user/.aether/plans/7f8d4f67.md",
    "old_string": "...",
    "new_string": "..."
  }
}
```

拒绝：

- 任何非当前 session plan path 的 `write_file` / `file_edit`。
- `notebook_edit`。
- `shell`，即使命令只是 `echo > plan.md`。
- `todo_write`。
- `task` / `task_stop`。
- memory write/update/forget。

实现位置：

- 放在 `aether.tools.registry.check_plan_mode_block` 或 engine pre-permission gate 的同一策略路径中。
- 不要在 `write_file` / `file_edit` tool 内部绕过 blocker。
- engine pre-permission gate 和 registry final gate 必须共用同一判断函数，避免一个允许、另一个拒绝。

路径安全要求：

- 从 tool arguments 中读取 `path`。
- 当前 session plan path 使用 `get_plan_path(session_id)`。
- 比较前做 canonicalization。对已存在文件使用 resolved real path；对待创建文件使用 parent resolved path + filename。
- 目标必须等于当前 session plan path，不能只是 prefix match。
- 拒绝 sibling path，例如 `<prefix>.md.bak`。
- 拒绝其他 session plan path。
- 拒绝 symlink escape。若 plan path 已经是 symlink 且指向 plans 目录外，应拒绝写入并返回明确错误。
- 文件扩展名必须是 `.md`，但不要只靠扩展名判断。

返回 metadata：

```json
{
  "plan_mode_blocked": true,
  "tool_executed": false,
  "allowed_plan_path": "/home/user/.aether/plans/7f8d4f67.md"
}
```

允许写 plan file 时可加 metadata：

```json
{
  "plan_mode_plan_file_write": true,
  "plan_path": "/home/user/.aether/plans/7f8d4f67.md"
}
```

### 4. `exit_plan_mode` artifact-first

修改 `exit_plan_mode` 工具实现：

1. `plan` 参数改为 optional。
2. 如果提供非空 `plan` 参数，先 `write_plan(session_id, plan)`，这是兼容旧模型和 approval UI edited-plan 回传。
3. 如果没有 `plan` 参数，从 `read_plan(session_id)` 读取 artifact。
4. 如果 artifact 不存在或为空，返回明确错误：要求先写 plan file 或传入 `plan`。
5. 再触发 approval request。
6. approval approve/reject 后仍确保 artifact 中保留最近 plan。

这样即使用户 reject，`/plan` 也能展示最近被拒的计划，模型可以继续修订并再次调用 `exit_plan_mode`。

tool descriptor 建议：

```json
{
  "properties": {
    "plan": {
      "type": "string",
      "description": "Optional compatibility field. Prefer writing the plan to the session plan file; if provided, this content replaces the artifact before approval."
    }
  },
  "required": []
}
```

approve result 内容必须包含：

- 简短批准文案。
- plan path。
- approved plan 内容。

示例：

```text
Plan approved. Returning to agent mode.

Your plan has been saved to: /home/user/.aether/plans/7f8d4f67.md

## Approved Plan:
# Plan
...
```

reject result 内容必须短，并保持 mode 为 plan：

```text
Plan rejected. Revise the plan file and call exit_plan_mode again.
```

### 5. `plan.current` 接入 artifact

扩展 PR 12.1 的 `plan.current`：

```json
{
  "session_id": "...",
  "mode": "plan",
  "plan_path": "/home/user/.aether/plans/7f8d4f67.md",
  "has_plan": true,
  "plan_content": "# Plan\n..."
}
```

要求：

- 文件不存在时 `has_plan=false`、`plan_content=null`，但 `plan_path` 仍应返回稳定路径。
- 文件存在但为空时 `has_plan=true`、`plan_content=""`。
- 读取失败时返回 RPC error，错误中包含 path 和原因，但不要泄漏无关环境信息。
- 内容过大时可以后续加截断字段；本 PR 不改变 `plan_content` 语义。

### 6. Open-Claude-Code Parity Notes

- OCC 使用 readable word slug 和 optional settings `plansDirectory`；Aether 第一版使用 session id prefix 和 `${AETHER_HOME}/plans`。
- OCC 的 `ExitPlanMode` V2 schema 对模型基本不要求传 plan；Aether 为兼容既有模型保留 optional `plan`。
- OCC 允许 plan file write 是权限系统里的 internal editable path；Aether 必须把例外放在 registry / engine gate 统一路径。
- OCC 支持 subagent-specific plan file；Aether 本 sprint 明确阻断 subagent dispatch，不实现 agent-specific plan file。

## 测试 / Tests

Python 单测建议：

- `test_plan_path_is_stable`：同 session id 返回同一路径。
- `test_plan_path_uses_aether_home_plans_dir`：临时 `AETHER_HOME` 下路径为 `<home>/plans/<prefix>.md`。
- `test_write_plan_creates_directory`：目录不存在时自动创建并写入。
- `test_read_plan_missing_returns_none`：缺失文件返回 `None`。
- `test_clear_plan_is_idempotent`：缺失和存在文件都不报错。
- `test_exit_plan_mode_writes_artifact_before_approval`：调用 tool 后 artifact 立即可读。
- `test_exit_plan_mode_without_plan_reads_artifact`：模型先写 plan file，再调用空参数 `exit_plan_mode`。
- `test_exit_plan_mode_empty_missing_artifact_errors`：无 artifact 或空 plan 时明确失败并保持 plan mode。
- `test_exit_plan_mode_plan_arg_overwrites_artifact`：兼容参数仍可更新 artifact。
- `test_exit_plan_mode_reject_keeps_artifact`：reject 后 plan 文件仍保留。
- `test_plan_current_reads_artifact`：gateway `plan.current` 返回 path、has_plan、content。
- `test_plan_mode_non_plan_file_write_still_blocked`：plan mode 下普通写文件仍拒绝。
- `test_plan_mode_allows_write_file_to_current_plan_path`。
- `test_plan_mode_allows_file_edit_to_current_plan_path`。
- `test_plan_mode_rejects_write_file_to_other_path`。
- `test_plan_mode_rejects_file_edit_to_other_path`。
- `test_plan_mode_rejects_other_session_plan_path`。
- `test_plan_mode_rejects_plan_path_sibling_prefix`。
- `test_plan_mode_rejects_plan_path_symlink_escape`。
- `test_plan_mode_still_blocks_shell_todo_task_memory`。

## 验收 / Acceptance

- 模型在 plan mode 下可以用 `write_file` 创建 `${AETHER_HOME}/plans/<session-prefix>.md`。
- 模型在 plan mode 下可以用 `file_edit` 修订同一 plan file。
- 模型在 plan mode 下不能写任何非 plan artifact 文件。
- `exit_plan_mode()` 能从 artifact 读取 plan 并触发 approval。
- `exit_plan_mode(plan="...")` 兼容旧调用，写入 artifact 后触发 approval。
- `/plan` 能通过 `plan.current` 读到 artifact。
- `/plan open` 有稳定文件路径可打开。
- plan 文件目录不存在时自动创建。
- plan mode 下非 plan artifact 的写入仍被拒绝。

## 不在本 PR / Deferred

- `/clear` 清理 artifact、`/resume` 恢复 artifact：PR 12.6。
- approval modal markdown 渲染：PR 12.5。
- readable slug、版本化 plan history、plan diff：后续 sprint。
