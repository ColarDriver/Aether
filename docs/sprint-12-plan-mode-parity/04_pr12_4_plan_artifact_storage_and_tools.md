# PR 12.4 — Plan Artifact Storage and Tools

## 目标 / Goal

为每个 session 提供一个稳定的 Markdown plan artifact，让 `/plan` 能查看最近计划，`/plan open` 能打开文件，`exit_plan_mode(plan=...)` 能把模型提交的计划保存下来。

第一版使用 `${AETHER_HOME}/plans/<session-prefix>.md`，不引入随机 word slug 或额外配置面。

## 当前问题 / Current Problem

`exit_plan_mode` 可以携带 plan 文本触发 approval，但 plan 内容没有稳定文件承载：

- `/plan` 无法展示最近计划。
- `/plan open` 没有可打开目标。
- resume 后无法按 session 查看历史计划。
- clear/new 生命周期无法定义 "旧 plan 是否还在"。

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

def write_plan(session_id: str, content: str) -> None:
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

### 3. `exit_plan_mode` 写入 artifact

修改 `exit_plan_mode(plan=...)` 工具实现：

1. 收到 plan 参数后立即 `write_plan(session_id, plan)`。
2. 再触发 approval request。
3. approval approve/reject 后仍确保 artifact 中保留最近 plan。

这样即使用户 reject，`/plan` 也能展示最近被拒的计划，模型可以继续修订并再次调用 `exit_plan_mode`。

如果 approval bridge 在 approve 后会转换 mode，本 PR 不负责 UI 文案；但 artifact 写入必须发生在 mode 切换前后都稳定。

### 4. `plan.current` 接入 artifact

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

- 文件不存在时 `has_plan=false`、`plan_content=null`。
- 文件存在但为空时 `has_plan=true`、`plan_content=""`。
- 读取失败时返回 RPC error，错误中包含 path 和原因，但不要泄漏无关环境信息。

### 5. 可选增强：只允许写 plan 文件

可选，不建议第一版默认开启，除非产品明确希望模型在 plan mode 中持续编辑 artifact 文件。

如果要允许模型在 plan mode 通过写工具更新 plan file：

- 只允许目标 path 精确等于 `get_plan_path(session_id).resolve()`。
- 其他所有写路径仍由 `WRITE_TOOLS_BLOCKED_IN_PLAN` 拒绝。
- 该例外必须放在现有 blocker 的同一策略路径中，不能在 tool 实现里绕过。
- 测试必须覆盖 symlink、relative path、path normalization。

推荐第一版只让 `exit_plan_mode(plan=...)` 写 artifact，降低权限策略复杂度。

## 测试 / Tests

Python 单测建议：

- `test_plan_path_is_stable`：同 session id 返回同一路径。
- `test_plan_path_uses_aether_home_plans_dir`：临时 `AETHER_HOME` 下路径为 `<home>/plans/<prefix>.md`。
- `test_write_plan_creates_directory`：目录不存在时自动创建并写入。
- `test_read_plan_missing_returns_none`：缺失文件返回 `None`。
- `test_clear_plan_is_idempotent`：缺失和存在文件都不报错。
- `test_exit_plan_mode_writes_artifact_before_approval`：调用 tool 后 artifact 立即可读。
- `test_exit_plan_mode_reject_keeps_artifact`：reject 后 plan 文件仍保留。
- `test_plan_current_reads_artifact`：gateway `plan.current` 返回 path、has_plan、content。
- `test_plan_mode_non_plan_file_write_still_blocked`：plan mode 下普通写文件仍拒绝。

如果实现可选 plan-file write exception，追加：

- `test_plan_mode_allows_only_current_session_plan_path`。
- `test_plan_mode_rejects_plan_path_symlink_escape`。
- `test_plan_mode_rejects_other_session_plan_path`。

## 验收 / Acceptance

- `exit_plan_mode(plan="...")` 后，`${AETHER_HOME}/plans/<session-prefix>.md` 存在且内容为最新 plan。
- `/plan` 能通过 `plan.current` 读到 artifact。
- `/plan open` 有稳定文件路径可打开。
- plan 文件目录不存在时自动创建。
- plan mode 下非 plan artifact 的写入仍被拒绝。

## 不在本 PR / Deferred

- `/clear` 清理 artifact、`/resume` 恢复 artifact：PR 12.6。
- approval modal markdown 渲染：PR 12.5。
- readable slug、版本化 plan history、plan diff：后续 sprint。

