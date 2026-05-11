# PR 7.5 - Shell + Plan-mode Policy Integration

## 目标

把 `shell` 和 plan-mode 写入阻塞接入同一个 permission policy。用户确认 shell 命令时，命令不应该先进入输出框，更不能提前启动 subprocess。

## shell 权限策略

`backend/harness/aether/tools/builtins/shell.py` 当前在 `execute()` 中直接 `subprocess.run(...)`。PR 7.2 gate 必须保证 `execute()` 只在批准后调用。

shell preview：

```python
ToolPermissionPreview(
    title="Run command",
    subtitle=str(cwd) if cwd else None,
    command=command,
    body=f"timeout: {timeout}s",
    metadata={"cwd": str(cwd) if cwd else None, "timeout_sec": timeout},
)
```

首版不做智能危险分类，所有 `shell` 默认 ask。可以加最小风险标签：

| pattern | risk |
|---|---|
| `rm`, `mv`, `chmod`, `chown`, `sudo`, redirect write | `high` |
| `git commit`, `git push`, package install | `medium` |
| `ls`, `pwd`, `find`, `grep`, test command | `shell` |

风险标签只影响 UI 文案，不自动放行。

## accept-session for shell

session allow 不应该粗暴允许所有 shell。建议首版只允许 prefix-scoped rule：

| 用户选择 | Rule |
|---|---|
| allow this command once | 不写 rule |
| allow commands with this exact prefix in session | `command_prefix` |
| allow all shell this session | 首版不提供，避免过宽 |

可选项文案：

| 选项 | 说明 |
|---|---|
| `Yes, run this command` | 单次批准 |
| `Yes, allow this command prefix this session` | prefix session 批准 |
| `No` | 拒绝 |

prefix 的默认值可以取命令第一个 segment，例如 `pytest`、`git status`、`find .`。不要默认为整条 shell compound command 的任意前缀，避免把 `pytest && rm -rf` 放进宽规则。

## Plan-mode 接入

当前 `tools/registry.py`：

```python
WRITE_TOOLS_BLOCKED_IN_PLAN = {...}
_check_plan_mode_block(...)
```

PR 7.5 需要明确 plan mode 与 permission 的优先级：

| 状态 | 行为 |
|---|---|
| session 在 plan mode 且 tool 是写类 | 直接 block，不弹 permission |
| session 不在 plan mode 且 tool 是危险工具 | 走 permission ask/allow/deny |
| `exit_plan_mode` | 仍走现有 plan approval prompter，不纳入通用 tool permission |

理由：plan mode 是更高层状态机约束。用户没有退出 plan mode 前，不能通过普通 tool permission 绕过“先给计划再写入”的约束。

实现上建议把 `_check_plan_mode_block()` 从 `ToolRegistry.dispatch()` 前移到 engine permission gate 之前：

| 位置 | 原因 |
|---|---|
| engine gate 前 | 可以避免 UI 先弹 permission，再被 registry plan-mode block |
| registry 保留兜底 | 防止未来绕过 engine dispatch 的调用路径直接 dispatch |

## UI 行为

shell 确认 pending 时：

| 区域 | 内容 |
|---|---|
| overlay title | `Run command` |
| command block | 原始命令，保持换行和缩进 |
| metadata | cwd、timeout、risk |
| footer | `Enter approve · Esc reject` |

批准后才调用现有 `render_tool_call("shell", args)`。拒绝后不渲染 shell 输出块，因为没有 subprocess 输出。

## 测试

新增 `tests/tools/test_shell_permission.py` 或 engine 集成测试：

```python
def test_shell_reject_does_not_call_subprocess_run(monkeypatch): ...
def test_shell_accept_calls_subprocess_once(monkeypatch): ...
def test_shell_non_interactive_denied_by_default(monkeypatch): ...
def test_shell_accept_session_prefix_skips_second_prompt(monkeypatch): ...
def test_shell_plan_mode_block_happens_before_permission_prompt(monkeypatch): ...
```

扩展 registry/plan tests：

```python
def test_plan_mode_still_blocks_write_file_even_if_session_rule_allows(): ...
def test_exit_plan_mode_approval_still_uses_approval_prompter(): ...
```

## 验收门

- `shell` 拒绝路径绝不启动 subprocess。
- `shell` 批准路径启动一次 subprocess。
- plan mode 写类工具不弹普通权限确认，直接返回 plan-mode block。
- session shell allow 不支持无边界 allow-all，除非后续 PR 有更强 policy。
- 现有 shell timeout、cwd、spill、interrupt 测试不回归。

