# PR 12.7 — Tests, Observability, and Acceptance

## 目标 / Goal

收敛 Sprint 12 的跨层验收：Python 单测、TS 单测、手工脚本、观测字段和回归清单都要覆盖 plan mode 的完整用户链路。

这个 PR 可以是独立的测试补丁，也可以在前 6 个 PR 中逐步落地测试，最终用本清单做 merge gate。

本清单按 open-claude-code parity 复核后的目标编写：plan artifact 是 plan mode 中唯一允许写的文件，`exit_plan_mode` 优先读取 artifact，`plan` 参数只是兼容路径。

## Python 测试范围

### Session state

- 默认 mode 为 `agent`。
- `set_mode(session_id, "plan")` 后 `get_mode(session_id) == "plan"`。
- 未知 mode 被拒绝。
- 新 session 不继承旧 session mode。
- clear 后 mode 回到 `agent`。

### Plan RPC

- command catalog 包含 `/plan`。
- `plan.mode_get` 返回当前 mode。
- `plan.mode_set` 更新 session state。
- `plan.mode_set` 持久化 session record mode。
- `plan.current` 返回 `mode`、`plan_path`、`has_plan`、`plan_content`。
- `plan.current` 在 artifact 缺失时仍返回稳定 `plan_path`。
- `plan.clear` 删除 artifact、重置 mode 为 `agent`、持久化 record。
- 缺 `session_id`、未知 session、未知 mode 都返回明确 RPC error。

### Registry / engine blocker

- plan mode 下 `shell` 被 permission prompt 前 blocker 拒绝。
- plan mode 下 `write_file` / `file_edit` 只允许当前 session plan path。
- plan mode 下 `write_file` / `file_edit` 写其他路径被拒绝。
- plan mode 下 `write_file` / `file_edit` 写 sibling prefix、其他 session plan、symlink escape 被拒绝。
- plan mode 下 memory write/update/forget 被拒绝。
- plan mode 下 subagent dispatch 被拒绝。
- plan mode 下 `todo_write` 被拒绝。
- `exit_plan_mode` 不被 blocker 拦截。
- blocker result metadata 包含 `plan_mode_blocked`。
- blocker result metadata 包含 `tool_executed=false`。

### Engine prompt injection

- plan mode 下 provider 收到 `<plan_mode_reminder>`。
- reminder 包含 plan path。
- reminder 包含 plan exists / no-plan 状态。
- reminder 明确唯一可编辑文件是 plan file。
- reminder 明确 `ask_user_question` 不能用于 plan approval。
- agent mode 下不注入。
- reminder 文本多轮稳定。
- provider/run metadata 可观察 `plan_mode_active`。

### Exit approval

- `exit_plan_mode()` 从 artifact 读 plan 并触发 approval request。
- `exit_plan_mode(plan=...)` 兼容旧调用，先写 artifact 再触发 approval request。
- artifact 缺失或为空时，`exit_plan_mode()` 返回明确错误并保持 plan mode。
- approve 返回 `{ confirmed: true }` 或现有等价结构。
- reject 返回 `{ confirmed: false }` 或现有等价结构。
- approve 后 mode 切到 `agent`。
- reject 后 mode 保持 `plan`。
- approve result 包含 `plan_path` 和 approved plan 内容。
- approval edited plan 回传时覆盖 artifact。

### Plan artifact

- plan path 基于 `${AETHER_HOME}/plans/<session-prefix>.md`。
- path 对同一 session 稳定。
- `write_plan` 自动创建目录。
- `read_plan` 对缺失文件返回 `None`。
- `clear_plan` 幂等。
- `exit_plan_mode` 写入 artifact。
- `exit_plan_mode` 无参数读取 artifact。
- `plan.current` 能读 artifact。
- 当前 session plan 文件可在 plan mode 下被 `write_file` / `file_edit` 更新。
- 非 plan 文件写入仍在 plan mode 下被拒绝。

## TS 测试范围

### Slash dispatcher

- dispatcher 注册 `planCommand`。
- `/help` 包含 `/plan`。
- slash completer 包含 `/plan`。
- `/plan open` 被识别为子命令。
- `/plan open auth` 不误触 open 子命令，按 PR 12.2 固定规则处理。
- 如果采用 OCC parity，agent mode 下 `/plan open` 只进入 plan mode，不打开 editor。

### Plan command

- `/plan` 在 agent mode 下调用 `plan.mode_set(plan)`。
- `/plan` 更新 session store mode 为 `plan`。
- `/plan <description>` 返回 `kind: "query"`。
- query 内容保留用户 description，不包含 `/plan` 前缀。
- 已在 plan mode 时裸 `/plan` 调 `plan.current` 并渲染 plan。
- `/plan` 渲染 plan path 和 `/plan open` hint。
- 无 plan 时输出 no-plan 提示。
- `/plan open` 无 plan、有 plan、editor 失败三种情况都有可读输出。

### Approval modal

- kind=plan 使用 plan approval 样式。
- plan markdown 被渲染。
- plan path 被显示。
- Ctrl+G 打开 plan file。
- Ctrl+G 编辑后刷新 modal 预览。
- approve after edit 回传 edited plan 或保证 backend 读取 edited artifact。
- Enter/Y approve。
- N/Esc reject。
- approve 后 store mode 为 `agent`。
- reject 后 store mode 为 `plan`。
- transcript 输出 approve/reject 简短提示。

### Session store mode

- 初始加载从 `session.current.mode` 或 `plan.mode_get` 同步。
- `/new` 设置 mode 为 `agent`。
- `/resume` 同步 resumed session mode。
- `/clear` 清理 plan state 并设置 mode 为 `agent`。
- banner / activity 区域在 plan mode 显示 `plan`，agent mode 不误显示。

## 手工验收脚本 / Manual Acceptance

1. 启动 TUI：

```bash
npm run dev
```

2. 输入：

```text
/plan add auth flow
```

3. 确认 TUI：

- transcript 显示 `Enabled plan mode`。
- 状态区域显示 `plan`。
- 紧接着发起一轮内容为 `add auth flow` 的 agent run。

4. 确认 engine：

- provider request 含 `<plan_mode_reminder>`。
- metadata 可见 `plan_mode_active=true`。

5. 让模型尝试探索：

- read/search/list 可以执行。
- 写当前 session plan file 可以执行。
- 写其他文件、shell、commit、subagent、todo、memory mutation 被 plan blocker 拒绝。
- 拒绝不弹普通 permission modal。

6. 让模型先写 plan artifact：

```text
write_file(path="${AETHER_HOME}/plans/<session-prefix>.md", content="# Plan\n...")
```

7. 让模型调用：

```text
exit_plan_mode()
```

兼容路径也要验收：

```text
exit_plan_mode(plan="...")
```

8. 确认 TUI 弹出 plan approval：

- plan markdown 可读。
- plan path 可见。
- Ctrl+G 可外部编辑并刷新预览。
- Enter/Y approve。
- N/Esc reject。

9. approve 路径：

- modal 返回 confirmed。
- session mode 切到 `agent`。
- 状态区域不再显示 `plan`。
- transcript 显示 `Plan approved. Returning to agent mode.`。
- tool result 包含 plan path 和 approved plan。
- 后续写操作进入正常 permission / execution flow。

10. reject 路径：

- modal 返回 rejected。
- session mode 仍为 `plan`。
- transcript 显示 `Plan rejected. Revise and call exit_plan_mode again.`。
- 下一轮仍注入 plan reminder。
- `/plan` 仍能看到被拒的 plan，模型可以继续编辑 plan file。

11. artifact 验收：

- `${AETHER_HOME}/plans/<session-prefix>.md` 存在。
- `/plan` 能显示最近 plan。
- `/plan` 显示 plan path。
- `/plan open` 能打开该文件，或在 editor 不可用时给出清晰提示。

12. lifecycle 验收：

- `/clear` 后 `/plan` 不显示旧 plan。
- `/new` 后不继承 plan mode。
- `/resume` 对有 plan artifact 的 session 可以查看计划。

## 观测 / Observability

新增或确认以下可观察字段：

| 位置 | 字段 | 说明 |
|---|---|---|
| engine request metadata | `plan_mode_active` | 当前 provider request 是否处于 plan mode |
| tool result metadata | `plan_mode_blocked` | 工具是否因 plan mode blocker 被拒绝 |
| tool result metadata | `tool_executed` | blocker 拒绝时必须为 false |
| tool result metadata | `plan_mode_plan_file_write` | plan mode 下允许写 plan file 时可观察 |
| gateway `session.current` | `mode` | 当前 session mode，缺省兼容旧 session |
| gateway `plan.current` | `mode`, `plan_path`, `has_plan`, `plan_content` | plan mode 和 artifact 状态 |
| TUI session store | `mode` | 前端状态区域和 command 行为的本地来源 |
| approval result metadata | `plan_path`, `approved`, `new_mode` | approval outcome 和 artifact path |

日志要求：

- blocker 日志说明 tool name、session id、mode。
- plan-file write exception 日志说明 allowed path，但不要记录完整 plan 内容。
- approval 日志说明 approve/reject 和最终 mode。
- artifact 写入失败必须可见，但不要把整段 plan 打进普通日志。

## 回归确认 / Regression Checklist

- 普通 agent mode 不注入 plan reminder。
- 普通 agent mode 写工具 permission flow 不变。
- `/model` 不受 session mode 字段影响。
- `/resume` 能继续恢复旧 session。
- `/clear` 不破坏普通 transcript clear。
- permission modal 的普通 tool approval 样式不变。
- `todo_write` 不重新变成普通 permission prompt；如其在 plan mode 被禁止，仍走 blocker。
- subagent / skill / memory 既有测试全绿。
- `write_file` / `file_edit` 在 agent mode 的普通 permission flow 不受 plan-file exception 影响。
- `exit_plan_mode` 在 agent mode 下仍返回 "not in plan mode" 类明确错误，不弹 approval。

## 命令 / Suggested Verification Commands

按实际 test layout 调整，目标是覆盖 Python 和 TS 两侧：

```bash
python -m pytest aether/tests
npm test
npm run type-check
npm run lint
```

如果仓库使用 `uv`：

```bash
uv run pytest aether/tests
uv run pyright
```

## Sprint Acceptance

Sprint 12 完成条件：

- 以上 Python / TS 自动化测试全绿。
- 手工验收脚本 approve 和 reject 两条路径都通过。
- plan artifact 在 clear/new/resume 下行为一致。
- plan mode guardrail 由 prompt 和 engine blocker 双重覆盖：只读探索 + 唯一可写 plan file。
- `exit_plan_mode` artifact-first 语义通过测试，兼容 `plan` 参数。
- 所有新增 wire 字段后向兼容，旧 session 不需要迁移即可运行。
