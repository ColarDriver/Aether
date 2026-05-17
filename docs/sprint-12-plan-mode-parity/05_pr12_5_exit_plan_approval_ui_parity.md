# PR 12.5 — Exit Plan Approval UI Parity

## 目标 / Goal

把 `exit_plan_mode` 的 approval UI 做到和 Aether 权限体验一致：plan 内容以 Markdown 渲染，approve 后切回 agent mode，reject 后保持 plan mode 并要求模型修订。

本 PR 不改变 approval bridge 的基本协议，只补 plan approval 的呈现、键盘操作、mode transition 和 transcript 文案。

## 当前问题 / Current Problem

当前 plan approval 虽然能走 bridge，但 UI 与常规 permission 体验不一致，且 approve/reject 后 session mode 的语义需要明确：

- plan approval 内容如果按纯文本显示，可读性差。
- 当前黄色风格与 Aether 现有 permission modal 不一致。
- reject 后必须继续 plan mode，否则模型会在未批准计划的情况下开始执行。
- approve 后必须切回 agent mode，否则模型拿到批准也无法写。

## 改动 / Changes

### 1. 改造 ApprovalModal plan 样式

修改：

```text
tui/src/overlays/ApprovalModal.tsx
```

当 approval request `kind === "plan"` 或等价字段表示 plan approval 时：

- 使用 Aether 现有 permission modal 的边框、标题、按钮风格。
- 不使用当前突兀的黄色 warning 风格。
- 标题短且明确，例如 `Plan approval`。
- 内容区域支持滚动或截断，避免长 plan 撑破终端布局。

### 2. Markdown 渲染

plan 内容应按 Markdown 渲染，而不是普通纯文本。

要求：

- headings、lists、code spans/code fences 至少可读。
- 渲染失败时 fallback 到纯文本，不阻塞 approval。
- 长代码块不能破坏 modal 宽度。
- 不执行 HTML 或任意终端 escape。

### 3. 键盘操作

支持：

- `Enter` / `Y`：approve。
- `N` / `Esc`：reject。

要求：

- 大小写都可接受。
- reject 不应关闭 session，也不应清理 plan artifact。
- modal 关闭后焦点回到输入区域。

### 4. approve 后 mode transition

`exit_plan_mode` approved 后：

1. gateway / backend 将 session mode 切回 `agent`。
2. TUI session store 同步为 `agent`。
3. transcript 输出简短提示：

```text
Plan approved. Returning to agent mode.
```

如果 mode transition 由 backend 完成，TUI 仍应用 `plan.mode_get` 或 approval result payload 校准本地 store。

### 5. reject 后 mode transition

`exit_plan_mode` rejected 后：

1. session mode 保持 `plan`。
2. TUI session store 保持或同步为 `plan`。
3. transcript 输出：

```text
Plan rejected. Revise and call exit_plan_mode again.
```

模型下一轮仍会收到 PR 12.3 的 plan reminder。

### 6. 可选 plan path display

可选：modal 中显示 plan path 或提供复制能力。但本 PR 不做 in-modal 编辑；编辑计划仍通过模型修订并再次调用 `exit_plan_mode`，或用户用 `/plan open` 打开 artifact。

## 测试 / Tests

TS 单测建议：

- `plan approval renders markdown`：传入 heading/list/code，断言 markdown renderer 被使用或输出结构可读。
- `plan approval uses permission styling`：snapshot 或 style assertion 确认不走黄色 warning variant。
- `enter approves plan`：按 Enter 返回 `{ confirmed: true }`。
- `y approves plan`：按 `Y` / `y` 返回 `{ confirmed: true }`。
- `n rejects plan`：按 `N` / `n` 返回 `{ confirmed: false }`。
- `escape rejects plan`：Esc 返回 `{ confirmed: false }`。
- `approve updates mode to agent`：approval success 后 store mode 为 `agent`。
- `reject keeps mode plan`：reject 后 store mode 仍为 `plan`。
- `approve transcript message is concise`：输出 `Plan approved. Returning to agent mode.`。
- `reject transcript message is concise`：输出 `Plan rejected. Revise and call exit_plan_mode again.`。

Python / gateway 单测建议：

- `test_exit_plan_mode_approve_sets_agent_mode`。
- `test_exit_plan_mode_reject_keeps_plan_mode`。
- `test_exit_plan_mode_result_payload_exposes_mode`，如果 bridge result schema 支持。

## 验收 / Acceptance

- 模型调用 `exit_plan_mode(plan=...)` 后，TUI 弹出 plan approval modal。
- plan markdown 可读，列表和代码块不会撑坏布局。
- Enter/Y approve；N/Esc reject。
- approve 后状态区域不再显示 `plan`，写工具可进入正常 permission / execution flow。
- reject 后状态区域仍显示 `plan`，写工具仍被 plan blocker 拒绝。

## 不在本 PR / Deferred

- plan artifact 管理：PR 12.4。
- `/resume` 后恢复 approval modal：不在本 sprint，除非现有 approval bridge 已支持 pending approval persistence。
- 用户在 modal 内直接编辑计划：后续产品需求再做。

