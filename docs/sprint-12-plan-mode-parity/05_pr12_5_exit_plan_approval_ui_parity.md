# PR 12.5 — Exit Plan Approval UI Parity

## 目标 / Goal

把 `exit_plan_mode` 的 approval UI 做到和 Aether 权限体验一致：plan 内容以 Markdown 渲染，展示 plan path，支持用户在外部 editor 中修订 plan，approve 后切回 agent mode，reject 后保持 plan mode 并要求模型修订。

本 PR 会扩展 approval bridge 的 plan payload，但不引入 remote / ultraplan / teammate approval。它补 plan approval 的呈现、键盘操作、mode transition、edited plan round-trip 和 transcript 文案。

## 当前问题 / Current Problem

当前 plan approval 虽然能走 bridge，但 UI 与常规 permission 体验不一致，且 approve/reject 后 session mode 的语义需要明确：

- plan approval 内容如果按纯文本显示，可读性差。
- 当前黄色风格与 Aether 现有 permission modal 不一致。
- 用户看不到 plan artifact 路径，不知道 `/plan open` 或外部编辑的目标。
- 如果 approval UI 不支持编辑，用户只能 reject 后让模型改，无法像 OCC 一样快速修订措辞或补充要求。
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

### 3. Plan path display

approval request payload 建议扩展：

```json
{
  "kind": "plan",
  "session_id": "...",
  "run_id": "...",
  "tool_call_id": "...",
  "plan_text": "# Plan\n...",
  "plan_path": "/home/user/.aether/plans/7f8d4f67.md",
  "deadline_ms": 600000
}
```

要求：

- modal 中显示 plan path，使用 dim text 即可。
- `plan_path` 缺失时 UI 仍可 approval，但应降级为不可 external edit。
- bridge payload 加字段要后向兼容：旧 TUI 忽略新字段，旧 gateway 不提供字段时新 TUI 不崩。

### 4. 外部编辑

对齐 OCC 的本地能力：plan approval modal 支持打开当前 plan file 到 `$AETHER_EDITOR || $VISUAL || $EDITOR`。

建议快捷键：

- `Ctrl+G`：打开 plan path 到外部 editor。

行为：

1. 仅当 `plan_path` 存在时启用。
2. 打开失败时在 modal 内或 notification 中显示可读错误。
3. editor 返回后重新读取 plan file。
4. 如果文件内容变化，modal 中的 markdown 预览更新。
5. approve 时把编辑后的 plan 作为 updated input / response payload 回传给 backend，backend 写回 artifact 后再完成 `exit_plan_mode`。

approval response 建议：

```json
{
  "confirmed": true,
  "answers": {},
  "updated_input": {
    "plan": "# Edited Plan\n..."
  }
}
```

如果当前 bridge 只能返回 `{ confirmed, answers }`，本 PR 应扩展 schema；短期也可让 TUI 先写文件，backend 在 `exit_plan_mode` approval 后重新读 artifact，但必须避免 approve 后模型看到旧 plan。

### 5. 键盘操作

支持：

- `Enter` / `Y`：approve。
- `N` / `Esc`：reject。
- `Ctrl+G`：编辑 plan file，如果 `plan_path` 存在。

要求：

- 大小写都可接受。
- reject 不应关闭 session，也不应清理 plan artifact。
- modal 关闭后焦点回到输入区域。

### 6. approve 后 mode transition

`exit_plan_mode` approved 后：

1. gateway / backend 将 session mode 切回 `agent`。
2. TUI session store 同步为 `agent`。
3. transcript 输出简短提示：

```text
Plan approved. Returning to agent mode.
```

如果 mode transition 由 backend 完成，TUI 仍应用 `plan.mode_get` 或 approval result payload 校准本地 store。

approve 后 tool result 应包含：

- `approved=true`
- `new_mode="agent"`
- `plan_path`
- approved plan 内容或可解析的 `## Approved Plan:` 块

这样模型批准后能继续引用计划，尤其在后续 compact / clear context 场景中更稳。

### 7. reject 后 mode transition

`exit_plan_mode` rejected 后：

1. session mode 保持 `plan`。
2. TUI session store 保持或同步为 `plan`。
3. transcript 输出：

```text
Plan rejected. Revise and call exit_plan_mode again.
```

模型下一轮仍会收到 PR 12.3 的 plan reminder。

### 8. Explicitly deferred OCC UI variants

open-claude-code 的 plan approval 还有更多分支，本 sprint 不实现：

- `ultraplan` launch。
- approve with keep-context / clear-context 多选。
- auto / bypass permission mode 选择。
- teammate mailbox approval。
- image paste / feedback threading。

如果 Aether 未来引入对应产品能力，再单独设计，不要在本 PR 偷塞半成品。

## 测试 / Tests

TS 单测建议：

- `plan approval renders markdown`：传入 heading/list/code，断言 markdown renderer 被使用或输出结构可读。
- `plan approval uses permission styling`：snapshot 或 style assertion 确认不走黄色 warning variant。
- `plan approval displays plan path`：payload 有 `plan_path` 时显示。
- `plan approval survives missing plan path`：payload 无 path 时仍可 approve/reject。
- `ctrl_g_opens_plan_file`：mock editor，断言使用 `plan_path`。
- `ctrl_g_editor_failure_is_readable`。
- `ctrl_g_refreshes_markdown_after_edit`：编辑后 UI 使用新内容。
- `approve_after_edit_returns_updated_plan`：approval response 包含 edited plan，或 backend 重新读到 edited artifact。
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
- `test_exit_plan_mode_result_payload_exposes_mode_and_plan_path`。
- `test_exit_plan_mode_approve_result_contains_approved_plan`。
- `test_exit_plan_mode_updated_plan_from_approval_overwrites_artifact`，如果 bridge 支持 updated input。

## 验收 / Acceptance

- 模型调用 `exit_plan_mode()` 且 artifact 存在时，TUI 弹出 plan approval modal。
- 模型调用兼容形式 `exit_plan_mode(plan=...)` 后，TUI 弹出同样 modal。
- plan markdown 可读，列表和代码块不会撑坏布局。
- modal 显示 plan path。
- Ctrl+G 可打开 plan file；编辑保存后 modal 预览更新。
- Enter/Y approve；N/Esc reject。
- approve 后状态区域不再显示 `plan`，写工具可进入正常 permission / execution flow。
- reject 后状态区域仍显示 `plan`，写工具仍被 plan blocker 拒绝。
- approve 后模型收到 plan path 和 approved plan 内容。

## 不在本 PR / Deferred

- plan artifact 管理：PR 12.4。
- `/resume` 后恢复 approval modal：不在本 sprint，除非现有 approval bridge 已支持 pending approval persistence。
- clear-context / keep-context approval variants：后续产品需求再做。
