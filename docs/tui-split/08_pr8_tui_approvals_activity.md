# PR 8 · TUI approvals + activity bar + picker

## 摘要

补齐 TS 端能跑长 turn 所需的所有 overlay 与状态可视化：approval modal（plan / questions）、permission modal（diff preview + 三选项规则）、activity bar（思考 / 工具 / token 计数）、tool call 折叠面板、session picker。本 PR 之后 TS TUI 在功能面上已与 Python TUI 对齐。

## Scope

In scope:

- `aether-tui/src/overlays/`：overlay framework + 4 个 overlay
  - `OverlayFrame.tsx`：键盘焦点管理、ESC 优先级链
  - `ApprovalModal.tsx`（plan / questions）
  - `PermissionModal.tsx`（diff preview + accept-once / accept-session / deny）
  - `SessionPicker.tsx`
  - `HelpOverlay.tsx`（`/help` 升级版）
- `aether-tui/src/components/ActivityBar.tsx`：替换 PR 7 的 StatusLine
- `aether-tui/src/components/ToolCallPanel.tsx`：可折叠的工具调用渲染
- `aether-tui/src/store/{overlayStore,activityStore,permissionRulesStore}.ts`
- `aether-tui/src/hooks/useReverseRpc.ts`：消费 `approval.request` / `permission.request`，调度对应 overlay
- `aether-tui/src/lib/diffRender.ts`：unified diff 渲染（彩色 +/-）
- 单元测试

Out of scope:

- 文件 IDE diff 集成（sprint-7 也明确为非目标，留到未来）
- 远程审批（手机端 / 网页）
- session 编辑 / 重命名（picker 只看 + 选）

## Contracts

### Overlay framework

```ts
type OverlayKind = 'approval' | 'permission' | 'session-picker' | 'help'

interface OverlayState {
  kind: OverlayKind
  id: string                       // matches reverse-rpc id when applicable
  payload: unknown
  createdAt: number
  priority: number                 // higher = front
  onDismiss: (reason: 'cancel' | 'commit') => void
}

const overlayStore = atom<OverlayState[]>([])
```

ESC 优先级链（与 sprint-7 已建立的链对齐）：

1. permission overlay → reject + close
2. approval overlay → reject + close
3. session picker → cancel
4. help overlay → close
5. composer 内输入 → 清空草稿
6. 顶层 → confirm exit

只有最顶层 overlay 接收键盘；其它 overlay 与 composer 暂时锁定。

### `ApprovalModal` 协议

- 由 `useReverseRpc` 在收到 `approval.request` 时 push 进 overlayStore。
- 用户操作后调 `GatewayClient.respond(id, ApprovalResponse)`。
- Timeout 倒计时显示在 modal 底部，过期后自动 `confirmed=false`。

`questions` kind：

- `open` → `<TextInput>` 输入框
- `select` → 用 ↑/↓ 选项、Enter 确认
- 多条 question 顺序展示，一次性提交。

### `PermissionModal` 协议

- 顶部：tool 名称 + risk badge + reason
- 中部：preview
  - 有 `diff` → `diffRender(diff)` 彩色显示，最多 40 行（超出折叠）
  - 有 `command` → 灰底显示命令
  - 否则 → `body` 文本
- 底部：三选项 + ESC
  - `Allow once` (Y) / `Allow for session` (S) / `Deny` (N) / ESC → abort

session allow 选择时弹出 prefix 编辑器：

```
Allow for session matching:
  tool: file_edit
  path prefix: src/foo/
[Confirm] [Cancel]
```

确认后客户端把 rule 一起放进 response（`rule` 字段，由 PR 5 协议承载）。

客户端**额外**把这条 rule 存进 `permissionRulesStore`（前端 cache），未来同 session 内同类 request 自动 prefill 决策（仍然显示但默认选项变成 allow）。注意：server 端仍然是权威，本 cache 只用于 UX 提示。

### `ActivityBar`

```
┌──────────────────────────────────────────────────────────────────────┐
│ ◐ thinking · iter 2/8 · 1.2k in / 480 out · ses_abc · sonnet-4-6     │
└──────────────────────────────────────────────────────────────────────┘
```

数据源：

| 字段 | 事件 |
|---|---|
| 状态图标 + 文字 | `status` event |
| iter | `iteration.start` / `iteration.end` |
| token in/out | `usage` event 累加 |
| session | `session.current` |
| model | `prefs.get('model')` + session info |

Loop state 异常（`error`, `cancelled`）→ activity bar 红色。

### `ToolCallPanel`

- 默认折叠，显示一行：`▸ shell · ls -la · ⏱ 0.4s`
- 展开后显示 arguments JSON（已截断）、result 文本（已截断）
- 失败工具 → 红色 + 标记 `error`

折叠 / 展开通过键盘 `Tab` 在 transcript 中循环聚焦工具行，按 Enter 切换。

### `SessionPicker`

- 由 `/resume` 触发或启动参数 `--resume`（PR 9 接入）。
- 列出 `session.list` 最近 20 条，每条显示 `session_id`、updated_at（相对时间）、provider/model、summary。
- ↑/↓ 选择，Enter `session.resume`，结果填进 transcript（`replace-history`）。
- ESC 取消。

## 设计要点

**为什么所有 overlay 都进同一个 store + 渲染层。** 焦点管理只在一个地方做。Hermes 在早期把每个 overlay 各管自己的输入，结果出现两个 overlay 同时接 key、ESC 错误穿透，难调。改成统一调度后这类 bug 几乎消失。

**为什么 permission cache 在前端而不是后端。** sprint-7 已经在后端有 `ToolPermissionRule` session store。本 PR 的前端 cache 只用于"UI 默认选项"——server 端会重新询问（默认拒绝是安全侧），cache 让用户体验流畅。如果未来后端把 session rule 也通过 `permission.request` 直接返回 decision，本 cache 可以下线。

**Diff 渲染。** 不引入完整 diff lib（`diff`、`unidiff` 都过重）。手写 line-based ANSI 着色，遇到非 unified diff 退回 plain text。对 sprint-7.4 的 file_edit / write_file diff 已经够用。

**activity bar 与 status 事件的节流。** Provider 在长 streaming 期间每 ~50ms 一条 `usage` 更新，全量重渲让人眼花。activity bar 内部对 token 计数节流 100ms。

**ToolCallPanel 的 transcript 集成。** 它本质上是一个特殊的 ChatItem，但用单独组件以便控制折叠状态。`chatStore` 里它仍然是 `tool-call` + `tool-result` 两条记录；折叠状态在组件本地 state，进程退出消失。

**为什么 picker 不在自己的 PR。** picker 与其它 overlay 共享 framework 与焦点管理。单独拉一个 PR 会重复一遍 overlay framework 的代码 review。

**Help overlay vs `/help` 文本。** PR 7 的 `/help` 是把 catalog 写进 transcript；PR 8 的 HelpOverlay 是一个分类、可滚动、按字母索引的全屏 overlay。两者并存：`/help` 默认走 overlay；`?` 也触发。`--no-overlay-help` 启动参数（开发用）退化到 transcript。

## Files touched

- new: `aether-tui/src/overlays/*.tsx`
- new: `aether-tui/src/components/{ActivityBar,ToolCallPanel}.tsx`
- new: `aether-tui/src/store/{overlayStore,activityStore,permissionRulesStore}.ts`
- new: `aether-tui/src/hooks/useReverseRpc.ts`
- new: `aether-tui/src/lib/diffRender.ts`
- modified: `aether-tui/src/app.tsx`（挂 overlay layer、替换 StatusLine 为 ActivityBar）
- modified: `aether-tui/src/components/ChatTranscript.tsx`（接入 ToolCallPanel）
- new: `aether-tui/src/__tests__/{overlayFocus,approvalModal,permissionModal,sessionPicker,activityBar,diffRender}.test.tsx`

## Dependencies

PR 5（reverse RPC），PR 7（chat + composer 已有）。

## Acceptance criteria

- 触发 plan-mode 的 turn → approval overlay 弹出在 composer 上方，ESC 与 No 都让 server 收到 `confirmed=false`，Yes 让 turn 继续。
- file_edit turn → permission overlay 显示 unified diff（绿增红删），三选项均能正确把决定传回 server，文件 mutation 只发生在 allow 路径。
- shell turn → permission overlay 显示 command，session allow + prefix 选择路径在第二次同 prefix 命令时直接走 cache hinted default。
- activity bar 在 streaming 期间实时更新 token 计数，状态切换平滑（无闪烁、无重排）。
- 工具面板默认折叠；Tab + Enter 能切换；失败工具红色显示。
- session picker 启动后能 resume；ESC 取消干净。
- ESC 优先级链：permission > approval > picker > help > composer > exit confirm，单测覆盖完整。
- sprint-7 的 acceptance 矩阵在新 TUI 上同样通过（重新跑那一份 manual checklist）。

## Manual verification

```bash
cd aether-tui
npm start

# 在 TUI 里：
# 1. 让模型 plan 后退出 plan → approval overlay 出现 → Y 通过 / N 拒绝
# 2. 让模型 edit 文件 → permission overlay 出现 → 看 diff → Y/N/S
# 3. 让模型 ls -la → permission overlay 显示 command → S 设 session rule
# 4. 长 turn → 观察 activity bar 实时数据
# 5. Tab 在工具行间切换 → Enter 展开
# 6. /resume → picker → 选一条恢复
# 7. ESC 链：多 overlay 叠起来时按 ESC 一层层退
```
