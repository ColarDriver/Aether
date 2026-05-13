# PR 7 · TUI chat view + slash commands

## 摘要

把 PR 6 的 `GatewayClient` 接到 Ink UI 上：渲染 banner、聊天记录、流式 markdown、输入框；实现 slash 命令分发（本地直接执行 + 通过 `session.*` / `prefs.*` / `providers.*` 等 RPC 远程执行）。这个 PR 跑完，用户可以从 TS 端真的开聊，但还看不到 approval / tool-call 面板（PR 8）。

## Scope

In scope:

- `aether-tui/src/app.tsx`：顶层组件
- `aether-tui/src/components/Banner.tsx`
- `aether-tui/src/components/ChatTranscript.tsx`
- `aether-tui/src/components/ChatMessage.tsx`（含 streaming markdown）
- `aether-tui/src/components/Composer.tsx`（输入框 + 多行 + 历史）
- `aether-tui/src/components/StatusLine.tsx`（迷你状态条；完整 activity bar 在 PR 8）
- `aether-tui/src/store/chatStore.ts`（nanostores）
- `aether-tui/src/store/composerStore.ts`
- `aether-tui/src/store/sessionStore.ts`
- `aether-tui/src/hooks/useGatewayEvents.ts`：把 `GatewayClient` 事件汇入 stores
- `aether-tui/src/slash/dispatcher.ts`：slash 命令解析与路由
- `aether-tui/src/slash/commands/*.ts`：每个 slash 命令一个文件
- 单元测试：reducer 覆盖、slash 分发、composer 编辑路径

Out of scope:

- approval modal / permission overlay（PR 8）
- 活动条、tool 面板、session picker（PR 8）
- 多 session 切换 UI（不在本 sprint 范围）

## Contracts

### App 状态分层

```
chatStore        list of ChatItem (user message | assistant message | tool call | tool result | system note)
composerStore    draft text, mode (single-line | multi-line), history index
sessionStore     current session info, model, provider, usage running total
```

`ChatItem` discriminated union：

```ts
export type ChatItem =
  | { kind: 'user'; id: string; text: string; ts: number }
  | { kind: 'assistant'; id: string; text: string; streaming: boolean; ts: number }
  | { kind: 'tool-call'; id: string; toolName: string; argsPreview: string; ts: number; iteration: number }
  | { kind: 'tool-result'; id: string; toolCallId: string; text: string; isError: boolean; ts: number }
  | { kind: 'note'; id: string; text: string; level: 'info' | 'warn' | 'error'; ts: number }
```

### Event → store mapping

```ts
function mapEvent(ev: GatewayEvent): Action[] {
  switch (ev.type) {
    case 'text.delta': return [{ kind: 'appendAssistant', id: ev.run_id, text: ev.text }]
    case 'tool.call': return [{ kind: 'pushToolCall', id: ev.tool_call_id, toolName: ev.tool_name, argsPreview: previewArgs(ev.arguments), iteration: ev.iteration }]
    case 'tool.result': return [{ kind: 'pushToolResult', toolCallId: ev.tool_call_id, text: ev.content, isError: ev.is_error }]
    case 'status': return [{ kind: 'setStatus', kind2: ev.kind, detail: ev.detail }]
    case 'usage': return [{ kind: 'addUsage', input: ev.input_tokens, output: ev.output_tokens }]
    // ...
  }
}
```

### Streaming markdown 策略

- 不引入完整 markdown 渲染（复杂、Ink 不友好）。
- 处理：粗体 `**...**` → `chalk.bold`；行内 code `` `...` `` → `chalk.gray`；fence `` ``` `` → 单独 box 渲染。
- 列表、表格、链接保持原样（终端文本）。
- 关键约束：increment-only。streaming 期间不重排版面，只在 buffer 后面 append。完成态再做一次"漂亮化"重渲。

### Slash dispatcher

```ts
type SlashCommand = {
  name: string                 // "/model"
  category: 'local' | 'remote' | 'control'
  execute(args: string[], ctx: SlashCtx): Promise<SlashResult>
}

type SlashResult =
  | { kind: 'note'; text: string; level: 'info' | 'warn' | 'error' }
  | { kind: 'replace-history'; messages: TranscriptMessage[] }
  | { kind: 'exit' }
  | { kind: 'noop' }
```

PR 7 实现的命令清单（与 server 的 `commands.catalog` 一致）：

| Slash | category | 实现 |
|---|---|---|
| `/help` | local | 渲染 catalog 表 |
| `/exit` | local | 触发 Ink unmount + GatewayClient.stop |
| `/clear` | local | `chatStore.set([])`；不动 server |
| `/refresh` | local | 重渲（清屏） |
| `/session` | remote | `session.current` → note |
| `/sessions` | remote | `session.list` → 渲染列表（不进 picker；picker 在 PR 8） |
| `/new` | remote | `session.create` |
| `/system` | remote | `session.create` 时已传；本命令编辑当前 session（如果允许） |
| `/model` | remote | `prefs.set` + `providers.models` 校验 |
| `/stats` | local | 从 sessionStore 累计读出 |
| `/resume` | remote | `session.resume` → `replace-history` |
| `/verbose` | local | 翻转 `chatStore.verboseMode` |
| `/interrupt` | control | `agent.cancel` |
| `/tools` | remote-tbd | 留到 PR 8（依赖 `tools.catalog`） |

### Composer 行为

- 单行模式：Enter 发送。
- 多行模式：触发条件 = 输入末尾以 `\\` 结尾，或显式 `Alt+Enter`；Enter 换行；`Ctrl+D` 发送。
- 上 / 下方向：单行 + 空 buffer 时翻历史；多行模式下走文本移动。
- `Ctrl+C`：发送中 → `agent.cancel`；空闲时 → `/exit` 等价。

## 设计要点

**为什么用 nanostores 而不是 useState / useReducer。** Ink 重渲粒度粗，nanostores 能让"只有 assistant 文本变了"时只重渲那一个 message 组件。Hermes 实测下来在长 transcript 里这是肉眼可见的差异。

**为什么 streaming markdown 不上完整解析。** 流式过程中频繁 reparse 让 CPU 飙升、闪屏；终端里全功能 markdown 无意义（链接不能点）。最小集已经能盖 95% 的可读性。完成态可以保留 raw text，让未来 export 用。

**Slash dispatcher 的 category。** `local` 命令完全不打 server；`remote` 走单次 RPC；`control` 是会影响活跃 turn 的命令（cancel / interrupt），需要特殊路由。三类划分让单测覆盖更可控。

**`/system` 的处理。** 现在 Python REPL 里 `/system <text>` 是把本地 ReplState 的系统提示词改掉，下次发请求带上。本 PR 改成把 system override 缓存在 `sessionStore`，下一次 `agent.run` 调用作为 `system_override` 参数透传。如果用户切 session，缓存清空。

**`/clear` vs `/new` 的语义差别。** 沿用 Python REPL：clear = 清当前 session 的可见 transcript，server 端 session 不变；new = `session.create` 一个新 session_id 并把旧的归档。

**为什么 PR 7 不做 picker。** session 列表用文本展示在 transcript 区域里是 OK 的过渡；picker 涉及单独的覆盖层、键盘焦点管理，与 PR 8 的 overlay 框架共用，所以放到 PR 8 一起做。

**输入历史的存储。** 内存 ring buffer，进程退出消失。落盘留给后续 PR。

**TTY guard.** 启动时 `process.stdin.isTTY === false` → 立刻 `console.error('aether requires an interactive terminal')` 然后 `process.exit(2)`。Hermes 也是这条路径，避免 CI 误启 TUI。

## Files touched

- new: `aether-tui/src/app.tsx`
- new: `aether-tui/src/components/{Banner,ChatTranscript,ChatMessage,Composer,StatusLine}.tsx`
- new: `aether-tui/src/store/{chatStore,composerStore,sessionStore}.ts`
- new: `aether-tui/src/hooks/useGatewayEvents.ts`
- new: `aether-tui/src/slash/dispatcher.ts`
- new: `aether-tui/src/slash/commands/*.ts`
- new: `aether-tui/src/lib/markdownLite.ts`
- modified: `aether-tui/src/entry.tsx`（接入 App）
- new: `aether-tui/src/__tests__/{chatStore,slashDispatcher,markdownLite,composer}.test.tsx`

## Dependencies

PR 3（slash 命令对应的 RPC 方法），PR 4（agent.run 流式事件 + agent.cancel），PR 6（GatewayClient）。

## Acceptance criteria

- 启动 `npm start` 后看到 banner，光标在输入框，状态条显示 `idle`。
- 发一条 `hello`，能看到 assistant 消息逐 token 流出；状态条 `thinking → responding → idle` 切换。
- 期间按 Ctrl+C → 服务器收到 `agent.cancel`，UI 显示 cancelled note，状态回 idle。
- `/help` 渲染表格，命令数量与 `commands.catalog` 长度一致。
- `/sessions` 输出最近 N 个 session（受 `session.list` 限制）。
- `/clear` 清空 transcript 但保留 sessionStore 信息。
- `/new` 创建新 session_id，UI 顶部 session 摘要更新。
- `/model <id>` 调 `prefs.set` 后再发消息走新模型；不存在的 model 给出可读 note。
- 单测覆盖：streaming append、slash 分发、composer 编辑、cancellation 路径。

## Manual verification

```bash
cd aether-tui
npm start

# 在 TUI 里：
# 1. 输入 hello，回车 → 看流
# 2. 输入 /help → 看命令表
# 3. 输入 /new → 创建新 session
# 4. 长消息 → Ctrl+C → 看 cancellation 路径
# 5. /model claude-haiku-4-5-20251001 → 再发消息看是否换模型
# 6. /exit → 干净退出，无 zombie python 进程
```
