# PR 9.2 — Chat-Side Edit Summary + Inline Diff

## 目标 / Goal

让 `file_edit` / `write_file` 在用户批准并执行后，在 chat 流里留下一条 Claude Code 风格的持久行：

```
● Edited tui/src/overlays/PermissionModal.tsx (+7 −15)
```

并允许焦点展开（Tab+Enter）后在同一位置内联展开折叠的 unified diff —— 复用 PR 9.1 透过来的 `result.metadata.diff` 与 `result.metadata.lines_added/removed/hunks`。

参考截图：`/workspace/Aether/tmp/issues4/image copy 18.png`。

依赖：**PR 9.1 必须先合入**（否则 `result.metadata` 是空 dict）。

## 当前问题 / Current Problem

### 1. ChatItem.tool-call 没有摘要槽

`tui/src/store/chatStore.ts:9-24`：

```typescript
| {
    kind: 'tool-call'
    id: string
    toolCallId: string
    toolName: string
    args: JsonObject
    argsPreview: string
    ts: number
    iteration: number
    coalesce: boolean
    durationMs: number | null
    result?: { text: string; isError: boolean }
  }
```

没有 `summary` 字段，也没有 `result.metadata`（后者在 PR 9.1 添加）。

### 2. ExploredTree 渲染过于简化

`tui/src/components/ExploredTree.tsx:36-53` 只渲染 `verb + detail`，例如：

```
● Explored
  ⎿ Wrote /workspace/Aether/tmp/calc.py
```

没有 `+N −M` 后缀，没有 diff 入口。

### 3. ChatMessage 在 tool-call coalesce=true 时直接渲染空节点

`tui/src/components/ChatMessage.tsx:66-75`：

```typescript
if (item.kind === 'tool-call') {
  if (item.coalesce) {
    return <></>
  }
  return <ToolCallPanel item={item} expanded={expanded} focused={focused} />
}
```

`coalesce=true` 路径完全交给 ExploredTree 渲染，但 ExploredTree 拿不到具体 ChatItem，只能看 `ToolGroupEntry`。

## 改动 / Changes

### 1. 扩展 ChatItem 结构

**`tui/src/store/chatStore.ts:9-24`**：

```typescript
export interface ToolCallSummary {
  path: string                 // 文件路径（来自 metadata.path）
  linesAdded: number           // 来自 metadata.lines_added
  linesRemoved: number         // 来自 metadata.lines_removed
  hunks?: number               // 来自 metadata.hunks
  diff?: string                // 来自 metadata.diff（<4 KiB 才会有）
}

| {
    kind: 'tool-call'
    id: string
    toolCallId: string
    toolName: string
    args: JsonObject
    argsPreview: string
    ts: number
    iteration: number
    coalesce: boolean
    durationMs: number | null
    result?: { text: string; isError: boolean; metadata?: ToolResultMetadata }
    summary?: ToolCallSummary    // 由 useGatewayEvents 在收到 result 时填入
  }
```

`addToolResult`（约 215-245 行）：

```typescript
addToolResult(input: {...}): void {
  const items = chatItems.get()
  const index = items.findIndex(
    (item) => item.kind === 'tool-call' && item.toolCallId === input.toolCallId
  )
  if (index < 0) {
    // 已有的 orphan-result fallback 逻辑保持不动
    ...
    return
  }
  const existing = items[index] as Extract<ChatItem, { kind: 'tool-call' }>
  const summary = buildSummary(existing.toolName, input.metadata)
  const next: ChatItem = {
    ...existing,
    durationMs: Date.now() - existing.ts,
    result: { text: input.text, isError: input.isError, metadata: input.metadata },
    summary: summary ?? existing.summary,
  }
  chatItems.set([...items.slice(0, index), next, ...items.slice(index + 1)])
}
```

`buildSummary` 新增 helper（同文件）：

```typescript
function buildSummary(
  toolName: string,
  metadata: ToolResultMetadata | undefined,
): ToolCallSummary | undefined {
  if (!metadata) return undefined
  const category = categoryFor(toolName)
  if (category !== 'edit' && category !== 'write') return undefined
  const path = typeof metadata.path === 'string' ? metadata.path : undefined
  if (!path) return undefined
  return {
    path,
    linesAdded: numberOr(metadata.lines_added, 0),
    linesRemoved: numberOr(metadata.lines_removed, 0),
    hunks: typeof metadata.hunks === 'number' ? metadata.hunks : undefined,
    diff: typeof metadata.diff === 'string' ? metadata.diff : undefined,
  }
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
```

### 2. 新组件 `EditSummary.tsx`

**新文件 `tui/src/components/EditSummary.tsx`**：

```typescript
import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { DiffView } from '../lib/diffView.js'
import { theme } from '../lib/theme.js'
import type { ToolCallSummary } from '../store/chatStore.js'

export interface EditSummaryProps {
  summary: ToolCallSummary
  toolName: string  // 'file_edit' | 'write_file' | …
  expanded: boolean
  focused: boolean
}

export function EditSummary({ summary, toolName, expanded }: EditSummaryProps): ReactElement {
  const brand = theme.colorProps('brand')
  const verb = toolName === 'write_file' ? 'Wrote' : 'Edited'
  const basename = summary.path.split('/').pop() || summary.path
  const counts = `(+${summary.linesAdded} −${summary.linesRemoved})`
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text bold {...brand}>{theme.icon('assistant') || '●'} </Text>
        <Text bold>{verb} </Text>
        <Text>{basename}</Text>
        <Text dimColor> {counts}</Text>
        {summary.hunks ? (
          <Text dimColor> · {summary.hunks} hunk{summary.hunks === 1 ? '' : 's'}</Text>
        ) : null}
      </Box>
      {expanded && summary.diff ? (
        <Box marginLeft={2}>
          <DiffView diff={summary.diff} expanded={true} showHeader={false} />
        </Box>
      ) : null}
    </Box>
  )
}
```

`theme.icon('assistant')` 已经在多处使用（`ChatMessage.tsx:55`、`ExploredTree.tsx:20`），无需新加 icon。

### 3. 接入 ChatMessage

**`tui/src/components/ChatMessage.tsx:66-75`** 改为：

```typescript
if (item.kind === 'tool-call') {
  if (item.coalesce) {
    // 旧行为：被 ExploredTree 吸收，渲染空节点。
    // 新增：如果是 edit/write 且有 summary，则单独渲染 EditSummary
    // —— 与 Claude Code 一致，edit 摘要永远独立成行，不被 Explored 吸收。
    if (item.summary) {
      return (
        <EditSummary
          summary={item.summary}
          toolName={item.toolName}
          expanded={expanded}
          focused={focused}
        />
      )
    }
    return <></>
  }
  return <ToolCallPanel item={item} expanded={expanded} focused={focused} />
}
```

**`tui/src/components/ChatTranscript.tsx`** —— 当前 ExploredTree 通过 `tool-group` ChatItem 渲染；需要确认 edit/write 的 `tool-call` 不被 ToolGroupTracker 算进 entries（避免双渲染）。检查 `tui/src/store/toolGroupStore.ts:78-113` 的 `startCall`：

- 当 `category === 'write'` 或 `'edit'` 时，依旧加入 group（保持 `coalesce=true` 标记），但是 ExploredTree 要跳过这些 entry（见下一节）。

### 4. ExploredTree 跳过 edit/write 项 + 失败保留

**`tui/src/components/ExploredTree.tsx:34-53`**：

```typescript
{group.entries
  .filter((entry) => !(
    (entry.category === 'write' || entry.category === 'edit')
    && !entry.isError
  ))
  .map((entry, idx) => (...)
)}
```

成功的 write/edit 不进 Explored 树（EditSummary 已独立显示）；失败的仍留在 Explored 树里以 `(failed)` 标记，因为 EditSummary 在失败时不会渲染（无 `summary`）。

如果过滤后 entries 为空且没有其他类别 entry，整个 `tool-group` ChatItem 也应被跳过 —— 在 `ChatMessage.tsx` 的 `tool-group` 分支添加：

```typescript
if (item.kind === 'tool-group') {
  if (item.group.entries.length === 0) return <></>
  return <ExploredTree group={item.group} />
}
```

`item.group.entries.length` 是序列化时已过滤后的长度，所以这条短路在 chatStore 维护 group 时也能生效。等价做法：在 `toolGroupStore` 的 flush 步骤跳过 write/edit-only group。两条路径选其一即可，**推荐在 store 层裁剪**（让 UI 层保持 dumb）。

### 5. 动词表保持

**`tui/src/lib/toolCategory.ts:183-194`** 已经正确：

```typescript
const VERBS_PAST: Record<ToolCategory, [string, string]> = {
  ...
  write: ['Wrote', 'wrote'],
  edit: ['Edited', 'edited'],
  ...
}
```

无需修改。EditSummary 通过 `toolName === 'write_file'` 直接判断，未走 verb 表 —— 与现有命名保持解耦。

### 6. 失败路径 / Error path

EditSummary 仅在 `summary` 存在时渲染；`buildSummary` 只在 `metadata.path` 与 `lines_added`/`lines_removed` 都齐时返回非 undefined。失败时（is_error=true）通常 metadata 缺 `lines_added` 等，所以自然回落到旧的 ExploredTree `(failed)` 标记。

如果 PR 9.1 实现把 `path` 也写到失败的 metadata，再加一条防御：

```typescript
function buildSummary(...) {
  if (input.isError) return undefined  // 失败不显示摘要
  ...
}
```

## 测试 / Tests

### TS 单测

新建 `tui/src/__tests__/editSummary.test.tsx`：

- `renders Edited basename (+N −M)` —— 给定 summary `{path: '/a/b.tsx', linesAdded: 7, linesRemoved: 15}`，断言渲染包含 `Edited b.tsx (+7 −15)`
- `renders Wrote when toolName=write_file` —— 同上但 `toolName='write_file'` 时应为 `Wrote`
- `shows hunk count when present` —— `hunks: 2` 时渲染 ` · 2 hunks`
- `inlines DiffView when expanded and diff present` —— `expanded=true` 且 `diff='--- /a\n+++ /b\n@@ -1 +1 @@\n-old\n+new\n'` 时 DiffView 被渲染
- `omits DiffView when collapsed` —— `expanded=false` 时不渲染 DiffView

新建 `tui/src/__tests__/chatTranscript.test.tsx` 已存在；补一个 case：

- `edit tool-call with summary renders EditSummary instead of empty in coalesce mode`
- `failed edit tool-call still appears in ExploredTree`

修改 `tui/src/__tests__/chatStore.test.ts`：

- `addToolResult populates summary when toolName=file_edit and metadata has lines counts`
- `addToolResult does NOT populate summary when toolName=shell`

### Python

无 Python 侧改动 —— 本 PR 完全在 TUI 层。

## 验收 / Acceptance

- `npm test --prefix tui` 全绿
- `npm run typecheck --prefix tui` 零新增告警
- 手动场景（已 PR 9.1 合入的前提下）：
  1. 让模型 edit 一个文件，批准 —— chat 中出现 `● Edited <basename> (+N −M)` 行
  2. 上下方向键 focus 到该 EditSummary 行，Tab+Enter 展开 —— 看到内联 unified diff
  3. 让模型连续 edit 三个文件 —— 三条 `● Edited` 行依次出现，**不被合并到 `● Explored` 里**
  4. 让模型故意 edit 一个不存在的字符串（失败）—— chat 中没有 EditSummary，但 Explored 树里有 `(failed)` 标记

## 不在本 PR

- 失败的 edit 也显示 diff（暂不做；失败用例 metadata 通常不完整）
- 多文件 batch summary（单文件场景足够 cover Claude Code 当前展示）
- diff 的语法着色（保留 PR 9.+ polish）
