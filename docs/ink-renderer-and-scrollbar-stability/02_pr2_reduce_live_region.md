# PR 2 — Shrink the Live Region (Aggressive Static Promotion)

## 目标 / Goal

让 Ink live region (即 React tree 中 `<Static>` 之外、每帧都会重绘的部分) 的高度
**几乎永远 ≤ terminal rows**，从而让 `log-update.js:47-51` 那段
`ansiEscapes.eraseLines(previousLineCount)` 不会让 cursor 跑到 viewport 顶以上。

这是和 PR 1 互补的根因层面的修复：PR 1 让每一帧的 write 变成原子，PR 2 让被
写出的内容本身就不会触发 terminal 的 scroll-on-output 行为。

## 当前问题 / Current Problem

### 物理机制再确认

`tui/node_modules/ink/build/log-update.js:47-51` 标准路径每帧执行：

```js
stream.write(returnPrefix +
    ansiEscapes.eraseLines(previousLineCount) +
    str +
    cursorSuffix);
```

其中 `ansiEscapes.eraseLines(N)` 等于：

```
('\x1b[2K\x1b[1A' × (N - 1)) + '\x1b[2K\x1b[G'
```

也就是 `previousLineCount - 1` 次 cursor-up。**只要 `previousLineCount ≤ rows`，
cursor 永远在 viewport 内，terminal 不会因为 cursor 越界而 scroll。** 反之只要
有一帧的 live region 在 wrap 后超过 `rows`，cursor-up 上溢就会触发 scroll，把
用户手动拖动的滚动条拽回。

shimmer 的伤害不在于"频率高"而在于"频率高 × 每帧都有 cursor 越界"。把
live region 高度压到 rows 以下，shimmer 就算每 50ms tick 也不会再 yank。

### Aether 现状

`tui/src/components/ChatTranscript.tsx:26`：

```ts
const LIVE_CONTEXT_ITEMS = 10
```

`splitStaticPrefix` (line 415-434)：

```ts
function splitStaticPrefix(items, enabled) {
  if (!enabled) return { staticItems: [], liveItems: items }
  let stablePrefixEnd = 0
  while (stablePrefixEnd < items.length && isStaticTranscriptItem(items[stablePrefixEnd])) {
    stablePrefixEnd += 1
  }
  const staticEnd = Math.max(0, stablePrefixEnd - LIVE_CONTEXT_ITEMS)
  return {
    staticItems: items.slice(0, staticEnd),
    liveItems: items.slice(staticEnd)
  }
}
```

问题：

1. **`LIVE_CONTEXT_ITEMS = 10` 是按 "条数" 数的，不是按 "行数" 数的**。一个
   长 reasoning + 长 tool-call result + 长 assistant message 可能各自就 30+ 行，
   10 个 stable item 加上正在 streaming 的 tail，整 live region 轻松达到 200+
   行，远超 24 行的常见 terminal 高度。
2. **`isStaticTranscriptItem` 把 streaming assistant 排除在外**，但 reasoning
   是 `tool-call`，只要 `durationMs !== null` 就算稳定。流式过程中所有非稳定
   item 都会留在 live region 中，叠加 LIVE_CONTEXT_ITEMS 的 10 条 buffer。
3. **`ActivityBar` / `Composer` / `permission overlay` / leading banner** 都在
   live region 内 (`app.tsx` 中它们不在 `<Static>`)，每帧也都算 `previousLineCount`。

我们要把策略从 "保留最近 10 条不静态化" 改成 **"保留最少够用的行数"**，并且当
viewport scroll 不活跃 (用户没在主动拖) 时，可以进一步压缩。

## 改动 / Changes

### MOD `tui/src/components/ChatTranscript.tsx`

#### 1. 按行数而非条数计 stable tail

```diff
-const LIVE_CONTEXT_ITEMS = 10
+/**
+ * live region 最多保留 LIVE_CONTEXT_ROWS 行已稳定内容做"上下文垫底"。
+ * 配合 PR 1 的 atomic-sync stdout，保证 log-update 的 eraseLines 不会让
+ * cursor 越过 viewport 顶。物理上限选 12 行：足够让用户看到上一轮 user
+ * prompt + 简短 assistant 回复，又不至于把整 viewport 占满。
+ */
+const LIVE_CONTEXT_ROWS = 12
+/** 保底至少留 1 条已稳定 item 做语义连续 (用户上一句话不要消失太快) */
+const LIVE_CONTEXT_MIN_ITEMS = 1
```

```diff
 function splitStaticPrefix(
   items: ChatItem[],
-  enabled: boolean
+  enabled: boolean,
+  width: number
 ): { staticItems: ChatItem[]; liveItems: ChatItem[] } {
   if (!enabled) {
     return { staticItems: [], liveItems: items }
   }
   let stablePrefixEnd = 0
   while (stablePrefixEnd < items.length && isStaticTranscriptItem(items[stablePrefixEnd])) {
     stablePrefixEnd += 1
   }
-  // Keep the recent stable tail live so the user can still see the previous
-  // turn and the just-submitted prompt while the next assistant response
-  // streams. Older stable rows are printed once into terminal scrollback.
-  const staticEnd = Math.max(0, stablePrefixEnd - LIVE_CONTEXT_ITEMS)
+  // 按"行数"从尾部往前累，凑够 LIVE_CONTEXT_ROWS 或 LIVE_CONTEXT_MIN_ITEMS 为止
+  let liveRows = 0
+  let liveItems = 0
+  let staticEnd = stablePrefixEnd
+  for (let i = stablePrefixEnd - 1; i >= 0; i--) {
+    const rows = estimateItemRows(items[i], width)
+    if (liveItems >= LIVE_CONTEXT_MIN_ITEMS && liveRows + rows > LIVE_CONTEXT_ROWS) break
+    liveRows += rows
+    liveItems += 1
+    staticEnd = i
+  }
   return {
     staticItems: items.slice(0, staticEnd),
     liveItems: items.slice(staticEnd)
   }
 }
```

#### 2. 引入 `estimateItemRows`

放在 `ChatTranscript.tsx` 末尾或单独 `lib/transcriptRows.ts`：

```ts
/**
 * 粗估一个 ChatItem 在 width 列下需要多少行。准确度不要求像素级 —
 * 用于判断"是否要把这个 item 留在 live region"，宽松点反而更安全 (估多 →
 * 更激进静态化 → 更小的 live region)。
 *
 * 参考 ChatMessage 里既有的 measureContent / wrapByWidth — 但避免循环
 * 引用，先用最朴素的 split('\n') + 字符宽度近似。
 */
function estimateItemRows(item: ChatItem, width: number): number {
  const content = previewText(item)
  if (!content) return 1
  const lines = content.split('\n')
  let rows = 0
  for (const line of lines) {
    const w = stringWidth(line)
    rows += Math.max(1, Math.ceil(w / Math.max(40, width - 2)))
  }
  // 元信息开销：tool-call 框 +2, tool-result +1, 间距 spacer +1 等
  if (item.kind === 'tool-call' || item.kind === 'tool-group') rows += 2
  if (item.kind === 'tool-result') rows += 1
  return rows
}

function previewText(item: ChatItem): string {
  switch (item.kind) {
    case 'user':
    case 'assistant':
    case 'note':
      return item.text ?? ''
    case 'tool-call':
      return [item.title, item.summary ?? ''].filter(Boolean).join('\n')
    case 'tool-group':
      return (item.summary ?? '')
    case 'tool-result':
      return item.text ?? ''
  }
}
```

#### 3. 流式期间也做 collapse

当 `assistant` item 正在 streaming 时它本身不能静态化，但它"曾经的内容"前缀其实
已经稳定，只是 React render 把整段算到一个 live item 里。**本 sprint 不动这条
路径** — 它已经是 historical failure mode #4 的雷区 ("过早静态化导致黑块")。
PR 4 (direct-write shimmer) 才考虑绕开 streaming row 的 rerender。

#### 4. `<Static>` 不动 key/epoch 策略

我们保留 `chatEpoch` 作为 `<Static>` 的 key (`ChatTranscript.tsx` 现有逻辑)，
不让本 PR 触发 reflow。所有改动只在 "splitStaticPrefix 的拆分点" 这一个函数内。

### MOD `tui/src/components/ChatTranscript.tsx` 调用点

把 `width` 传进去：

```diff
   const split = useMemo(
-    () => splitStaticPrefix(allVisible, usesStaticScrollback),
-    [allVisible, usesStaticScrollback]
+    () => splitStaticPrefix(allVisible, usesStaticScrollback, width),
+    [allVisible, usesStaticScrollback, width]
   )
```

### NEW unit tests `tui/src/__tests__/splitStaticPrefix.test.ts`

```ts
import { describe, it, expect } from 'vitest'
import { splitStaticPrefix } from '../components/ChatTranscript.js'
// 如果 splitStaticPrefix 不 export，本 PR 顺带改成 export const

describe('splitStaticPrefix', () => {
  it('promotes prefix to static when live tail exceeds 12 rows', () => {
    const items = mkRows([20, 5, 2, 3, 4]) // 5 stable items, 34 rows total
    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80)
    // 应该把第一个 20 行的 item 推进 static，留下 5+2+3+4=14 行 — 但 14>12，
    // 所以继续推：5 也进 static，留 2+3+4=9 < 12 但 +5=14 越线，停。
    expect(staticItems.length).toBe(2)
    expect(liveItems.length).toBe(3)
  })

  it('always keeps at least LIVE_CONTEXT_MIN_ITEMS in live', () => {
    const items = mkRows([100]) // 单个超大 item
    const { staticItems, liveItems } = splitStaticPrefix(items, true, 80)
    expect(staticItems.length).toBe(0)
    expect(liveItems.length).toBe(1)
  })

  it('returns all items live when staticScrollback disabled', () => {
    const items = mkRows([1, 1, 1])
    const { staticItems, liveItems } = splitStaticPrefix(items, false, 80)
    expect(staticItems).toEqual([])
    expect(liveItems).toEqual(items)
  })
})
```

## 风险与边界 / Risks & Boundaries

| 风险 | 缓解 |
|---|---|
| 静态化太激进导致 user 看不到上一句话 | `LIVE_CONTEXT_MIN_ITEMS = 1` 保底；`LIVE_CONTEXT_ROWS = 12` 在常见 terminal (24 行) 仍能留一半空间给 leading + activity |
| `<Static>` 重复打印 / 黑块 (history #3) | `splitStaticPrefix` 的输入只有 `isStaticTranscriptItem` 已经稳定的 item — 即 streaming 中的 assistant / pending tool-call 永远不会被推进 `<Static>` |
| streaming 时 stable tail 反复抖动导致 epoch 跳变 | 本 PR 不改 `<Static>` 的 key/epoch，只改前缀切分；`<Static>` 内容只会单调增长 |
| `estimateItemRows` 估错 | 估多比估少安全：估多 → 多推进 static → live 更小。坏情况是 stable tail 比期望短 1-2 行 |
| 测试覆盖不足导致 transcript 渲染回归 | 新增 splitStaticPrefix.test.ts + 跑既有 `chatTranscript.test.tsx` 保证不破现有行为 |

## 不修改的内容 / Out of Scope

- 不动 `chatEpoch` / `<Static>` 的 key 策略 — history #3、#4 雷区。
- 不动 streaming assistant row 的渲染逻辑 — PR 4 处理 shimmer，PR 3 处理 viewport
  暂停。
- 不动 composer / banner / activity bar 在 React tree 中的位置 — history #1、#2
  雷区。
- 不引入新的 "scroll mode" / "quiet mode" store — 用户没要求，且增加复杂度。

## 文件清单 / File Manifest

- MOD `tui/src/components/ChatTranscript.tsx` (替换 `LIVE_CONTEXT_ITEMS` 为 row-
  based 计算，新增 `estimateItemRows` / `previewText`)
- NEW `tui/src/__tests__/splitStaticPrefix.test.ts`

## 验收 / Acceptance

1. `npm --prefix tui test` 全绿，新单测覆盖 split 边界。
2. 手测：发一个会产生长 reasoning 的提问，观察 transcript：稳定段尽早进 scrollback，
   live region 在 24 行 terminal 内 **从不超过 12-15 行**。
3. 手测：长会话来回滚动后再发新提问，banner 不重复、不消失、不闪烁。
4. 手测：长 reasoning streaming 中拖 native scrollbar — 即使没有 PR 1 的 atomic
   sync (临时取消)，yank 频率也应显著降低 (因为 `previousLineCount` 物理上不再
   > rows)。
5. 视觉对比 `master`：transcript 内容、上一轮 user prompt 可见性、tool-call 框、
   tool-result 框、note 行视觉无回归。
