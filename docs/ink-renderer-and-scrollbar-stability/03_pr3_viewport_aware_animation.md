# PR 3 — Viewport-Aware Animation (Shared Clock + Visibility Gating)

## 目标 / Goal

让 spinner / shimmer / reasoning ticker 在 **滚出 terminal viewport 后停止 tick**，
不再触发 React rerender，进而不再让 Ink log-update 写新一帧 stdout — 即使用户拖
着原生 scrollbar 看历史，没人也没东西在底下默默催 Ink 重绘。

同时引入 **共享时钟 (shared clock)**：多个动画组件 (`ActivityBar` 的 spinner、
shimmer，`ReasoningLine` 的 elapsed ticker，未来的 progress bar 等) 共用一个
`setInterval`，每次 tick 只触发一次顶层重绘，避免多个独立 `setInterval` 错峰
触发的 "重绘风暴"。

参考实现：

- `open-claude-code/src/ink/hooks/use-animation-frame.ts`
- `open-claude-code/src/ink/hooks/use-terminal-viewport.ts`

## 当前问题 / Current Problem

### 多个独立 setInterval

```ts
// tui/src/components/ActivityBar.tsx:103-105
const handle = setInterval(() => {
  setAnimationTick((tick) => tick + 1)
}, SPINNER_INTERVAL_MS)  // 150ms
```

```ts
// tui/src/components/ReasoningLine.tsx (1s ticker)
```

两个独立 interval，相位不同步 → 每秒会出现"两次小重绘"+"周期性大重绘"，
shimmer 帧间隔与 reasoning 秒钟切换相互错峰，每个 cycle 内 Ink 会被触发 2-3
次重绘。每次重绘都把 PR 1 / PR 2 的好处稀释一点 — 帧少 → yank 概率小，帧多 →
即使原子 sync 也压不住非支持终端。

### 滚出 viewport 仍 tick

`ActivityBar` 只在 `!ACTIVE_STATES.has(activity.status)` 时停止 interval。
用户拖原生 scrollbar 看历史时，`status` 仍是 `thinking` / `responding`，组件
仍在 tick，**即使它已经滚出 terminal 可视区域** — 每 150ms 仍然产 1 帧重绘，每
帧仍然在底下重画 (从 terminal 视角不可见但 stdout 还在 write)。

Ink 的 `log-update` 不知道 ActivityBar 是不是滚出去了，它只看到 React tree 还
在变 → 仍然走 `eraseLines + content` 的全 live region rewrite。yank 仍然会发生。

OCC 的解法是用 yoga layout 走 DOM 父链算 element 的绝对 top，比较 `viewportY`：

```ts
// open-claude-code/src/ink/hooks/use-terminal-viewport.ts:46-93
const height = element.yogaNode.getComputedHeight()
const rows = terminalSize.rows
let absoluteTop = element.yogaNode.getComputedTop()
let parent = element.parentNode
let root = element.yogaNode
while (parent) {
  if (parent.yogaNode) {
    absoluteTop += parent.yogaNode.getComputedTop()
    root = parent.yogaNode
  }
  if (parent.scrollTop) absoluteTop -= parent.scrollTop
  parent = parent.parentNode
}
const screenHeight = root.getComputedHeight()
const bottom = absoluteTop + height
const cursorRestoreScroll = screenHeight > rows ? 1 : 0
const viewportY = Math.max(0, screenHeight - rows) + cursorRestoreScroll
const viewportBottom = viewportY + rows
const visible = bottom > viewportY && absoluteTop < viewportBottom
```

Aether 上游 Ink 6 的 `DOMElement.yogaNode` 也有 `getComputedHeight` /
`getComputedTop`，所以这套移植**可行**，但 OCC hook 还依赖：

- `TerminalSizeContext` (Aether 没有，要 reuse `useStdout().rows`)
- 自己的 `ClockContext` (Aether 要新写一个)
- React Compiler runtime (Aether 不引入)

我们把 hook 重写成 **不依赖 Compiler、不依赖 OCC 的 context、能在 Ink 6 上直接
跑** 的版本。

## 改动 / Changes

### NEW `tui/src/lib/animationClock.ts`

```ts
import { useEffect, useState } from 'react'

/**
 * 单一全局时钟。订阅者 ≥ 1 时启动 setInterval，= 0 时关闭。
 * 所有 spinner 共用一个 16ms (或 50ms) 心跳，组件内自己用 modulo 控制
 * 实际重绘间隔。
 */

type Listener = () => void
const listeners = new Set<Listener>()
let timer: NodeJS.Timeout | null = null
let startedAt = 0
const HEART_BEAT_MS = 50

export function nowMs(): number {
  return Date.now()
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn)
  if (timer === null) {
    startedAt = nowMs()
    timer = setInterval(() => {
      for (const l of listeners) l()
    }, HEART_BEAT_MS)
  }
  return () => {
    listeners.delete(fn)
    if (listeners.size === 0 && timer !== null) {
      clearInterval(timer)
      timer = null
    }
  }
}

export function clockStartedAt(): number {
  return startedAt
}

/** Test seam: stop the clock and clear all listeners. */
export function _resetClockForTests(): void {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
  listeners.clear()
}
```

### NEW `tui/src/lib/useTerminalViewport.ts`

```ts
import { useCallback, useLayoutEffect, useRef } from 'react'
import { useStdout } from 'ink'
import type { DOMElement } from 'ink'

interface ViewportEntry { isVisible: boolean }

/**
 * Ink 6 DOMElement 暴露 yogaNode（getComputedHeight/getComputedTop）。
 * 通过 walk DOM 父链算出元素相对 root 的绝对 top，与当前 terminal rows
 * 推得的 viewport 区间比较得出 isVisible。
 *
 * 不触发 setState；只更新 ref。调用者 (useAnimationFrame) 在自己的 tick
 * 里读 ref。
 */
export function useTerminalViewport(): [
  ref: (el: DOMElement | null) => void,
  entry: ViewportEntry
] {
  const { stdout } = useStdout()
  const elRef = useRef<DOMElement | null>(null)
  const entryRef = useRef<ViewportEntry>({ isVisible: true })

  const setEl = useCallback((el: DOMElement | null) => {
    elRef.current = el
  }, [])

  useLayoutEffect(() => {
    const el = elRef.current
    if (!el?.yogaNode || !stdout) return
    const rows = stdout.rows ?? 24
    const height = el.yogaNode.getComputedHeight()
    let absoluteTop = el.yogaNode.getComputedTop()
    let root = el.yogaNode
    let parent: DOMElement | undefined = el.parentNode as DOMElement | undefined
    while (parent) {
      if (parent.yogaNode) {
        absoluteTop += parent.yogaNode.getComputedTop()
        root = parent.yogaNode
      }
      // Aether 当前没有自定义 ScrollBox；保留这条路径方便后续接入
      const sTop = (parent as { scrollTop?: number }).scrollTop
      if (sTop) absoluteTop -= sTop
      parent = parent.parentNode as DOMElement | undefined
    }
    const screenHeight = root.getComputedHeight()
    const bottom = absoluteTop + height
    const cursorRestoreScroll = screenHeight > rows ? 1 : 0
    const viewportY = Math.max(0, screenHeight - rows) + cursorRestoreScroll
    const viewportBottom = viewportY + rows
    const visible = bottom > viewportY && absoluteTop < viewportBottom
    if (visible !== entryRef.current.isVisible) {
      entryRef.current = { isVisible: visible }
    }
  })

  return [setEl, entryRef.current]
}
```

### NEW `tui/src/lib/useAnimationFrame.ts`

```ts
import { useEffect, useState } from 'react'
import { nowMs, subscribe } from './animationClock.js'
import { useTerminalViewport } from './useTerminalViewport.js'
import type { DOMElement } from 'ink'

/**
 * 共享时钟驱动的动画 hook。
 *
 * - intervalMs = null → 暂停 (不订阅时钟)
 * - 滚出 viewport → 暂停
 * - 否则按 intervalMs 节流重绘
 *
 * 返回 [ref, time]：ref 挂到要追踪可见性的 element；time 是单调毫秒
 * 时间 (调用方用 `Math.floor(time / FRAME_MS) % N` 取帧)。
 */
export function useAnimationFrame(
  intervalMs: number | null
): [ref: (el: DOMElement | null) => void, time: number] {
  const [vpRef, vp] = useTerminalViewport()
  const [time, setTime] = useState(() => nowMs())

  const active = intervalMs !== null && vp.isVisible

  useEffect(() => {
    if (!active) return
    let lastFire = nowMs()
    return subscribe(() => {
      const t = nowMs()
      if (t - lastFire >= intervalMs!) {
        lastFire = t
        setTime(t)
      }
    })
  }, [active, intervalMs])

  return [vpRef, time]
}
```

### MOD `tui/src/components/ActivityBar.tsx`

替换 `setInterval` 为 `useAnimationFrame`：

```diff
-import { useEffect, useRef, useState, type ReactElement } from 'react'
+import { useRef, type ReactElement } from 'react'
+import { useAnimationFrame } from '../lib/useAnimationFrame.js'
```

```diff
 export function ActivityBar({ animate = true }: { animate?: boolean } = {}): ReactElement {
   const activity = useStore(activityState)
   const session = useStore(sessionState)
   const ascii = !theme.isUnicodeAllowed()
   const tokenTurnStartedAtRef = useRef<number | null | undefined>(undefined)
   const displayedResponseLengthRef = useRef(0)
-  const [animationTick, setAnimationTick] = useState(0)
-
-  useEffect(() => {
-    if (!animate) {
-      return
-    }
-    if (!ACTIVE_STATES.has(activity.status)) {
-      return
-    }
-    const handle = setInterval(() => {
-      setAnimationTick((tick) => tick + 1)
-    }, SPINNER_INTERVAL_MS)
-    return () => {
-      clearInterval(handle)
-    }
-  }, [animate, activity.status])
+  const shouldAnimate = animate && ACTIVE_STATES.has(activity.status)
+  const [animationRef, animationTime] = useAnimationFrame(
+    shouldAnimate ? SPINNER_INTERVAL_MS : null
+  )
+  const animationTick = Math.floor(animationTime / SPINNER_INTERVAL_MS)
```

```diff
   return (
-    <Box flexDirection="column">
+    <Box flexDirection="column" ref={animationRef}>
       <Box>
```

### MOD `tui/src/components/ReasoningLine.tsx`

同样替换 1s ticker：

```diff
-import { useEffect, useState } from 'react'
+import { useAnimationFrame } from '../lib/useAnimationFrame.js'
```

```diff
-  const [, setNow] = useState(Date.now())
-  useEffect(() => {
-    const handle = setInterval(() => setNow(Date.now()), 1000)
-    return () => clearInterval(handle)
-  }, [])
+  const [tickRef, time] = useAnimationFrame(1000)
+  void time
```

把 `tickRef` 挂到 reasoning line 的容器 Box。

### NEW tests

`tui/src/lib/__tests__/animationClock.test.ts`:

- 验证 0 订阅 → 不启动 timer
- 1 订阅 → 启动；2 订阅 → 仍只一个 timer
- 全部 unsubscribe → 关闭 timer

`tui/src/lib/__tests__/useAnimationFrame.test.tsx`：

- 使用 `ink-testing-library` 渲染一个挂 `useAnimationFrame(150)` 的组件
- `vi.useFakeTimers()` 推进时间，验证 tick 触发
- 模拟 viewport 不可见 (mock useTerminalViewport)，验证暂停

## 风险与边界 / Risks & Boundaries

| 风险 | 缓解 |
|---|---|
| yoga `getComputedTop` 在某些 React 重渲染时返回 stale 值 | hook 每次渲染都重新 walk 父链 (OCC 同样做法)，不缓存中间结果 |
| `useTerminalViewport` 返回的 `isVisible` 在边界情况下抖动 | hook 只在值变化时更新 ref，没有 setState → 不会引起 cascading rerender |
| Aether 当前无 ScrollBox，但代码留了 `scrollTop` 路径 | 保留路径但不依赖 — 当前所有 yogaNode `scrollTop` 都是 undefined，fast path 直接 falsy |
| 共享时钟在 50ms 心跳上对所有订阅者唤起 | 订阅者用 `intervalMs` 自己 modulo；50ms 心跳本身只是 setInterval 调度，开销极小 |
| 引入 ref 到 ActivityBar / ReasoningLine 后 yoga 重排 | 挂载位置 (外层 Box) 和原结构一致；ref 不改变 yoga 节点数量 |
| 历史失败 #6 (降低 spinner 频率破坏 shimmer) | 不降低频率，只是滚出 viewport 才暂停 — 用户实际看到 spinner 时频率不变 |

## 不修改的内容 / Out of Scope

- 不改 `shimmer()` 算法本身。
- 不动 `chatEpoch` / `<Static>` 拆分逻辑 (PR 2 已经处理)。
- 不引入 React Compiler runtime — hook 写成不依赖 Compiler 的纯 React 版本。
- 不动 permission overlay 下方隐藏 activity 的现有规则 (`app.tsx:87-93`)。
- 不引入 `ClockContext` (OCC 用法)。Aether 用模块级 singleton 更简单，未来需要
  test isolation 时再加 context。

## 文件清单 / File Manifest

- NEW `tui/src/lib/animationClock.ts`
- NEW `tui/src/lib/useTerminalViewport.ts`
- NEW `tui/src/lib/useAnimationFrame.ts`
- NEW `tui/src/lib/__tests__/animationClock.test.ts`
- NEW `tui/src/lib/__tests__/useAnimationFrame.test.tsx`
- MOD `tui/src/components/ActivityBar.tsx` (替换 setInterval)
- MOD `tui/src/components/ReasoningLine.tsx` (替换 setInterval)

## 验收 / Acceptance

1. `npm --prefix tui test` 全绿；新 hook 单测覆盖：
   - 共享时钟订阅 / 取消订阅引用计数
   - intervalMs=null 暂停
   - viewport 不可见暂停
2. `npm --prefix tui run build` 通过。
3. 手测：长任务进行中，把 transcript 滚到上面 (拖原生 scrollbar 离开底部)，
   shimmer **物理上 (从 stdout 写入次数看) 暂停**。可以临时打 `--inspect`
   或 `script -q` 抓 stdout 看 frame 数。
4. 手测：scroll 回底部后，shimmer 立即恢复 tick (无卡顿)。
5. 手测：reasoning line 的"thought for Ns"在滚出后停止增长，回到底部恢复
   (符合预期，因为它本来就是"看着才更新"的指示)。
6. 视觉对比 `master`：shimmer / spinner 在正常使用 (用户没拖) 时帧率视觉无回归。
