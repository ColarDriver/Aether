# PR 4 — Direct-Write Shimmer (Bypass Ink for Cell-Level Animation)

## 目标 / Goal

把 ActivityBar 的 shimmer 高亮窗口 **从 React tree 中拆出来**，让它不再经过 Ink
的 log-update 全 live-region 重绘，而是通过我们自己控制的 `cursorTo +
重写那一行` 直接写 stdout。这是借鉴 `open-claude-code/src/ink/log-update.ts` 的
cell-level diff 思路的局部应用 — 只对 shimmer 这一行做，不重写整个 Ink。

这样即使 PR 1-3 全部生效，shimmer 仍是 Ink live region 中**唯一**每 150ms tick
的元素 (其他都已经稳定了)。把它从 Ink 重绘路径里拿走后，**spinner 持续动画
期间，Ink 的 React tree 实际上是稳定的 → log-update 完全不写 stdout → 即使在
不支持 DEC 2026 的 terminal 上，原生 scrollbar 也不再被任何东西干扰**。

这是 sprint 中风险最高的一个 PR，所以放在最后，并且默认 **opt-in** (env flag)。

## 当前问题 / Current Problem

经过 PR 1-3：

- live region 高度 ≤ 12 行，cursor-up 不再越过 viewport (PR 2)
- shimmer 滚出 viewport 暂停 (PR 3)
- 用户在 viewport 内时 shimmer 仍要 tick → React rerender → Ink log-update →
  `eraseLines(previousLineCount)` (PR 1 atomic-sync 保护)

仍然有一个剩余场景：**用户的 transcript 内容塞满 viewport，shimmer 在
viewport 内仍 tick，shimmer cell 变动触发 React rerender，Ink rewrite 整个 live
region**。在 DEC 2026 支持的 terminal 上 PR 1 已经消解了 yank；在不支持 DEC
2026 但行为正常的 terminal 上 (例如老 macOS Terminal.app, 一些 Linux x11
terminal)，rewrite 仍会触发 scroll-on-output 抢回滚动条。

如果我们**根本不让 React rerender**，那 Ink 的 log-update 不会被叫醒，rewrite
不会发生。shimmer 通过侧道写到那个 cell 的对应屏幕坐标即可。

OCC 的 `log-update.ts` 做了一套完整的 screen diff，能力远超我们需要的。本 PR
只实现一个最小的子集：**仅当唯一变化是 shimmer highlight 的位置时，绕开 Ink
直写 ANSI**。

## 改动 / Changes

### NEW `tui/src/lib/shimmerWriter.ts`

```ts
import { nowMs, subscribe } from './animationClock.js'
import { shimmer, type ShimmerSlices } from './shimmer.js'
import { theme } from './theme.js'

/**
 * 把 shimmer 高亮直接写到指定屏幕坐标 (row, col)，绕开 Ink。
 *
 * 关键约束：
 * - 只能在 ActivityBar 已经被 Ink 渲染过一帧 (我们知道它的坐标) 之后启动。
 * - 受 stdout 的 atomic-sync proxy 保护：BSU + cursor save + cursorTo +
 *   colored slice + cursor restore + ESU 一次性 write，与 PR 1 的 proxy
 *   配合实现单 syscall。
 * - 写入的 ANSI 序列不通过 Ink 的 log-update，因此 Ink 的 previousOutput
 *   仍然是"上一帧的字符串"。下一次 React rerender 时 Ink 仍会基于这个旧
 *   字符串做 diff — 这是可以接受的，因为：
 *     1. shimmer 区域的字符内容 (label) 没变，只是颜色不同；
 *     2. 下一次 React 真的 rerender 时它会写出当前帧的完整 string，覆盖
 *        我们的副作用；
 *     3. 我们不会在 Ink 即将 rerender 的同一 tick 里再写一次。
 */

import { BSU, ESU, SYNC_SUPPORTED } from './atomicSyncStdout.js'
const SAVE = '\x1b7'
const RESTORE = '\x1b8'

export interface ShimmerWriterOptions {
  /** 当前 shimmer cell 在 terminal 中的绝对 (row, col) — 由调用方测量后传入 */
  row: number
  col: number
  /** shimmer 文本 (verb) */
  label: string
  /** 颜色：base = 灰，highlight = 亮 */
  baseColor: string
  highlightColor: string
  /** tick 间隔；典型 150ms */
  intervalMs: number
}

interface RunningShimmer {
  stop: () => void
}

function isEnabled(): boolean {
  return process.env.AETHER_SHIMMER_DIRECT_WRITE === '1'
}

export function startShimmerWriter(opts: ShimmerWriterOptions): RunningShimmer | null {
  if (!isEnabled() || !process.stdout.isTTY) return null

  let lastTick = -1
  const intervalSubscription = subscribe(() => {
    const tick = Math.floor(nowMs() / opts.intervalMs)
    if (tick === lastTick) return
    lastTick = tick
    writeShimmerFrame(tick, opts)
  })

  return {
    stop: () => {
      intervalSubscription()
    }
  }
}

function writeShimmerFrame(tick: number, opts: ShimmerWriterOptions): void {
  const slices: ShimmerSlices = shimmer(opts.label, tick)
  const base = colorize(opts.baseColor)
  const hi = colorize(opts.highlightColor)
  const reset = '\x1b[0m'

  // cursorTo(row, col): \x1b[<row>;<col>H
  const cursorTo = `\x1b[${opts.row};${opts.col}H`

  const payload =
    cursorTo +
    base + slices.before + reset +
    hi + slices.highlight + reset +
    base + slices.after + reset

  // 包 BSU/ESU 让 PR 1 的 proxy 一次 syscall 写完；不支持的 terminal 上
  // proxy 会丢字节但保留 payload。再加 save/restore 防止移动 cursor 影响
  // 正在输入的 composer 光标位置。
  const frame =
    (SYNC_SUPPORTED ? BSU : '') +
    SAVE +
    payload +
    RESTORE +
    (SYNC_SUPPORTED ? ESU : '')

  process.stdout.write(frame)
}

function colorize(name: string): string {
  // 简化映射；真实代码可以读 theme.color(name) 然后映射到 ANSI 256/truecolor
  return name === 'bright' ? '\x1b[1;97m' : '\x1b[2;37m'
}
```

### MOD `tui/src/components/ActivityBar.tsx`

只在 `AETHER_SHIMMER_DIRECT_WRITE=1` 时启用：

```ts
import { useEffect, useRef } from 'react'
import { startShimmerWriter } from '../lib/shimmerWriter.js'

// ...inside ActivityBar:
const shimmerHandleRef = useRef<{ stop: () => void } | null>(null)
useEffect(() => {
  if (!shouldAnimate) return
  // 等 Ink 至少绘一帧后再启动，否则坐标取不到
  const raf = setTimeout(() => {
    // 测量当前 shimmer cell 的屏幕坐标 (row, col)：
    // - col 来自 icon + space 的字符宽度
    // - row 来自 stdout.rows - 当前 live region 高度 + 0
    // 实际实现需要从 yogaNode.getComputedTop/Left 拿
    shimmerHandleRef.current = startShimmerWriter({
      row: /* measured */ 0,
      col: /* measured */ 3,
      label: verb,
      baseColor: 'dim',
      highlightColor: 'bright',
      intervalMs: SPINNER_INTERVAL_MS
    })
  }, 50)
  return () => {
    clearTimeout(raf)
    shimmerHandleRef.current?.stop()
    shimmerHandleRef.current = null
  }
}, [shouldAnimate, verb])
```

当 direct-write 启用时，**React 端就不再渲染 shimmer 高亮的不同颜色**，shimmer
区域统一渲染成 base color (避免和 direct write 打架)。React tree 在 spinner
tick 期间稳定。

```ts
// 在 direct-write 模式下，shimmerSlices 替换为单色 verb：
const directWriteActive = process.env.AETHER_SHIMMER_DIRECT_WRITE === '1' && shouldAnimate
const shimmerSlices = directWriteActive ? null : (isActive && animate ? shimmer(verb, animationTick) : null)
```

### Opt-in gating / 默认关闭

- env `AETHER_SHIMMER_DIRECT_WRITE=1` 才启用。
- 默认 React 端走老路径 (PR 3 之后的共享时钟版本)。
- 文档建议：先在 PR 1-3 都已稳定的 terminal 上灰度，确认没坐标对不齐的 case
  后再考虑默认开启。
- **不修改任何 Ink 内部逻辑** — direct-write 是"额外的 cursor save/restore
  片段"，与 Ink 的 log-update 状态机正交。

## 风险与边界 / Risks & Boundaries

| 风险 | 缓解 |
|---|---|
| Ink 重绘时把我们的 direct-write 覆盖 | 接受 — Ink rerender 应当少 (因为我们在 React 端把 shimmer 变成单色)；偶尔的 rerender 会把 cell 重置回 base color，下一 tick 我们再写回 |
| 屏幕坐标 (row, col) 测错 → 写到错误位置 | 用 `yogaNode.getComputedTop/Left` 测量；50ms delay 等 Ink 首帧；坐标变化时由 useEffect 重启 |
| 与 composer 输入的 cursor 位置冲突 | 用 `\x1b7` (save) / `\x1b8` (restore) 包裹；写完恢复 cursor |
| 用户 resize 终端 → 坐标过期 | resize event 触发 React rerender → useEffect re-fire → 重新测量 |
| 不支持 DEC 2026 的 terminal 上无法 atomic write | PR 1 proxy 丢 BSU/ESU 后剩 `save + cursorTo + payload + restore`；这本就是一个 ANSI escape 序列，单次 write 即可，原子性已经够 |
| 历史失败 #8 (escape 进 composer) | 我们的 payload 全部是写出去 (output) 的 escape，不读 stdin；与 mouse/key escape 完全无关 |
| Windows conhost cursor-up yank | direct-write 不发 cursor-up，只发绝对 cursor position；Windows 表现优于 master |
| 测试覆盖 | 写一个 `_writeShimmerFrame` export，在 Node Writable mock 上断言 frame 内容；不测真实坐标 |

## 不修改的内容 / Out of Scope

- 不替换 Ink 的 log-update — 仍走上游路径，只是 shimmer cell 不触发它。
- 不实现 OCC 的 full cell-diff 算法 — 那是 PR 5 调研路径。
- 不动 banner / composer / overlay 渲染。
- 不动 React tree 中 ActivityBar 其他部分的样式 / token segments。
- 不默认启用 — 默认 `AETHER_SHIMMER_DIRECT_WRITE` 未设置 → 走 PR 3 的安全
  路径。

## 文件清单 / File Manifest

- NEW `tui/src/lib/shimmerWriter.ts`
- NEW `tui/src/lib/__tests__/shimmerWriter.test.ts`
- MOD `tui/src/components/ActivityBar.tsx` (opt-in direct write 路径 + 测量
  坐标 + React 端 shimmer 单色化)

## 验收 / Acceptance

1. `npm --prefix tui test` 全绿。
2. 默认情况 (`AETHER_SHIMMER_DIRECT_WRITE` 未设置) 与 PR 3 行为完全一致。
3. `AETHER_SHIMMER_DIRECT_WRITE=1` 启动：
   - shimmer 视觉效果正确，高亮窗口按预期扫过 verb。
   - composer 输入光标位置不被打乱。
   - 用 `script` 抓 stdout：spinner tick 期间 **没有 `eraseLines` 序列**，
     只有 `cursorTo + colored slice + restore` 的小片段。
4. 在不支持 DEC 2026 的 terminal (例如老 macOS Terminal.app) 上启用 direct-
   write，spinner tick 期间拖原生 scrollbar **不被拽走**。
5. resize 终端，shimmer 自动重新定位到新坐标，无视觉错位。
6. opt-out 简单：unset env var 后立刻回到 PR 3 行为。
