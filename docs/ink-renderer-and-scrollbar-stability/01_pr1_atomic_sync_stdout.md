# PR 1 — Atomic Sync Stdout Proxy

## 目标 / Goal

在不动 Ink 源代码的前提下，给 `process.stdout` 套一层 proxy，使得每一帧的
`BSU + log-update.output + ESU` 走 **一次** 底层 `write()` 系统调用，并用显式
allowlist 决定哪些 terminal 真正启用 DEC 2026 synchronized output (其余 terminal
直接丢弃 BSU/ESU 字节，避免无用的 CSI 噪声打进 tmux / 老 macOS Terminal)。

期望效果：在主流现代 terminal (iTerm2 / WezTerm / Kitty / Ghostty / Alacritty /
WindowsTerminal / GNOME Terminal VTE ≥ 0.68 / VS Code / Warp / Contour / Zed /
Foot) 上，spinner 每 150ms tick 时，**单次 write 整帧**进入 terminal；同步窗口
内 terminal 不刷新 viewport，因此用户拖动原生 scrollbar 不会被拽走。

## 当前问题 / Current Problem

Ink 6 内置的 `write-synchronized.js` 只做了非常薄的一层：

```js
// tui/node_modules/ink/build/write-synchronized.js
export const bsu = '[?2026h';
export const esu = '[?2026l';
export function shouldSynchronize(stream) {
    return 'isTTY' in stream && stream.isTTY === true && !isInCi;
}
```

调用点 (`tui/node_modules/ink/build/ink.js:149-156` 类似多处) 长这样：

```js
const sync = shouldSynchronize(this.options.stdout);
if (sync) this.options.stdout.write(bsu);
this.log(outputToRender);          // log-update 内部还会再 write 一次
if (sync) this.options.stdout.write(esu);
```

两个问题：

1. **检测过粗**。`isTTY && !isInCi` 在 tmux 下也返回 `true`，但 tmux 实际上会把
   BSU/ESU 字节按字符切碎转发，终端收到的不是原子序列；在不支持 DEC 2026 的老
   terminal 上则会出现未识别 CSI 噪声 (一般无害但污染 `script` / `cat -v` 抓取)。
2. **非原子写**。`bsu` / `output` / `esu` 是三次独立的 `stream.write()`。Node
   `process.stdout` 在 TTY 上虽然是同步的，但终端 emulator 的 input parser 可
   能在三次 syscall 之间已经做了一部分渲染工作。OCC 的对照实现
   `open-claude-code/src/ink/terminal.ts:190-248` 是把 BSU + 内容 + ESU buffer
   成一个 string 后一次性 write — atomicity 强于现状。

我们不改 Ink 的源代码 (修 node_modules 会被 npm 重置)。改法是把外部 stdout 替换
成一个 proxy：把 BSU / ESU 之间的所有 write 攒成一段 buffer，识别到 ESU 后整段
flush；如果当前 terminal 不支持 sync mode，BSU / ESU 直接丢弃。

## 改动 / Changes

### NEW `tui/src/lib/atomicSyncStdout.ts`

```ts
import type { Writable } from 'node:stream'

export const BSU = '[?2026h'
export const ESU = '[?2026l'

/**
 * 显式 allowlist：判断当前 terminal 是否真正实现 DEC 2026 synchronized
 * output。env 在 session 内不会变，所以模块加载时 cache 一次即可。
 *
 * 参考 open-claude-code/src/ink/terminal.ts:70-118
 */
export function isSynchronizedOutputSupported(): boolean {
  // tmux 会把 CSI 切碎转发，atomicity 已被破坏 — 直接关闭。
  if (process.env.TMUX) return false

  const termProgram = process.env.TERM_PROGRAM
  const term = process.env.TERM

  if (
    termProgram === 'iTerm.app' ||
    termProgram === 'WezTerm' ||
    termProgram === 'WarpTerminal' ||
    termProgram === 'ghostty' ||
    termProgram === 'contour' ||
    termProgram === 'vscode' ||
    termProgram === 'alacritty'
  ) return true

  if (term?.includes('kitty') || process.env.KITTY_WINDOW_ID) return true
  if (term === 'xterm-ghostty') return true
  if (term?.startsWith('foot')) return true
  if (term?.includes('alacritty')) return true
  if (process.env.ZED_TERM) return true
  if (process.env.WT_SESSION) return true

  const vte = process.env.VTE_VERSION
  if (vte) {
    const v = parseInt(vte, 10)
    if (!Number.isNaN(v) && v >= 6800) return true
  }

  return false
}

export const SYNC_SUPPORTED = isSynchronizedOutputSupported()

type WriteArgs =
  | [chunk: string | Uint8Array]
  | [chunk: string | Uint8Array, cb: (err?: Error | null) => void]
  | [chunk: string | Uint8Array, encoding: BufferEncoding, cb?: (err?: Error | null) => void]

/**
 * 包一层 stdout。BSU 之间的 chunk 全部 buffer，遇到 ESU flush 成一次 write。
 *
 * - 在 BSU 之外的 write 透传给底层 stdout。
 * - 在 !SYNC_SUPPORTED 的 terminal 上，BSU / ESU 直接丢弃；其余 chunk 透传。
 * - 严格相等比较即可，因为 Ink 总是单独 write BSU / ESU (不会拼在 chunk 内)。
 */
export function wrapStdoutWithAtomicSync(stdout: NodeJS.WriteStream): NodeJS.WriteStream {
  if (!SYNC_SUPPORTED) {
    // 仍然包一层，只是行为是 "丢弃 BSU/ESU，其他透传"，避免 CSI 噪声
    return makeProxy(stdout, /* sync */ false)
  }
  return makeProxy(stdout, /* sync */ true)
}

function makeProxy(stdout: NodeJS.WriteStream, sync: boolean): NodeJS.WriteStream {
  let buffer: string | null = null

  const write = (...args: WriteArgs): boolean => {
    const chunk = args[0]
    const cb = typeof args[args.length - 1] === 'function'
      ? (args[args.length - 1] as (err?: Error | null) => void)
      : undefined

    const asString = typeof chunk === 'string' ? chunk : null

    // === Sync supported path ===
    if (sync) {
      if (asString === BSU) {
        buffer = BSU
        cb?.()
        return true
      }
      if (asString === ESU) {
        if (buffer !== null) {
          const full = buffer + ESU
          buffer = null
          return stdout.write(full as never, cb as never)
        }
        // 防御：ESU 没匹配到 BSU，按透传
        return stdout.write(chunk as never, cb as never)
      }
      if (buffer !== null) {
        buffer += typeof chunk === 'string' ? chunk : chunk.toString()
        cb?.()
        return true
      }
      return stdout.write(chunk as never, cb as never)
    }

    // === Sync not supported path: 丢弃 BSU/ESU，其他透传 ===
    if (asString === BSU || asString === ESU) {
      cb?.()
      return true
    }
    return stdout.write(chunk as never, cb as never)
  }

  return new Proxy(stdout, {
    get(target, prop, receiver) {
      if (prop === 'write') return write
      const value = Reflect.get(target, prop, receiver)
      return typeof value === 'function' ? value.bind(target) : value
    }
  }) as NodeJS.WriteStream
}
```

要点：

- 用 `Proxy` 而不是 `Object.create`，保证 `isTTY` / `columns` / `rows` / `on` /
  `off` / `getColorDepth` / resize event 全部透传到底层 (Ink 监听
  `process.stdout.on('resize')` 来取最新 `rows`，包错会破内部宽度逻辑)。
- 用 **严格相等** 比较 BSU / ESU。Ink 6 的现状 (验证过 `ink.js:149-156`、`ink.js:322-336`、`ink.js:340-348`、`throttled-log` 的多处实现) 都是单独 write
  BSU / ESU，从不和正文拼接，所以严格比较已足够。如果上游未来改成拼接 write，
  我们的 detection 应当回退而不是误判 — 这种情况下 ESU 会走到 "ESU 没匹配 BSU"
  的防御透传分支，行为退化为 master。
- `cb` 在 buffer 阶段直接同步调用 — 这模仿 stdout TTY 同步语义，下游不会真正
  依赖 cb 异步。

### MODIFY `tui/src/entry.tsx:36-41`

```diff
 import { applyCliArgs } from './lib/cliArgs.js'
 import { terminateDevWatchRunner } from './lib/devExit.js'
+import { wrapStdoutWithAtomicSync } from './lib/atomicSyncStdout.js'
```

```diff
   const instance = render(
     <App client={client} repoRoot={repoRoot} workspaceCwd={workspaceCwd} />,
     {
-      exitOnCtrlC: false
+      exitOnCtrlC: false,
+      stdout: wrapStdoutWithAtomicSync(process.stdout)
     }
   )
```

注意：

- TTY 探测仍用 `process.stdin.isTTY` / `process.stdout.isTTY` (`entry.tsx:31`)，
  没必要包过 proxy 才判断。
- 不动 `process.stdout.write` 全局 monkey-patch — proxy 只作用于传入 Ink 的那
  个 stream，避免污染日志 / 第三方库。

### NEW `tui/src/lib/__tests__/atomicSyncStdout.test.ts`

```ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { Writable } from 'node:stream'
import { BSU, ESU, wrapStdoutWithAtomicSync } from '../atomicSyncStdout.js'

function fakeStdout(): NodeJS.WriteStream & { writes: string[] } {
  const writes: string[] = []
  const w = new Writable({
    write(chunk, _enc, cb) {
      writes.push(chunk.toString())
      cb()
    }
  }) as never as NodeJS.WriteStream & { writes: string[] }
  ;(w as { isTTY: boolean }).isTTY = true
  ;(w as { columns: number }).columns = 80
  ;(w as { rows: number }).rows = 24
  w.writes = writes
  return w
}

describe('atomicSyncStdout', () => {
  const origEnv = { ...process.env }
  beforeEach(() => {
    for (const k of Object.keys(process.env)) delete process.env[k]
  })
  afterEach(() => {
    for (const k of Object.keys(process.env)) delete process.env[k]
    Object.assign(process.env, origEnv)
  })

  it('coalesces BSU + content + ESU into a single underlying write on supported terminal', async () => {
    process.env.TERM_PROGRAM = 'WezTerm'
    vi.resetModules()
    const { wrapStdoutWithAtomicSync: wrap } = await import('../atomicSyncStdout.js')
    const inner = fakeStdout()
    const wrapped = wrap(inner)

    wrapped.write(BSU)
    wrapped.write('hello')
    wrapped.write('world')
    wrapped.write(ESU)

    expect(inner.writes).toEqual([BSU + 'hello' + 'world' + ESU])
  })

  it('drops BSU/ESU on unsupported terminal but passes content through', async () => {
    delete process.env.TERM_PROGRAM
    delete process.env.TERM
    process.env.TMUX = '/tmp/tmux-1000/default,123,0'
    vi.resetModules()
    const { wrapStdoutWithAtomicSync: wrap } = await import('../atomicSyncStdout.js')
    const inner = fakeStdout()
    const wrapped = wrap(inner)

    wrapped.write(BSU)
    wrapped.write('payload')
    wrapped.write(ESU)

    expect(inner.writes).toEqual(['payload'])
  })

  it('passes non-BSU writes through outside any sync window', async () => {
    process.env.TERM = 'xterm-kitty'
    vi.resetModules()
    const { wrapStdoutWithAtomicSync: wrap } = await import('../atomicSyncStdout.js')
    const inner = fakeStdout()
    const wrapped = wrap(inner)

    wrapped.write('plain')
    expect(inner.writes).toEqual(['plain'])
  })

  it('preserves isTTY / columns / rows on the proxy', async () => {
    process.env.TERM_PROGRAM = 'iTerm.app'
    vi.resetModules()
    const { wrapStdoutWithAtomicSync: wrap } = await import('../atomicSyncStdout.js')
    const inner = fakeStdout()
    const wrapped = wrap(inner)

    expect(wrapped.isTTY).toBe(true)
    expect(wrapped.columns).toBe(80)
    expect(wrapped.rows).toBe(24)
  })

  it('forwards resize events from underlying stdout', async () => {
    process.env.TERM_PROGRAM = 'iTerm.app'
    vi.resetModules()
    const { wrapStdoutWithAtomicSync: wrap } = await import('../atomicSyncStdout.js')
    const inner = fakeStdout()
    const wrapped = wrap(inner)
    const handler = vi.fn()
    wrapped.on('resize', handler)
    inner.emit('resize')
    expect(handler).toHaveBeenCalledTimes(1)
  })
})
```

## 风险与边界 / Risks & Boundaries

| 风险 | 缓解 |
|---|---|
| Ink 未来把 BSU 和正文拼成一个 chunk write | proxy 严格比较失败 → 走 "BSU 不匹配" 的透传分支，不会卡死。回归即"和 master 等价"。 |
| 上游 `write-synchronized.js` 加入更多 wrap 点 | 不影响 — proxy 只关心 chunk 内容是 BSU/ESU 与否。 |
| stdout proxy 阻塞 cb 调用 | 在 buffer 阶段同步调用 cb，Ink/log-update 都不依赖异步行为。 |
| 终端不在 allowlist 内但实际支持 DEC 2026 | 行为退化为 master：用户无收益，但也无回归。后续 PR 5 调研 XTVERSION 探测。 |
| Windows conhost cursor-up yank bug (OCC 标注的) | 本 PR 不处理，PR 5 调研。 |

## 不修改的内容 / Out of Scope

- `ActivityBar.tsx`、`ReasoningLine.tsx`、`ChatTranscript.tsx`、composer
  布局 — PR 2/3/4 处理。
- Ink 源文件 — 修 `node_modules/` 会被 install 重置，且毁了未来升级路径。
- `process.stdout` 全局 monkey-patch — proxy 范围严格限制在传入 Ink 的 stream。

## 文件清单 / File Manifest

- NEW `tui/src/lib/atomicSyncStdout.ts`
- NEW `tui/src/lib/__tests__/atomicSyncStdout.test.ts`
- MOD `tui/src/entry.tsx` (+2 行 import / +1 行 stdout option)

## 验收 / Acceptance

1. `npm --prefix tui test` 全绿，新 atomicSyncStdout 测试覆盖率 100%。
2. `npm --prefix tui run build` 通过。
3. 手测：iTerm2 / WezTerm / GNOME Terminal (VTE ≥ 6800) 下，发一个长任务，
   shimmer 持续 tick 期间拖动原生 scrollbar 到中间位置，**滚动条不被拽走**。
4. 手测：tmux 内运行，`script` 抓取 stdout，确认 `\x1b[?2026h` / `\x1b[?2026l`
   字节 **不出现**。
5. 手测：`master` 上同样场景对比，确认本 PR 实际消除了 yank (而非环境侥幸)。
