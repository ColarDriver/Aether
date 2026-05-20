# Ink Renderer & Scrollback Stability — Sprint Overview

## 背景 / Motivation

Aether TUI 出现的反复回归是：长会话进行中，用户拖动 terminal 自带的滚动条 (native
scrollbar) 想往上看历史，结果每隔 ~150ms 滚动位置就被拽回到底部，或者拽到最顶部
卡住。表现成 "spinner 一直抓着下拉条"。

`/workspace/Aether/tmp/drag-error/error.md` 记录了过去九次试错的失败模式 (alt-screen
误用、composer 固定底部、banner 重复消失、`<Static>` 误用、过早静态化、permission
overlay 与 activity bar 同时活、暂停 spinner 反而破了 shimmer、自研 viewport scroll
和原生 scroll 打架、mouse escape 进 composer、diff 绑死 permission)。所有这些都不
是"修一个 bug"，而是 **Ink 周期性 live-region 重绘 vs 原生 terminal scrollback 的
机制级冲突**。

我们必须保留 shimmer / spinner / 实时 streaming 的交互体验，**不能**简单冻结
spinner 来交换 scrollback 稳定性。用户已多次确认这一点。

## 当前状态 / Current State

Aether 直接依赖 npm 上游 Ink 6.x (`tui/node_modules/ink/build/`)，配合
`log-update` 做 live region 重绘。每一轮 React 重渲染都走以下路径：

```
ink.js:render() → ink.js:onRender() → log-update.render(str)
  → stream.write(returnPrefix + ansiEscapes.eraseLines(previousLineCount) + str + cursorSuffix)
```

其中 `ansiEscapes.eraseLines(N)` 展开为：

```
('\x1b[2K\x1b[1A' × (N - 1)) + '\x1b[2K\x1b[G'
```

即每次重绘都让 cursor 上移 `previousLineCount - 1` 行，逐行擦除再重新打印。

### 真正的 root cause / Real Root Cause

> **不是 `clearTerminal`**。用户已确认 scrollback 没有被擦掉。
>
> 真正的元凶是上面那段 `eraseLines(previousLineCount)`：
>
> 1. 当 live region 高度 > terminal 可视行数 (例如 streaming 长 reasoning + 长
>    transcript + activity bar)，`previousLineCount` 自然超过 `rows`。
> 2. `\x1b[1A` 在 cursor 已经位于 viewport 顶行时，多数终端会让 viewport
>    跟随 cursor **向上滚** — 这就把用户已经拖到的滚动条位置一并拽走。
> 3. shimmer 每 150ms tick 一次 → 每 150ms 都重放这套 cursor-up 序列 →
>    用户的拖动永远停不下来。

旁路 root cause 是 `ink.js:322-336` 那个 fullscreen 分支：当 `lastOutputHeight >=
rows` 时它会走 `ansiEscapes.clearTerminal` (`\x1b[2J\x1b[3J\x1b[H`)，3J 会擦掉
scrollback。但用户本次报告的不是这个分支 (scrollback 仍在)。本 sprint 仍要识别并
回避这个分支被触发的条件。

### Ink 6 自带的 DEC 2026 包装 / Built-in Synchronized Output

Ink 6 已经在 `tui/node_modules/ink/build/write-synchronized.js` 里实现了 DEC 2026
BSU/ESU 包装，并在 `ink.js:149-156` 把 `bsu` / `log(output)` / `esu` 拆成 **3 次
独立 `stream.write()`**。这导致两个问题：

1. **检测过粗**：只用 `isTTY && !isInCi`，在不支持 DEC 2026 的 terminal 上仍照发
   BSU/ESU 字节，在 tmux 下还会被切碎成无效序列。
2. **非原子**：三次 syscall 之间 terminal 可以已经处理掉中间 frame，atomicity 被
   削弱。

参考实现 `open-claude-code/src/ink/terminal.ts:190-248` 是把 `BSU + diff + ESU`
合并成 **一次 write**，并用显式 allowlist (`isSynchronizedOutputSupported()`,
lines 70-118) 判断是否启用同步模式。

## Reference Model

我们调研了 `/workspace/open-claude-code/src/ink/`，它对同一问题的应对组合是：

| 技术 | 文件 | 作用 |
|---|---|---|
| Cell-level diff | `src/ink/log-update.ts` | 只在差异 cell 上写，避免 eraseLines 整块擦除 |
| BSU/ESU 单写 | `src/ink/terminal.ts:190-248` | 一次 syscall 写完整 frame，避免 partial paint |
| 终端 allowlist | `src/ink/terminal.ts:70-118` | 显式判断 DEC 2026 支持，tmux/老 terminal 不发字节 |
| Viewport-aware animation | `src/ink/hooks/use-terminal-viewport.ts` + `use-animation-frame.ts` | 滚出可视区域的 shimmer 暂停 tick |
| 共享时钟 | `src/ink/hooks/use-animation-frame.ts` | 多个 spinner 共用一个 interval，减少 rerender 风暴 |

### 是否直接整包替换 OCC 的 Ink? / Should we wholesale-replace Ink?

**结论：不替换**。证据：

| 维度 | 现状 |
|---|---|
| OCC ink 目录文件数 | 101 个 `.ts/.tsx` |
| 隐藏依赖 | `src/utils/{debug, sliceAnsi, intl, env, envUtils, semver}.js`、`src/bootstrap/state.js`、`src/native-ts/yoga-layout/index.js` |
| Runtime 依赖 | React Compiler runtime (Aether 没有引入) |
| yoga | 自带 native build (`src/native-ts/yoga-layout/`)，不是 npm `yoga-layout` |
| Aether 中 `from 'ink'` 的 import 站点 | 27 个文件 (`entry.tsx`、`app.tsx`、`components/*`、`overlays/*`、`slash/commands/model.tsx`、`__tests__/*`) |
| 风险 | 高：tests/snapshots 全部要重写、ChakraUI-like 二级组件不可用、未来 OCC 上游变动锁死 Aether |
| 工期 | 2-3 周纯接管，且没有 vendored Ink 的回归手术刀 |

替代方案：**借鉴 OCC 的几个关键技术，落地到独立的 Aether 模块，让 Ink 原样保留**。
本 sprint 的 5 个 PR 全部沿这条路走。

## Sprint Goals

1. **彻底解决 scrollbar yank**：在 DEC 2026 支持的 terminal 上，shimmer 持续 tick
   时拖动 native scrollbar 不被拽走。
2. **缩小 live region 高度**：让 `previousLineCount` 几乎不会超过 `rows`，让上面
   的 cursor-up overflow 物理上不再发生。
3. **不可见就不 tick**：滚出 viewport 的 ActivityBar / ReasoningLine 暂停动画，
   减少无用 rerender。
4. **shimmer cell 不走 Ink 重绘**：把仅仅做单 cell 颜色变化的 shimmer 用直写
   `cursorTo` 旁路，不触发 log-update。
5. **明确长期路线**：调研 Ink fork / vendor 方案，记录 trigger 条件和成本，但
   不在本 sprint 执行。

## Non-Goals

- 不替换 Ink 为 OCC 的 ink 整包 (理由见上)。
- 不重写 transcript 渲染、composer 布局、permission overlay、focus 仲裁。
- 不引入 alt-screen / mouse capture (历史失败模式 #1、#8)。
- 不动 `<Static>` 的 epoch / key 策略 (历史失败模式 #3、#4)。
- 不在 React 层引入 Compiler runtime (PR 5 调研的产物不入主线)。
- 不重做 `/scroll` 或自研 viewport scroll (历史失败模式 #7)。

## Roadmap

| # | 文档 | 内容 | 难度 | 依赖 |
|---|---|---|---|---|
| 1 | [`01_pr1_atomic_sync_stdout.md`](./01_pr1_atomic_sync_stdout.md) | 在 Ink 之外包一层 stdout proxy，单 write 发 BSU+diff+ESU，显式 allowlist 检测 DEC 2026 支持 | S | 无 |
| 2 | [`02_pr2_reduce_live_region.md`](./02_pr2_reduce_live_region.md) | 把 transcript 稳定段尽早入 `<Static>`，保留极小 stable tail，让 live region 高度 ≤ rows | M | 无 |
| 3 | [`03_pr3_viewport_aware_animation.md`](./03_pr3_viewport_aware_animation.md) | 移植 OCC 的 `useAnimationFrame` + `useTerminalViewport`，滚出可视区域时 spinner 暂停 | M | PR 1 |
| 4 | [`04_pr4_direct_write_shimmer.md`](./04_pr4_direct_write_shimmer.md) | 把 1×N cell 的 shimmer 用 `cursorTo` 直写，不再触发 React rerender | M | PR 1 |
| 5 | [`05_pr5_ink_fork_evaluation.md`](./05_pr5_ink_fork_evaluation.md) | 调研 vendor / fork ink 的成本与 trigger 条件，产出 decision doc | S | 无 |
| - | [`99_acceptance_matrix.md`](./99_acceptance_matrix.md) | 终端 × PR 的验收矩阵、回归清单、复测脚本 | S | 1–4 |

## Dependency Graph

```
PR 1 (atomic sync stdout)
  ├─→ PR 3 (viewport-aware animation)
  └─→ PR 4 (direct-write shimmer)

PR 2 (reduce live region)  — 独立，但和 PR 1 叠加效果最好

PR 5 (fork evaluation)     — 独立 research 文档，不入主代码路径

99 acceptance              — 跨 PR 1–4 验收
```

推荐合入顺序：

1. **PR 1** 先落，因为它是 "其他 PR 的安全网"：即使 PR 2-4 出现回归，atomic sync
   也能在主流 terminal 上消解 90% 的 yank。
2. **PR 2** 紧随，因为 `previousLineCount > rows` 是 cursor-up 上溢的物理前提，
   缩小 live region 能从源头削弱症状。
3. **PR 3** 在 PR 1 落定后接入，shared clock + viewport gating。
4. **PR 4** 最后接，因为它绕过 Ink 的部分；要在 PR 1 的 atomicity 保证之上做。
5. **PR 5** 与上面并行，是 research / decision doc。
6. **99** 跨所有 PR 跑一遍终端矩阵验收。

## Acceptance Summary

- 在 iTerm2 / WezTerm / Kitty / Ghostty / Alacritty / WindowsTerminal / GNOME
  Terminal (VTE ≥ 6800) / VS Code terminal 上，长 reasoning streaming 期间，用户
  拖动原生 scrollbar 到中间位置，**不被拽走**。
- shimmer 视觉效果保留 (与 master 对比无回归)。
- composer 不被固定到底部，banner 不重复 / 不消失，`<Static>` 行为不变。
- permission / approval overlay 弹出时，下方 activity / composer 仍按
  `app.tsx:87-93` 的现有规则隐藏，不与新机制冲突。
- 在 tmux / 老 macOS Terminal 上，BSU/ESU 字节不出现在 `script` 抓的输出里，TUI
  其余表现与 master 一致。
- `npm test` 全绿；新加测试覆盖 stdout proxy 与 viewport hooks。

## Historical Failure Modes To Avoid

引用 `tmp/drag-error/error.md`，本 sprint **不得**重蹈以下覆辙：

1. composer 固定底部 → alt-screen 类布局，输入框和 banner 中间出现大空白。
2. banner 一会儿在 fullscreen leading content，一会儿在 transcript/static
   scrollback，导致重复 / 消失 / 闪烁。
3. 把仍在变化的内容塞进 `<Static>` 导致重复打印 / 黑块。
4. 把刚发送的用户输入过早静态化，导致下一轮 streaming 时前面内容黑块。
5. permission/approval 下方还挂 spinner/composer → 滚动条仍被抢。
6. 降低 spinner 频率却破坏 shimmer 视觉。
7. 自己实现 viewport scroll，和原生 terminal scrollbar 打架。
8. 启用 terminal mouse capture / alt-screen 后 escape 进 composer。
9. diff 绑死 permission panel，approve 后 diff 跟着消失。

本 sprint 的 5 个 PR 都是 **additive**：不动 composer 位置、不动 `<Static>` 拆分
策略、不动 alt-screen / mouse mode、不动 permission overlay 渲染规则。

## Parity Gaps To Track

| Gap | Sprint decision |
|---|---|
| OCC 用 cell-diff log-update，Aether 用上游 log-update | 不替换：PR 4 仅在 shimmer cell 范围内做局部直写 |
| OCC 自带 yoga build，Aether 用 npm yoga | 不替换：依赖太深，PR 5 调研 trigger |
| OCC 用 React Compiler runtime，Aether 没引入 | 不引入：PR 3 hook 写成不依赖 Compiler 的版本 |
| OCC viewport hook 知道 alt-screen / multi-screen，Aether 单 screen | 简化：PR 3 只实现 single-screen viewport |
| OCC 有 `hasCursorUpViewportYankBug` for Windows | 暂缓：PR 5 记录，等 Windows Terminal 上验收时回看 |
