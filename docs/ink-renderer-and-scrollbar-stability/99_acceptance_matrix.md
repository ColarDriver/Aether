# 99 — Acceptance Matrix & Regression Checklist

> 跨 PR 1-4 的端到端验收清单。每次合入后用本 doc 跑一遍 → 把表格填出来。
> 99 这一档只跑 manual end-to-end，单测在各自 PR 内已覆盖。

## 验收原则 / Acceptance Principles

1. **症状级验收**：本 sprint 的存在意义是 "用户拖原生 scrollbar 不被拽走"。
   所有自动化测试都不能完全证明这一点 — 必须真人手测。
2. **多终端矩阵**：同一行为在不同 terminal 上表现不一致是本问题的本质。
   每个 PR 必须在至少 4 个 terminal 上验收。
3. **回归对照**：每个手测同时跑 `master` 和当前 PR，区分 "本 PR 改善了"
   vs "本来就这样"。
4. **历史失败模式不复发**：跑完 acceptance 后必须 cross-check `error.md`
   的 9 个 failure modes。

## Terminal × PR 矩阵 / Terminal × PR Matrix

每格填 ✅ / ❌ / ⚠ (有限改善 / 仍有 yank 但频率降低)。

| Terminal | DEC 2026? | PR1 only | PR1+2 | PR1+2+3 | PR1-4 (direct-write opt-in) | Master baseline |
|---|---|---|---|---|---|---|
| iTerm2 (TERM_PROGRAM=iTerm.app) | ✅ | | | | | |
| WezTerm | ✅ | | | | | |
| Kitty (TERM=xterm-kitty) | ✅ | | | | | |
| Ghostty (TERM_PROGRAM=ghostty 或 TERM=xterm-ghostty) | ✅ | | | | | |
| Alacritty | ✅ | | | | | |
| WindowsTerminal (WT_SESSION) | ✅ | | | | | |
| GNOME Terminal (VTE_VERSION ≥ 6800) | ✅ | | | | | |
| GNOME Terminal (VTE_VERSION < 6800) | ❌ | | | | | |
| VS Code integrated terminal (TERM_PROGRAM=vscode) | ✅ | | | | | |
| Foot (TERM=foot*) | ✅ | | | | | |
| Zed (ZED_TERM) | ✅ | | | | | |
| Warp (TERM_PROGRAM=WarpTerminal) | ✅ | | | | | |
| Contour | ✅ | | | | | |
| 老 macOS Terminal.app | ❌ | | | | | |
| tmux outer (其内运行 Aether) | ❌ (proxy 主动关) | | | | | |
| screen | ❌ | | | | | |
| xterm 默认 | ❌ | | | | | |

填表说明：
- `PR1 only` = atomic-sync proxy 单独效果
- `PR1+2` = + 缩小 live region
- `PR1+2+3` = + viewport-aware animation
- `PR1-4` = + direct-write shimmer (`AETHER_SHIMMER_DIRECT_WRITE=1`)
- ✅ = 拖动 scrollbar 时永不被拽走
- ⚠ = 拖动可暂停，但松手后偶有几次 yank
- ❌ = 拖动直接被拽回底部

## 手测脚本 / Manual Test Script

每个 terminal 在每个 PR stage 跑以下步骤：

### Setup

1. `git checkout <branch>`
2. `npm --prefix tui run build`
3. 在目标 terminal 中 `cd /workspace/Aether && node tui/dist/entry.js`
4. 等 banner 出现，TTY 确认可用。

### Scenario A: Long Reasoning Streaming + Scrollbar Drag

1. 发送提问：`"please give me a detailed 2000-word essay about the history of bash"`
2. spinner 开始动；reasoning 区域开始出文字。
3. 在 spinner tick 期间，**用鼠标拖原生 scrollbar 到中间**。
4. 保持鼠标按住 5 秒。
5. 观察：scrollbar 是否被拽走 / 是否抖动 / 视觉位置是否稳定。
6. 松开鼠标，观察是否回弹。
7. 滚到底部，观察 transcript 渲染是否回归正常。

**期望**：PR 1-4 合入后，supported terminal 上 5 秒内 scrollbar 完全不动；
unsupported terminal 上至少不再被持续拽走 (一次拖动后能稳住)。

### Scenario B: Tool Call Streaming + Scrollbar Drag

1. 发送：`"run \`find /workspace -name '*.ts' -type f | head -200\`"`
2. tool-call panel 出现，shimmer 在那里 tick。
3. tool 输出大量行，scrollback 累计。
4. 拖 scrollbar 到中间，保持 5 秒。
5. 观察同 Scenario A。

### Scenario C: Permission Modal Up + Scrollbar Drag

1. 发送一个会触发 permission 弹窗的请求 (例如写文件)。
2. permission modal 弹出。
3. 此时 `app.tsx:87-93` 应隐藏下方 activity bar。
4. 拖 scrollbar 到上方，**观察是否 yank**。

**期望**：permission modal 下 ActivityBar 已隐藏，shimmer 不 tick → scrollbar
完全稳定 (与 master 一致)。这条主要是 "不引入新回归" 的对照。

### Scenario D: Idle State Scroll

1. 不发请求，TUI 处于 idle (status = idle, spinner 静止)。
2. 拖 scrollbar 上下。

**期望**：完全稳定。本 sprint 不应当影响 idle 状态行为。

### Scenario E: tmux Inside

1. `tmux new -s aether-test`，在 tmux session 内运行 Aether。
2. 重复 Scenario A。
3. 用 `script -q /tmp/aether-out.log -c 'node tui/dist/entry.js'` 抓 stdout。
4. `cat -v /tmp/aether-out.log | grep -c '\[?2026'`：**应为 0** (PR 1 在
   tmux 下主动关闭 BSU/ESU 发送)。

### Scenario F: Resize During Streaming

1. 发起长任务。
2. 在 spinner tick 期间 resize terminal 窗口 (拖窗口边缘)。
3. 观察：
   - shimmer 是否在新尺寸下重新对齐
   - scrollbar 是否稳定
   - transcript 是否 reflow 正确

**期望**：PR 3 的 useTerminalViewport 在 resize 后正确读取新 rows；PR 4 的
direct-write shimmer 重新测量坐标。

## 历史失败模式回归 / Historical Failure Mode Regression Check

参考 `tmp/drag-error/error.md`。每项必须明确 "未复发"：

| # | 失败模式 | 复测方法 | 状态 |
|---|---|---|---|
| 1 | composer 被固定到底部，banner 与 input 之间出现空白 | 启动 TUI，观察 input 位置 / banner / 留白 | |
| 2 | banner 重复 / 消失 / 闪烁 | 发 3 轮提问，每轮 banner 在 scrollback 出现且不重复 | |
| 3 | `<Static>` 中出现重复打印 / 黑块 | 滚动多次后回到底部，观察 scrollback 中无重复 | |
| 4 | 刚发送的 user input 在下轮 streaming 时显示为黑块 | 发提问，下一轮开始时上一句 user input 仍可见 | |
| 5 | permission/approval 下方 spinner/composer 仍活 | 弹出 permission 时观察下方为空 | |
| 6 | 试图降低 spinner 频率破坏 shimmer | 视觉对比 master 的 shimmer | |
| 7 | 自研 viewport scroll 和原生 scrollbar 打架 | 应用内 `↑/↓` 不影响原生 scrollbar | |
| 8 | mouse / 箭头 escape 进入 composer | 拖滚动条 / 按方向键 / 滚轮，composer 文本无乱码 | |
| 9 | diff 绑死 permission，approve 后 diff 消失 | approve permission 后 diff summary 仍在 scrollback | |

## 单测清单 / Unit Test Inventory

| PR | 测试文件 | 覆盖点 |
|---|---|---|
| 1 | `tui/src/lib/__tests__/atomicSyncStdout.test.ts` | BSU/ESU coalescing, drop on unsupported, proxy properties, resize forward |
| 2 | `tui/src/__tests__/splitStaticPrefix.test.ts` | 行数预算、最少 1 个 item、staticScrollback off |
| 3 | `tui/src/lib/__tests__/animationClock.test.ts` | 订阅计数、空集关闭 |
| 3 | `tui/src/lib/__tests__/useAnimationFrame.test.tsx` | intervalMs=null 暂停、visibility=false 暂停 |
| 4 | `tui/src/lib/__tests__/shimmerWriter.test.ts` | frame 内容含 cursor save/restore、不启用时返回 null |
| 5 | (无 — 纯 docs) | — |

跑：

```bash
npm --prefix tui test -- --run
npm --prefix tui run build
```

两条命令必须全绿，没有新 lint / TS error。

## 性能基线 / Performance Baseline

| 指标 | master | PR 1-4 | 目标 |
|---|---|---|---|
| 长流式期间 stdout write 次数 (5s 窗口) | 测 | 测 | ≤ master 的 1.2× (PR 4 启用后应 < master) |
| ActivityBar setInterval 数量 | 1 (固定) | 0 (共享时钟) | 0 |
| ReasoningLine setInterval 数量 | 1 (固定) | 0 (共享时钟) | 0 |
| `previousLineCount` 峰值 (live region) | 测 | 测 | ≤ stdout.rows |

测量方法：临时 patch `process.stdout.write` 计数；不需要进生产。

## 决策签字 / Decision Sign-off

| 项 | 状态 |
|---|---|
| PR 1 合入 | |
| PR 2 合入 | |
| PR 3 合入 | |
| PR 4 合入 (默认 opt-out) | |
| PR 5 evaluation doc 完成 | |
| 终端矩阵跑完一遍 (上表填齐) | |
| 历史失败模式 9 项全部确认未复发 | |
| Sprint 完成 | |

## 未解决问题 / Open Issues

下面这些已知但本 sprint 不处理，作为 follow-up 候选：

- Windows conhost 的 cursor-up viewport yank bug — OCC 用
  `hasCursorUpViewportYankBug()` 单独走 fullscreen 路径，Aether 暂未实现。
  trigger: 用户报告 Windows Terminal / WSL 仍 yank。
- `<Static>` reflow 时 banner 短暂消失 — 历史 #2 的边角，PR 2 未直接处理；
  如果再现，先看 `chatEpoch` 何时跳变。
- ScrollBox 支持 — 当前 Aether 没用 ScrollBox，PR 3 的 hook 已留口子；如未来
  接入需补 `scrollTop` 校验。
- XTVERSION 异步探测 (OCC 用法) — 走 SSH 时 TERM_PROGRAM 可能不传过来；本
  sprint 不处理。
