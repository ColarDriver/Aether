# PR 5 — Ink Fork Evaluation (Research / Decision Document)

> 本 PR **不入主线代码**，只产出一份 decision doc。任务是评估 "把 Ink fork 或
> vendor 进来" 的成本、收益、trigger 条件，作为本 sprint 的最终归档。

## 目标 / Goal

回答两个问题：

1. **本 sprint 结束后 (PR 1-4 合入) 是否还需要 fork / vendor Ink？**
2. **如果未来要做，什么样的信号应当触发我们启动 fork 工作？**

输出物：

- `docs/ink-renderer-and-scrollbar-stability/notes/fork-evaluation.md`
- 决策表格、信号清单、迁移路径分阶段拆解
- 不修改任何 `tui/` 下的代码

## 背景 / Motivation

历次试错之后我们已知：

- Aether 用上游 npm Ink 6.x，其 `log-update` / `write-synchronized` 在
  特定场景下与原生 scrollback 有结构性冲突。
- `open-claude-code` 的解决思路是 vendored ink (`src/ink/`，101 个文件，
  cell-diff log-update、viewport-aware hooks、自带 yoga build、与 React
  Compiler 强耦合)。
- 整包替换会把 Aether 卡在 OCC 上游变动节奏上，且要重写 27 个 `from 'ink'`
  import 站点的 props 边界 (主要是 OCC 加了不在上游 Ink 的属性，例如自定义
  `<Box ref>` 行为)。

PR 1-4 用了 **"向 Ink 周围加适配层"** 的策略。本 doc 评估这个策略的天花板。

## 调研对象 / Scope

调研三种长期路径，各自的工作量、收益、风险：

### A. 维持现状 — Ink 上游 + Aether-side 适配 (PR 1-4 的路径)

| 维度 | 评估 |
|---|---|
| 工作量 | 已完成 (PR 1-4) |
| 收益 | 90% yank case 在主流 terminal 上消除；opt-in 模式可覆盖剩余 10% |
| 风险 | Ink 上游变 `log-update` / `write-synchronized` API 时要重测我们的 proxy 严格相等比较 |
| 维护成本 | 低：每次 ink 升级跑 `tui/src/lib/__tests__/atomicSyncStdout.test.ts` 验证 |
| 退路 | 直接 unset env / revert PR 4；其他 PR 都是单测覆盖的纯 TS 代码 |

### B. Vendor Ink — 把上游 Ink 6 拷进 `tui/src/vendor/ink/`

| 维度 | 评估 |
|---|---|
| 工作量 | 1-2 天：复制源码、替换 import 路径、跑测试 |
| 收益 | 可以直接改 `log-update.js` (例如 cell-diff)，不再 wrap proxy |
| 风险 | 失去 ink 升级路径；维护责任落到 Aether；diff 与 upstream 难追 |
| 维护成本 | 中：每季度 rebase upstream，三方依赖 (cli-cursor / ansi-escapes / yoga) 还得跟 |
| 退路 | 删除 vendor 目录，恢复 npm dependency |

trigger 条件 (满足任一即启动 B)：

- PR 1-4 合入后，**仍有终端在压测中 yank**，且不在 OCC 的 allowlist 内
  (说明 OCC 自己也没处理)。
- 我们要做的事 (例如 message inline edit、inline mouse selection、固定
  composer 在底部但保留 scrollback) 必须改 ink internals 才能实现。
- ink 上游停止维护或出现明显 bug 我们没法 patch。

### C. Fork Ink — 类似 OCC 的做法，但只 fork 我们必需的部分

| 维度 | 评估 |
|---|---|
| 工作量 | 2-3 周：log-update cell-diff、viewport hooks、清掉 OCC 的 React Compiler 依赖 |
| 收益 | 上限最高：cell-diff 后大多数 rerender 不会写 stdout；scrollbar 永远不被抢 |
| 风险 | 高：要自己 fuzz 多终端兼容；npm 包名/版本管理；测试矩阵爆炸 |
| 维护成本 | 高：拥有自己的 ink 等于拥有自己的 React renderer |
| 退路 | 痛苦：所有 `from 'ink'` import 已改成 `from '@aether/ink'`，回退要全 sed |

trigger 条件 (满足全部即启动 C)：

- B (vendor) 已经做了，并且 vendored 代码累计 patch 已经 > 500 行 diff vs
  upstream。
- 我们需要 **多个** 仅靠 wrap proxy 做不到的特性 (cell-diff、自定义 scroll、
  mouse selection、inline-edit cursor)。
- Aether 已有专门的 TUI 团队 (现在没有)。
- 团队评审同意承担额外维护成本。

### D. 替换为 OCC ink (放弃 Aether 自治)

| 维度 | 评估 |
|---|---|
| 工作量 | 2-3 周纯接管 + 持续追 OCC 主干 |
| 收益 | 立刻拿到 OCC 所有 ink 能力 |
| 风险 | 极高：被 OCC 的 React Compiler 假设、yoga 自带 build、bootstrap state、debug、intl 等绑死；Aether 失去差异化空间 |
| 维护成本 | 同步 OCC = 每次 upstream 改 internals 我们都要跟 |
| 退路 | 难：要全部回退到 upstream ink，事实上等于再做一次 D |

**结论：D 排除。** 即便 OCC 的能力诱人，绑定到一个不为我们设计的第三方
内部模块的代价高于自己 fork。

## 当前推荐决策 / Recommendation

**短期 (本 sprint)**：A，即合入 PR 1-4，不做 vendor / fork。

**中期 (下一季度)**：保持 A，监测 trigger 信号；如果 PR 1-4 在 1-2 个具体
terminal 上仍有 yank 且 OCC allowlist 没覆盖，准备 B。

**长期 (一年以上)**：仅当 trigger C 的条件全部满足时启动 C；否则停留在 A 或 B。

## 触发信号清单 / Trigger Signals

记录这些信号，未来 maintainer 看到就知道该升级路径：

| 信号 | 升级到 | 备注 |
|---|---|---|
| PR 1-4 合入后某终端仍 yank | B (vendor 改 log-update) | 先在 vendor 改一个 hot patch 验证可行性 |
| 需要 inline edit / mouse selection | B 或 C | 看改动复杂度 |
| `tui/src/lib/atomicSyncStdout.ts` 的严格相等比较失败 (上游改了 wrap 行为) | B (vendor 锁定 ink 版本) | 临时回退措施 |
| Ink 上游 6 个月未发布且有 bug | B | 主动接管 |
| Aether TUI 团队人数 ≥ 2 名专职 | 可以考虑 C | 但仍要严格评审 |

## 调研产物 / Deliverables

`docs/ink-renderer-and-scrollbar-stability/notes/fork-evaluation.md` 应包含：

1. **现状基线**：截至本 doc 完成日 (`2026-05-19`)，PR 1-4 在以下 terminal 验收
   通过 / 失败：
   - iTerm2: ?
   - WezTerm: ?
   - Kitty: ?
   - Ghostty: ?
   - Alacritty: ?
   - WindowsTerminal: ?
   - GNOME Terminal (VTE ≥ 6800): ?
   - VS Code integrated terminal: ?
   - tmux 内运行 (作为 outer container): ?
   - 老 macOS Terminal.app: ?
2. **OCC ink internals 文件清单 / 依赖图** — 拷下来供以后 trigger 时参考。
3. **fork 路径的最小可行步骤** — 如果未来真要做，按这个 checklist 走。
4. **决策签字** — 谁批准了 "现在不 fork"。

## 不修改的内容 / Out of Scope

- 不动任何代码。
- 不写 vendor 目录骨架 — 真要做时再写。
- 不引入 `@aether/ink` 之类的别名 — 同上。

## 文件清单 / File Manifest

- NEW `docs/ink-renderer-and-scrollbar-stability/notes/fork-evaluation.md`
  (本 PR 的可交付产物)
- NEW `docs/ink-renderer-and-scrollbar-stability/notes/.gitkeep`

## 验收 / Acceptance

1. `fork-evaluation.md` 写完，包含上面四块内容。
2. PR 1-4 终端验收矩阵 (99 acceptance) 填好，作为本 doc 的基线。
3. team review 在 PR 描述里明确签字 "短期不 fork"。
4. 无代码变更，PR 只动 docs。
