# Sprint 9 — Coding-UX Parity with Claude Code (Overview)

## 背景 / Motivation

Aether 的 TUI（`tui/`，基于 React + Ink）已经具备：

- 完整的权限请求流（`PermissionModal` + reverse-RPC `permission.request`）
- 着色 unified diff 渲染器（`tui/src/lib/diffView.tsx` + `tui/src/lib/diffRender.ts`）
- 工具调用聚合视图（`ExploredTree` + `ToolCallPanel`）
- 工具分类与动词表（`tui/src/lib/toolCategory.ts`）

但在 `/workspace/Aether/tmp/issues4/` 下的截图对比中，相对于 Claude Code（截图中的对照组）暴露出 **4 个具体差距**。这些差距都不是基础设施缺失，而是末端拼装的疏漏——数据被工具层算出来了，但没有沿链路输送到用户眼前。

### 截图导览 / Screenshot Tour

| 截图 | 展示内容 | 对比结论 |
|---|---|---|
| `image copy 9.png` | Claude Code 的 `write_file` 权限弹框：`Create file /tmp/calc.py` + `+73 · −0 · 1 hunk` + 完整 unified diff（绿色 `+` 行） | 这是目标形态 |
| `image copy 11.png` | Aether 的 `Overwrite file /workspace/Aether/tmp/calculator.py`：**原始 JSON dump**，含字面 `\n` 转义 | G1 fallback 被错误地命中 |
| `image copy 13.png` | Claude Code 的 shell 结果块：`▸ shell · $ command · ⏱ 0ms`，失败时跟 `[exit 1 · 39ms · stderr_lines=0]` 多行 | 我们没有这种 footer |
| `image copy 14.png` | Aether 的 `Run command $ python3 /tmp/calculator.py` 权限弹框：单行命令，OK | shell 单命令分支可以；问题在多命令分支 |
| `image copy 16.png` | Aether 的 `$ cd /workspace/Aether cp .env.example .env uv sync uv run aether`：四条命令被拼成一行 | G4：换行/链式符未拆分 |
| `image copy 18.png` | Claude Code 聊天里的 `● Edited tui/src/overlays/PermissionModal.tsx (+7 −15)` + 内联 diff | G2：我们批准后的 chat 只有一行 `⎿ Wrote …`，没有 `+N −M` 和内联 diff |

### Gap Matrix

| # | 现象 | 根因（带行号） |
|---|---|---|
| **G1** | `write_file` 覆写在权限弹框中显示原始 JSON 而非 diff | `write_file._build_diff()` 在 `old_content == new_content` 等边缘情况下返回 `""`，使 `preview.diff` 在 JS 中变成 falsy，逐级 fallback 到 `JSON.stringify(arguments)`（`tui/src/overlays/PermissionModal.tsx:248`） |
| **G2** | 批准后 chat 中没有 `● Edited X (+N −M)` 持久行，也没有内联 diff | `ChatItem.tool-call` 缺 `summary` 字段（`tui/src/store/chatStore.ts:9-24`）；`ExploredTree.tsx:36-53` 只渲染 `verb + detail` |
| **G3** | `ToolResult.metadata`（`change_count`、`bytes_before/after`、`exit_code`、`duration_ms`…）从未送出 wire | `ToolResult(AgentEventBase)`（`aether/gateway/protocol.py:216-222`）schema 缺 `metadata` 字段；`agent_methods.py:226-239` 显式只拷贝 `content`/`is_error` |
| **G4** | 多命令 shell 在权限弹框里挤成一行；结果 panel 没有 `[exit · ms]` footer | `PermissionModal.PreviewBody.command` 分支只 `split('\n')`（无链式符切分）；shell 结果的 exit/duration 卡在 G3 没出来 |

## 设计原则 / Design Principles

1. **不重造已有渲染器** — `DiffView` / `parseUnifiedDiff` / `formatStats` 全部复用，PR 9.2 的 EditSummary 只是新外壳。
2. **wire schema 只加不改** — `ToolResult.metadata` 走 additive 扩展（Pydantic 默认空 dict），旧客户端不受影响。
3. **fallback 必须可观察** — PR 9.3 在 JSON-args fallback 触发时打 telemetry，避免再次悄悄退化。
4. **shell 元数据走结构化字段，不走文本解析** — PR 9.4 把 `[exit X · Ums]` 从 content 头部移到 `result.metadata`，TUI 端不做字符串解析。
5. **保留与现有 Sprint 7 权限合约的兼容** — `ToolPermissionPreview` 形状不动；`PermissionPreview.diff` 仍是唯一权威字段。

## Sub-PR 路线图 / Roadmap

按依赖顺序：

```
PR 9.1 (wire metadata)  ──┬─→ PR 9.2 (chat edit summary)
                          └─→ PR 9.4 (shell footer)

PR 9.3 (modal fallback)  ── 独立，可与 9.1 并行
```

| PR | 标题 | 主要文件 | Gap |
|---|---|---|---|
| 9.1 | Forward `ToolResult.metadata` over the wire | `aether/gateway/protocol.py`, `agent_methods.py`, `tui/src/gatewayTypes.ts` | G3 |
| 9.2 | Chat-side edit summary + inline diff | `tui/src/components/EditSummary.tsx`(NEW), `ChatMessage.tsx`, `ExploredTree.tsx`, `chatStore.ts` | G2 |
| 9.3 | `write_file` permission modal fallback fix | `aether/tools/builtins/write_file.py`, `PermissionModal.tsx` | G1 |
| 9.4 | Shell command rendering parity | `PermissionModal.tsx`, `ShellResultFooter.tsx`(NEW), `aether/tools/builtins/shell.py` | G4 |

每个 PR 都对应本目录中一个独立文档，包含 **目标 / 当前问题 / 改动 / 测试** 四节。

## Out of Scope（延后到 Sprint 10+）

- **Notebook diff 渲染** — 内容模型不同（cell 数组而非文本），需要独立设计。
- **Diff 行内语法高亮** — 现有 `cli-highlight` 集成（`tui/src/lib/markdown.tsx`）理论上可以套到 addition/deletion 行，但属于 polish 而非 parity。
- **流式 diff 动画** — Claude Code 本身也不做，跳过。
- **多文件 batch diff** — 当前 `file_edit` / `write_file` 都是单文件，无须考虑跨文件汇总。

## 验收 / Acceptance（汇总见 `99_acceptance_matrix.md`）

- 端到端：5 个场景全部通过（覆写同内容文件、edit、链式 shell、失败 shell、新建文件）
- 单元测试：Python `pytest aether/tests/gateway aether/tests/tools` + TS `npm test` 全绿
- 类型检查：`uv run pyright` 与 `npm run typecheck` 零新增告警
- 遥测：完整跑一次冒烟后，gateway 日志中 `permission_preview_fallback` 事件计数为 0
