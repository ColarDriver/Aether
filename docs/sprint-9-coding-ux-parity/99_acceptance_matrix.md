# Sprint 9 — Acceptance Matrix

按场景 × PR 的二维矩阵；行是端到端测试用例，列是相关 PR，单元格记录该场景在该 PR 完成后应有的可见行为。

## 端到端 / E2E

| # | 场景 | PR 9.1 | PR 9.2 | PR 9.3 | PR 9.4 |
|---|---|---|---|---|---|
| **E1** | 覆写已有文件为完全相同内容 | metadata 携带 `no_op: True` | EditSummary 不出现（短路前） | **不弹权限框**，直接 no-op result | — |
| **E2** | 创建一个新的空文件 | metadata `bytes_after=0`, `lines_added=0` | EditSummary `● Wrote x (+0 −0)` | 弹框显示 `Create /x · size: 0 bytes (0 lines)` | — |
| **E3** | 覆写为完全不同的多行内容 | metadata `lines_added=N`, `lines_removed=M`, `diff=<text>` | EditSummary `● Wrote x (+N −M)`，展开内联 diff | 弹框彩色 unified diff（保持现状） | — |
| **E4** | `file_edit` 普通替换 | metadata `change_count`, `lines_added`, `lines_removed` | EditSummary `● Edited x (+N −M)`，展开内联 diff | 弹框 unified diff（保持现状） | — |
| **E5** | `file_edit` 替换不存在的字符串（失败） | `metadata.path` 可能仍在；`is_error=true` | EditSummary 不渲染；ExploredTree 留 `(failed)` | 弹框照常 | — |
| **E6** | 链式 shell `ls && pwd && date` | metadata `exit_code=0`, `duration_ms=...` | — | 弹框命令 trim().length 通过 | 弹框 3 行；ToolCallPanel footer `[exit 0 · Nms]` |
| **E7** | 失败 shell `ls /nope` | metadata `exit_code≠0`, `stderr_lines>0` | — | — | Footer 红色 `[exit 2 · Nms · stderr_lines=N]` |
| **E8** | 模型送出空格拼接多命令 `cd /a cp x y uv sync` | metadata 照常 | — | — | 弹框第一行末尾 ⚠ 标记 |
| **E9** | shell 命令被 spill 截断 | metadata `truncated=true` | — | — | Footer 含 `truncated` |
| **E10** | shell 命令超时 | metadata `timed_out=true` | — | — | Footer 含 `timed out` |
| **E11** | 模型连续 edit 3 个不同文件 | 每次都填 metadata | 3 条独立 `● Edited` 行；不被 Explored 吸收 | — | — |
| **E12** | 关键 fallback 触发（构造一个 preview 三字段都空的工具）| — | — | 弹框显示 synthetic body；gateway 收到 `permission_preview_fallback` telemetry 事件 | — |

## 单元测试 / Unit

| 文件 | 覆盖 PR |
|---|---|
| `aether/tests/gateway/test_tool_result_wire.py` (NEW) | 9.1 |
| `aether/tests/tools/test_file_tool_metadata.py` (NEW 或扩展现有) | 9.1 |
| `aether/tests/tools/test_file_permission_preview.py` (扩展) | 9.3 |
| `aether/tests/gateway/test_permission_preview_validator.py` (NEW) | 9.3 |
| `aether/tests/tools/test_shell_tool.py` (扩展) | 9.1 + 9.4 |
| `tui/src/__tests__/gatewayClient.test.ts` (扩展) | 9.1 |
| `tui/src/__tests__/chatStore.test.ts` (扩展) | 9.2 |
| `tui/src/__tests__/editSummary.test.tsx` (NEW) | 9.2 |
| `tui/src/__tests__/chatMessage.test.tsx` (扩展) | 9.2 |
| `tui/src/__tests__/chatTranscript.test.tsx` (扩展) | 9.2 |
| `tui/src/__tests__/permissionModal.test.tsx` (扩展) | 9.3 + 9.4 |
| `tui/src/__tests__/shellCommandPreview.test.tsx` (NEW) | 9.4 |
| `tui/src/__tests__/shellResultFooter.test.tsx` (NEW) | 9.4 |

## 类型检查 / Static

| 命令 | 期望 |
|---|---|
| `uv run pyright` | 无新增告警 |
| `npm run typecheck --prefix tui` | 无新增告警 |
| `uv run ruff check aether` | 通过 |
| `npm run lint --prefix tui` | 通过（如配置） |

## 性能 / Performance Sanity

| 检查 | 阈值 |
|---|---|
| `ToolResult.metadata` 序列化耗时 | 单次 < 1 ms（典型 metadata < 1 KiB） |
| `EditSummary` 折叠态首次渲染 | < 16 ms |
| `ShellCommandPreview` 渲染 100-行命令 | < 50 ms |
| `_safe_metadata` 处理非可序列化值 | O(N) keys，N ≤ 32 时不可观测 |

## 遥测 / Telemetry

| 事件 | 期望计数（一次完整冒烟）|
|---|---|
| `permission_preview_fallback` | 0 |
| `tool_result_metadata_drop` (PR 9.1 引入) | 仅在工具实现不规范时出现；冒烟时 0 |

## 回归 / Regression

冒烟跑完后，确保以下旧行为仍工作：

- `/clear` / `/new` / `/resume` / `/interrupt` 全部正常
- ApprovalModal（plan / questions 类型，不是 PermissionModal）行为不变
- `ExploredTree` 在仅有 read/list/search 操作时正常显示（不被本 sprint 影响）
- 旧 wire 客户端（如有）忽略 metadata 字段后照常工作

## 文档 / Docs

- `/workspace/Aether/docs/sprint-9-coding-ux-parity/` 全部 7 个文件齐备
- `README.md` 链接到所有子文档
- 每个子 PR 文档列出 critical files + 测试清单
- 此 acceptance matrix 在每个 PR 完成后被勾选更新
