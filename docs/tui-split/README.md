# TUI Split — Python gateway + TypeScript Ink frontend

本目录把 Aether 的 TUI 拆分计划拆成 9 个可独立 review 的 PR。目标是把现在 `aether/cli/`（Python + `prompt_toolkit` + `rich`）的 TUI 替换成 TypeScript Ink 前端，同时把 agent engine、memory、tools、providers、sessions 全部保留在 Python 后端，并为未来 web UI 预留同一套 gateway。

## 建议阅读顺序

1. `00_overview.md` — 背景、目标架构、PR 依赖图、术语表
2. `01_pr1_gateway_skeleton.md` — `aether/gateway/` 骨架 + `Transport` 协议 + stdio 实现
3. `02_pr2_protocol_and_dispatcher.md` — JSON-RPC 协议、事件 schema、`@method` 注册表、线程池
4. `03_pr3_backend_methods_sessions_config.md` — sessions / prefs / providers / slash catalog 的 RPC 包装
5. `04_pr4_run_loop_streaming.md` — `AgentEngine.run_loop()` 的 RPC 封装与事件流
6. `05_pr5_approval_permission_bridge.md` — approvals / tool permissions 的 server-initiated RPC
7. `06_pr6_tui_scaffold_gateway_client.md` — `aether-tui/` Ink 工程脚手架 + `GatewayClient`
8. `07_pr7_tui_chat_and_slash.md` — 聊天主视图、流式 markdown、slash 命令分发
9. `08_pr8_tui_approvals_activity.md` — 确认 overlay、permission overlay、activity bar、tool 面板、session picker
10. `09_pr9_cutover_remove_python_tui.md` — 切换入口并删除 Python TUI

验收矩阵见 `99_acceptance_matrix.md`。

## PR 状态

| PR | 标题 | 状态 |
|---|---|---|
| PR 1 | Gateway skeleton + Transport | 待实施 |
| PR 2 | JSON-RPC protocol + dispatcher | 待实施 |
| PR 3 | Backend methods (sessions/prefs/providers/commands) | 待实施 |
| PR 4 | Run-loop streaming | 待实施 |
| PR 5 | Approval / permission bridge | 待实施 |
| PR 6 | TS scaffold + GatewayClient | 待实施 |
| PR 7 | TUI chat + slash commands | 待实施 |
| PR 8 | TUI approvals + activity bar | 待实施 |
| PR 9 | Single-shot cutover | 待实施 |

## 参考实现

- Hermes (`/workspace/hermes-agent`) — TS Ink TUI + Python gateway via JSON-RPC over stdio。
- open-claude-code (`/workspace/open-claude-code`) — 纯 TS Ink，无 Python，可借鉴的部分是 Ink 组件结构、NDJSON 帧、transport 抽象。
