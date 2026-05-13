# TUI Split — Acceptance Matrix

## PR 验收矩阵

| PR | 必须通过的专项测试 | 必须确认的回归点 |
|---|---|---|
| 1 Gateway skeleton + Transport | `test_transport.py`、`test_entry_boot.py`：peer-gone 静默、并发写、SIGPIPE 不秒杀、shutdown grace、crash log 落盘 | 不引入新 runtime 依赖；`aether-gateway` 干净启动与退出 |
| 2 Protocol + dispatcher | `test_protocol_roundtrip.py`、`test_dispatcher.py`、`test_long_method_pool.py`：错误码分类、`extra=forbid`、long 走线程池、handler 异常不退出进程 | `gateway.ping` 联通、批量请求不支持的语义被锁死、contextvar 在 worker 内可见 |
| 3 Sessions/prefs/providers/commands | `test_session_methods.py`、`test_prefs_methods.py`、`test_providers_methods.py`、`test_commands_methods.py`：CRUD round-trip、未知 id 错误、catalog 长度与 REGISTRY 一致 | `aether/cli/sessions.py` 等源文件语义不变；handler 不 import prompt_toolkit / rich |
| 4 Run-loop streaming | `test_agent_run_streaming.py`、`test_agent_cancel.py`：事件序列、cancel `exit_reason`、并发 run 拒绝、session/run id 全携带 | engine 测试套件不回归；`StreamSilentCallback` 语义不变 |
| 5 Approval / permission bridge | `test_reverse_rpc.py`、`test_prompter_bridge.py`、`test_permission_bridge.py`：accept/reject/timeout/disconnect、未知 id、重复 response | sprint-7 的 `test_tool_permissions.py` 与 `test_tool_permission_gate.py` 继续通过；engine 调 prompter 路径无 patch |
| 6 TS scaffold + GatewayClient | `gatewayClient.test.ts`：启动、解析、超时、断流、并发 pending、stderr 捕获、subscribe 回放、Python 解析器优先级 | `npm run type-check`、`npm run lint`、`npm test` 全绿；TTY guard 非 TTY 退出码 2 |
| 7 TUI chat + slash | `chatStore.test.tsx`、`slashDispatcher.test.ts`、`markdownLite.test.ts`、`composer.test.tsx`：streaming append、slash 路由、composer 编辑、cancel 路径 | `gateway.ping` 联通；mock provider 单 turn 端到端可见 |
| 8 TUI approvals + activity + picker | `overlayFocus.test.tsx`、`approvalModal.test.tsx`、`permissionModal.test.tsx`、`sessionPicker.test.tsx`、`activityBar.test.tsx`、`diffRender.test.ts` | sprint-7 acceptance manual checklist 在新 TUI 上重新跑通；ESC 优先级链无穿透 |
| 9 Cutover | 全 backend 套件 + `aether-tui` 全 TS 套件；新装环境 `aether` 入口启动 TS TUI | 没有 `prompt_toolkit` 导入残留；纯 UI 文件全部删除；git history 用 rename 保留 |

## 用户可见完成标准

| 场景 | 预期 |
|---|---|
| `aether` 启动 | Banner 出现，光标在输入框，状态条 idle |
| 单 turn 聊天 | 文本逐 token 流出，状态条 thinking → responding → idle |
| 工具调用 | tool.call panel 默认折叠，Tab+Enter 展开 |
| Plan 模式退出 | approval overlay 弹出，Yes/No 决定继续与否，UI 不打印 JSON |
| 模型请求 file_edit | permission overlay 显示 unified diff，三选项可用 |
| 模型请求 shell | permission overlay 显示 command，session-allow + prefix 可设 |
| 长 turn 中按 Ctrl+C | 进程不被秒杀，server 收到 `agent.cancel`，UI 状态回 idle |
| `/new` | 新 session_id，UI 顶部摘要更新 |
| `/resume` | session picker 弹出，选条恢复 transcript |
| `/model` | prefs.set + next turn 用新 model |
| `/exit` | TS 进程退出，Python gateway 在 2s 内退出，无 zombie |
| 非 TTY 启动 | exit code 2，提示需要交互终端 |

## 推荐测试命令

```bash
# Python 后端
pytest aether/tests/gateway -v
pytest aether/tests/agents aether/tests/runtime aether/tests/engine aether/tests/tools

# TS 前端
(cd aether-tui && npm test && npm run type-check && npm run lint)

# 端到端联通（PR 6+）
(cd aether-tui && AETHER_PYTHON_SRC_ROOT="$PWD/.." npm start)

# 兼容性
pytest aether/tests/sessions aether/tests/providers
```

## 手测脚本

```text
完整 sprint 走查（PR 9 之后）：

1. 全新 venv 安装 aether，运行 `aether`，看 banner。
2. 发 hello，确认流式渲染。
3. 输入 /help，确认命令清单。
4. 让模型 read 一个文件，看 tool 折叠面板。
5. 让模型 edit 文件，看 permission overlay + diff，按 N 拒绝，确认文件未变化。
6. 让模型再次 edit，按 Y 允许 once，文件变化；再次 edit 仍弹 overlay。
7. 让模型 edit 第三次，按 S 设 session rule，下一次同范围 edit 走 hinted default。
8. 让模型 shell touch 一个文件，按 N，确认未创建。
9. 输入 /interrupt 中断当前生成。
10. 输入 /new 开新 session。
11. 输入 /resume 选回原 session，确认 transcript 重现。
12. Ctrl+C 在空闲状态 → 确认退出。
13. 再次 `aether`，验证启动稳定无 stderr 残留。
```

## Sprint 完成定义

- 9 个 PR 全部合并到 master。
- TS TUI 在功能面与原 Python TUI 对齐（聊天、工具、approval、permission、session、slash）。
- Python 端只剩 launcher，纯 UI 代码全部删除，`prompt_toolkit` 不在 runtime deps。
- `aether/gateway/` 包通过 `Transport` 抽象为未来 web UI 留好接口。
- 文档 `docs/tui-split/` 12 个文件齐全，可作为 sprint 完成存档。

## 与 Python TUI 已知偏差

后续 parity 工作 (`/workspace/Aether/tui/src/components/Banner.tsx` 等)
落地之后，TS TUI 几乎与 Python prompt_toolkit TUI 一致，唯一保留的
架构偏差：

| 偏差 | 原因 |
|---|---|
| 无固定式 streaming 预览帧 | Python 用 Rich Live 在 composer 上方 pin 一个 tail-crop 的 assistant 缓冲区；Ink 自上而下的 reconciliation 没有等价的 "pinned below transcript" 容器。改成把流式 delta 直接追加到 transcript，用 `streaming: true` 标记区分。功能等价，视觉不同。 |
| Reasoning 摘要不在 activity bar 下方 | Python 把 `<thinking>` 内容渲染在 activity bar 下方；TS 用独立的 `ReasoningLine` 组件以保留焦点管理简单性。 |
| 中文/CJK 字符的 shimmer 索引按 code point | Python 同样按 code point；视觉宽度近似但不对齐，对 ASCII 动词无影响。 |

## 新增的 CLI 启动器旗标

PR 9 的 launcher 文档列了 6 个旗标。Parity 工作把全部 14 个 Python
`aether/cli/main.py` 旗标都接通了，通过 `aether.cli.launcher` 模块转译
成 env vars 交给 TS subprocess。完整对照表：

| Python 旗标 | 对应 env var | TS 消费点 |
|---|---|---|
| `--provider X` | `AETHER_PROVIDER` | `sessionStore.provider` 初始值 |
| `--model X` | `AETHER_MODEL` | `sessionStore.model` 初始值 |
| `--api-key X` | `AETHER_API_KEY` | (gateway 端 provider 构造) |
| `--base-url X` | `AETHER_BASE_URL` | `sessionStore.baseUrl` → `session.create.base_url` |
| `--system X` | `AETHER_SYSTEM` | `sessionStore.systemOverride` |
| `--system-file PATH` | `AETHER_SYSTEM_FILE` | 启动时 `readFileSync` → systemOverride |
| `--session ID` | `AETHER_SESSION_ID` | 启动后自动 `session.resume` |
| `--resume [ID]` | `AETHER_RESUME` | 启动后 picker 或 `session.resume` |
| `--max-iterations N` | `AETHER_MAX_ITERATIONS` | `agent.run.max_iterations` 透传 |
| `--temperature T` | `AETHER_TEMPERATURE` | `agent.run.temperature` 透传 |
| `--max-tokens M` | `AETHER_MAX_TOKENS` | `agent.run.max_tokens` 透传 |
| `--verbose` | `AETHER_VERBOSE` | 启动时 `chatActions.toggleVerbose` |
| `--no-banner` | `AETHER_NO_BANNER` | `Banner` 走 `BootLine` 路径 |
| `--no-builtin-tools` | `AETHER_NO_BUILTIN_TOOLS` | `agent.run.disable_builtin_tools` 透传 |
| `--log-level X` | `AETHER_LOG_LEVEL` | gateway env，影响 Python logging |
