# Sprint 11 — Acceptance Matrix

场景 × PR 二维矩阵。每行是一个端到端测试用例，每列是相关 PR；单元格记录该场景在该 PR 完成后应有的可见行为。

## 端到端 / E2E

| # | 场景 | 11.1 | 11.2 | 11.3 | 11.4 | 11.5 | 11.6 |
|---|---|---|---|---|---|---|---|
| **E1** | 在 Python 仓库 prompt "rename `count` to `total` in module X" | 模型主动 grep 引用并修；report 含 "I checked all callers" | — | — | — | — | — |
| **E2** | 完不成的 prompt（依赖缺失等） | 模型显式说 "无法验证因为 …"，不糊弄性"完成" | — | — | — | — | — |
| **E3** | engine 跑一个普通 `file_edit` 成功 | — | `post_tool_use` hook 触发一次；参数完整 | — | — | — | — |
| **E4** | engine 跑一个会抛 ToolNotFoundError 的工具 | — | `post_tool_use_failure` 触发一次 | — | — | — | — |
| **E5** | 用户写的 hook 实现自身抛错 | — | 主循环继续推进；日志含 `hook.post_tool_use_raised` | — | — | — | — |
| **E6** | mock LSP 在 edit 前后诊断变化 `[A] → [A, B]` | — | — | `tracker.get_new_diagnostics([p])` 返回仅 `B` | — | — | — |
| **E7** | LSP 永久 hang | — | — | `before_file_edited` 在 settle_timeout_ms 后退化为 `()` baseline，不抛 | — | — | — |
| **E8** | 没装 LSP server 的环境 | — | — | `DiagnosticTracker(None).enabled == False`；所有 API no-op | — | — | — |
| **E9** | `file_edit` 成功执行 | — | — | — | `result.metadata["edited_paths"] == [abs_path]`；`tracker.before_file_edited` 被调用 | — | — |
| **E10** | engine 内部 `DiagnosticDispatchHook` 监听到 edited_paths | — | — | — | `tracker.notify_file_changed(p, content)` 被调度（fire-and-forget） | — | — |
| **E11** | turn 0 edit 引入 NameError → turn 1 LLM 请求 | — | — | — | — | messages 末尾含 user-role `<diagnostics>` 含 NameError | — |
| **E12** | turn 1 模型没修 → turn 2 | — | — | — | — | messages 不再追加同条诊断（已 delivered） | — |
| **E13** | turn 1 模型修了 → turn 2 | — | — | — | — | 既无新诊断也无旧诊断 | — |
| **E14** | `DiagnosticTracker(None)` | — | — | — | — | messages 中永远不含 `<diagnostics>` | — |
| **E15** | 模型尝试 `task(subagent_type="Verifier", ...)` | — | — | — | — | — | child registry 不含 `file_edit` / `write_file` / `notebook_edit` / `task` / `send_message` |
| **E16** | session 累计编辑 3 个文件 | — | — | — | — | — | 第 4 个 turn 出现 `<system-reminder>` 提示 spawn Verifier |
| **E17** | spawn 一次 Verifier 后 reminder 消失 | — | — | — | — | — | `_verifier_invoked` 置位 True；后续 turn 不再 nudge |
| **E18** | `verifier_gate_enabled=False` | — | — | — | — | — | 编辑 10 个文件也不出现 reminder |
| **E19** | 子 agent 在 worktree 里编辑（Sprint 10.8） | — | — | tracker 只看 worktree 内 path | — | child engine 的 messages 注入自己的 diagnostics；父 engine 不被污染 | — |
| **E20** | 父开 verification directive，子未 override | — | — | — | — | — | child engine 第一次请求 system 含 `<verification_directive>` |
| **E21** | 顺序：tool → middleware.after_tool → user post_tool_use → internal diagnostic hook | — | hook 看到 middleware 处理过的 ToolResult | — | DiagnosticDispatchHook 在 user hook 之后跑 | `<diagnostics>` 出现在 *下一轮* PRE_LLM 而非本轮 | — |
| **E22** | 性能：edit → tracker round-trip | — | — | mock 立即响应：≤ 5ms | mock 立即响应：≤ 20ms（含 file read） | — | — |

## 单元测试 / Unit

| 文件 | 覆盖 PR |
|---|---|
| `aether/tests/agents/core/test_system_prompt.py`（扩展） | 11.1 |
| `aether/tests/agents/test_engine_system_prompt_wiring.py`（NEW） | 11.1 |
| `aether/tests/agents/test_post_tool_use_hook.py`（NEW） | 11.2 |
| `aether/tests/runtime/diagnostics/test_tracker.py`（NEW） | 11.3 |
| `aether/tests/runtime/resources/test_lsp_manager_high_level.py`（NEW） | 11.3 |
| `aether/tests/tools/builtins/test_file_edit_diagnostic_wire.py`（NEW） | 11.4 |
| `aether/tests/agents/test_internal_diagnostic_hook.py`（NEW） | 11.4 |
| `aether/tests/integration/test_edit_to_diagnostic_pipeline.py`（NEW） | 11.4 |
| `aether/tests/runtime/diagnostics/test_attachments.py`（NEW） | 11.5 |
| `aether/tests/runtime/attachments/test_dispatcher.py`（NEW） | 11.5 |
| `aether/tests/agents/test_diagnostic_attachment_pipeline.py`（NEW） | 11.5 |
| `aether/tests/agents/types/test_verifier_definition.py`（NEW） | 11.6 |
| `aether/tests/agents/test_verifier_gate_reminder.py`（NEW） | 11.6 |
| `aether/tests/integration/test_verifier_end_to_end.py`（NEW） | 11.6 |

## 类型 / Type Checking

- `uv run pyright`：零新增告警，全 sprint 各 PR 单独检查。
- `npm run typecheck --prefix tui`：仅 PR 11.5 因 `phantomTool.ts` 正则扩展可能触动；零新增告警。

## 遥测 / Telemetry

PR 11.2 落地后增加结构化 log 事件：

| 事件名 | 触发点 | 字段 |
|---|---|---|
| `hook.post_tool_use_raised` | 用户 hook 抛错 | `tool_name`, `tool_call_id` |
| `hook.post_tool_use_failure_raised` | 用户 failure hook 抛错 | 同上 |
| `internal_hook.diagnostic_dispatch` | 内置诊断 hook 抛错 | `tool_name`, `edited_paths_count` |

PR 11.3 落地后：

| 事件名 | 触发点 | 字段 |
|---|---|---|
| `diagnostic.baseline_captured` | `before_file_edited` 成功 | `path`, `baseline_size` |
| `diagnostic.fetch_timeout` | `_fetch_blocking` 在 settle_timeout 内拿不到 | `path`, `timeout_ms` |
| `diagnostic.new_delivered` | `get_new_diagnostics` 返回非空 | `path`, `new_count` |

PR 11.5 落地后：

| 事件名 | 触发点 | 字段 |
|---|---|---|
| `attachment.diagnostics_attached` | dispatcher 返回非空诊断 attachment | `paths`, `total_diagnostics` |
| `attachment.skill_nudge_attached` | skill_nudge producer 命中 | `iteration` |

PR 11.6 落地后：

| 事件名 | 触发点 | 字段 |
|---|---|---|
| `verifier.gate_reminder_sent` | reminder attachment 注入 | `edited_files_count`, `threshold` |
| `verifier.invoked` | `task(subagent_type="Verifier")` 触发 | `parent_session_id`, `child_session_id` |
| `verifier.gate_bypassed` | session 结束且 `edited_files_count >= threshold` 但 `_verifier_invoked is False` | `edited_files_count` |

## 手测脚本 / Manual

### M1（PR 11.1 之后即可跑）

```bash
cd /workspace/Aether
uv run aether
# Prompt:
> 把 aether/runtime/tools/skill_catalog.py 里的 _format_entry 重命名为 _render_listing_entry
```

期望：agent 改完后 *主动* `grep -n "_format_entry" aether/` 并修补所有调用点，最后报告含 `pyright` 或 `python -c` 输出。

### M2（PR 11.5 之后跑）

```bash
cd /workspace/Aether
uv run aether
# Prompt:
> 把 aether/runtime/tools/skill_listing.py 中的变量 listing 重命名为 rendered，并故意把其中一处保留旧名
```

期望：agent edit 完后下一轮自动看到 `<diagnostics>` 含 NameError 并修复，无需用户手动指出。

### M3（PR 11.6 之后跑）

```bash
cd /workspace/Aether
uv run aether
# Prompt:
> 在 aether/subagents/default_builder.py / aether/tools/builtins/agent_tool.py / aether/agents/core/agent.py 这三处把 inherit_tools 行为统一改为 deny-list
```

期望：agent 改完后 spawn `task(subagent_type="Verifier", ...)`，verifier 子 agent 跑 `pyright` + `pytest` 并以 PASS / FAIL / PARTIAL 之一返回；父 agent 据此决定最终 report。
