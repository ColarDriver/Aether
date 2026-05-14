# Sprint 10 — Acceptance Matrix

场景 × PR 二维矩阵。每行是一个端到端测试用例，每列是相关 PR；单元格记录该场景在该 PR 完成后应有的可见行为。

## 端到端 / E2E

| # | 场景 | 10.1 | 10.2 | 10.3 | 10.4 | 10.5 | 10.6 | 10.7 | 10.8 |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | 模型在不知 skill 名时被问 "use the TDD skill" → 自发调 `skill(skill="test-driven-development")` | turn 0 system 含 `<system-reminder>` 列出 skill | — | — | — | — | — | — | — |
| **E2** | 长会话 50 轮后模型重新看见 skill 列表 | `skill_nudge_interval=10` 时第 10/20/... 轮前 messages 含 nudge | — | — | — | — | — | — | — |
| **E3** | `task(subagent_type="Explore", prompt="find usages of ToolRegistry")` | — | descriptor.enum 含 `Explore`；未注册 type → error | child registry 不含 `write_file/file_edit/shell` | — | — | — | — | — |
| **E4** | 项目级 `.claude/agents/code-reviewer.md` 加载 | — | registry.get("code-reviewer") 命中；source=="project" | child 用 markdown body 作 system_prompt；tools/model 应用 | — | — | — | — | — |
| **E5** | model override：`task(subagent_type="Explore", model="haiku")` | — | — | child `ModelCallConfig.model == claude-haiku-4-5-20251001` | — | — | — | — | — |
| **E6** | `task(prompt="...", run_in_background=true)` → 立即返回 task_id | — | — | — | TaskStore 中 RUNNING 记录 | tool 返回 status="async_launched" 且耗时 < 100ms | — | — | — |
| **E7** | `task_output(task_id, block=true)` 阻塞直到 child 完成 | — | — | — | result.json 写入 | finalize 把 status→COMPLETED | 阻塞返回 summary | — | — |
| **E8** | `task_output(task_id, block=false)` 立即返回 partial 状态 | — | — | — | output.log 流式追加 | — | content 含 progress + tail | — | — |
| **E9** | 杀掉 gateway → 重启 → task 自动 KILLED | — | — | — | mark_orphaned heartbeat>60s | — | block=false 立即返回 status="killed" | — | — |
| **E10** | `send_message(to=task_id, message="...")` → child 下个 iteration 看到 | — | — | — | pending.jsonl 多一行 | drain hook 在 PRE_LLM 执行 | — | engine PRE_LLM 注入 user turn | — |
| **E11** | async child 完成 → 父 PRE_LLM 收到 `<task-notification>` | — | — | — | — | _on_async_done 触发 | — | manager 写入 parent 队列；父 drain | — |
| **E12** | `isolation=worktree` 时 child CWD 在新 worktree；干净退出后自动清理 | — | — | — | TaskRecord.worktree_path 写入 | — | — | — | git worktree list 出/入一行 |
| **E13** | 嵌套 async：父 async A → A 启 async B；B 完成通知冒泡到 A | — | — | — | 两条 RUNNING → 两条 COMPLETED 记录 | depth=2 不抛限制 | — | A 的 pending.jsonl 含 notification | — |
| **E14** | 自定义 type 的 `definition.background=True` 强制 async | — | registry.get(name).background==True | builder 不检查 run_in_background，强制走 async | — | force-async branch 命中 | — | — | — |
| **E15** | `Explore` child 尝试 `write_file` → tool not found error 而非 silent edit | — | — | child registry 过滤生效 | — | — | — | — | — |
| **E16** | `send_message` 到 COMPLETED task → 报错且不入队 | — | — | — | — | — | — | tool returns is_error=True | — |
| **E17** | `task_output` 大输出 spill 到文件 | — | — | — | — | — | maybe_spill_for_tool 触发 | — | — |
| **E18** | dirty worktree 自动保留供用户审阅 | — | — | — | record.worktree_path 不为 None | — | task_output content 含 worktree path | — | cleanup report reason=="dirty" |

## 单元测试 / Unit

| 文件 | 覆盖 PR |
|---|---|
| `aether/tests/runtime/tools/test_skill_listing.py` (NEW) | 10.1 |
| `aether/tests/agents/test_skill_listing_injection.py` (NEW) | 10.1 |
| `aether/tests/agents/test_skill_nudge_interval.py` (NEW) | 10.1 |
| `aether/tests/agents/types/test_definition.py` (NEW) | 10.2 |
| `aether/tests/agents/types/test_markdown_loader.py` (NEW) | 10.2 |
| `aether/tests/agents/types/test_registry.py` (NEW) | 10.2 |
| `aether/tests/tools/test_agent_tool_type_resolution.py` (NEW) | 10.2 |
| `aether/tests/tools/test_filter.py` (NEW) | 10.3 |
| `aether/tests/subagents/test_default_builder_typed.py` (NEW) | 10.3 |
| `aether/tests/subagents/test_subagent_type_e2e.py` (NEW) | 10.3 |
| `aether/tests/runtime/tasks/test_store.py` (NEW) | 10.4 |
| `aether/tests/runtime/tasks/test_recovery.py` (NEW) | 10.4 |
| `aether/tests/subagents/test_async_lifecycle.py` (NEW) | 10.5 |
| `aether/tests/tools/test_agent_tool_async.py` (NEW) | 10.5 |
| `aether/tests/tools/test_task_output.py` (NEW) | 10.6 |
| `aether/tests/tools/test_task_output_e2e.py` (NEW) | 10.5 + 10.6 |
| `aether/tests/tools/test_send_message.py` (NEW) | 10.7 |
| `aether/tests/agents/test_pending_message_drain.py` (NEW) | 10.7 |
| `aether/tests/subagents/test_task_notification.py` (NEW) | 10.7 |
| `aether/tests/runtime/tasks/test_worktree.py` (NEW) | 10.8 |
| `aether/tests/subagents/test_worktree_subagent.py` (NEW) | 10.8 |

## 类型检查 / Static

| 命令 | 期望 |
|---|---|
| `uv run pyright` | 无新增告警 |
| `uv run ruff check aether` | 通过 |
| `npm run typecheck --prefix tui` | 无新增告警（仅事件类型扩展） |

## 性能 / Performance Sanity

| 检查 | 阈值 |
|---|---|
| `format_skill_listing(catalog, budget=4096)` | < 1 ms（typical catalog ≤ 30 skills） |
| `AgentTypeRegistry.discover()` | < 50 ms（typical .claude/agents/ ≤ 20 files） |
| `TaskStore.create()` | < 5 ms（atomic JSON write） |
| `TaskStore.append_output()` | < 1 ms per call（buffered append） |
| `run_task_async()` 返回耗时 | < 50 ms（即便 child 跑分钟级） |
| `TaskOutput(block=True, timeout=200ms)` 抖动 | < 400 ms |
| `cleanup_worktree(clean)` | < 500 ms（含 `git worktree remove`） |

## 遥测 / Telemetry

| 事件 | 期望计数（一次完整冒烟）|
|---|---|
| `task.launched` | ≥ 1 |
| `task.progress` | 与子 agent iteration 数相等 |
| `task.notification` | 等于完成的 async 任务数 |
| `subagent_type_unknown` (新增 telemetry，记录 unknown type) | 0（除非测试主动构造） |
| `worktree_cleanup_kept` | 等于"dirty 退出"的 async 任务数 |

## 回归 / Regression

冒烟跑完后，确保以下旧行为仍工作：

- **同步 `task()` 路径**（不传 `run_in_background`）行为与本 sprint 之前完全一致 —— 同步阻塞、返回 SubagentResult 摘要。
- `task_stop` 仍能终止正在跑的 async / sync 任务。
- `skill` 工具的错误路径（unknown name）仍列出 first 30 个 skill 名（PR 10.1 不破坏此行为）。
- `<tool_use_contract>` 块仍正常注入，工具列表与 child 实际拥有的 registry 一致。
- 旧 wire 客户端（不解析 `task.progress` / `task.notification`）继续工作（事件 union 是 additive）。
- TUI 端代码无变更需求 —— `tui/` 目录在本 sprint 内除 `gatewayTypes.ts` 加几行 type definition 外不动。

## 文档 / Docs

- `/workspace/Aether/docs/sprint-10-subagent-and-skill-parity/` 全部 11 个文件齐备：
  - `README.md`
  - `00_overview.md`
  - `01_pr10_1_skill_listing_injection.md`
  - `02_pr10_2_agent_type_registry.md`
  - `03_pr10_3_per_type_child_config.md`
  - `04_pr10_4_on_disk_task_store.md`
  - `05_pr10_5_async_subagent_lifecycle.md`
  - `06_pr10_6_task_output_tool.md`
  - `07_pr10_7_send_message_and_notifications.md`
  - `08_pr10_8_worktree_isolation.md`
  - `99_acceptance_matrix.md`
- `README.md` 链接到所有子文档。
- 每个子 PR 文档列出 critical files + 测试清单。
- 此 acceptance matrix 在每个 PR 完成后被勾选更新。

## 提交清单 / Final Checklist

合入最后一个 PR 前请确认：

- [ ] 所有 E1-E18 端到端场景手测通过
- [ ] `uv run pytest aether/tests/` 全绿
- [ ] `uv run pyright` 零新告警
- [ ] `npm run typecheck --prefix tui` 零新告警
- [ ] `~/.aether/tasks/` 目录布局与 PR 10.4 描述一致
- [ ] `git worktree list` 在 worktree 任务完成后干净（dirty 任务保留属预期）
- [ ] 至少一份 `.claude/agents/<name>.md` 示例落库（如 `code-reviewer.md`）
- [ ] 旧同步 subagent 行为零回归
