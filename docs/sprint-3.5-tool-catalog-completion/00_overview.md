# 00 — Sprint 3.5 总览

## 一、动机与目标

### 1.1 当前的不一致

PR 3.2 在 `EngineConfig.cheap_tool_names` 默认值里写了 7 个工具名，
其中 4 个（`todo_write`、`memory*`、`skill_manage`、`session_search`）在 Aether 里**根本不存在**。
这意味着 PR 3.2 的 cheap-tool refund 机制对默认配置形同虚设：

```python
# config/schema.py（PR 3.2 落地的默认值）
cheap_tool_names: tuple[str, ...] = (
    "update_todo",
    "todo_write",       # ← 工具不存在
    "memory",
    "memory_write",
    "memory_read",
    "skill_manage",     # ← 工具不存在
    "session_search",   # ← 工具不存在
)
```

**修复路径**：Sprint 3.5 落地 `TodoWriteTool`（PR 3.5.3），
让 `update_todo` / `todo_write` 这条 cheap_tool 路径真正可用。`memory_*` / `skill_manage` /
`session_search` 留给后续 sprint（涉及 memory/skill/session 子系统设计）。

### 1.2 与 claude-code 的工具集差距

claude-code 默认开启 17 个工具（不算 ant-only / feature-flagged 实验性条目）。
Aether 当前只有 6 个，差距如下：

| 类别 | claude-code | Aether 现状 | Sprint 3.5 后 |
|---|---|---|---|
| Shell | `BashTool` | `shell` ✓ | ✓ |
| 文件读 | `FileReadTool` | `read_file` ✓ | ✓（spill 升级） |
| 文件写（覆盖） | `FileWriteTool` | `write_file` ✓ | ✓（spill 升级） |
| 文件编辑（局部） | `FileEditTool` | — | ✅ +PR 3.5.2 |
| Notebook 单元格 | `NotebookEditTool` | — | ✅ +PR 3.5.4 |
| Glob | `GlobTool` | `glob` ✓ | ✓（spill 升级） |
| Grep | `GrepTool` | `grep` ✓ | ✓（spill 升级） |
| 目录列表 | （走 Bash） | `list_dir` | ✓ Aether 独有 |
| Web 抓取 | `WebFetchTool` | — | ✅ +PR 3.5.5 |
| Web 搜索 | `WebSearchTool` | — | ✅ +PR 3.5.5 |
| 浏览器 | `WebBrowserTool` | — | ✅ +PR 3.5.10 |
| 任务清单 | `TodoWriteTool` | — | ✅ +PR 3.5.3 |
| 询问用户 | `AskUserQuestionTool` | — | ✅ +PR 3.5.7 |
| 进入计划 | `EnterPlanModeTool` | — | ✅ +PR 3.5.7 |
| 退出计划 | `ExitPlanModeV2Tool` | — | ✅ +PR 3.5.7 |
| Subagent 派发 | `AgentTool` | （SubagentManager 未暴露） | ✅ +PR 3.5.6 |
| Subagent 输出 | `TaskOutputTool` | — | ✅ +PR 3.5.6 |
| Subagent 停止 | `TaskStopTool` | — | ✅ +PR 3.5.6 |
| Skill 调用 | `SkillTool` | — | ✅ +PR 3.5.8 |
| LSP 查询 | `LSPTool` | — | ✅ +PR 3.5.9 |

**剔除范围**（claude-code 产品特定基础设施，不适配 Aether）：

`RemoteTriggerTool` / `ScheduleCronTool` / `PushNotificationTool` / `MonitorTool` /
`SendUserFileTool` / `SubscribePRTool` / `TeamCreateTool` / `SendMessageTool` / `BriefTool` /
`CtxInspectTool` / `SuggestBackgroundPRTool` / `OverflowTestTool` / `TungstenTool` /
`ConfigTool` / `ReviewArtifactTool` / `REPLTool` / `VerifyPlanExecutionTool` /
`ToolSearchTool` / `ListPeersTool` / MCP-related tools。

### 1.3 同步合并 Tier 1 持久化的理由

如果先做工具补齐再做 spill，每个新工具都要被改两次（一次实现、一次加 spill）。
**做事更划算的顺序**：先把 spill 基础设施落地（PR 3.5.1），
后续每个新工具一开始就长在 spill 之上。

## 二、Sprint 范围与依赖图

```
                                    ┌─────────────┐
                                    │  PR 3.5.1   │ ← spill 基础设施
                                    │ Foundation  │   (runtime/tool_result_storage.py)
                                    │  + 6 升级   │   (6 现有工具加 spill 段)
                                    └──┬────────┬─┘
                                       │        │
                  ┌────────────────────┼────────┼────────────────────┐
                  ▼                    ▼        ▼                    ▼
            ┌──────────┐         ┌──────────┐         ┌──────────────────┐
            │ PR 3.5.2 │         │ PR 3.5.3 │         │     PR 3.5.4     │
            │ FileEdit │         │ TodoWrite│         │ NotebookEdit     │
            │  (+spill)│         │ (closes  │         │   (+spill)       │
            │          │         │  3.2 loop)│        │                  │
            └──────────┘         └──────────┘         └──────────────────┘
                  │                    │                       │
                  └────────────────────┼───────────────────────┘
                                       ▼
                            ┌─────────────────────┐
                            │      PR 3.5.5       │
                            │ Web Fetch + Search  │
                            │      (+spill)       │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │      PR 3.5.6       │
                            │  Subagent triplet   │
                            │  (Agent + Output    │
                            │   + Stop)           │
                            │ 复用 SubagentManager │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │      PR 3.5.7       │
                            │ Interaction triplet │
                            │ (PlanMode +         │
                            │  AskUserQuestion)   │
                            │  CLI 模式扩展        │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │      PR 3.5.8       │
                            │  Skill catalog +    │
                            │     SkillTool       │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │      PR 3.5.9       │
                            │ LSP server 集成     │
                            │     LSPTool         │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │     PR 3.5.10       │
                            │  Playwright 集成 +  │
                            │   WebBrowserTool    │
                            └─────────────────────┘
```

**关键依赖说明**：

* PR 3.5.2-4 之间无依赖，可**并行开发**（合并时仍然按顺序）。
* PR 3.5.5（Web 工具）依赖 3.5.1 的 spill；与 3.5.6+ 无依赖。
* PR 3.5.6 假设 `SubagentManager` 已经能跑（已有），只是新增工具入口。
* PR 3.5.7 依赖 CLI 已有的 prompt_toolkit 交互层；不需要新基础设施。
* PR 3.5.8 是**第一个引入新子系统**的 PR（skill catalog），独立设计。
* PR 3.5.9-10 引入外部进程依赖（LSP server / Playwright），可能需要 optional dep 配置。

## 三、统一设计约定（所有工具共享）

### 3.1 spill 阈值矩阵

| 工具类别 | `MAX_RESULT_CHARS` | 触发条件 |
|---|---|---|
| `shell` | 40_000 | stdout + stderr 拼接后超阈值 |
| `read_file` | 60_000 | 文件内容超阈值，且**不**是已 spilled 文件（防递归） |
| `write_file` | — | 输出短，不 spill |
| `list_dir` | 20_000 | 极少触发，巨型目录会 |
| `grep` | 30_000 | 行边界截断（不在半行处） |
| `glob` | 20_000 | 路径列表，几千条触发 |
| `FileEditTool` | — | 输出是 diff 摘要，不 spill |
| `NotebookEditTool` | 60_000 | 与 read_file 一致（可能产生整 notebook 输出） |
| `TodoWriteTool` | — | 输出是 ack message，不 spill |
| `WebFetchTool` | 80_000 | 网页 markdown 通常较大；超阈值 spill |
| `WebSearchTool` | 30_000 | 搜索结果 JSON |
| `AgentTool` | 60_000 | Subagent 完整对话历史可能很大 |
| `TaskOutputTool` | 40_000 | 取一段已经在跑的任务输出 |
| `TaskStopTool` | — | 状态确认信息，不 spill |
| `AskUserQuestionTool` | — | 用户回答通常很短 |
| `EnterPlanModeTool` | — | 状态切换确认，不 spill |
| `ExitPlanModeTool` | — | 同上 |
| `SkillTool` | 60_000 | skill 输出可能是文档 |
| `LSPTool` | 40_000 | 符号定义/引用列表 |
| `WebBrowserTool` | 80_000 | 同 WebFetch |

### 3.2 共用 spill 模板

所有"会输出大内容"的工具遵循同一模板：

```python
from aether.runtime.tool_result_storage import (
    spill_to_disk,
    build_truncation_notice,
)

class FooTool(ToolExecutor):
    MAX_RESULT_CHARS = 40_000

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        full_output = self._do_work(...)

        config = self._resolve_config(context)
        if (
            config.tool_result_spill_enabled
            and len(full_output) > self.MAX_RESULT_CHARS
        ):
            preview = full_output[: self.MAX_RESULT_CHARS]
            try:
                receipt = spill_to_disk(
                    full_output,
                    session_id=context.session_id,
                    call_id=call.id,
                    extension="txt",
                    config_dir=config.tool_result_spill_dir,
                    preview_chars=len(preview),
                )
                content = preview + build_truncation_notice(
                    receipt,
                    full_lines=full_output.count("\n") + 1,
                )
                context.metadata["tier1_spilled_count"] = (
                    int(context.metadata.get("tier1_spilled_count", 0)) + 1
                )
            except OSError as exc:
                content = preview + (
                    f"\n\n... [output truncated: {len(full_output)} chars total, "
                    f"could not spill to disk: {exc}] ..."
                )
        else:
            content = full_output

        return ToolResult(...)
```

### 3.3 EngineConfig 新字段（PR 3.5.1 落地）

```python
# Sprint 3.5 / PR 3.5.1
tool_result_spill_enabled: bool = True
tool_result_spill_dir: Path | None = None  # None = ~/.aether/tool_results
```

### 3.4 TurnContext 注入 EngineConfig 引用

PR 3.5.1 在 `_prepare_turn_entry` 中把 `EngineConfig` 写入 `context.metadata["_engine_config"]`，
所有工具通过 `context.metadata.get("_engine_config")` 读取。
（不直接给 `TurnContext` 加字段是为了避免破坏现有 dataclass 形状 — 已有大量代码 `TurnContext(session_id, iteration)` 构造。）

## 四、错误预期与回滚开关

| 风险 | 触发条件 | 缓解 / 回滚 |
|---|---|---|
| `~/.aether/tool_results` 占用越来越多磁盘 | 长期使用 | `cleanup_session_spills` 默认 7 天清理；可在 `/clear` 时调用 |
| Subagent 工具与现有 SubagentManager 行为分裂 | PR 3.5.6 实现错误 | 复用现有 `SubagentManager.run_tasks` 入口，不重新实现派发 |
| LSP server 启动失败导致 LSPTool 不可用 | PR 3.5.9 缺失 LSP binary | 工具返回明确错误"LSP not available"，不 crash |
| Playwright 初始化失败 | PR 3.5.10 缺失浏览器 | 工具返回错误，建议用 WebFetch 作为降级 |
| FileEditTool 的 search/replace 误匹配多处 | PR 3.5.2 实现错误 | 强制 unique-match 校验（claude-code 同款），匹配多处时报错让模型重写更具体的 old_string |
| TodoWriteTool 状态污染 turn 间 | PR 3.5.3 错误使用全局状态 | 状态绑定到 session_id，由 SessionStore 管理 |
| Web 工具被误用做 SSRF | PR 3.5.5/3.5.10 | URL 黑名单（loopback/private IP）+ 默认白名单选项 |
| AskUserQuestion 在非交互场景 hang | PR 3.5.7 | 检测 stdin TTY，非交互时返回错误"non-interactive mode" |

每个 PR 都会带自己的回滚开关（`EngineConfig` 加一个 `*_enabled` 字段），紧急情况下可以单工具关闭。

## 五、验收门（sprint 整体）

参见 [`99_acceptance_matrix.md`](99_acceptance_matrix.md) — 跨 PR 的端到端验收清单。

每个 PR 自身的验收门写在各自文档的"验收门"小节。
