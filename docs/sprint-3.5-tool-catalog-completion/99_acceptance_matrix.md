# 99 — Sprint 3.5 跨 PR 验收矩阵

> sprint 整体完成的判定标准。逐 PR 验收门见各自文档；本文是跨 PR 的端到端清单。

## 一、量化验收

### 1.1 工具计数

| 来源 | sprint 前 | sprint 后 |
|---|---|---|
| Aether builtins（全部） | 6 | **20** |
| 与 claude-code 对齐的 default tool | 6 / 17 (35%) | **17 / 17 (100%)** |
| 共享 spill 能力 | 0 | **15** |
| 不需要 spill 的轻量工具（write_file / TodoWrite / PlanMode 等） | — | 5 |

### 1.2 测试计数

| PR | unit 测试 case | 其他 |
|---|---|---|
| 3.5.1 | 30+ | spill 基础 |
| 3.5.2 | 25+ | FileEdit |
| 3.5.3 | 16+ | TodoWrite + cheap_tool 集成 |
| 3.5.4 | 25+ | NotebookEdit + nbformat 校验 |
| 3.5.5 | 30+ | Web 双工具 + SSRF |
| 3.5.6 | 15+ | Subagent triplet + manager 升级 |
| 3.5.7 | 35+ | Plan/Ask + mode 强制 |
| 3.5.8 | 22+ | Skill catalog + tool |
| 3.5.9 | 25+ | LSP（unit）+ 选配集成 |
| 3.5.10 | 13+ | Browser（unit）+ 选配真实 |
| **合计** | **236+** | |

每个 PR 自己的 case 全绿 + 既有测试零回归。

### 1.3 配置增项

| 字段 | 默认 | 引入 PR |
|---|---|---|
| `tool_result_spill_enabled` | True | 3.5.1 |
| `tool_result_spill_dir` | None | 3.5.1 |
| `web_fetch_enabled` | True | 3.5.5 |
| `web_search_enabled` | True | 3.5.5 |
| `web_search_api_key` | None | 3.5.5 |
| `web_fetch_max_download_bytes` | 5 MiB | 3.5.5 |
| `web_fetch_timeout_seconds` | 30 | 3.5.5 |
| `plan_mode_enabled` | True | 3.5.7 |
| `ask_user_question_enabled` | True | 3.5.7 |
| `ask_user_question_timeout_seconds` | 600 | 3.5.7 |
| `skill_tool_enabled` | True | 3.5.8 |
| `skill_search_paths` | () | 3.5.8 |
| `skill_list_in_system_prompt` | False | 3.5.8 |
| `lsp_tool_enabled` | True | 3.5.9 |
| `lsp_request_timeout_seconds` | 15 | 3.5.9 |
| `lsp_initialization_timeout_seconds` | 10 | 3.5.9 |
| `web_browser_enabled` | **False** | 3.5.10 |
| `web_browser_idle_timeout_seconds` | 30 | 3.5.10 |
| `web_browser_navigation_timeout_seconds` | 30 | 3.5.10 |
| `allow_subagent_dispatch`（如需） | True | 3.5.6 |

合计 **19 个**新配置字段。

## 二、跨 PR 集成场景验收

### 2.1 闭环场景 A：spill + LLM 自洽读回

1. 用户：`shell: find / -type f 2>/dev/null | head -100000` （大量输出）
2. ShellTool spill 到 `~/.aether/tool_results/<sid>/<cid>.txt`
3. ToolResult preview 含 "use read_file to retrieve"
4. 模型自动调用 `read_file <spilled-path>` 拿完整结果
5. ReadFile 检测到 spilled 路径，**不**再 spill
6. **验证**：`result.metadata["compaction"]["tier1_spilled_count"] == 1`

### 2.2 闭环场景 B：cheap_tool refund 真实生效

1. 一轮 LLM 调用只用 `todo_write`
2. AgentEngine 检测到所有 tool 都是 cheap → 调 `IterationBudget.refund()`
3. **验证**：`result.metadata["iteration_budget"]["refund_count"] >= 1`
4. **验证**：`result.metadata["iteration_budget"]["used"] < consume_count`

### 2.3 闭环场景 C：Plan mode 强制

1. 模型 `enter_plan_mode` → `session_mode=plan`
2. 模型尝试 `shell: rm -rf /tmp/x` → 工具被拒
3. 模型 `read_file` / `grep` 探索（允许）
4. 模型 `exit_plan_mode(plan="...")` → CLI 弹 prompt → 用户 approve
5. 模型恢复执行 `shell: ...`（允许）
6. **验证**：被拒的工具调用都返回 `is_error=True` + 明确提示

### 2.4 闭环场景 D：Subagent 派发 + Tier 1 spill

1. 主 agent `task(prompt="读这 30 个文件，找出含 print 的")`
2. SubagentManager 起 child agent
3. child 调 `read_file` × 30；每个 ReadFile 内部可能 spill
4. child 返回 summary
5. AgentTool 把 summary 包成 ToolResult；可能再 spill
6. 主 agent 拿到精简结果继续推理
7. **验证**：spill 计数累加合理；child + parent metadata 都有 `tier1_spilled_count`

### 2.5 闭环场景 E：完整工具能力联展（手动 smoke）

跑这个 prompt：
> "读 backend/harness/aether/cli/main.py 的前 50 行，理解一下结构，
> 然后用 grep 找所有 'def execute' 的函数定义，
> 然后写一个 todo list 把要 review 的 5 个函数列出来，
> 用 web_search 搜 'python click decorator' 看看有没有 best practice，
> 最后写一段总结"

应该看到：
* `read_file` ✓
* `grep` ✓
* `todo_write` ✓ → cheap-tool refund 触发
* `web_search` ✓（如有 API key）
* `result.metadata` 含 `compaction.tier1_spilled_count` / `iteration_budget` 信息
* 没有 crash；没有"unknown tool"错误

## 三、性能 / 资源验收

| 指标 | 目标 | 测量方式 |
|---|---|---|
| 6 个现有工具升级后单次 tool_call latency | +< 5ms | benchmark |
| spill 文件 7 天后清理 | 自动 | 手动 mtime 测试 |
| LSP server 进程数 | ≤ 5（语言数） | `ps aux` |
| 浏览器 idle 30s 后被关闭 | 是 | `ps aux` 检测 |
| `~/.aether/tool_results` 单 session 占用 | < 100 MB（典型用例） | 跑 5 轮估算 |
| sprint 加包大小 | + ~50MB（不含 Playwright） | `pip show` |

## 四、文档验收

* [ ] `docs/sprint-3.5-tool-catalog-completion/README.md` 列出全部 10 PR + 进度链接
* [ ] 每个 PR 文档（01-10）章节齐全：目标 / 设计 / 改动清单 / 测试 / 验收门 / 风险
* [ ] `00_overview.md` 联调图反映实际依赖
* [ ] `99_acceptance_matrix.md`（本文）覆盖跨 PR 场景
* [ ] `README.md`（项目级，如有）提到新工具集
* [ ] CHANGELOG（如有）记录 sprint 3.5 变更

## 五、回滚路径

每个 PR 的回滚开关都被验证可用（`*_enabled=False`）：

| 工具/能力 | 关闭开关 | 验证 |
|---|---|---|
| spill | `tool_result_spill_enabled=False` | 工具回到 plain truncation |
| FileEdit | 不注册 | 模型收到 "unknown tool" |
| TodoWrite | 不注册 | 同上；cheap_tool refund 回到 dangling 状态 |
| NotebookEdit | 不注册 | 同上 |
| Web tools | `web_*_enabled=False` | 工具返回明确 disabled 错误 |
| Subagent | `allow_subagent_dispatch=False` | 工具返回 disabled 错误 |
| Plan/Ask | `plan_mode_enabled=False` etc | 同上 |
| Skill | `skill_tool_enabled=False` | 同上 |
| LSP | `lsp_tool_enabled=False` | 同上 |
| Browser | `web_browser_enabled=False` (默认) | 默认就是 disabled |

## 六、Sprint 完成确认

✅ Sprint 3.5 完成 = 全部下面满足：

1. 10 个 PR 全部合并到 master
2. 跨 PR 集成场景 A-E **全部**手动验证通过
3. 全部测试 case 绿（236+）
4. 既有测试零回归
5. 文档全部 self-contained 且通过同行 review
6. 任意一个工具开关关闭后系统能正常工作（5 分钟手动 smoke）

至此 Aether 在工具能力上达到与 claude-code default tool 完全对齐 +
全套 Tier 1 持久化能力，为 Sprint 4（Tier 2-5 压缩流水线）打下数据基础。
