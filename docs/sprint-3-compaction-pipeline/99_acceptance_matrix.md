# Sprint 3 验收矩阵 — 端到端场景与回归

> Sprint 3 收尾时跑这份矩阵。每条场景标注：所属 Tier、需要的真实/mock 资源、
> 预期可观测信号、通过条件。所有场景应当在合并最后一个 PR 后 24 小时内全部 PASS。

## 一、端到端场景

### Scenario S1 — 真实长会话不再 fail（核心场景）

| 维度 | 内容 |
|---|---|
| 触发的 Tier | Tier 5（autocompact） |
| 资源 | 真实 GPT-4 Turbo / Claude 3.5 Sonnet |
| 步骤 | 1. 设 `compression_enabled=True`<br>2. 跑 50 轮真实对话（每轮 user 输入 200 字，assistant 回复 + 1-2 次 read_file/grep）<br>3. 累计 prompt 应到 100k+ token |
| 预期信号 | - CLI 日志出现 `compact[preflight] tier5_autocompact: 100000 → 60000 tokens`<br>- `result.metadata["compaction"]["tier5_summaries_generated"] >= 1`<br>- 第 51 轮对话仍可正常完成 |
| PASS 条件 | 第 50-100 轮无任何 ProviderInvocationError；token 减少 ≥ 40%；模型仍能引用第 1-10 轮的"用户原始任务" |
| FAIL 兜底 | 若失败，回滚 `compression_enabled=False`；定位是否 protect_first_n 太小导致丢失系统消息 |

### Scenario S2 — 巨量工具输出不撑爆上下文

| 维度 | 内容 |
|---|---|
| 触发的 Tier | Tier 1（per-tool spill） |
| 资源 | 本地 mock provider |
| 步骤 | 1. 设 `tool_result_spill_enabled=True`<br>2. 模型调 `shell: find / -type f 2>/dev/null` 输出 100MB+<br>3. 下一轮模型调 `read_file <spilled-path>` |
| 预期信号 | - shell 工具的 `result.content` ≤ 1KB（preview + notice）<br>- `~/.aether/tool_results/<sid>/<cid>.txt` 文件存在且 ≈ 100MB<br>- `result.metadata["compaction"]["tier1_spilled_count"] == 1`<br>- 模型用 read_file 能拿到完整内容 |
| PASS 条件 | 单 turn 上下文不超过模型 window；spill 文件可读；模型能继续调 |
| FAIL 兜底 | 检查 spill_dir 权限 / 磁盘空间；fallback 到 plain truncation 行为 |

### Scenario S3 — 闲置回来不重新上传整个 prefix

| 维度 | 内容 |
|---|---|
| 触发的 Tier | Tier 3（time-based microcompact） |
| 资源 | mock 时间（用 `freezegun` 或类似） |
| 步骤 | 1. 设 `microcompact_gap_threshold_minutes=5, microcompact_keep_recent=3`<br>2. 跑 5 轮，每轮调 read_file（共 5 次 read_file 调用）<br>3. mock 时间往前推 6 分钟<br>4. 跑第 6 轮 |
| 预期信号 | - 日志 `tier3: gap=6.0min > 5.0min; cleared 2 tool results, kept last 3`<br>- 第 6 轮发给 provider 的 messages 里前 2 个 read_file 的 tool_result content 是 `[Old tool result content cleared]`<br>- `result.metadata["compaction"]["tier3_cleared_count"] == 2` |
| PASS 条件 | 清理数 == 5 - 3 == 2；保留最后 3 个；模型仍能正常工作 |
| FAIL 兜底 | 检查 timestamp 注入是否到位；是否被 phantom_tool 干扰 |

### Scenario S4 — 冗余调用被自动修剪

| 维度 | 内容 |
|---|---|
| 触发的 Tier | Tier 2（snip） |
| 资源 | mock provider |
| 步骤 | 1. 设 `snip_enabled=True, snip_dupe_enabled=True`<br>2. mock provider 在一轮里调 read_file path=/x.txt 共 5 次（input 完全相同）<br>3. 流水线触发 |
| 预期信号 | - 日志 `tier2: snipped 4 tool-use/result pairs + 0 empty assistants`<br>- view messages 里只剩最后一次 read_file<br>- `result.metadata["compaction"]["tier2_snipped_count"] == 4` |
| PASS 条件 | 删除 4 对配对完整；保留最后一对；下游 provider 调用不报 tool_use_id 配对错误 |
| FAIL 兜底 | 检查 _normalize_name 实现；检查 snip_dupe_enabled 默认值 |

### Scenario S5 — 预算耗尽给 summary 而非沉默

| 维度 | 内容 |
|---|---|
| 触发的 Tier | 不属于流水线（PR 3.2） |
| 资源 | mock provider |
| 步骤 | 1. 设 `max_iterations=3, summary_on_budget_exhausted=True`<br>2. mock provider 每次返回 `tool_calls=[ToolCall("noop")]`，永远不结束<br>3. 第 4 次（grace）返回文本 "I summarised what I did" |
| 预期信号 | - `result.exit_reason == ExitReason.MAX_ITERATIONS`<br>- `result.final_response == "I summarised what I did"`<br>- `result.metadata["iteration_budget"]["grace_consumed"] == True`<br>- `result.metadata["summary_provided"] == True` |
| PASS 条件 | summary 文本进 final_response；状态仍为 MAX_ITERATIONS；不抛错 |
| FAIL 兜底 | `summary_on_budget_exhausted=False` 退回原"沉默 break"行为 |

### Scenario S6 — token 计数全程可读（基础设施）

| 维度 | 内容 |
|---|---|
| 触发的 Tier | 不属于流水线（PR 3.1） |
| 资源 | 真实 GPT-4 + Claude 各一次 |
| 步骤 | 1. 跑一次正常 turn，5 次 LLM 调用<br>2. 检查 result.metadata["usage"] |
| 预期信号 | - `result.metadata["usage"]` 含 `input_tokens / output_tokens / cache_read_tokens / cache_write_tokens / reasoning_tokens / prompt_tokens / completion_tokens / total_tokens`<br>- `total_tokens` ≈ 5 次单次累加<br>- `metadata["api_calls"] == 5` |
| PASS 条件 | 全部字段存在；JSON-serializable；与 provider 原始返回的 usage 数学一致 |
| FAIL 兜底 | 检查 normalize_usage 三家 provider 转换是否正确 |

### Scenario S7 — cheap tool refund 有效

| 维度 | 内容 |
|---|---|
| 触发的 Tier | 不属于流水线（PR 3.2） |
| 资源 | mock provider |
| 步骤 | 1. 设 `max_iterations=5`<br>2. mock provider 前 5 轮只调 update_todo<br>3. 第 6-10 轮调 shell |
| 预期信号 | - 第 5 轮结束时 `iteration_budget["used"] == 0`（refund 全部抵消）<br>- 第 10 轮结束时 `iteration_budget["used"] == 5` |
| PASS 条件 | budget 计算精准；不提前触发 max_iterations |
| FAIL 兜底 | 把 update_todo 从 cheap_tool_names 列表移除 |

### Scenario S8 — Tier 4 与 Tier 5 互斥

| 维度 | 内容 |
|---|---|
| 触发的 Tier | Tier 4 + Tier 5（应只有 Tier 4 跑） |
| 资源 | mock provider，model_window=10k |
| 步骤 | 1. 设 `compression_enabled=True, context_collapse_enabled=True`<br>2. 构造 30 条 messages 共 ≈ 9.5k token（>= 95% blocking_pct）<br>3. 触发流水线 |
| 预期信号 | - 日志 `tier4: committed segment[2-12], view 9500→6000 tokens`<br>- `metadata["collapse_owns_headroom"] == True`<br>- 日志中**没有** `tier5_autocompact: ...`（被互斥屏蔽） |
| PASS 条件 | Tier 4 commit 一段；Tier 5 完全不触发；下一轮 turn 仍正确应用 view |
| FAIL 兜底 | `context_collapse_enabled=False` 退回纯 Tier 5 |

### Scenario S9 — 全流水线协同（综合）

| 维度 | 内容 |
|---|---|
| 触发的 Tier | Tier 1 → 2 → 3 → 4 → 5（理想情况下只触发前几级） |
| 资源 | mock provider 多轮 |
| 步骤 | 1. 设全部开关启用<br>2. 跑 30 轮：包含 100MB shell 输出（Tier 1）+ 5 次重复 read_file（Tier 2）+ idle 10 分钟（Tier 3）+ 长 prompt（Tier 4/5）|
| 预期信号 | 流水线日志按顺序出现各级 → 各级 metadata counter 都 > 0 → 最终 token 控制在 75% 以下 |
| PASS 条件 | 30 轮无 fail；上下文无溢出；所有 metadata counter 正确递增 |
| FAIL 兜底 | 单独排查哪一级未生效 |

### Scenario S10 — CLI 文件路径不被误判为命令（PR 之前的 bug 回归）

| 维度 | 内容 |
|---|---|
| 触发的代码 | `cli/commands.py:is_slash` |
| 资源 | CLI REPL |
| 步骤 | 1. 在 REPL 输入 `/workspace/hermes-agent 帮我看下这个项目`<br>2. 输入 `/halp`<br>3. 输入 `/help` |
| 预期信号 | - 第 1 条作为正常 user 输入发给模型（不报 Unknown command）<br>- 第 2 条报 Unknown command（合法 typo）<br>- 第 3 条命中 help 命令 |
| PASS 条件 | 三条都按预期处理 |
| FAIL 兜底 | 检查 `is_slash` 是否正确读 commands.py 的修复 |

## 二、回归测试矩阵

每个 PR 合入后必须跑全量。

| 测试套件 | 文件 | case 数（估算） | 关键覆盖 |
|---|---|---|---|
| 既有 357 个测试 | `backend/harness/aether/tests/` | 357 | Sprint 0/1/1.5/2 全部行为 |
| PR 3.1 | `test_usage_normalize.py` + `test_engine_result_metadata.py` | ~50 | usage 标准化 + metadata schema |
| PR 3.2 | `test_iteration_budget.py` + `test_max_iterations_summary.py` + `test_cheap_tool_refund.py` | ~40 | budget + summary + refund |
| PR 3.3 | `test_tool_result_storage.py` + `test_tool_result_spill.py` | ~40 | spill 工具 + 6 builtin |
| PR 3.4 | `test_compaction_pipeline.py` + `test_autocompact_gate.py` + `test_llm_fork_summarizer.py` + `test_token_estimation.py` + `test_compaction_integration.py` | ~50 | 流水线骨架 + Tier 5 |
| PR 3.5 | `test_microcompact_time_based.py` | ~25 | Tier 3 时间触发 |
| PR 3.6 | `test_snipper.py` | ~32 | Tier 2 三规则 |
| PR 3.7 | `test_collapse_store.py` + `test_collapse_tier.py` + `test_collapse_integration.py` | ~38 | Tier 4 投影 |
| **合计** | | **~632** | |

### 跑法

```bash
# 全部
cd backend/harness
uv sync
cd aether
uv run python -m unittest discover -s tests -v

# 单文件（实施过程中加速反馈）
uv run python -m unittest aether.tests.test_compaction_pipeline -v
```

### 通过条件

- 全部 632 个 case green。
- 无 deprecation warning（如新增 dataclass `slots=True` 与 frozen 共用导致的 warning 等）。
- 总运行时长 ≤ 60s（流水线 mock provider 不要走真网络）。

## 三、性能基准（Sprint 3 收尾必跑一次）

### 3.1 流水线开销

| 指标 | 测量方式 | PASS 阈值 |
|---|---|---|
| Tier 1（spill）延迟 | 在 shell 工具 spill 100MB 时计时 | < 100ms |
| Tier 2（snip）延迟 | 100 条 messages 的修剪 | < 20ms |
| Tier 3（microcompact）延迟 | 50 条 messages（10 个 tool_use） | < 30ms |
| Tier 4（collapse）一段 | 单次 LLM fork（mock 10ms 延迟） | LLM 调用 + < 50ms 本地 |
| Tier 5（autocompact）一次 | 单次 LLM fork（mock 10ms 延迟） | LLM 调用 + < 80ms 本地 |
| `estimate_messages_tokens` | 100 条 messages | < 5ms |

### 3.2 内存

| 指标 | 测量方式 | PASS 阈值 |
|---|---|---|
| `CompactionPipeline` 单实例 | `tracemalloc.get_traced_memory()` | < 100KB |
| `CollapseStore` 含 10 segments | 同上 | < 50KB |

### 3.3 跑一次完整 turn 的总开销增量

| 指标 | 对比 | PASS 阈值 |
|---|---|---|
| 平均 turn 延迟 | 流水线 enabled vs disabled，30 turn 平均 | 增加 ≤ 50ms |
| 平均 turn token 消耗 | 同上（不算 fork 出来的 summariser 调用） | 持平或下降 |

测试方法：用 `time` + `tracemalloc`，benchmark 跑 30 turn 求平均，结果落到
`docs/sprint-3-compaction-pipeline/PERF.md`（Sprint 收尾时新建）。

## 四、监控指标

落地后纳入日常观测的指标（建议在后续 Sprint 4 / Sprint 6 实现 metrics 系统时加上）：

| 指标 | 来源 | 报警阈值（建议） |
|---|---|---|
| `compaction.tier5_summaries_per_session` | metadata 累加 | > 5/session 持续一周 |
| `compaction.consecutive_failures_max` | metadata 累加 | == max（熔断触发率） |
| `compaction.tier1_spilled_total_bytes` | spill_dir 大小 | > 1GB（清理告警） |
| `compaction.tier3_cleared_per_turn_avg` | metadata 累加 | > 10/turn |
| `compaction.tier4_collapse_segments_per_session` | metadata 累加 | > 3/session |
| `iteration_budget.grace_consumption_rate` | metadata 累加 | > 30%（用户经常碰预算上限）|

## 五、文档同步收尾清单

PR 3.7 合入后必须做：

- [ ] 更新 [`docs/run-loop-roadmap/03_p1_robustness_gaps.md`](../run-loop-roadmap/03_p1_robustness_gaps.md)：
      P1-1 / P1-2 / P1-3 / P1-4 / P1-11 全部标 `✅ 已完成`，附 PR 编号
- [ ] 更新 [`docs/run-loop-roadmap/07_sprint_execution_plan.md`](../run-loop-roadmap/07_sprint_execution_plan.md)：
      Sprint 3 标 `✅ 已完成`，"### 3. 输出（Deliverables）"小节列出 7 个 PR 的 commit hash
- [ ] 更新 [`docs/run-loop-roadmap/README.md`](../run-loop-roadmap/README.md)：
      Sprint 表 Sprint 3 状态改 `✅`
- [ ] 新建 [`docs/agent-engine/08_compaction_pipeline.md`](../agent-engine/)：
      引用本目录设计文档，作为 agent-engine 视角的横切关注点入口
- [ ] 在每个 PR 文档末尾追加"实施记录"小节，记录实际偏差 / 追加测试 / 踩坑

## 六、Sprint 3 完成的最终签收

满足以下全部条件视为 Sprint 3 工程完成：

1. 7 个 PR 全部合入 master，commit message 风格一致（动词开头，引用 PR 编号）。
2. 632 个测试全部 green，CI 跑两次连续成功。
3. § 一 的 10 个端到端场景全部手工验证 PASS。
4. § 三 的性能基准跑过且全部满足阈值；结果落到 `PERF.md`。
5. § 五 的文档同步全部勾选完成。
6. 在生产环境（`compression_enabled=True`）灰度运行至少 7 天，无新增 incident。

签收人：技术负责人 + 至少 1 名 reviewer。
