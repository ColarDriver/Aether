# Aether Run-Loop 演进路线图

## 一、本目录的目的

本目录用于把 Aether `AgentEngine.run_loop` 与 hermes-agent `AIAgent.run_conversation`
做一次彻底对齐的工程化记录，并把对齐过程拆成可逐步推进的 Sprint。

- **基准源码**
  - Aether：`backend/harness/aether/agents/core/agent.py` 的 `AgentEngine.run_loop`（约 155–405 行）
  - Hermes：`/workspace/hermes-agent/run_agent.py` 的 `AIAgent.run_conversation`（约 10569–14167 行，~3600 行）

- **方法论**
  1. 把 Hermes 的整段 `run_conversation` 抽象为 17 个独立的"loop 阶段"；
  2. 对每个阶段标注 Aether 现状：✅ 已有 / ⚠️ 部分实现 / ❌ 缺失；
  3. 把缺口按"严重性"分为 P0/P1/P2/P3 四档；
  4. 把这些缺口压成 6 个独立可合并的 Sprint。

## 二、文件导航

| 文档 | 内容 | 用途 |
|---|---|---|
| [01_stage_matrix.md](./01_stage_matrix.md) | **17 阶段对照表（核心交付物）** | 一张表看清 Hermes 每一步在 Aether 里的位置和缺口 |
| [02_p0_critical_gaps.md](./02_p0_critical_gaps.md) | P0 缺口（不修就坏） | Sprint 1–2 的输入 |
| [03_p1_robustness_gaps.md](./03_p1_robustness_gaps.md) | P1 缺口（长会话/规模化必须） | Sprint 3 的输入 |
| [04_p2_ux_observability_gaps.md](./04_p2_ux_observability_gaps.md) | P2 缺口（UX/可观测性） | Sprint 5 的输入 |
| [05_p3_advanced_features.md](./05_p3_advanced_features.md) | P3 高级功能 | Sprint 6+ 的输入 |
| [06_aether_structural_issues.md](./06_aether_structural_issues.md) | Aether 现状的结构性问题 | 在动手之前必须先修的"地基"问题 |
| [07_sprint_execution_plan.md](./07_sprint_execution_plan.md) | 6 个 Sprint 的执行计划 | 排期 + 文件改动清单 + 验收标准 |

## 三、关键结论速览

### 1. 整体差距评估
- **量化**：Aether 当前 `run_loop` ≈ 250 行；Hermes `run_conversation` ≈ 3600 行。差距并非简单的"代码量"，
  而是结构性的——Aether 把"重试 / 兜底 / 恢复"全部推给了 provider 内部，引擎层只剩"调一次、要么成要么死"。
- **可生产性**：当前 Aether 的 `run_loop` 在面对**长会话 / rate limit / 上下文超限 / 截断 tool_call /
  纯 reasoning 响应**这五个最常见的真实场景下都会直接返回失败。

### 2. 最优先要补的 4 件事（P0）
1. **错误分类与恢复**（17 阶段表的阶段 12，~12 个分支） —— 没有它，任何 4xx/5xx 都是单次失败。
2. **`finish_reason="length"` 处理**（阶段 10） —— 长输出场景必撞，必须有续写 + 回滚。
3. **截断 tool_call 检测**（阶段 15.②） —— 现状：半截 JSON 会被原样塞给 tool runtime。
4. **空响应 / thinking-only 9 步降级**（阶段 16） —— reasoning 模型（GLM-5/QwQ/DeepSeek-R1）必触发。

### 3. 推荐 Sprint 顺序（详见 `07_sprint_execution_plan.md`）

| Sprint | 主题 | 涉及阶段 | 验收信号 | 状态 |
|---|---|---|---|---|
| 0 | 地基修复（实例属性 → metadata、provider 单次发包、`RecoveryStrategy` 抽象） | 结构问题 1/3/5 | 多 session 并发隔离稳定 | ✅ 已完成 |
| 1 | 流式健康检查 + 截断处理 | 8、10、15.② | 模型撞 max_tokens 时能续写而不是失败 | ✅ 已完成 |
| 1.5 | 内置工具集 + 散文 → 结构化合成 | P0-9 | Kimi 类模型不再卡在 phantom-tool 警告 | ✅ 已完成 |
| 2 | 错误分类器 + Provider Fallback 链 + 工具派发 hardening | 9、12、13、P0-5/6/7 | 429 自动切 fallback、模糊修复 typo、cap/dedup | ✅ 已完成 |
| 3 | 上下文压缩 + Iteration Budget | 3、5、12.10–12.11、17 | 长会话不再因 413 失败 | 待启动 |
| 4 | 空响应/Thinking 9 步降级 + Tool 容错 | 14、15、16 | reasoning 模型能完整跑完一轮 | 待启动 |
| 5 | Prompt cache + Reasoning 续传 + 增量持久化 | 2、7、4、17 | Anthropic cache 命中率 ≥ 70% | 待启动 |
| 6 | `/steer` + 凭证池 + 后台 review | 6、11.x、22 | 多 session 并发不会互相干扰 | 待启动 |

### 4. 结构性"地基"问题（先修，详见 `06_aether_structural_issues.md`）
- `_empty_response_retries` / `_provider_error_retries` 是引擎实例属性 → 多 session 并发会互相覆盖，
  必须挪到 `TurnContext.metadata`。
- `_empty_response_retries` 累加但**没有任何代码读它来决定是否继续 retry**（死字段）。
- 重试控制权全部在 provider 内部 → 引擎层无法感知 budget / interrupt / 上下文压缩 / fallback 链。

## 四、维护约定

- 每个 Sprint 完成后，回到对应优先级文档把状态从 ❌/⚠️ 改为 ✅，并在 PR 描述里链接到本目录的对应章节。
- 新发现的差异如果不在 17 阶段表里，**先扩表**再写代码（保持单一事实来源）。
- 任何超出本目录规划的设计变更（例如换 provider 抽象、改 middleware 协议），必须先在 `agent-engine/`
  目录下补设计文档，再回到本目录更新阶段状态。
