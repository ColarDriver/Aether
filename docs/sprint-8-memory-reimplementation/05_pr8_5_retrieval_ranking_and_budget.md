# PR 8.5 - Retrieval Ranking + Token Budget

## 目标

实现无向量的 memory 检索和 packing。重点是稳定、可解释、低延迟，而不是最大召回率。

## 为什么 v1 不用 Vector DB

| 风险 | 说明 |
|---|---|
| 依赖复杂 | 需要 embeddings provider、索引迁移、存储版本管理 |
| 成本不透明 | 每次写入/重建可能触发额外 API 调用 |
| 误召回难排查 | 语义近但任务无关的 memory 容易污染上下文 |
| 稳定性不足 | 长任务 agent 更需要确定性和 provenance |

Sprint 8 先做 deterministic retrieval。后续可以在 `MemoryProvider`
后加 optional vector backend。

## 新增/修改文件

| 文件 | 内容 |
|---|---|
| `backend/harness/aether/memory/retrieval.py` | candidate recall 和 ranking |
| `backend/harness/aether/memory/budget.py` | token budget 和 packing |
| `backend/harness/aether/memory/tokenize.py` | token estimate helper |
| `backend/harness/aether/tests/memory/test_retrieval.py` | 检索测试 |

## Query Features

从 `MemoryQuery` 提取：

- user message keywords。
- 最近 user/assistant message keywords。
- active files 和路径片段。
- tool names 和 tool result path hints。
- task goal/constraints。
- explicit tags，例如用户提到 “architecture”、“permission”、“memory”。

## Candidate Recall

召回来源：

1. task snapshot：始终候选。
2. project index：按 tags、paths、kind、keyword 匹配。
3. topic files：只读取 index 命中的少量 topic entry。
4. user memory：默认不参与，personal assistant 模式才参与。

不做：

- 不全量读取所有 topic 正文。
- 不扫描整个 repo。
- 不调用 embeddings。

## Ranking

本地打分：

```text
score =
  scope_weight
  + keyword_overlap
  + path_affinity
  + tag_match
  + recency_boost
  + confidence_boost
  - stale_penalty
  - token_cost_penalty
```

默认权重：

| 因子 | 权重 |
|---|---:|
| task/session scope | +4.0 |
| project scope | +2.5 |
| exact path match | +3.0 |
| tag match | +1.5 |
| keyword overlap | +0.2 each |
| high confidence | +1.0 |
| stale > 90 days | -1.0 |
| block > 500 tokens | -2.0 |

阈值：

- `score < 1.5` 不注入。
- 最多注入 8 个 block。
- project memory 最多 5 个 block。

## Budget Packing

Budget 输入：

- `effective_budget` from PR 8.2。
- per-scope budget split。
- per-block max。

Packing 顺序：

1. 固定保留 task goal block。
2. 添加 task constraints/decisions。
3. 添加高分 project blocks。
4. 如果 personal mode，添加 user blocks。
5. 留 10% reserve，避免 estimator 偏差。

裁剪规则：

- 单 block 超限时按段落裁剪。
- 裁剪后追加 `[memory truncated]`。
- 无法裁剪到有效内容则丢弃。

## Skip Reasons

必须标准化：

| reason | 含义 |
|---|---|
| `disabled` | memory 关闭 |
| `mode_off` | `memory_mode=off` |
| `budget_too_small` | budget 不足 |
| `no_query_signal` | 用户消息无法形成检索信号 |
| `no_candidates` | 没有候选 |
| `no_relevant_blocks` | 候选低于阈值 |
| `timeout` | 检索超时 |
| `provider_error` | provider 异常 |

## LLM Rerank

Sprint 8 默认关闭 `memory_llm_rerank_enabled`。

如果后续开启，限制为：

- 只对已经本地高分的 top 20 candidates rerank。
- timeout 独立，不阻塞主 loop。
- 不允许 rerank 读取 topic 外的文件。
- rerank 失败回退本地排序。

## 测试

```python
def test_task_memory_wins_over_project_under_budget_pressure(): ...
def test_project_path_match_boosts_relevance(): ...
def test_low_score_candidates_are_not_injected(): ...
def test_user_memory_excluded_in_project_mode(): ...
def test_budget_packer_reserves_estimation_headroom(): ...
def test_large_block_is_truncated_before_injection(): ...
def test_retrieval_timeout_returns_skipped_bundle(): ...
def test_corrupted_project_index_rebuilds_or_skips_safely(): ...
```

## 验收门

- 检索结果可解释：metadata 能看到 candidate_count、injected_count、skipped_reason。
- 小问题不会注入大量 memory。
- 长上下文不会因为 memory 额外触发 overflow。
- 无向量依赖，不需要网络或 embedding key。
