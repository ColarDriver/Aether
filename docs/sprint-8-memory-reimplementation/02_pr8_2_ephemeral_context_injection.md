# PR 8.2 - Ephemeral Context Injection

## 目标

把 Memory 检索结果安全注入 provider-bound outbound messages，同时保证：

- 不修改 canonical transcript。
- 不污染 session replay。
- 不进入 compaction summary。
- 有严格 token budget 和 observability。

## 当前可复用边界

Aether 已有 `EngineHooks.pre_llm_call()` 和 `HookOutcome`：

```python
@dataclass(slots=True)
class HookOutcome:
    inject_user_context: str | None = None
    inject_system_addendum: str | None = None
    short_circuit_response: NormalizedResponse | None = None
```

AgentEngine 当前会对 `messages` 做 deepcopy，然后把
`inject_user_context` 追加到最近一条 user message：

```xml
<hook_context>
...
</hook_context>
```

Memory 注入必须复用这个机制或同等 outbound-copy 机制，不新增会改原始
messages 的拼接路径。

## 新增/修改文件

| 文件 | 内容 |
|---|---|
| `backend/harness/aether/memory/render.py` | `MemoryBundle` 渲染为 XML-ish context |
| `backend/harness/aether/memory/injection.py` | budget、skip reason、metadata helper |
| `backend/harness/aether/agents/core/agent.py` | 在 pre-LLM 阶段调用 memory provider |
| `backend/harness/aether/tests/memory/test_injection.py` | 注入边界测试 |

## 注入位置

顺序建议：

1. preflight compaction 先运行。
2. 得到当前 messages/projection。
3. 计算剩余 memory budget。
4. 调用 memory provider retrieve。
5. 将 bundle 渲染成 `HookOutcome.inject_user_context` 等价文本。
6. 调用现有 `_apply_hook_outcome_to_messages()`。
7. middleware `run_before_llm()` 继续处理 outbound messages。

这样 memory 不会影响 preflight compaction 的输入，也不会写回原 messages。

## 注入格式

统一渲染为：

```xml
<memory_context>
  <memory_policy>
    Retrieved memory may be stale. Prefer the current user message,
    current repository files, and fresh tool results when there is conflict.
  </memory_policy>
  <task_memory>
    <memory id="..." source="..." updated_at="...">...</memory>
  </task_memory>
  <project_memory>
    <memory id="..." source="..." updated_at="...">...</memory>
  </project_memory>
</memory_context>
```

要求：

- 没有 block 的 scope 不输出空 section。
- 不输出 user memory section，除非 personal assistant 模式启用。
- 每个 block 必须带 source。
- 单 block 超过 `memory_block_token_max` 时先裁剪再渲染。

## Budget 计算

输入：

- resolved model context window。
- estimated current outbound message tokens。
- `memory_token_budget_pct`。
- `memory_token_budget_max`。
- compaction threshold，例如 `compression_pre_llm_pct`。

规则：

```text
base_budget = min(model_window * pct, memory_token_budget_max)
remaining_before_threshold = model_window * compression_pre_llm_pct - estimated_prompt_tokens
effective_budget = min(base_budget, max(0, remaining_before_threshold * 0.5))
```

如果 `effective_budget < 300`，跳过 memory 注入，记录
`skipped_reason="budget_too_small"`。

## Metadata

在 `TurnContext.metadata["memory"]` 写入稳定形状：

```python
{
    "enabled": True,
    "mode": "project",
    "retrieval_ms": 18.4,
    "candidate_count": 12,
    "injected_count": 3,
    "injected_tokens": 980,
    "scopes": ["task", "project"],
    "skipped_reason": None,
    "error": None,
}
```

默认不记录 memory 内容。

## 与 Compaction 的边界

必须明确 pin 住以下行为：

| 阶段 | 是否包含 retrieved memory |
|---|---|
| canonical messages | 否 |
| preflight compaction input | 否 |
| recovery compaction input | 否 |
| provider outbound payload | 是 |
| session store | 否 |
| trajectory content | 默认否 |
| debug metadata | 只记录计数和 source |

如果 provider 因 context overflow 触发 recovery compaction，重试前应重新计算
memory budget；不能复用旧注入文本。

## 失败策略

- provider retrieve 超时：skip。
- provider retrieve 抛异常：skip，metadata 记录 error class。
- renderer 抛异常：skip，不能影响主 LLM call。
- token estimate 不可信：按保守倍率处理，必要时 skip。

## 测试

```python
def test_memory_injection_does_not_mutate_canonical_messages(): ...
def test_memory_context_appended_to_latest_user_message(): ...
def test_no_memory_injection_when_budget_too_small(): ...
def test_memory_provider_exception_skips_without_failing_loop(): ...
def test_memory_metadata_does_not_include_block_text_by_default(): ...
def test_recovery_compaction_recomputes_memory_budget(): ...
def test_compaction_input_excludes_retrieved_memory(): ...
```

## 验收门

- 注入路径只影响 outbound copy。
- 打开 compression 后 memory 不进入 summary。
- memory skip 不改变模型正常响应。
- metadata 足够 UI/日志判断 memory 是否参与本轮。
