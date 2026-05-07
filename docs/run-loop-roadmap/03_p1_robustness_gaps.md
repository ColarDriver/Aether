# P1 缺口（长会话/规模化必须）

> 范围：阶段 2、3、5、7、11、12.1、12.4、12.5、14、15.⑤–⑧、17.①、17.④、17.⑦
>
> 这一档不修也能跑，但用户一旦把会话堆到几十轮、或者并发几十个 session，
> 就会撞上"上下文炸了 / 凭证过期 / 没法计费 / Anthropic cache 命中率为 0"等问题。

---

## P1-1 上下文压缩抽象（阶段 3、12.10、12.11、15.⑦）

### 现状
Aether 完全没有压缩概念。`EngineConfig` 里没有压缩相关字段；provider 抛出 4xx 就直接失败。

### 期望

新增 `backend/harness/aether/runtime/context_compressor.py`：

```python
class ContextCompressor(Protocol):
    threshold_tokens: int          # 触发阈值
    context_length: int            # 模型实际上下文窗口
    last_prompt_tokens: int        # 上一次响应的 prompt_tokens（用于真实估算）
    protect_first_n: int = 2       # 头部保护（system + first user）
    protect_last_n: int = 6        # 尾部保护（最近的对话）

    def should_compress(self, approx_tokens: int) -> bool: ...
    def compress(
        self,
        messages: list[dict],
        system_prompt: str | None,
    ) -> tuple[list[dict], str | None]: ...
    def update_from_response(self, usage: dict) -> None: ...
    def update_model(self, *, model: str, context_length: int, ...) -> None: ...
```

### 三个触发点
1. **Preflight**（turn 入口）：在 `_prepare_session_and_system_prompt` 之后做一次预估。
2. **413 / context_overflow**（错误处理）：依赖 P0-7 的 ErrorClassifier，按 `payload_too_large` / `context_overflow` 分流到 `compress_loop(max=3)`。
3. **Post-tool**（每轮 tool 执行完毕后）：用 `last_prompt_tokens` 与 threshold 对比，超过即压缩。

### 默认实现
`SummarizingCompressor`：
- 保留 `protect_first_n` + `protect_last_n` 不动。
- 中段调用一次 LLM `summarize(messages_middle) -> str`，替换为单条 system "Earlier conversation summary: ..."。
- 总 token 预估仍超阈值时多次 pass。

### 数据结构变更
- `EngineConfig` 增加：`compression_enabled: bool = False`、`compression_threshold_pct: float = 0.5`、`compression_max_passes: int = 3`。
- `ExitReason.COMPRESSION_EXHAUSTED`。

### 验收
- 单测：`SummarizingCompressor.compress` 把 100 条消息压成 12 条（前 2 + 后 6 + 中段 summary + tool result 桩）。
- 集成测：长会话循环到第 30 轮自动触发 post-tool 压缩。

---

## P1-2 Iteration Budget（阶段 5）

### 现状
仅 `iterations < self.config.max_iterations` 硬上限。`_handle_max_iterations` 不存在。

### 期望

新增 `backend/harness/aether/runtime/iteration_budget.py`：

```python
@dataclass
class IterationBudget:
    max_total: int
    used: int = 0
    grace_consumed: bool = False

    @property
    def remaining(self) -> int: ...
    def consume(self) -> bool: ...    # False 表示用尽
    def refund(self) -> None: ...     # cheap tool 不扣预算
    def grace_call(self) -> bool: ...  # 用尽后再给一次
```

### Run-loop 改动
1. 在 `run_loop` 入口构造 `budget = IterationBudget(self.config.max_iterations)`，挂到 `context.metadata["iteration_budget"]` 而不是 `self`（多 session 安全）。
2. 每轮开头 `if not budget.consume(): break`。
3. 仅 cheap tool（如 `execute_code`）回合结束后 `budget.refund()`。可配置 cheap tool 名单。
4. 用尽后调用 `_handle_max_iterations`（见 P1-3）。

### 验收
- 单测：5 轮预算，第 6 轮 break 出循环。
- 单测：4 轮 cheap tool 都 refund，预算应能继续。

---

## P1-3 max_iterations 兜底 summary（阶段 17.①）

### 现状
直接返回 `MAX_ITERATIONS`，最终用户看到的对话没有结尾。

### 期望

`_handle_max_iterations(messages, api_call_count) -> str`：
1. 复制 `messages`；丢弃 tools=[]；追加一条 user：`[System: You've used your iteration budget. Provide a clean summary of what you did and what's left.]`
2. 调一次 provider（一定不带 tools）。
3. 拿响应 content 当 `final_response`，状态保持 `MAX_ITERATIONS`，`metadata["summary_provided"] = True`。

### 验收
- 单测：mock provider 第 N+1 次返回 summary 文本，断言 result.final_response 等于该文本且 status==MAX_ITERATIONS。
- 集成测：真实模型场景下 summary 包含此前关键操作。

---

## P1-4 Token / Cost 累计（阶段 11）

### 现状
`EngineResult` 完全没有 token/cost 字段。

### 期望

1. 新增 `backend/harness/aether/runtime/usage.py`：
   ```python
   @dataclass
   class CanonicalUsage:
       input_tokens: int
       output_tokens: int
       cache_read_tokens: int
       cache_write_tokens: int
       reasoning_tokens: int
       prompt_tokens: int
       completion_tokens: int
       total_tokens: int

   def normalize_usage(raw: Any, *, provider: str, api_mode: str) -> CanonicalUsage: ...
   ```
2. provider `_parse_response` 把 raw usage 透出到 `NormalizedResponse.metadata["usage"]`。
3. run_loop 在 `post_llm_call` hook 之前累加到 `context.metadata["usage_accumulator"]`。
4. `_build_result` 把累计值写入 `EngineResult.metadata["usage"]`。

### 数据结构变更
- `EngineResult.metadata` 标准字段：`prompt_tokens / completion_tokens / total_tokens / cache_read_tokens / cache_write_tokens / reasoning_tokens / api_calls / approx_cost_usd?`。

### 验收
- 单测：mock provider 返回 `usage={prompt_tokens: 100, completion_tokens: 50, prompt_tokens_details: {cached_tokens: 80}}`，断言 cache_read_tokens=80 被正确累加。

---

## P1-5 凭证池轮转 + OAuth 自动刷新（阶段 12.1、12.4）

### 现状
provider 持有单一 `api_key`，401 时直接失败。

### 期望

1. **`CredentialPool`**（新文件 `runtime/credential_pool.py`）：
   - 维护 `list[Credential]`，按调用轮询。
   - 在 401/403 + 配置允许时调用 `pool.rotate()`。
2. **`provider.refresh_credentials() -> bool`**：抽象方法，默认 NotImplementedError。
   - `OpenAICompatibleModel` 默认实现：从 env 读取最新 `OPENAI_API_KEY`，与当前不同则替换并返回 True。
   - 后续可加 `AnthropicProvider.refresh_oauth_token()` 等。
3. **run_loop 401 分支**：检查 `classified.is_auth and not auth_retry_attempted`，调一次 refresh，成功则 `continue`。

### 验收
- 单测：mock 401 + refresh 返回新 key，断言重发并成功。
- 集成测：env 滚 key 后 ≤ 1 个 turn 自动恢复。

---

## P1-6 thinking_signature 失效恢复（阶段 12.5）

### 现状
Anthropic 在压缩 / 截断后再发请求会因 thinking 签名失效返回 400，Aether 直接失败。

### 期望

ErrorClassifier 识别 `thinking_signature` reason；run_loop 一次性剥掉所有 `messages[*]["reasoning_details"]` 后 retry 一次。

### 验收
- 单测：mock 400 + body 含 "invalid thinking block"，断言 retry 前 reasoning_details 被清空。

---

## P1-7 API 消息构建中的 reasoning 续传（阶段 7 子项）

### 现状
- `_append_assistant_text_message` 直接丢掉 reasoning 字段。
- 没有 `_copy_reasoning_content_for_api` 的等价物。

### 期望

新增 `backend/harness/aether/runtime/message_builder.py`：

```python
class MessageBuilder:
    def build_api_messages(
        self,
        messages: list[dict],
        *,
        system_prompt: str | None,
        provider_profile: ProviderProfile,
    ) -> list[dict]: ...
```

至少做这几件事（按重要性）：
1. **reasoning 续传**：assistant 消息上若有 `reasoning`，按 provider profile 转换：
   - Anthropic Messages：`reasoning_details` 数组（保留 signature）。
   - OpenAI / OpenRouter：`reasoning_content`（Moonshot 也用这个）。
   - 严格 API（Mistral / Fireworks）：直接删。
2. **孤儿 tool 结果补桩**：发现 `assistant.tool_calls[i].id` 没有对应 `tool` 消息时插入桩。
3. **JSON 规整**：`tool_calls[*].function.arguments` 用 `json.dumps(obj, sort_keys=True, separators=(",",":"))` 重写以提升 KV cache 命中。
4. **删除内部字段**：`finish_reason / _thinking_prefill / reasoning（保留到 reasoning_details 之后）`。

### 验收
- 单测：构造 assistant 消息含 `reasoning="思考"`，按 anthropic_messages profile 输出应有 `reasoning_details=[{"type":"thinking",...}]`，按 strict profile 应彻底删除。
- 单测：缺失某个 tool 结果时输出包含一条 `role=tool, content="(no result captured)"` 桩。

---

## P1-8 Codex incomplete 续写（阶段 14）

### 现状
不存在 codex_responses 模式，但未来要接 Codex / o1-style 提供方时会撞这个问题。

### 期望

放在 `MessageBuilder` 之后的"响应后置归一化"阶段：
- 检测 `finish_reason=="incomplete"`（Codex Responses API 专有）。
- 把当前 assistant 消息（带 reasoning）追加进历史，标记 `_thinking_prefill=True`。
- `continue` 重新进 LLM_CALL。最多 3 次。
- 用尽后返回 `partial=True` + `error="Codex response remained incomplete after 3 continuation attempts"`。

### 验收
- 单测（仅 Codex provider）：mock 3 次 incomplete + 1 次 complete，断言能拿到完整响应且历史里有 3 条 prefill 占位。

---

## P1-9 Tool 路径增量持久化（阶段 15.⑧）

### 现状
`run_loop` 只在 `_build_result` 时返回 messages。Ctrl+C 时，已经做完的 tool 工作全丢。

### 期望

1. `EngineConfig.persist_after_each_tool: bool = True`。
2. 在 `_append_tool_result_message` 之后调一次 `self._session_store.append_messages(session_id, [assistant_msg, tool_msg])`（或全量 replace）。
3. 增加 `SessionStore.persist_partial(session_id, messages)` 接口。

### 验收
- 中断测：第 3 个工具执行完毕后立即 `interrupt()`，从 session_store 读出的 messages 应包含完整的前 3 个工具结果。

---

## P1-10 Turn-end 结构化诊断（阶段 17.④）

### 现状
`_build_result` 不写日志；`ExitReason` 只有 8 种值。

### 期望

1. **扩 ExitReason** 到至少 16 种（与 Hermes 的 `_turn_exit_reason` 对齐）：
   - `TEXT_RESPONSE / EMPTY_RESPONSE / LENGTH_EXHAUSTED / LENGTH_RECOVERED`
   - `MAX_ITERATIONS / BUDGET_EXHAUSTED / GRACE_CALL_USED`
   - `INTERRUPTED / INTERRUPTED_DURING_API / INTERRUPTED_DURING_TOOL`
   - `RATE_LIMITED / FALLBACK_EXHAUSTED / COMPRESSION_EXHAUSTED`
   - `TOOL_ERROR / UNKNOWN_TOOL / INVALID_TOOL_REPEATED / TOOL_CALL_TRUNCATED`
   - `PROVIDER_ERROR / MIDDLEWARE_ERROR / NETWORK_DROP`
   - `PARTIAL_STREAM_RECOVERY / FALLBACK_PRIOR_TURN_CONTENT`
   - `STUCK_AFTER_TOOL`（最后一条消息是 `role=tool`，等于 Hermes 的 WARNING 信号）

2. **`_build_result` 增加结构化日志**：
   ```python
   logger.info(
       "Turn ended: reason=%s model=%s iterations=%d/%d budget=%d/%d "
       "tool_turns=%d last_msg_role=%s response_len=%d session=%s",
       exit_reason, model, iterations, max_iterations, ...
   )
   ```
   当 `last_msg_role == "tool"` 且非 INTERRUPTED 时降级为 WARNING。

### 验收
- 单测：每种 ExitReason 至少 1 个用例触发，断言日志格式。
- 单测：构造一个 mock 让最后一条消息是 tool，断言日志走 WARNING 分支。

---

## P1-11 EngineResult 字段扩展（阶段 17.⑦）

### 现状
`EngineResult` 字段够用，但没有 reasoning 抽取、token、cost、pending_steer。

### 期望

`EngineResult.metadata` 标准化字段（保持向后兼容，新字段全在 metadata 里）：

```python
metadata = {
    "request": {...},                    # 已有
    "turn": {...},                       # 已有
    "runtime": {...},                    # 已有，扩展 retry 计数
    "usage": CanonicalUsage,            # 来自 P1-4
    "iteration_budget": {used, max},    # 来自 P1-2
    "exit": {
        "reason": str,                   # 与 ExitReason 同名
        "last_msg_role": str,
        "stuck_after_tool": bool,
    },
    "reasoning": {
        "last_reasoning": str | None,    # 当前 turn 内最近一次非空 reasoning
    },
    "pending_steer": str | None,        # 阶段 6
    "interrupt_message": str | None,    # 已有部分语义
}
```

### 验收
- 单测：覆盖 reasoning 抽取（不能跨 user 消息边界）。

---

## P1 总览：依赖关系

```
P1-1 (压缩) ──┬─→ P1-2 (budget) ─→ P1-3 (summary 兜底)
              │                         │
P1-4 (token) ─┘                         │
                                        ▼
P1-7 (msg builder) ──→ P1-6 (thinking sig) ──→ P1-8 (codex)
                                        │
P1-5 (cred pool) ──→ ↑                  │
P1-9 (增量持久化) ──→ ↑                  │
P1-10 (诊断) + P1-11 (result) ←─────────┘
```

P1-1、P1-2、P1-4 可以并行进 Sprint 3。
P1-7（MessageBuilder）是后续 prompt cache / Anthropic 相关功能的前置，必须在 Sprint 5 之前完成。

下一步：阅读 [04_p2_ux_observability_gaps.md](./04_p2_ux_observability_gaps.md)。
