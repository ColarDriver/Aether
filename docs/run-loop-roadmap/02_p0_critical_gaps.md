# P0 缺口（不修就坏）

> 范围：阶段 8、9、10、12.7、12.8、12.10、12.11、12.13、15.①②③④、16.全部
>
> 这一档对应"任何稍有规模的真实使用都会触发的失败"。当前 Aether 的 `run_loop`
> 在这些场景下返回的不是"降级体验"，而是直接失败。

---

## P0-1 流式健康检查（阶段 8）

### 现状
```51:104:backend/harness/aether/models/provider/openai_compatible.py
    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,  # noqa: ARG002
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        payload = self._build_payload(messages, tools=tools, config=config)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                with httpx.Client(timeout=self.request_timeout_sec) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    return self._parse_response(resp.json(), stream_callback=stream_callback)
            ...
```
- 整个 generate 是**阻塞** `httpx.Client.post`，没有 SSE。
- `stream_callback` 形参只在响应完成后才被一次性触发，**不是流式回调**。
- 没有 stale-stream 检测：模型生成中途服务端不发包就只能等到 `request_timeout_sec`。

### 期望
1. provider 增加 `_streaming_generate(api_kwargs)` 路径，逐 chunk 触发 `stream_callback(delta)`。
2. 增加 stale-stream 检测：若 90s 内没有任何 SSE 事件即抛 `StreamStallError`。
3. 增加 `on_first_delta` 回调（用于停 spinner / 关闭 thinking 提示）。
4. 流式失败自动降级为非流式重试一次，并把 `provider._disable_streaming` 置位以避免本会话重复尝试。

### 验收
- 单测：mock SSE 事件流，包含 `delta.content` / `delta.tool_calls.function.arguments` 增量片段，断言 `stream_callback` 被多次回调。
- 集成测：真实 `gpt-4o-mini` 调用，确认能在 1s 内拿到第一个 chunk。
- Stall 测：mock 一个 SSE 流在第一个 chunk 后停止发包 90s，断言抛出 `StreamStallError` 而不是 `httpx.ReadTimeout`。

---

## P0-2 响应壳校验（阶段 9）

### 现状
- `_parse_response` 在 `OpenAICompatibleModel` 里直接 `data["choices"][0]["message"]`，没有 None / 空 list 防御。
- 引擎层完全不校验 `NormalizedResponse` 形状。

### 期望
1. `ModelProvider` 增加 `validate_response(raw) -> tuple[bool, list[str]]`，返回是否合法 + 失败原因清单。
2. 引擎在 `LLM_CALL` 之后立即校验，invalid 触发 `ResponseInvalidError`。
3. invalid 场景**优先 eager fallback**而不是 generic retry（依赖 P0-7）。

### 验收
- 单测：构造 `{"choices": []}` / `{"choices": [{"message": null}]}` / 缺 `content` 的响应，断言归类为 invalid。
- 集成测：模拟 OpenRouter 的 `error` 字段返回（HTTP 200 但 body 是错误对象），确认能被识别。

---

## P0-3 `finish_reason="length"` 续写与回滚（阶段 10）

### 现状
`run_loop` 完全无视 `response.finish_reason`。

### 期望

引入 4 个分支（按 Hermes 11632–11848 的形态）：

1. **Thinking-budget 耗尽检测**：如果响应只有 `<think>` / `<reasoning>` 块且无 visible content 且 `finish_reason=="length"`，friendly-exit 并返回 `partial=True` + 提示用户降低 reasoning effort。
2. **续写 retry**（最多 3 次）：
   - 把当前 assistant 消息（带 `finish_reason="length"`）追加到 messages。
   - 追加一条 user 消息：`[System: Your previous response was truncated... Continue exactly where you left off.]`
   - 设置 `_ephemeral_max_output_tokens = max_tokens × (n+1)`（上限 32k）。
   - `continue` 重新进 LLM_CALL。
3. **截断 tool_call 单独 retry**（最多 1 次）：如果 `finish_reason=="length"` 且 `tool_calls` 非空，**不进消息历史**，重发当前请求一次。再失败就返回 partial。
4. **回滚到最后完整 assistant turn**：若续写 3 次仍失败，调用 `_get_messages_up_to_last_assistant(messages)`，返回 `partial=True` 给上层。

### 数据结构变更
- `NormalizedResponse.finish_reason` 已存在。需新增 `ExitReason.LENGTH_EXHAUSTED` 与 `ExitReason.LENGTH_RECOVERED`。
- `EngineResult.metadata` 增加 `length_continue_attempts: int`。
- `ModelCallConfig` 或 `EngineConfig` 增加 `max_length_continue_retries: int = 3`。

### 验收
- 单测：mock provider 连续 3 次返回 `finish_reason="length"`，断言进入回滚分支。
- 单测：mock provider 第 2 次返回非 length，断言续写 prefix 被正确拼接到 final_response。
- 单测：模型一次返回 `finish_reason="length"` + 截断 tool_call，断言执行单独 retry 而不进消息历史。

---

## P0-4 截断 tool_call 检测（阶段 15.②）

### 现状
- `dispatch(tool_call, context)` 直接拿原始 arguments 调用工具，没有 JSON 校验。
- 模型返回半截 JSON（`{"path": "/etc/pas`）会被原样喂给 tool runtime，工具炸出异常 → 模型把它当真错误 → 怪圈。

### 期望

在 `_append_assistant_tool_message` 之前插入 tool-call 校验阶段，按以下顺序处理：

1. **类型规整**：`arguments` 是 dict/list 时 `json.dumps`；非 str 时 `str()`；空字符串视为 `"{}"`。
2. **JSON 解析**：`json.loads(arguments)`；解析失败时记录到 `invalid_json_args` 列表。
3. **截断启发式**：参数 strip 后未以 `}` / `]` 结尾 → 视为 length 截断 → 走 P0-3 的回滚分支，不进 dispatch。
4. **JSON 错误 retry**：解析失败但未截断时，重发当前请求最多 3 次；用尽后回写**真实的 tool 角色 error 消息**让模型自纠（注意保持 role 轮替：先 append assistant_msg with tool_calls，再 append tool error）。

### 数据结构变更
- `ExitReason.TOOL_CALL_TRUNCATED`。
- `EngineResult.metadata` 增加 `invalid_json_retry_count: int`。

### 验收
- 单测：mock 模型返回 args=`{"path": "/etc/pas`，断言被识别为截断、不进 dispatch、走回滚。
- 单测：mock 模型返回 args=`{"path: "/foo"}`（缺引号但有 `}`），断言走 JSON retry 路径。
- 单测：3 次 JSON 错误后断言注入 tool error 消息且 role 轮替正确。

---

## P0-5 工具名校验与模糊修复（阶段 15.①）

### 现状
```292:306:backend/harness/aether/agents/core/agent.py
                                try:
                                    result = self.services.tool_registry.dispatch(tool_call, context)
                                except UnknownToolError:
                                    if self.config.fail_on_unknown_tool:
                                        error_text = f"Unknown tool: {tool_call.name}"
                                        exit_reason = ExitReason.UNKNOWN_TOOL
                                        state_machine.transition(LoopState.FAILED)
                                        tool_failed = True
                                        break
                                    result = ToolResult(
                                        tool_call_id=tool_call.id,
                                        name=tool_call.name,
                                        content=f"Unknown tool: {tool_call.name}",
                                        is_error=True,
                                    )
```
- `fail_on_unknown_tool=True` 时直接整轮失败；`False` 时只回一条 ToolResult 但模型可能继续重复同样错误。

### 期望

1. **模糊修复**：写一个 `_repair_tool_call(name) -> str | None`，按以下规则尝试：
   - 大小写归一（`readFile → read_file`）
   - 下划线/连字符互换（`read-file → read_file`）
   - 去掉常见前缀（`tool_read_file → read_file`）
   - Levenshtein 距离 ≤ 2 的最近匹配
   - 全部命中后 log warning + 替换 tool_call.name + 继续 dispatch
2. **不存在工具回写**：仍然不存在时，append assistant_msg + 一条 `role=tool` 消息说明 `Available tools: ...`，并在 `_invalid_tool_retries` 累加。
3. **3 次后失败退出**，返回 `partial=True`。

### 数据结构变更
- `ExitReason.INVALID_TOOL_REPEATED`。
- `EngineConfig.invalid_tool_max_retries: int = 3`（已存在 `fail_on_unknown_tool`，改为优先级更低的开关）。

### 验收
- 单测：mock 模型返回 name=`readFile`，断言被修复为 `read_file` 并执行。
- 单测：name=`reed_file`（拼写错误，距离 1），断言被修复为 `read_file`。
- 单测：3 次返回不存在工具名，断言 partial 退出。

---

## P0-6 工具调用上限与去重（阶段 15.③④）

### 现状
- 没有递归 subagent 数量上限：模型一次 `delegate_task` × 10 会真的 fan-out 10 个子代理。
- 没有去重：模型在同一轮 `read_file({"path":"/a"})` × 5 会真执行 5 次。

### 期望

1. **`_cap_delegate_task_calls`**：
   - 配置项 `EngineConfig.max_delegate_calls_per_turn: int = 4`。
   - 超出部分在 dispatch 之前替换为 `ToolResult(is_error=True, content="Skipped: exceeds delegate cap")`。

2. **`_deduplicate_tool_calls`**：
   - 按 `(name, sorted_keys_json(arguments))` 去重。
   - 重复项替换为指向首次结果的 stub `ToolResult`。

### 验收
- 单测：mock 模型返回 5 个 `delegate_task` call，断言只前 4 个被实际派发。
- 单测：mock 模型返回 3 个完全相同的 `read_file` call，断言只执行 1 次。

---

## P0-7 错误分类器 + Provider Fallback 链（阶段 12.7、12.8、12.10、12.11、12.13）

### 现状
- 任何异常都走 `_handle_pipeline_error → _provider_error_retries += 1` → 没救场就失败。
- `EngineServices.provider` 是单一的，不可热切换。

### 期望

1. **`ErrorClassifier`**（新文件 `backend/harness/aether/runtime/error_classifier.py`）：
   - 输入：`exception, provider, model, approx_tokens, context_length, num_messages`
   - 输出：`ClassifiedError(reason: FailoverReason, status_code, retryable, should_compress, should_rotate_credential, should_fallback)`
   - 至少识别：`rate_limit / billing / context_overflow / payload_too_large / long_context_tier / thinking_signature / overloaded / network_transient / auth_refreshable / unknown`

2. **`FallbackChain`**（新文件 `backend/harness/aether/runtime/fallback_chain.py`）：
   - 接受 `list[ProviderFactory]`，提供 `activate_next() -> bool`。
   - `EngineServices.provider` 改为 `FallbackChain.current_provider`。

3. **在 LLM_CALL 异常分支按 reason 分流**（顺序很关键）：
   ```
   classified = ErrorClassifier.classify(exc, ...)
   if classified.reason == rate_limit and chain.has_next():
       chain.activate_next(); continue
   if classified.reason == long_context_tier:
       compressor.cap(200_000); compressor.compress(); restart_compressed = True; break
   if classified.reason == payload_too_large:
       compress_loop(max=3); break or fail
   if classified.reason == context_overflow:
       step_down_or_compress(); break or fail
   if classified.reason == thinking_signature:
       strip_reasoning_details(messages); continue
   ...
   if classified.is_local_validation_error:
       fail_immediately
   else:
       jittered_backoff_then_retry
   ```

4. **保留 `Retry-After` 解析**：HTTP 429 响应头优先于 jitter backoff（`min(int(header), 120)`）。

5. **可中断 sleep**：retry 等待用 200ms 步长循环并定期检查 `interrupt_controller`。

### 数据结构变更
- `FailoverReason` 枚举（与 Hermes 一致命名）。
- `ExitReason` 增加 `RATE_LIMITED / CONTEXT_EXHAUSTED / PAYLOAD_TOO_LARGE / FALLBACK_EXHAUSTED`。

### 验收
- 单测：每种 `FailoverReason` 至少 1 个 mock exception → 校验分类正确。
- 集成测：fallback 链 2 个 provider，第一个返回 429，断言自动切到第二个。
- 集成测：context_overflow 错误 + 启用压缩，断言进入压缩→重发循环。
- 中断测：retry 等待中调用 `engine.interrupt(session_id)`，断言 200ms 内退出且 `EngineResult.status==INTERRUPTED`。

---

## P0-8 空响应/Thinking-only 9 步降级（阶段 16）

### 现状
```360:372:backend/harness/aether/agents/core/agent.py
                    if final_response:
                        self._empty_response_retries = 0
                        exit_reason = ExitReason.TEXT_RESPONSE
                    else:
                        self._empty_response_retries += 1
                        exit_reason = ExitReason.EMPTY_RESPONSE
                    state_machine.transition(LoopState.FINALIZE)
                    break
```
- `_empty_response_retries` 累加但**没有任何代码读它来决定是否继续 retry**。
- 直接进入 FINALIZE 退出，对 reasoning 模型几乎必失败。

### 期望

按 Hermes 13588–13845 实现 9 步降级（顺序严格）：

1. **partial stream recovery**：
   - 维护 `context.metadata["streamed_assistant_text"]` 由 `_build_stream_callback` 累计。
   - 当 final_response 为空但 `streamed_assistant_text` 已经累计了非空内容（去掉 `<think>` 后），直接当最终响应。
2. **上轮 housekeeping 内容 fallback**：
   - 维护 `context.metadata["last_content_with_tools"]` 与 `last_content_tools_all_housekeeping`。
   - housekeeping 工具集合：`{"memory", "todo", "skill_manage", "session_search"}`（可配置）。
   - 满足条件时直接拿上轮内容当最终响应。
3. **post-tool empty nudge**（最多 1 次）：
   - 检测到上 5 条消息里有 `role=tool`，注入：
     - assistant `(empty)` 占位 → user `[System: You just executed tool calls but returned an empty response. Please continue.]`
4. **thinking prefill**（最多 2 次）：
   - 检测到响应有 `reasoning_*` 字段或内容里有 `<think>`/`<reasoning>` 但无 visible content。
   - 把当前 assistant 消息（标记 `_thinking_prefill=True`）追加进历史，`continue`。
   - 后续成功响应时用 `pop_thinking_prefill_messages()` 清理。
5. **空响应 retry**（最多 3 次）：直接 `continue` 重发。
6. **fallback provider**（如果配置了）：`chain.activate_next() then continue`。
7. **终态 `(empty)`**：写入 history `(empty)`，break，返回 `EMPTY_RESPONSE`。
8. **Codex intermediate ack**（最多 2 次，仅 codex_responses 模式）：
   - 检测器：响应内容看起来像"好的，我马上去做"但工具未调用。
   - 注入 system 续写消息让模型推进。
9. **截断 prefix 拼接**：把阶段 10 留下的 `truncated_response_prefix` 拼到 `final_response` 头部；strip `<think>` 入历史。

### 数据结构变更
- `TurnContext.metadata` 标准化字段：
  - `streamed_assistant_text: str`
  - `last_content_with_tools: str`
  - `last_content_tools_all_housekeeping: bool`
  - `post_tool_empty_retried: bool`
  - `thinking_prefill_retries: int`
  - `empty_response_retries: int`（搬离 `self.`，参照 06 文档）

### 验收
- 单测：模型连续 3 次返回空 + reasoning 字段，断言进入 thinking prefill；第 4 次拿到非空，断言成功完成。
- 单测：mock 流式回调累计了 100 字符再断流，最终 final_response 取自 `streamed_assistant_text`。
- 单测：上轮 housekeeping fallback 路径完整。

---

## P0 总览：依赖关系

```
P0-1 (流式)  ─┐
P0-2 (校验)  ─┼─→ P0-3 (length 续写) ─┐
              │                       ├─→ P0-7 (错误分类 + fallback) ─→ P0-8 (空响应 9 步降级)
              └─→ P0-4 (截断 tool_call) ─┘
                                       │
P0-5 (名字模糊修复) ─→ P0-6 (上限 + 去重) ─┘
```

按这个依赖图，Sprint 1 应该聚焦 **P0-1、P0-2、P0-3、P0-4**（流式 + 截断处理），
Sprint 2 聚焦 **P0-5、P0-6、P0-7**（错误分类 + tool 容错），
Sprint 4 聚焦 **P0-8**（空响应 9 步降级，因为它依赖 P0-7 的 fallback chain）。

下一步：阅读 [03_p1_robustness_gaps.md](./03_p1_robustness_gaps.md)。
