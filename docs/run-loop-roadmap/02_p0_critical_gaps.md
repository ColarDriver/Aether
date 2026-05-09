# P0 缺口（不修就坏）

> 范围：阶段 8、9、10、12.7、12.8、12.10、12.11、12.13、15.①②③④、16.全部
>
> 这一档对应"任何稍有规模的真实使用都会触发的失败"。当前 Aether 的 `run_loop`
> 在这些场景下返回的不是"降级体验"，而是直接失败。

---

## P0-1 流式健康检查（阶段 8）

### ✅ 已完成（Sprint 1 / PR 1.1）

`OpenAICompatibleModel.generate` 现在按 `stream_callback` 是否提供分流：
有 callback 时走 `_streaming_generate`（SSE 解析 + 90s stale-stream watchdog
→ 抛 `StreamStallError`，is_network_error=True），失败一次后置位
`_disable_streaming` 并自动降级为非流式重试一次；`EngineConfig.streaming_enabled`
提供紧急回滚开关。`_parse_sse_stream` 模块级函数支持单元测试，`stream_options.include_usage`
默认开启以保留 token 计费。覆盖测试见
`tests/test_streaming_generate.py` 与 `tests/test_streaming_engine_gate.py`。

### 现状（已修复，存档原始描述）
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

### ✅ 已完成基础（Sprint 1 / PR 1.1） + ⚠️ 仍待 Sprint 2 的 eager fallback

`ModelProvider.validate_response()` 已成为 provider 抽象的一部分（默认实现返回
`(True, [])`，零成本 opt-out）；`OpenAICompatibleModel` 默认实现识别 OpenRouter
风格的 `error` 信封 + 空 `choices`。引擎在 LLM_CALL 之后立即校验，失败抛
`ResponseInvalidError`（继承自 `ProviderInvocationError`，`is_network_error=True`）
走 Sprint 0 的 recovery 层；用尽预算后落到 `ExitReason.RESPONSE_INVALID` 终态，
与 `PROVIDER_ERROR` 区分。eager fallback 仍走 Sprint 2 的 `FallbackChain`。
覆盖测试见 `tests/test_response_validation.py`。

### 现状（部分修复，存档原始描述）
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

### ✅ 已完成（Sprint 1 / PR 1.2）

`AgentEngine._handle_length_finish_reason` 在每次 LLM_CALL 之后命中
`finish_reason == "length"` 时分流：
1. **Thinking-budget 检测**：`_looks_like_thinking_only_length_response` →
   friendly exit（`ExitReason.LENGTH_EXHAUSTED` + `length_exit_reason="thinking_budget"`）。
2. **续写 retry**（最多 `max_length_continue_retries`=3 次）：把当前 assistant
   消息（带 `finish_reason="length"` 与 `partial=True` 元数据）+ 续写指令 user 消息
   写入历史；通过 `context.metadata["_ephemeral_max_output_tokens"] = base × (n+1)`
   抬高输出预算（上限 32k），由 `_invoke_provider_with_recovery` 在下次发包时拷贝
   到一次性 `ModelCallConfig` 而不污染原 `EngineRequest`。可见前缀累积到
   `truncated_response_prefix_parts`。
3. **回滚**：续写用尽后 `_get_messages_up_to_last_assistant` 剥离脚手架并附
   `partial=True` 助手消息，落到 `ExitReason.LENGTH_EXHAUSTED`。
4. **`length_continuation_enabled=False`** 紧急回滚开关。

`ExitReason.LENGTH_RECOVERED` / `LENGTH_EXHAUSTED` 已写入 contracts。
截断 tool_call 走 P0-4 的独立路径。覆盖测试见
`tests/test_run_loop_length_continuation.py`。

### 现状（已修复，存档原始描述）
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

### ✅ 已完成（Sprint 1 / PR 1.3）

`AgentEngine._validate_tool_call_arguments` 在 dispatch 之前对每个 `tool_call.arguments`
按 hermes 13335-13426 的顺序处理：
1. **类型规整**：dict/list 直接 pass-through 当作已解析；非 str 标量 `str()`；
   空串/纯空格 → `{}`。处理后 dispatch 看到的永远是规范化后的 `dict`/`list`。
2. **JSON 解析 + 错误收集**：`json.loads` 成功则 store 解码后形态；失败则记录
   `(name, error_msg)` 进 `invalid_json_args`。
3. **截断启发式**（`_detect_truncated_tool_call`，纯静态方法）：任一失败的
   args strip 后未以 `}`/`]` 结尾 → 整批视为截断。能命中 router 把
   `finish_reason` 从 `"length"` 改写成 `"tool_calls"` 的 case。
4. **截断分流**：仍有预算时 retry（不进 dispatch、不进历史）；用尽
   `max_truncated_tool_call_retries`(=1) 后回滚到上一完整 assistant turn，落到
   `ExitReason.TOOL_CALL_TRUNCATED` + `partial=True`。
5. **JSON 错误分流**：未截断 → 静默 retry `max_invalid_json_retries`(=3) 次；
   用尽后注入 assistant 消息（带 `_invalid_json_recovery=True` 标记）+
   每个失败 tool_call 一条 `role=tool` 错误说明（保持 OpenAI 严格的角色轮替），
   让模型自纠（hermes 13396-13423 的复刻）。`invalid_json_recovery_enabled=False`
   时回到 fail-fast。

`run_loop` 在 `finish_reason="length" && tool_calls` 命中时走
`_handle_length_with_tool_calls`（重试一次同样不进历史，超额后落到
`TOOL_CALL_TRUNCATED`）。`EngineResult.metadata.runtime` 暴露 `truncated_tool_call_retries`
与 `invalid_json_retries` 计数。`truncated_tool_call_detection_enabled=False` 紧急
回滚开关。覆盖测试见 `tests/test_run_loop_truncated_tool_call.py`（13 例：
`_detect_truncated_tool_call` 单元 + 7 个 run-loop 端到端场景）。

### 现状（已修复，存档原始描述）
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

### ✅ 已完成（Sprint 2 / PR 2.3）

`backend/harness/aether/agents/core/tool_hardening.py` 提供 `repair_tool_name`
四阶段管道：exact → prefix-strip（`tool_` / `functions_` / `mcp__server__`）→
`_normalize_name` 命中（与 Sprint 1.5 phantom-tool 共享 helper）→ Levenshtein
≤ 2 模糊匹配，tied 距离时拒绝以免误派发。`prepare_tool_calls` 把不可修复的
名字替换为 `synthetic_result`（含 `Available tools: …` 自纠提示），并维护
`TURN_KEY_INVALID_TOOL_RETRIES` 计数器；达到 `EngineConfig.invalid_tool_max_retries`
后 `agent.run_loop` 终止当前 turn 为 `ExitReason.INVALID_TOOL_REPEATED`
（partial=True）。覆盖测试见 `tests/test_tool_hardening.py`（24 例）。

### 现状（已修复，存档原始描述）
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

### ✅ 已完成（Sprint 2 / PR 2.3）

`tool_hardening.prepare_tool_calls` 在派发循环前依次执行 cap → dedup：
- **Cap**：把 `_normalize_name(call.name)` 命中
  `EngineConfig.delegate_tool_names` 的调用纳入 `TURN_KEY_DELEGATE_CALLS`
  per-turn 计数；超过 `max_delegate_calls_per_turn (默认 4)` 的部分替换为
  `is_error=True` 的 stub `ToolResult`，告诉模型合并工作。
- **Dedup**：用 `(_normalize_name(name), json.dumps(args, sort_keys=True))`
  作为 key（顺序无关），首次派发，后续同 key 的调用替换为 `is_error=False`
  的 stub 指向首次 call_id。受 `EngineConfig.tool_dedup_enabled` 控制。

覆盖测试 `tests/test_tool_hardening.py`：5 个 `delegate_task` cap 到 3、3 个相同
`read_file` 去重为 1 实际派发、跨迭代 cap 计数累加、order-independent dedup。

### 现状（已修复，存档原始描述）
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

### ✅ 已完成（Sprint 2 / PR 2.1 + PR 2.2）

- `runtime/error_classifier.py` 提供 `classify_api_error(exc, *, provider, model,
  approx_tokens, context_length, num_messages) -> ClassifiedError`，覆盖
  `auth / billing / rate_limit / overloaded / server_error / timeout /
  context_overflow / payload_too_large / model_not_found / format_error /
  thinking_signature / long_context_tier / response_invalid / stream_stalled /
  unknown` 共 15 种 `FailoverReason`（与 hermes-agent 字符串值 1:1 对齐）。
- `runtime/fallback_chain.py` 提供 `FallbackChain` + `ProviderSlot`，懒加载
  factory、RLock 线程安全、失活 slot 自动跳过、cumulative observability counters。
- `runtime/services.py` 把 `EngineServices.provider` 改为 property：当
  `fallback_chain` 存在时从 `chain.current_provider` 读，单 provider 场景行为
  保持 Sprint 0 不变（向后兼容）。
- `runtime/recovery.py` 的 `ClassifiedRecoveryStrategy` 按 `FailoverReason`
  分发到对应行为（rate_limit + 长 retry-after → fallback；response_invalid →
  立即 fallback；context_overflow / payload_too_large → 压缩 hint；
  thinking_signature → strip-thinking hint）。`Retry-After` 头通过
  `ProviderInvocationError.retry_after_seconds` 直接保留，可中断 sleep 由
  Sprint 0 的 `wait_interruptible` 提供。
- `agent.py` 的 `_invoke_provider_with_recovery` 按 `decision.activate_fallback /
  compress_context / strip_thinking` 三轴正交分发；命中 fallback 时重置
  `attempt_state` 让新 provider 拿到完整 retry 预算；`recovery_terminal_exit_reason`
  metadata 经 `_resolve_terminal_exit_reason` 映射到 `RATE_LIMITED /
  CONTEXT_EXHAUSTED / PAYLOAD_TOO_LARGE / FALLBACK_EXHAUSTED`。
- 回滚开关：`EngineConfig.classified_recovery_enabled: bool = True`、
  `EngineConfig.fallback_chain_enabled: bool = False`（默认关闭，验证后再开）、
  `EngineConfig.max_fallback_activations_per_turn: int = 4`、
  `EngineConfig.rate_limit_fallback_threshold_seconds: float = 30.0`。

覆盖测试：`tests/test_error_classifier.py`（14 例 + 60 fixture，准确率 100%）、
`tests/test_classified_recovery_strategy.py`（14 例）、`tests/test_fallback_chain.py`
（17 例，含端到端 engine 集成）。

### 现状（已修复，存档原始描述）
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

## P0-9 内置工具集 + 散文工具调用合成（Sprint 1.5）

### ✅ 已完成（Sprint 1.5 / 本次 PR）

历史背景：在引入 P0-9 之前，`AgentEngine` 默认 `tool_registry=ToolRegistry()`（**空注册表**），
CLI 也没有给它注册任何工具。模型被告知它"是 agent"，但事实上**没有任何工具可调**——
对于 Kimi 这一类训练时见过工具调用模板的模型，结果就是它把工具调用以散文（markdown bash
fence、`<function=NAME>`、`<functions.shell:0>`、`<invoke>`）的形式写在 `content`
里。原本的 phantom-tool recovery（PR 1.5 早期）通过"注入 corrective user message + 重试"
**让模型重新尝试**，但模型没有真正可用的工具，只能再写一遍 prose，最终在 `max_phantom_tool_retries`
后退出 `PHANTOM_TOOL_INTENT`。用户看到的就是终端里反复出现 `└ attempted: Running command…`
然后 loop 中断。

### 解决方案

1. **`aether/tools/builtins/`** 新增六个内置 executor：`shell` / `read_file` / `write_file` /
   `list_dir` / `grep` / `glob`，对齐 claude-code Day-1 工具面（Bash / Read / Write / LS /
   Grep / Glob）。每个都纯 Python，无外部依赖（`grep` 在有 ripgrep 时优先调用），输出做了字节
   截断保护（shell 16KB/流，read_file 256KB 文件，write_file 1MB content）。
2. **`build_default_tool_registry(cwd=...)`** 工厂在 `tools/builtins/__init__.py`，
   把六个工具一次性注册到一个新的 `ToolRegistry` 并返回。
3. **`AgentEngine.__init__`** 当 `tool_registry is None` 且 `EngineConfig.use_builtin_tools`
   为 `True`（新默认）时，自动调用工厂；否则维持空注册表。这一行就让 SDK 消费者、子 Agent、
   测试 fixture 全部继承同一份默认工具集。
4. **`<tool_use_contract>` 系统提示** —— `agents/core/system_prompt.py:augment_system_with_tool_contract`
   会在用户传入的 system message 之前 prepend 一段中英双语契约，列出注册的工具名并明确禁止
   prose-style emission。该契约从 `tool_registry.list_descriptors()` 派生，注册表为空时
   自动跳过。在 `EngineConfig.tool_use_contract_enabled=True`（默认）时启用。
5. **散文工具合成** —— `phantom_tool.synthesize_tool_calls_from_phantom(intent, registry)`
   把已经解析出的 `PhantomToolIntent` 翻译成结构化 `ToolCall`：
   - bash fence → 注册表中第一个 shell 别名工具（`shell` / `bash` / `execute_command` /
     `run_command` / …）。
   - `<function=NAME>` / `<functions.NAME:N>` / `<invoke>` / `<tool_call>` → 通过
     case-fold + 去 namespace 前缀（`mcp__server__`、`functions.`、`ns:`）解析到注册的工具名。
   - 名字解析失败时返回 `None`，让 corrective-message 回退路径处理。
6. **Run-loop 切入** —— `AgentEngine._maybe_recover_phantom_tool_intent` 在原有"注入
   corrective user message"分支之前先尝试合成；成功时把 `response_to_store.tool_calls`
   注满，回到现有的 dispatch 路径继续 loop（新 helper `_dispatch_synthesized_tool_calls`
   做 streamlined 的 dispatch，跳过 truncated/invalid-JSON 校验，因为参数是我们自己造的）。
   `EngineConfig.phantom_tool_synthesis_enabled` 控制是否启用（默认 `True`），
   `TURN_KEY_PHANTOM_TOOL_SYNTHESIZED` 计数本回合合成的工具调用次数。
7. **UI 处理** —— `cli/ui.py` 把"model emitted inline tool tags"警告从 `end_stream` /
   `render_assistant_block` 推迟到 `end_turn`：合成成功时显示一条柔和的
   `↻ synthesized N tool call(s) from prose`，并把警告抑制掉；只有在合成失败（注册表里没有
   匹配工具）时才仍然展示原本的醒目警告。
8. **CLI 仅暴露 opt-out** —— `cli/main.py` 加了 `--no-builtin-tools` flag，映射到
   `EngineConfig(use_builtin_tools=False)`。CLI 不再 import 工具工厂、不再组装契约；
   它只是 engine 的"前台"。

### 验收

- 单测：`tests/test_builtin_tools.py`（26 用例）覆盖六个工具的 happy/error/truncation/schema。
- 单测：`tests/test_phantom_synthesis.py`（16 用例）覆盖：
  - 每种 phantom 句法（bash / function= / functions.NAME:N / invoke）→ ToolCall 转换。
  - 模糊解析（大小写、namespace 前缀、shell 别名）。
  - typo 不会误解析（`read_files` ≠ `read_file`）。
  - run-loop 集成：合成成功时 `phantom_tool_retries=0` / `phantom_tool_synthesized=1`、
    spy tool 真的被调用、`exit_reason=TEXT_RESPONSE`。
  - run-loop 集成：禁用 synthesis 时回退到 corrective-retry 路径仍然工作。
  - run-loop 集成：注册表里没有 shell 工具时合成跳过、正确转入 corrective 路径。
- 单测：`tests/test_run_loop_phantom_tool.py` 新增两个 case 校验 synthesis 走通时 retries
  仍为 0、metadata 中 `phantom_synth_notes` 包含 per-call 描述。
- 单测：`tests/test_cli_main.py` 校验 `--no-builtin-tools` flag 解析、默认 `EngineConfig`
  把 `use_builtin_tools` / `tool_use_contract_enabled` / `phantom_tool_synthesis_enabled`
  全部置为 `True`。
- 集成（手动）：用 Kimi-class provider 触发 `<functions.shell:0>{...}` 散文输出，断言不再看到
  `└ attempted: Running command…` 半截截断的警告，loop 顺利推进到下一轮。

### 数据结构变更

- `EngineConfig` 新增 `use_builtin_tools` / `tool_use_contract_enabled` /
  `phantom_tool_synthesis_enabled`，默认 `True`。
- `TURN_KEY_PHANTOM_TOOL_SYNTHESIZED` 加入 `runtime/session_runtime.py`。
- `phantom_tool.SynthesisOutcome` 新数据类。
- `EngineResult.metadata["runtime"]["phantom_tool_synthesized"]` 暴露给观察方。
- `EngineResult.metadata["turn"]["phantom_synth_notes"]` 携带 per-call 描述给 UI。

---

## P0 总览：依赖关系

```
P0-1 (流式)  ─┐
P0-2 (校验)  ─┼─→ P0-3 (length 续写) ─┐
              │                       ├─→ P0-7 (错误分类 + fallback) ─→ P0-8 (空响应 9 步降级)
              └─→ P0-4 (截断 tool_call) ─┘
                                       │
P0-5 (名字模糊修复) ─→ P0-6 (上限 + 去重) ─┘
                                       │
                                       └─→ P0-9 (内置工具集 + 散文合成)
```

按这个依赖图，Sprint 1 应该聚焦 **P0-1、P0-2、P0-3、P0-4**（流式 + 截断处理），
Sprint 1.5 加入 **P0-9**（内置工具集 + 散文合成，与 P0-5 的模糊修复语义共享 `_normalize_name`），
Sprint 2 聚焦 **P0-5、P0-6、P0-7**（错误分类 + tool 容错），
Sprint 4 聚焦 **P0-8**（空响应 9 步降级，因为它依赖 P0-7 的 fallback chain）。

下一步：阅读 [03_p1_robustness_gaps.md](./03_p1_robustness_gaps.md)。
