# Sprint 执行路线图

> 把所有缺口压成 7 个独立可合并的 Sprint（含 Sprint 0 的地基修复）。
>
> 每个 Sprint 给出：目标、依赖、文件改动清单、新增单测、验收标准、回滚开关。

---

## Sprint 0 — 地基修复（必做）

### 目标
把"动手前必修的 3 个结构性问题"先解决，为后续所有 Sprint 扫清障碍。

### 涉及
- 结构问题 1：实例属性 → `TurnContext.metadata` + `SessionRuntimeState`
- 结构问题 3：重试控制权上移到引擎层（provider 单次发包）
- 结构问题 5：引入 `RecoveryStrategy` 抽象（替代 middleware 的 on_error 恢复语义）

### 文件改动
- `backend/harness/aether/runtime/contracts.py`
  - `TurnContext` 加注释说明标准 metadata key；可选地引入 `TypedDict` 描述
- `backend/harness/aether/runtime/session_runtime.py`（新文件）
  - `SessionRuntimeState`、按 session_id 分槽的 registry
- `backend/harness/aether/runtime/recovery.py`（新文件）
  - `RecoveryStrategy` Protocol、`RecoveryOutcome`、空实现 `BackoffRetryStrategy`
- `backend/harness/aether/models/provider/openai_compatible.py`
  - `generate()` 改为单次 `httpx.post`，结构化异常 `ProviderInvocationError`
- `backend/harness/aether/agents/core/agent.py`
  - 把 `_memory_nudge_counter / _skill_nudge_counter / _empty_response_retries / _provider_error_retries` 全部搬到 `SessionRuntimeState` 与 `context.metadata`
  - LLM_CALL 异常分支改为：`classified = ErrorClassifier.classify(exc)` →（先用空实现）→ 走 `RecoveryStrategy.first_match(classified).recover(...)`

### 新增测试
- `tests/runtime/test_session_runtime_isolation.py`：双 session 并发计数器隔离
- `tests/runtime/test_provider_invocation_error.py`：provider 单次发包后抛结构化异常
- `tests/runtime/test_recovery_strategy.py`：空策略链下 fallthrough 到 abort

### 验收
- 现有所有单测通过
- 双 session 隔离测试 ≥ 100 次稳定通过

### 回滚开关
无需开关。是纯重构性质 PR；如发现回归直接 revert。

---

## Sprint 1 — 流式 + 截断处理（P0 第一波） ✅ 已完成

> 完成情况：
> - PR 1.1 ✅ — P0-1 流式 SSE + 90s stale-stream 检测 + 自动降级；P0-2 `validate_response` 抽象 + OpenAI 默认实现 + 引擎层校验 + `RESPONSE_INVALID` 终态
> - PR 1.2 ✅ — P0-3 `finish_reason="length"` 续写（×3）+ thinking-budget friendly exit + 回滚到最后完整 assistant turn
> - PR 1.3 ✅ — P0-4 `_detect_truncated_tool_call` 截断启发式 + `_validate_tool_call_arguments` 类型规整 + JSON 错误静默 retry + tool error 自纠注入 + `_handle_length_with_tool_calls`（不进历史 retry 一次）
>
> 所有验收用例通过；测试套件 149 例通过（含 13 例新增）。

### 目标
让 `finish_reason="length"` 不再变成"整轮失败"，让流式回调真的是"流式的"。

### 涉及
- P0-1：流式健康检查 ✅
- P0-2：响应壳校验 ✅（基础落地，eager fallback 走 Sprint 2）
- P0-3：finish_reason="length" 续写与回滚 ✅
- P0-4：截断 tool_call 检测 ✅

### 文件改动
- `backend/harness/aether/models/provider/openai_compatible.py`
  - 增 `_streaming_generate(api_kwargs)`：SSE 解析、stale-stream 90s 检测、on_first_delta 回调
  - 增 `validate_response(raw) -> tuple[bool, list[str]]`
- `backend/harness/aether/models/provider/base.py`
  - 抽象方法 `validate_response`；可选方法 `streaming_generate`
- `backend/harness/aether/runtime/contracts.py`
  - `ExitReason` 增加：`LENGTH_RECOVERED / LENGTH_EXHAUSTED / TOOL_CALL_TRUNCATED`
  - `EngineConfig` 增加：`max_length_continue_retries: int = 3`
- `backend/harness/aether/agents/core/agent.py`
  - `run_loop` 在 LLM_CALL 后插入 `validate_response` 校验
  - 新方法 `_handle_length_finish_reason(...)`：thinking-budget 检测、续写 retry、回滚
  - 新方法 `_detect_truncated_tool_call(tool_calls)`：JSON 末尾启发式
  - 新方法 `_get_messages_up_to_last_assistant(messages)`：回滚助手

### 新增测试
- `tests/agents/core/test_run_loop_length_continuation.py`：
  - 续写 1 次成功
  - 续写 3 次失败 → partial=True
  - thinking-budget 耗尽 → friendly exit
- `tests/agents/core/test_run_loop_truncated_tool_call.py`：
  - 半截 JSON args → 不进 dispatch
  - 截断 tool_call 单独 retry 1 次后失败
- `tests/models/provider/test_streaming_generate.py`：
  - SSE chunk 多次回调
  - stale-stream 90s 抛 `StreamStallError`
  - on_first_delta 在第一个 chunk 上触发

### 验收
- 模型撞 max_tokens 时能完整拿到响应（续写成功路径）
- mock 半截 JSON args 不会进 tool runtime
- stream_callback 真的按 chunk 触发（用真实 OpenAI 调用验证）

### 回滚开关
- `EngineConfig.streaming_enabled: bool = True`：False 时强制走非流式
- `EngineConfig.length_continuation_enabled: bool = True`

---

## Sprint 1.5 — 内置工具集 + 散文工具调用合成（P0-9） ✅ 已完成

> 计划外补丁：Sprint 1 期间用户跑 Kimi 模型时反复看到 `└ attempted: Running command…`
> 然后 loop 中断，溯源发现根因是 `AgentEngine` 默认空注册表 + 模型把工具调用写在 prose 里。
> 这一档把"内置工具集 + phantom 合成"补齐，让每个 engine 默认都有工具可用、并能从散文修复
> 回结构化 `tool_calls`。

### 目标
- 把 `shell` / `read_file` / `write_file` / `list_dir` / `grep` / `glob` 六个工具默认装上
  `AgentEngine`，让"engine = 没有工具的 agent"这一长期失配消失。
- 当模型输出 ` ```bash …`、`<function=NAME>`、`<functions.shell:N>`、`<invoke>` 等
  prose-style 工具调用时，**就地把它们合成为结构化 `ToolCall`** 并 dispatch，避免循环中断。
- 把"工具组合 + tool-use 契约提示"放在 engine 骨架里，CLI / SDK / 子 agent 一致继承。

### 涉及
- P0-9：内置工具集 + 散文工具调用合成（新章节，见 02 文档）

### 文件改动
- `backend/harness/aether/tools/builtins/`（新目录）
  - `shell.py` / `read_file.py` / `write_file.py` / `list_dir.py` / `grep.py` / `glob.py`
  - `__init__.py`：`build_default_tool_registry(*, cwd: Path | None = None)` 工厂
- `backend/harness/aether/agents/core/system_prompt.py`（新文件）
  - `augment_system_with_tool_contract(system, descriptors)` 在 user system message 之前
    prepend `<tool_use_contract>` 块
- `backend/harness/aether/agents/core/phantom_tool.py`
  - 新 `_FUNCTIONS_SLOT_TAG_RE`、扩展 `<function=NAME>` body 支持 JSON 解析
  - 新 `SynthesisOutcome` dataclass + `synthesize_tool_calls_from_phantom(intent, registry)`
  - `_resolve_registered` / `_resolve_shell_tool` / `_normalize_name` / `_SHELL_ALIASES` 工具
- `backend/harness/aether/agents/core/agent.py`
  - `__init__` 默认调 `build_default_tool_registry()`，gate by `EngineConfig.use_builtin_tools`
  - `_prepare_session_and_system_prompt` 通过 `augment_system_with_tool_contract` 注入契约
  - `_maybe_recover_phantom_tool_intent` 加新 `"synthesized"` 出口
  - 新 `_dispatch_synthesized_tool_calls` helper：streamlined dispatch（跳过 truncated /
    invalid-JSON 校验，因为参数是引擎自己造的）
  - run-loop 新分支：`phantom_outcome == "synthesized"` 时走 helper、`continue`
- `backend/harness/aether/runtime/session_runtime.py`
  - 新 `TURN_KEY_PHANTOM_TOOL_SYNTHESIZED`
- `backend/harness/aether/config/schema.py`
  - `EngineConfig`：`use_builtin_tools: bool = True`、`tool_use_contract_enabled: bool = True`、
    `phantom_tool_synthesis_enabled: bool = True`
- `backend/harness/aether/cli/main.py`
  - 新 `--no-builtin-tools` flag → `EngineConfig(use_builtin_tools=False)`
- `backend/harness/aether/cli/ui.py`
  - phantom warning 推迟到 `end_turn`；新 `_render_phantom_synth_note` 渲染柔和的
    `↻ synthesized N tool call(s) from prose`
- `backend/harness/aether/cli/repl.py`
  - 从 `result.metadata` 取 `phantom_synth_count` / `phantom_synth_notes` 传给 `end_turn`

### 新增测试
- `tests/test_builtin_tools.py`（26 用例）：六工具的 happy / error / 输出截断 / schema 校验
- `tests/test_phantom_synthesis.py`（16 用例）：合成函数 + run-loop 集成；含 fuzzy 解析、
  typo 拒绝、无 shell 工具时跳过、disable 时回退到 corrective
- `tests/test_run_loop_phantom_tool.py`：新增 2 用例校验 synthesis 走通时 retries=0、
  metadata 暴露 `phantom_synth_notes`
- `tests/test_cli_main.py`（新文件）：`--no-builtin-tools` flag 解析、默认 EngineConfig flags

### 验收
- 现有 239 + 新增 49 = 288 用例通过
- Kimi-class 模型输出 `<functions.shell:0>{...}` 时不再看到截断警告，loop 顺利推进
- 子 agent / SDK 消费者 / 测试 fixture 自动继承六个内置工具（不需要任何 CLI 端改动）

### 回滚开关
- `EngineConfig.use_builtin_tools: bool = True`：False 时引擎退回空注册表
- `EngineConfig.tool_use_contract_enabled: bool = True`：False 时不注入系统提示契约
- `EngineConfig.phantom_tool_synthesis_enabled: bool = True`：False 时回到 corrective-message
  retry 路径
- CLI: `--no-builtin-tools` 等价于上面三项 + 兼容旧测试 fixture

---

## Sprint 2 — 错误分类器 + Provider Fallback 链（P0 第二波） ✅ 已完成

> 完成情况（branch `feat/engine-error-recovery-and-tool-hardening`）：
> - PR 2.1 ✅ — `runtime/error_classifier.py` + 60-fixture 测试套件，分类准确率 100%（验收要求 ≥ 95%）
> - PR 2.2 ✅ — `runtime/fallback_chain.py` + `ClassifiedRecoveryStrategy`（recovery.py 扩展）+ `EngineServices.provider` 改为 property + `agent.py` 接入；新增 31 例测试
> - PR 2.3 ✅ — `agents/core/tool_hardening.py`（`prepare_tool_calls` / `repair_tool_name`）+ `agent.py` 集成；新增 24 例测试
>
> 测试套件 302 → 357（+55 新增）全部通过。

### 目标
让任何错误都有明确的恢复路径，让 rate limit 时能自动切到下一家 provider。

### 涉及
- P0-5：工具名模糊修复 ✅
- P0-6：工具调用上限 + 去重 ✅
- P0-7：错误分类器 + Fallback 链 ✅
- P0 中阶段 12.7、12.8、12.10、12.11、12.13 已具体实现

### 文件改动
- `backend/harness/aether/runtime/error_classifier.py`（新文件 ✅）
  - `FailoverReason` 枚举（与 hermes-agent 字符串值 1:1 对齐，便于跨引擎日志关联）
  - `ClassifiedError` dataclass（4 个正交 hint：`retryable / should_compress / should_rotate_credential / should_fallback`）
  - `classify_api_error(exc, *, provider, model, approx_tokens, context_length, num_messages) -> ClassifiedError`
- `backend/harness/aether/runtime/fallback_chain.py`（新文件 ✅）
  - `FallbackChain`、`ProviderSlot`、`ProviderFactory`、`FallbackChainExhausted`
  - 懒加载 factory + RLock 线程安全 + 失活 slot 跳过 + cumulative observability counters
- `backend/harness/aether/runtime/recovery.py`（扩展 ✅）
  - `RecoveryDecision` 增加 `activate_fallback / compress_context / strip_thinking / classified_reason` 字段
  - 新 `ClassifiedRecoveryStrategy`：dispatch table 按 `FailoverReason` 决定行为（rate_limit / response_invalid / context_overflow / payload_too_large / thinking_signature / overloaded / server_error / timeout / unknown / hard-stops）
  - 移除 `GenericBackoffStrategy._is_retriable` 中针对 `ResponseInvalidError` 的 Sprint 1 stop-gap
- `backend/harness/aether/runtime/services.py`（重写 ✅）
  - `EngineServices` 改为非 dataclass + `__slots__`（dataclass `slots=True` 与 `@property` 冲突）
  - `provider` 改为 property：当 `fallback_chain` 存在时从 `chain.current_provider` 读，否则回落 constructor-time 实例
- `backend/harness/aether/runtime/contracts.py`（扩展 ✅）
  - `ExitReason` 新增：`RATE_LIMITED / CONTEXT_EXHAUSTED / PAYLOAD_TOO_LARGE / FALLBACK_EXHAUSTED / INVALID_TOOL_REPEATED`
- `backend/harness/aether/config/schema.py`（扩展 ✅）
  - `EngineConfig`：`classified_recovery_enabled: bool = True`、`fallback_chain_enabled: bool = False`、`max_fallback_activations_per_turn: int = 4`、`rate_limit_fallback_threshold_seconds: float = 30.0`、`invalid_tool_max_retries: int = 3`、`max_delegate_calls_per_turn: int = 4`、`delegate_tool_names: tuple[str, ...]`、`tool_dedup_enabled: bool = True`
- `backend/harness/aether/agents/core/agent.py`（扩展 ✅）
  - 默认注入 `ClassifiedRecoveryStrategy`（受 `classified_recovery_enabled` 控制）
  - `_invoke_provider_with_recovery` 按 `decision.activate_fallback / compress_context / strip_thinking` 三轴正交分发；命中 fallback 时重置 `attempt_state` 让新 provider 拿到完整 retry 预算
  - 新 `_resolve_terminal_exit_reason` helper：把 `recovery_terminal_exit_reason` metadata 映射到正确的 `ExitReason`
  - 工具派发循环前调用 `prepare_tool_calls`（修复 + 上限 + 去重一站式），按 `PreparedToolCall.synthetic_result` 决定派发或注入合成结果
- `backend/harness/aether/agents/core/tool_hardening.py`（新文件 ✅）
  - `repair_tool_name`：四阶段管道（exact → prefix-strip → normalise → Levenshtein ≤ 2，tied 时拒绝）
  - `prepare_tool_calls`：repair → cap → dedup 顺序执行；返回 `ToolDispatchPlan`（含 observability 计数器 + 可选 `exit_reason`）

### 新增测试
- `tests/test_error_classifier.py`（14 用例 + 60 fixture 子用例 ✅）：每种 `FailoverReason` 至少 3 个真实 fixture，外加 helper extractor / metadata / 命名契约测试
- `tests/test_fallback_chain.py`（17 用例 ✅）：链原语单测 + `EngineServices.provider` property 行为 + 端到端 429 → 切换、`ResponseInvalidError` 立即切换、链耗尽 → `FALLBACK_EXHAUSTED`
- `tests/test_classified_recovery_strategy.py`（14 用例 ✅）：每个 `FailoverReason` 分支独立断言（rate_limit / response_invalid / stream_stalled / context_overflow / payload_too_large / thinking_signature / 5 个 hard-stop / 3 个 backoff / budget exhaustion / unknown）
- `tests/test_tool_hardening.py`（24 用例 ✅）：`repair_tool_name` 全 4 阶段 + tied refusal；`prepare_tool_calls` cap / dedup / counter / order-independence；3 个端到端 engine 集成

### 验收 ✅
- 模拟 429 自动切 fallback 在毫秒级完成（`test_429_rotates_to_fallback_provider_and_succeeds`）
- ErrorClassifier 在 60 个真实异常 fixture 上分类准确率 100%（验收要求 ≥ 95%）
- 5 个 delegate cap 到 4 个、3 个相同 read 去重为 1 实际派发
- `readFile` / `reed_file` / `tool_read_file` 等典型 typo 自动修复为 `read_file`
- 测试套件 357 例全部通过（无回归）

### 回滚开关
- `EngineConfig.fallback_chain_enabled: bool = False`：默认关闭，生产 chain 配置完毕后再开
- `EngineConfig.classified_recovery_enabled: bool = True`：紧急回退到 Sprint 0 `GenericBackoffStrategy`
- `EngineConfig.tool_dedup_enabled: bool = True`：False 时跳过 dedup 阶段保留原派发顺序
- `EngineConfig.max_delegate_calls_per_turn: int`：设 0 禁用 delegate 派发；设 999 等价于不限制

---

## Sprint 3 — 上下文压缩 + Iteration Budget + Token 累计（P1 第一波）

### 目标
长会话不再因 413 / 上下文超限失败；用户能从 result 拿到完整 token / cost；预算用尽时给 summary 而非沉默。

### 涉及
- P1-1：ContextCompressor
- P1-2：IterationBudget
- P1-3：max_iterations summary 兜底
- P1-4：Token / Cost 累计
- P1-11：EngineResult 字段扩展

### 文件改动
- `backend/harness/aether/runtime/context_compressor.py`（新文件）
  - `ContextCompressor` Protocol、`SummarizingCompressor` 默认实现
- `backend/harness/aether/runtime/iteration_budget.py`（新文件）
- `backend/harness/aether/runtime/usage.py`（新文件）
  - `CanonicalUsage`、`normalize_usage(...)`
- `backend/harness/aether/runtime/contracts.py`
  - `EngineConfig`：`compression_enabled / compression_threshold_pct / compression_max_passes / cheap_tool_names`
  - `EngineResult.metadata` 标准字段约定
- `backend/harness/aether/agents/core/agent.py`
  - 入口 preflight 压缩
  - LLM_CALL 异常分支按 `payload_too_large / context_overflow` 触发压缩循环
  - tool 执行后阈值压缩
  - max_iterations 兜底 summary：`_handle_max_iterations(messages) -> str`

### 新增测试
- `tests/runtime/test_context_compressor.py`：100 → 12 条
- `tests/runtime/test_iteration_budget.py`：consume / refund / grace
- `tests/runtime/test_usage_normalize.py`：anthropic / openai / codex 三种 usage 形式
- `tests/agents/core/test_max_iterations_summary.py`：用尽预算后 summary 文本进 final_response
- `tests/agents/core/test_preflight_compression.py`：长 messages 进入 run_loop 自动压缩

### 验收
- 在一个 50 轮真实长会话上 token 用量减少 ≥ 60%
- max_iterations 触发时 final_response 非空且有意义

### 回滚开关
- `EngineConfig.compression_enabled: bool = False`：默认关
- `EngineConfig.summary_on_budget_exhausted: bool = True`

---

## Sprint 4 — 空响应 9 步降级 + Tool 容错完善（P0 收尾 + P1）

### 目标
让 reasoning 模型（GLM-5/QwQ/DeepSeek-R1）能稳定跑完一轮；让 JSON 错误能被模型自纠。

### 涉及
- P0-8：空响应 / Thinking-only 9 步降级
- P0-4 收尾：JSON 错误 retry + 注入 tool error 让模型自纠
- P1-8：Codex incomplete 续写
- 结构问题 2 收尾：`_empty_response_retries` 移到 metadata

### 文件改动
- `backend/harness/aether/agents/core/agent.py`
  - 新方法 `_handle_empty_response(...)`：9 步降级（按 02 文档顺序）
  - tool 路径中的 JSON 错误 retry + tool error 注入（构造 role=tool 的修复消息）
  - housekeeping vs substantive 工具区分（默认集合：`{memory, todo, skill_manage, session_search}`）
- `backend/harness/aether/runtime/contracts.py`
  - `ExitReason` 加 `PARTIAL_STREAM_RECOVERY / FALLBACK_PRIOR_TURN_CONTENT / POST_TOOL_NUDGE`
  - `EngineConfig.housekeeping_tool_names: frozenset[str]`
  - `EngineConfig.thinking_prefill_max_retries: int = 2`
  - `EngineConfig.empty_response_max_retries: int = 3`

### 新增测试
- `tests/agents/core/test_empty_response_9_step_degradation.py`：
  - 每个分支至少 1 个用例
  - partial stream recovery 路径
  - thinking prefill 2 次后 fallback
  - housekeeping fallback 命中
- `tests/agents/core/test_invalid_json_recovery.py`：
  - 3 次 JSON 错误后注入 tool error；模型下一轮自纠成功

### 验收
- 在 GLM-5 / DeepSeek-R1 上跑一个长 turn，整轮不会因为某次空响应失败

### 回滚开关
- `EngineConfig.empty_response_recovery_enabled: bool = True`

---

## Sprint 5 — Prompt Cache + Reasoning 续传 + 增量持久化 + 诊断（P1 第二波）

### 目标
Anthropic prompt cache 命中率 ≥ 70%；Ctrl+C 不丢工作进度；turn-end 日志能告诉运维"为什么停了"。

### 涉及
- P1-7：MessageBuilder（reasoning 续传 + 孤儿 tool 补桩 + JSON 规整 + Anthropic prompt cache）
- P1-9：增量 session 持久化
- P1-10：扩 ExitReason + 结构化 Turn-end 日志
- 结构问题 4：System prompt 写入策略
- 结构问题 6：ProviderProfile

### 文件改动
- `backend/harness/aether/runtime/message_builder.py`（新文件）
- `backend/harness/aether/models/provider/profile.py`（新文件）
  - `ProviderProfile` 与默认 profile 表
- `backend/harness/aether/agents/core/agent.py`
  - 入口前调用 `MessageBuilder.build_api_messages(...)`
  - tool 路径每轮调 `session_store.persist_partial(...)`
  - `_build_result` 增加结构化 `Turn ended:` 日志（last_msg_role=tool 时 WARNING）
  - System prompt 写入策略：仅新 session 或 `force_system_prompt_replace=True` 时覆盖
- `backend/harness/aether/runtime/session_store.py`
  - 增 `persist_partial(session_id, messages)` 接口
- `backend/harness/aether/runtime/contracts.py`
  - 扩 ExitReason 至 ≥ 16 项
  - `EngineRequest.force_system_prompt_replace: bool = False`

### 新增测试
- `tests/runtime/test_message_builder_reasoning.py`：
  - anthropic profile 输出 reasoning_details
  - openai profile 输出 reasoning_content
  - strict profile 删除两者
- `tests/runtime/test_message_builder_orphan_tool.py`：缺失 tool 结果时插入桩
- `tests/agents/core/test_session_prompt_byte_stable.py`：5 轮调用 system 字节完全相同
- `tests/agents/core/test_partial_persist.py`：tool 中断后 session_store 含完整前序

### 验收
- 真实 Claude 调用 5 轮后 cache_read_tokens ≥ 70% prompt_tokens
- 中断测试：第 3 个 tool 完后 interrupt，重启后能从 session_store 恢复

### 回滚开关
- `EngineConfig.persist_after_each_tool: bool = True`
- `EngineConfig.use_message_builder: bool = True`

---

## Sprint 6 — 多 session 安全 + 凭证池 + 后台 review（P2 + P3 选做）

### 目标
SaaS 化场景下多 session 互不干扰；凭证过期不再失败；可选启用后台自我提升。

### 涉及
- P2-1：Plugin 钩子协议升级
- P2-2：`/steer` 异步引导
- P2-3：Surrogate / ASCII codec 兜底
- P2-7：跨会话 rate guard
- P2-9：Task-scoped 资源清理
- P2-10：当前 turn reasoning 抽取
- P2-11：失败请求落盘
- P2-12：死连接清理
- P1-5：凭证池 + OAuth 自动刷新
- P1-6：thinking_signature 失效恢复
- P3-1（可选）：Memory provider 协议
- P3-2（可选）：后台 review fork

### 文件改动
（按子 PR 拆）
- `runtime/steer_inbox.py`（新）
- `runtime/credential_pool.py`（新）
- `runtime/rate_guard.py`（新）
- `runtime/unicode_recovery.py`（新）
- `runtime/hooks.py`：`HookOutcome` + 增加 `pre_api_request / post_api_request`
- `agents/core/agent.py`：上述钩入

### 验收
单独写每个子 PR 的验收（参考 04 / 05 文档对应小节）。

### 回滚开关
全部默认关闭，按需开启。

---

## 推荐时间线

| 周次 | 内容 |
|---|---|
| W1 | Sprint 0（地基） |
| W2 | Sprint 1（流式 + length） |
| W2½ | Sprint 1.5（内置工具集 + 散文合成 — P0-9 补丁） |
| W3 | Sprint 2 上半（ErrorClassifier + Fallback） |
| W4 | Sprint 2 下半（Tool 容错） |
| W5 | Sprint 3（压缩 + budget + token） |
| W6 | Sprint 4（空响应 + JSON 容错） |
| W7 | Sprint 5（cache + builder + 持久化） |
| W8+ | Sprint 6 按需 |

每个 Sprint 完成后回到对应优先级文档（`02_/03_/04_/05_`），把状态从 ❌/⚠️ 改为 ✅，
并在 PR 描述里链接到本目录的对应章节，保持文档与代码同步演进。

---

## 落地原则

1. **每个 Sprint 都必须能独立合入主分支**，不引入未来 Sprint 才能修复的回归。
2. **回滚开关默认值要保守**：新功能默认关闭，验证后再开。
3. **测试用例必须先于实现**：每个 Sprint 的"新增测试"清单是验收凭据。
4. **文档与代码同步**：每完成一项缺口，更新 `01_stage_matrix.md` 的状态列。
5. **跨越 Sprint 的字段（如 ExitReason）允许多次扩**，但**不允许重命名/删除**。
