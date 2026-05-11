# Sprint 4 验收矩阵 — 端到端场景与回归

> Sprint 4 收尾时跑这份矩阵。每条场景标注：所属 PR、需要的真实/mock 资源、
> 预期可观测信号、通过条件。所有场景应当在合并最后一个 PR 后 24 小时内全部 PASS。
>
> **版本说明**：v1.1（含 Codex 反馈 8 项采纳修订）。变化要点：
> - S6 Codex ack 改为 finalise pre-hook（非 7 步内的 step 8），触发条件改为
>   `_is_codex_responses_provider(provider)`（Codex 反馈 #2/#3）。
> - S5 加入 forced fallback upgrade 路径（Codex 反馈 #6）。
> - S7 housekeeping 改用 SessionRuntimeState 路径（Codex 反馈 #5）。
> - S8 partial_stream + thinking 互斥改为 truncated_prefix vs partial_stream 排序验证（Codex 反馈 #4）。
> - S3/S4 read_file 参数名改为 `path`（Codex 反馈 #7）。
> - 真实 DeepSeek-R1 / Codex fixture 联网验收降为 P2，不阻塞 merge（见 § 七）。

## 一、端到端场景

### Scenario S1 — Reasoning 模型整轮稳定（核心场景）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.1 + 4.2 |
| 触发的步骤 | `is_legitimate_empty` THINKING_ONLY classification → 7 步 step 5 thinking_prefill |
| 资源 | 真实 DeepSeek-R1 / 真实 GLM-5（或 mock 等价 fixture） |
| 步骤 | 1. 设 `empty_response_recovery_enabled=True, thinking_prefill_enabled=True, thinking_prefill_max_retries=2`<br>2. 用一个需要复杂推理的提示（如 "解一道 5 步数学题，每步思考完才说答案"）<br>3. 模型预期前 2 次返回 `content="" + reasoning="..."` |
| 预期信号 | - 第 1 次 → step 4 prefill；`metadata["empty_recovery"]["last_step"]="thinking_prefill"`；`THINKING_PREFILL_RETRIES=1`<br>- 第 2 次 → 同<br>- 第 3 次 → 拿到非空 content；prefill messages 被清；exit_reason=TEXT_RESPONSE<br>- `metadata["empty_recovery"]["thinking_prefill_cleaned"]=2` |
| PASS 条件 | 整轮 exit_reason 不为 EMPTY_RESPONSE；最终 final_response 非空且包含答案 |
| FAIL 兜底 | `thinking_prefill_enabled=False` 退回 PR 4.1 行为（thinking-only 直接 TEXT_RESPONSE 但 final_response="") |

### Scenario S2 — 流式断流后 partial 内容能被救活

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.1 + 4.2 |
| 触发的步骤 | step 1 partial_stream_recovery |
| 资源 | mock provider（注入 stream callback 累计 + 然后 raise/return empty） |
| 步骤 | 1. 设 `empty_response_partial_stream_recovery_enabled=True`<br>2. mock provider 在 stream callback 里 emit "Hello world this is partial" 后立刻 return `NormalizedResponse(content="", finish_reason="length")` |
| 预期信号 | - `metadata["empty_recovery"]["last_step"]="partial_stream_recovery"`<br>- `metadata["empty_recovery"]["classification"]="bug_empty"`<br>- exit_reason=PARTIAL_STREAM_RECOVERY<br>- final_response="Hello world this is partial" |
| PASS 条件 | final_response 非空；UI 看到流式 + 最终结果一致 |
| FAIL 兜底 | `empty_response_partial_stream_recovery_enabled=False` 退回 EMPTY_RESPONSE |

### Scenario S3 — JSON 错误触发结构化提示

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.4 |
| 触发的步骤 | `_validate_tool_call_arguments` inject 路径 → `format_invalid_tool_args_error` |
| 资源 | mock provider |
| 步骤 | 1. 设 `tool_error_structured_format_enabled=True, max_invalid_json_retries=3`<br>2. mock provider 连续 3 次返回 `tool_calls=[{name:"read_file", arguments:'{"path":"/x" "bad":1}'}]`（缺 comma）<br>3. 第 4 次返回正确 args（**Codex 反馈 #7**：read_file 真实参数是 `path`）|
| 预期信号 | - 第 4 次注入的 tool message content 含 "Invalid JSON arguments for tool `read_file` at line 1 column XX"<br>- 含 "Hint:"<br>- 含 raw snippet `..."path": "/x" "bad...`<br>- `metadata["tool_errors"]["by_category"]["json_syntax"]==1`（第 4 次注入计 1） |
| PASS 条件 | 注入消息含位置 + 上下文；模型下一轮自纠成功 |
| FAIL 兜底 | `tool_error_structured_format_enabled=False` 退回 PR 1.3 字符串 |

### Scenario S4 — Schema 错误（缺参）触发结构化提示

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.4 |
| 触发的步骤 | `_maybe_inject_schema_errors` → `format_schema_error` |
| 资源 | mock provider；read_file tool descriptor `parameters.required=["path"]`（**Codex 反馈 #7**） |
| 步骤 | 1. 设 `tool_schema_precheck_enabled=True`<br>2. mock provider 返回 `tool_calls=[{name:"read_file", arguments:'{}'}]`<br>3. 下一轮模型修正为 `{"path":"/x"}` |
| 预期信号 | - 注入的 tool content 含 "The required parameter `path` is missing"<br>- `metadata["_tool_error_category"]="schema_missing"`<br>- 第二轮 dispatch 成功<br>- 验证 `ToolRegistry.get_descriptor("read_file")` 与 `list_names()`（**Codex 反馈 #8** 新 API）可用 |
| PASS 条件 | 模型在 1 轮内自纠 |
| FAIL 兜底 | `tool_schema_precheck_enabled=False` → tool 自身报错（旧行为） |

### Scenario S5 — prompt_too_long 错误不污染 UI（withholding cascade）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.3 |
| 触发的步骤 | `_invoke_provider_with_recovery` cascade |
| 资源 | mock provider 连续 raise 413；CompactionPipeline 启用；FallbackChain 配置 1 个 fallback |
| 步骤 | 1. 设 `error_withholding_enabled=True, compression_enabled=True`<br>2. mock provider 第一次 raise 413（payload_too_large）<br>3. compaction Tier 4 释放 5000 token<br>4. mock provider 第二次返回 `content="OK"` |
| 预期信号 | - `metadata["recovery"]["cascade_log"]` 含 `["compress_context(freed=5000,reason=payload_too_large)"]`<br>- `metadata["recovery"]["terminal"]="success"`<br>- stream_callback 没看到任何错误 emit<br>- exit_reason=TEXT_RESPONSE |
| PASS 条件 | UI 看到 "OK"；没看到 "Error → Retry → OK" 闪烁 |
| FAIL 兜底 | `error_withholding_enabled=False` 退回 PR 2.2 单步 apply |

### Scenario S5b — Withholdable 错误连续两次：引擎强制升级 fallback（**Codex 反馈 #6**）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.3 |
| 触发的步骤 | `_maybe_upgrade_decision_for_repeat_withholding` |
| 资源 | mock provider 连续 raise 413；CompactionPipeline 启用但 freed 量小；FallbackChain 配 1 个 fallback；strategy 始终给 `compress_context=True / activate_fallback=False` |
| 步骤 | 1. 设 `error_withholding_enabled=True, compression_enabled=True, fallback_chain_enabled=True`<br>2. attempt 1：raise 413 → strategy 给 compress → compress freed=200（不够）<br>3. attempt 2：raise 同样的 413 → 引擎检测到 `compression_attempted_for` 已含 payload_too_large → 不询问 strategy，强制合成 `activate_fallback=True` → fallback 切换<br>4. attempt 3：fallback provider 返回 `content="OK"` |
| 预期信号 | - `metadata["recovery"]["cascade_log"]` 含 `compress_context(freed=200,reason=payload_too_large)`、`force_fallback_upgrade(reason=payload_too_large,after_compression=true)`、`fallback(...)`<br>- `metadata["recovery"]["terminal"]="success"`<br>- exit_reason=TEXT_RESPONSE |
| PASS 条件 | 引擎层强制升级而非依赖 strategy；UI 没看到 error 闪烁；不需要修改 RecoveryDecision 契约 |
| FAIL 兜底 | `fallback_chain_enabled=False` 时不升级，cascade exhaust 后 surface 413 |

### Scenario S6 — Codex "好的我马上去做" 能被 finalise pre-hook 推进（**Codex 反馈 #2/#3**）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.2 |
| 触发的步骤 | `_maybe_continue_codex_intermediate_ack` finalise pre-hook（**不在 7 步 pipeline 内**）|
| 资源 | mock provider with `provider_name="codex"` and `api_mode="responses"`（不需要 ProviderProfile 抽象）|
| 步骤 | 1. 设 `codex_intermediate_ack_enabled=True`<br>2. mock provider 第一次返回 `content="好的，我马上读文件", tool_calls=[]`，**finish_reason="stop"** 即非空也非错<br>3. 引擎走 `_finalize_empty_response` 的 `final_response != ""` 分支 → finalise pre-hook 命中<br>4. 第二次 LLM 返回 `tool_calls=[{name:"read_file", arguments:{"path":"/x"}}]`<br>5. 工具 dispatch 成功；第三次 LLM 返回 final |
| 预期信号 | - 第一次：finalise pre-hook 命中；注入 system 续写消息；`metadata["empty_recovery"]["codex_ack_finalise_hook"]="attempt=1"`；`CODEX_ACK_RETRIES=1`<br>- run-loop 转 PRE_LLM 继续（**不**走 EMPTY_RESPONSE pipeline）<br>- 第二次：tool dispatch 正常<br>- exit_reason=TEXT_RESPONSE |
| PASS 条件 | 模型从 ack 推进到 tool 调用；整轮成功；不依赖 `provider.profile` 抽象 |
| FAIL 兜底 | `codex_intermediate_ack_enabled=False` 或 provider_name/api_mode 不匹配 → 退回原行为（直接 finalise 为 TEXT_RESPONSE，content=ack 文本）|

### Scenario S7 — Housekeeping fallback 命中（**Codex 反馈 #5：SessionRuntimeState 路径**）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.2 |
| 触发的步骤 | step 3 housekeeping_fallback |
| 资源 | mock provider；同一 session_id 跨两轮 |
| 步骤 | 1. 设 `housekeeping_fallback_enabled=True`<br>2. **第 N 轮**：模型 emit `tool_calls=[update_todo({...})]` + `content="Updated todo"`<br>3. 第 N 轮 `_build_result` 之前 `_record_last_content_for_housekeeping_fallback` 写入 `SessionRuntimeState.last_assistant_text_with_tools="Updated todo"` + `last_assistant_tools_all_housekeeping=True`<br>4. **第 N+1 轮**：新 TurnContext.metadata 由 `dict(request.metadata)` 重置；mock provider 返回 `content="", finish_reason="length"`<br>5. step 3 reader 读 `SessionRuntimeRegistry.get(session_id)` 拿到上一轮快照 |
| 预期信号 | - 第 N+1 轮 `metadata["empty_recovery"]["last_step"]="housekeeping_fallback"`<br>- exit_reason=FALLBACK_PRIOR_TURN_CONTENT<br>- final_response="Updated todo"<br>- **关键**：调用方 NOT 需要把 N 轮 result.metadata 回灌到 N+1 轮 request.metadata（验证 SessionRuntimeState 自动跨 turn） |
| PASS 条件 | final_response 取上一轮内容；调用方代码不改 |
| FAIL 兜底 | `housekeeping_fallback_enabled=False` 退回 EMPTY_RESPONSE |

### Scenario S8 — Truncated prefix 优先于 partial stream（**Codex 反馈 #4 排序**）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.2 |
| 触发的步骤 | step 1 truncated_prefix_concat 必须在 step 2 partial_stream_recovery 之前 |
| 资源 | mock provider |
| 步骤 | 1. 上一轮 length 续写产生 `TURN_KEY_TRUNCATED_RESPONSE_PREFIX="Previous answer prefix..."`<br>2. 本轮 mock callback emit "PartialContinuation"<br>3. response = `content="", finish_reason="length"` |
| 预期信号 | - classification.kind=BUG_EMPTY<br>- step 1 命中 → final_response="Previous answer prefix...PartialContinuation"<br>- step 2 partial_stream **不**单独 fire（验证排序）<br>- exit_reason=LENGTH_RECOVERED<br>- prefix 在 step 1 命中后被清空 |
| PASS 条件 | 排序正确；prefix 不留在 metadata 让下一轮误用 |
| FAIL 兜底 | 调整 pipeline 数组顺序使 step 1 在 step 2 之前 |

### Scenario S8b — Partial stream 与 thinking prefill 不互踩（无 prefix 场景）

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.1 + 4.2 |
| 触发的步骤 | classification.has_streamed_partial=True ∧ has_thinking=True → BUG_EMPTY → step 2 优先于 step 5 |
| 资源 | mock provider；NO truncated prefix |
| 步骤 | 1. mock callback emit "PartialText"<br>2. response = `content="", metadata={"reasoning_content": "thinking..."}` |
| 预期信号 | - classification.kind=BUG_EMPTY（不是 THINKING_ONLY，因为 has_streamed_partial=True）<br>- step 2 命中 → final_response="PartialText"<br>- step 5 thinking_prefill **不**触发；THINKING_PREFILL_RETRIES=0<br>- thinking 检测来自 metadata["reasoning_content"]，不读不存在的 `response.reasoning`（**Codex 反馈 #1**）|
| PASS 条件 | 优先级正确；不浪费 thinking_prefill 预算 |
| FAIL 兜底 | 调整 step 顺序回到 PR 4.2 设计 |

### Scenario S9 — 上下文压缩 → tool dispatch 全链路不回归

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.3 + Sprint 3 / PR 3.4 |
| 触发的步骤 | recovery cascade（compress） + 完整 tool 流 |
| 资源 | mock provider；CompactionPipeline 全开 |
| 步骤 | 1. 历史 messages 累计 80k tokens<br>2. 第一次 generate 抛 413；cascade 触发 Tier 5 fork summarizer 释放 50k<br>3. 第二次 generate 返回 `tool_calls=[shell({command:"ls"})]`<br>4. tool dispatch 成功；返回 `content="ls output: ..."` |
| 预期信号 | - `metadata["recovery"]["cascade_log"]` 含 `compress_context(freed=50000,...)`<br>- `metadata["compaction"]["tier5_summaries_generated"]=1`<br>- 整轮成功 |
| PASS 条件 | Sprint 3 + Sprint 4 两层 metadata 各自独立；行为正确 |
| FAIL 兜底 | 任一层关闭即可 isolate |

### Scenario S10 — 7 步降级与 phantom-tool 不双触发

| 维度 | 内容 |
|---|---|
| 触发的 PR | 4.2 + Sprint 1.5 |
| 触发的步骤 | phantom-tool 优先于 7 步 |
| 资源 | mock provider |
| 步骤 | 1. mock provider 返回 `content='```bash\nls\n```', tool_calls=[]`（phantom-tool 散文形态）<br>2. phantom_tool_synthesis_enabled=True；synthesis 成功合成 shell 调用 |
| 预期信号 | - phantom-tool synthesis 走通<br>- 7 步降级**不**触发；`metadata["empty_recovery"]["last_step"]` 缺失或为 `n/a`<br>- TEXT_RESPONSE / TOOL_RESPONSE 路径 |
| PASS 条件 | phantom-tool 优先级保持 |
| FAIL 兜底 | 顺序错误时调整 run-loop dispatch 表 |

### Scenario S11 — 全部回滚开关关闭后行为退回 Sprint 3

| 维度 | 内容 |
|---|---|
| 触发的 PR | 全部 4 个 |
| 资源 | mock provider |
| 步骤 | 1. 设 `empty_response_recovery_enabled=False, codex_intermediate_ack_enabled=False, error_withholding_enabled=False, tool_error_structured_format_enabled=False, tool_schema_precheck_enabled=False, legitimate_empty_passthrough_enabled=False`<br>2. 跑 Sprint 3 验收套件中所有场景 |
| 预期信号 | - 行为与 Sprint 3 完成时完全一致<br>- 不引入 PR 4.x metadata 字段（或字段值都是默认/空）|
| PASS 条件 | 100% 与 Sprint 3 commit 行为一致 |
| FAIL 兜底 | 任一开关导致行为变化即修复实现 |

## 二、单元测试覆盖矩阵

| PR | 测试文件 | 用例数 | 覆盖功能 |
|---|---|---|---|
| 4.1 | `test_response_classification.py` | 22 | EmptyKind 4 类、has_thinking（**只读 metadata**，Codex 反馈 #1）、has_streamed_partial |
| 4.1 | `test_tool_error_format.py` | 8 | JSONDecodeError 友好化、snippet、边界 |
| 4.1 | `test_finalize_empty_response.py` | 9 | helper 抽取后行为 |
| 4.1 | `test_session_runtime_state_housekeeping_fields.py` | 4 | SessionRuntimeState 新加 2 字段（**Codex 反馈 #5**）|
| 4.2 | `test_empty_response_step1_truncated_prefix.py` | 5 | step 1 truncated_prefix（原 step 9，提到首位）|
| 4.2 | `test_empty_response_step2_partial_stream.py` | 5 | step 2 partial_stream（原 step 1）|
| 4.2 | `test_empty_response_step3_housekeeping_fallback.py` | 6 | step 3 housekeeping + SessionRuntimeState 跨 turn 写入路径 |
| 4.2 | `test_empty_response_step4_post_tool_nudge.py` | 5 | step 4 post_tool_nudge |
| 4.2 | `test_empty_response_step5_thinking_prefill.py` | 7 | step 5 thinking_prefill + pop 清理（**只读 metadata**）|
| 4.2 | `test_empty_response_step6_retry_fallback.py` | 8 | step 6 retry/fallback（原 step 5+6 合并）|
| 4.2 | `test_empty_response_step7_terminal.py` | 4 | step 7（不写占位）|
| 4.2 | `test_empty_response_codex_ack_finalise_hook.py` | 8 | finalise pre-hook（**Codex 反馈 #3**：不在 7 步 pipeline 内）+ `_is_codex_responses_provider` gate（**Codex 反馈 #2**）|
| 4.2 | `test_empty_response_full_pipeline.py` | 9 | 端到端组合（含 truncated_prefix 优先级验证）|
| 4.3 | `test_recovery_cascade.py` | 29 | cascade 单步/多步/回退/observability/边界 + **forced fallback upgrade**（Codex 反馈 #6，4 case）|
| 4.3 | `test_recovery_cascade_integration.py` | 8 | 与 PR 2.2 / 3.4 / 4.1/4.2 集成 |
| 4.4 | `test_tool_error_format_schema.py` | 22 | schema 错误三类、unknown_tool、Levenshtein |
| 4.4 | `test_invalid_json_inject_structured.py` | 8 | invalid-JSON 注入升级（**read_file schema 用 `path`**，Codex 反馈 #7）|
| 4.4 | `test_schema_precheck.py` | 12 | precheck 路径 + 与 invalid-JSON 互斥 |
| 4.4 | `test_unknown_tool_inject.py` | 5 | unknown_tool synthetic message 升级 |
| 4.4 | `test_tool_registry_public_api.py` | 4 | `get_descriptor` / `list_names` 新公共 API（**Codex 反馈 #8**）|

**Sprint 4 总计**：≈ 188 个新增测试用例。

## 三、性能基准

### 3.1 7 步降级延迟（重排后）

| 路径 | 触发条件 | 期望延迟 |
|---|---|---|
| step 1 truncated_prefix concat | prefix 存在 + 字符串拼接 | < 1 ms |
| step 2 partial_stream | streamed_text 非空 | < 5 ms（纯字符串处理） |
| step 3 housekeeping | SessionRuntimeState 已就绪 | < 1 ms |
| step 4 post_tool_nudge | 注入 2 条消息 | < 1 ms |
| step 5 thinking prefill | 注入 1 条 prefill | < 5 ms（含 metadata 拷贝） |
| step 6 retry / fallback | 一次额外 LLM 调用 | + 1 次 LLM round-trip |
| step 7 terminal | metadata 写入 | < 1 ms |
| finalise hook codex_ack | 注入 system 消息 + 重新跑 | + 1 次 LLM round-trip |

**关键基准**：step 1-5 + 7 是"零额外 LLM"，单次 7 步执行时间应 < 50 ms。

### 3.2 Recovery cascade 延迟

| 场景 | 期望延迟 |
|---|---|
| 一次 attempt 成功 | 与 PR 2.2 一致 |
| 一次 cascade 含 compress（Tier 2/3 本地） | + 50 ms |
| 一次 cascade 含 compress（Tier 4 LLM fork） | + 1 次 fork LLM 调用 |
| 一次 cascade 含 compress（Tier 5 LLM fork） | + 1 次 fork LLM 调用 |
| 一次 cascade 含 fallback rotate | + provider switch overhead（< 10 ms） |

### 3.3 JSON 结构化错误

| 函数 | 输入规模 | 期望延迟 |
|---|---|---|
| `format_invalid_tool_args_error` | raw_args 10KB | < 5 ms |
| `format_schema_error` | 10 个字段 | < 1 ms |
| `format_unknown_tool_error` | 50 个 tool 名 | < 5 ms（含 Levenshtein） |

## 四、监控指标

> **来源说明**：当前 Aether 没有独立 metrics 系统。Sprint 4 输出全部落到
> `EngineResult.metadata` 的稳定 sub-dict（`empty_recovery` / `recovery` / `tool_errors`）。
> 后续 Sprint 引入正式 metrics 系统时，从这三个 sub-dict 拉数据即可，
> 不会破坏 Sprint 4 实施成果。
>
> 这三个 key 都被列入 `EngineResult.metadata` 的"reserved key shape，additive only"
> 契约（与 Sprint 3 的 `compaction` 子字典同等地位）。

完成 Sprint 4 后**建议**接入下列监控指标（外部 metrics 系统就绪时对接）：

| 指标 | 来源 | 报警阈值 |
|---|---|---|
| `empty_recovery.last_step` 各类分布 | EngineResult.metadata | terminal_empty 占比 > 5% 报警 |
| `empty_recovery.retries.empty` 平均值 | EngineResult.metadata | > 1.5 报警（说明 retry 太多） |
| `empty_recovery.retries.thinking_prefill` p99 | EngineResult.metadata | > 1 报警（说明 prefill 不够用） |
| `recovery.cascade_log` 长度分布 | EngineResult.metadata | p99 > 3 报警 |
| `recovery.terminal="exhausted"` 占比 | EngineResult.metadata | > 1% 报警 |
| `tool_errors.by_category["json_syntax"]` 频次 | EngineResult.metadata | 某 provider 突增 > 10x 报警 |
| `tool_errors.by_category["schema_missing"]` 频次 | EngineResult.metadata | 长期高 = 提示该 tool 描述需要改进 |
| ExitReason `EMPTY_RESPONSE` 占比 | EngineResult.exit_reason | > 0.5% 报警 |
| ExitReason `PARTIAL_STREAM_RECOVERY` 频次 | EngineResult.exit_reason | 偶发可接受；持续高表明 provider stream 不稳定 |
| ExitReason `FALLBACK_PRIOR_TURN_CONTENT` 频次 | EngineResult.exit_reason | 偶发可接受 |

## 五、回归测试套件

完成 Sprint 4 后**必须**跑通的既有测试集：

- [ ] Sprint 0 全套（runtime 基础设施）
- [ ] Sprint 1 全套（流式 + length 续写 + 截断 tool_call）
- [ ] Sprint 1.5 全套（内置工具 + phantom 合成）
- [ ] Sprint 2 全套（错误分类 + fallback chain + tool 容错）
- [ ] Sprint 3 全套（5 级压缩流水线）
- [ ] Sprint 4 全套新测试（≈ 173 case）

**总计**：合并完 PR 4.4 后跑 `uv run python -m unittest discover` 应全 green。

## 六、文档同步清单

合并 PR 4.4 后**同步**更新以下文档：

- [ ] [`02_p0_critical_gaps.md` § P0-8](../run-loop-roadmap/02_p0_critical_gaps.md) 状态从 ❌ 改 ✅
- [ ] [`02_p0_critical_gaps.md` § P0-4](../run-loop-roadmap/02_p0_critical_gaps.md) 状态从 ⚠️ 改 ✅
- [ ] [`03_p1_robustness_gaps.md` § P1-8](../run-loop-roadmap/03_p1_robustness_gaps.md) 状态从 ❌ 改 ✅
- [ ] [`07_sprint_execution_plan.md` § Sprint 4](../run-loop-roadmap/07_sprint_execution_plan.md) 加完成标记 ✅
- [ ] [`docs/agent-engine/`](../agent-engine/) 追加 `09_empty_response_pipeline.md` 描述 7 步状态机 + Codex finalise pre-hook 拓扑
- [ ] CHANGELOG.md 追加：
  - "Sprint 4 / PR 4.1: deliberate-empty responses no longer surface as EMPTY_RESPONSE"
  - "Sprint 4 / PR 4.2: 9-step empty-response degradation now active by default"
  - "Sprint 4 / PR 4.3: API errors now go through cheap-first cascade recovery"
  - "Sprint 4 / PR 4.4: tool-error messages now structured (missing/unexpected/type_mismatch)"

## 七、Sprint 4 完成的全局信号

Sprint 4 视为完成的判据：

**P0（阻塞 merge）**：

1. **所有 12 个端到端场景 PASS**（S1-S11，含 S5b 升级路径）——全部基于 mock fixture
2. **所有新增 ≈ 188 单测 PASS**
3. **既有 Sprint 0-3 测试 0 回归**
4. **prompt_too_long 场景下 UI 不再看到错误闪烁**（人工验证基于 mock provider）
5. **所有回滚开关验证可用**（Scenario S11）

**P1（建议合 merge 后跟进，不阻塞）**：

6. Mock DeepSeek-R1 fixture（thinking-only 反复出现）整轮 exit=TEXT_RESPONSE
7. Mock Codex fixture（codex_response_mode "好的我读"）finalise hook 推进成功
8. metadata schema（`empty_recovery` / `recovery` / `tool_errors`）完整性外部消费者验证

**P2（nice-to-have，需要用户提供资源）**：

9. **真实 DeepSeek-R1 fixture（联网）整轮成功率 ≥ 90%** ——
   需要用户提供 provider 配置、API key、是否允许联网。
10. **真实 Codex fixture（联网）整轮成功率 ≥ 95%** ——
    需要用户提供 codex CLI 凭据。

> 真实 fixture 在 mock fixture 全绿后跑，作为额外信心信号。
> 任一 P0/P1 失败即回归 implementation；P2 失败提交 issue 跟进，但不撤回 PR。

## 八、与上下游 Sprint 的衔接

### 8.1 Sprint 5 接力

Sprint 5 将引入 `MessageBuilder`，本 Sprint 引入的两个 marker 必须保持兼容：

```python
# Sprint 4 marker:
{"role": "assistant", "_thinking_prefill": True, ...}

# Sprint 5 MessageBuilder 必须实现：
def pop_thinking_prefill_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if not m.get("_thinking_prefill")]
```

并把本 Sprint 内 agent.py 实现的 `_pop_thinking_prefill_messages` 迁到 MessageBuilder。

### 8.2 Sprint 6 接力

Sprint 6 引入 plugin 钩子 `pre_api_request / post_api_request`：本 Sprint 的
`_observe_recovery_cascade` 输出可以作为 `post_api_request` 的标准输入；
7 步状态机的 `_handle_empty_response` 也可以暴露 `pre_empty_recovery / post_empty_recovery`
钩子供 plugin 干预。
