# 17 阶段对照表（核心交付物）

> 本文是整套路线图的"单一事实来源"。所有 P0/P1/P2/P3 文档与 Sprint 计划都以此表为基准。
>
> 状态约定：✅ 已对齐 / ⚠️ 部分实现 / ❌ 完全缺失

## 阅读指引

- **Hermes 列**给出每个阶段在 `run_agent.py` 中的代码区间，便于直接 `Read` 对照。
- **Aether 现状**写明当前 `run_loop` / 周边模块里有/没有什么。
- **缺口** 列直接给出待实现内容；同时附"严重性"与"对应 Sprint"。

---

## 阶段 1 — Turn 入口准备

| 项 | 内容 |
|---|---|
| Hermes 行号 | 10599–10720 |
| Hermes 关键动作 | `_install_safe_stdio()`；session log context；surrogate 字符 sanitize；reset 全部 per-turn retry 计数器；清死 TCP 连接（`_cleanup_dead_connections`）；replay compression warning；reset stream scrubber 与 think scrubber；触发 `_user_turn_count` |
| Aether 现状 | `_prepare_turn_entry()`：仅 `_sanitize_messages` + 重置 2 个 retry 计数器 |
| 缺口 | ① surrogate sanitize 现仅遍历字符串，没有覆盖 retry 时新生成的字段；② 没有死连接清理；③ 没有 stream scrubber；④ retry 计数器在引擎实例上（多 session 并发互相覆盖，详见 06 文档） |
| 严重性 | P2（多 session 安全 → P0） |
| 对应 Sprint | Sprint 6（多 session 隔离）+ Sprint 5（持久化） |

---

## 阶段 2 — System prompt 缓存与重建

| 项 | 内容 |
|---|---|
| Hermes 行号 | 10747–10797 |
| Hermes 关键动作 | session 级缓存 `_cached_system_prompt`；新 session 走 `_build_system_prompt`；老 session 从 DB 读出原 prompt 以保证 Anthropic prefix-cache 不破；首次构建后调用 plugin `on_session_start`；写回 `session_db.update_system_prompt` |
| Aether 现状 | `_prepare_session_and_system_prompt()` 已有 SessionStore，但**只要 `request.system_message` 非空就覆盖**，会破坏 prefix cache |
| 缺口 | ① 写入策略改为"仅新 session 或显式传入时覆盖"；② 加 `_cached_system_prompt` 引擎级缓存；③ 引入"压缩后失效"语义 |
| 严重性 | P1（命中率影响 cost） |
| 对应 Sprint | Sprint 5 |

---

## 阶段 3 — Preflight context compression

| 项 | 内容 |
|---|---|
| Hermes 行号 | 10799–10865 |
| Hermes 关键动作 | 估算 `messages + tools + system_prompt` 总 tokens；超阈值时多 pass 压缩；压缩后清空 `conversation_history` 引用；重置 retry 计数器；重新估算确认是否继续压 |
| Aether 现状 | ❌ 完全没有压缩抽象 |
| 缺口 | ① 引入 `ContextCompressor` 接口；② 三个触发点（preflight / 413 / post-tool 阈值）；③ `parsed_limit` 学习与持久化 |
| 严重性 | P1 |
| 对应 Sprint | Sprint 3 |

---

## 阶段 4 — Plugin pre_llm_call & 外部 memory prefetch

| 项 | 内容 |
|---|---|
| Hermes 行号 | 10867–10951 |
| Hermes 关键动作 | 调用 `pre_llm_call` plugin hook → 收集 `{"context": "..."}` 返回值；调用 `memory_manager.prefetch_all`；以 fenced block 注入到**当前 user message**而不是 system prompt（保护 prefix cache） |
| Aether 现状 | `EngineHooks.pre_llm_call` 只是通知钩子（返回 None，不收集任何上下文） |
| 缺口 | ① hook 协议升级为可返回 `{"context": "..."}`；② 引入 `MemoryProvider` 接口；③ 注入策略明确"user message 末尾 + fenced block" |
| 严重性 | P3（取决于是否要做 memory）；hook 协议升级是 P2 |
| 对应 Sprint | Sprint 6 |

---

## 阶段 5 — 主循环：Iteration budget

| 项 | 内容 |
|---|---|
| Hermes 行号 | 10953–10978 |
| Hermes 关键动作 | `IterationBudget(max_iterations)`；每轮 `consume()`；cheap tool（`execute_code`）走 `refund()`；`_budget_grace_call` 在用尽后再给一次"宽限"调用 |
| Aether 现状 | 仅 `iterations < self.config.max_iterations` 硬上限 |
| 缺口 | ① 引入 `IterationBudget` 类；② 提供 refund/grace 接口；③ 在 `EngineResult.metadata` 里返回 `budget_used / budget_max` |
| 严重性 | P1 |
| 对应 Sprint | Sprint 3 |

---

## 阶段 6 — `/steer` drain（异步用户引导）

| 项 | 内容 |
|---|---|
| Hermes 行号 | 11014–11062 |
| Hermes 关键动作 | API call 之前从 `_pending_steer` 读出引导文本；找到最近一条 `tool` 角色消息追加 `\n\nUser guidance: ...`；如果没有 tool 消息则把 steer 放回；保证不破坏 role 轮替 |
| Aether 现状 | ❌ 没有 steer 概念 |
| 缺口 | ① `EngineRequest` 增加 `pending_steer` 字段；② 提供线程安全的注入接口；③ 收尾若仍有未消费 steer 写回 result.metadata.pending_steer |
| 严重性 | P2 |
| 对应 Sprint | Sprint 6 |

---

## 阶段 7 — API 消息构建（每轮 LLM 调用前的最重要一步）

| 项 | 内容 |
|---|---|
| Hermes 行号 | 11064–11215 |
| Hermes 关键动作 | ① 复制 `messages` → 注入 ephemeral 内容；② `_copy_reasoning_content_for_api`（多轮 reasoning 续传）；③ 删去仅本地用的字段（`reasoning` / `finish_reason` / `_thinking_prefill`）；④ 严格 API 字段裁剪（Codex/Mistral 等不接受未知字段）；⑤ Anthropic prompt cache breakpoint 注入；⑥ `_sanitize_api_messages`（孤儿 tool 结果补桩）；⑦ `_drop_thinking_only_and_merge_users`；⑧ JSON 规整为 sort_keys/separators 提升 KV cache 命中；⑨ `_sanitize_messages_surrogates` 兜底 |
| Aether 现状 | ❌ 完全没有这一层；provider 内部也没做 |
| 缺口 | ① 引入 `MessageBuilder` 抽象；② 8 个子步骤里至少必做：reasoning 续传、孤儿 tool 补桩、JSON 规整、Anthropic prompt cache、严格 API 字段裁剪 |
| 严重性 | P0（因为孤儿 tool 补桩缺失会让 Anthropic 直接 400）/ P1（其余） |
| 对应 Sprint | Sprint 5 |

---

## 阶段 8 — 流式 vs 非流式选择

| 项 | 内容 |
|---|---|
| Hermes 行号 | 11369–11398 |
| Hermes 关键动作 | 默认走流式以获得 90s stale-stream 检测和 60s 读超时；针对 ACP/Mock 客户端禁用流式；流式失败可降级为非流式 |
| Aether 现状 | ✅ Sprint 1 / PR 1.1 已对齐：`OpenAICompatibleModel.generate` 在有 `stream_callback` 时走 `_streaming_generate`（SSE 解析 + 90s stale-stream watchdog → `StreamStallError`），失败一次后置位 `_disable_streaming` 自动降级为非流式；`EngineConfig.streaming_enabled` 提供紧急回滚开关 |
| 缺口 | — |
| 严重性 | P0 |
| 对应 Sprint | Sprint 1 ✅ |

---

## 阶段 9 — 响应壳校验

| 项 | 内容 |
|---|---|
| Hermes 行号 | 11418–11630 |
| Hermes 关键动作 | 按 api_mode 分别用 transport 的 `validate_response` 校验形状（codex 检查 `output`/`output_text`；anthropic 检查 `content` 列表非空；chat-completions 检查 `choices`）；invalid 响应**直接 eager fallback** 而不是 generic retry |
| Aether 现状 | ⚠️ Sprint 1 / PR 1.1 已落地基础：`ModelProvider.validate_response()` 抽象 + `OpenAICompatibleModel` 默认实现（识别嵌入式 `error` 信封 / 空 `choices`）；引擎在 LLM_CALL 之后立即校验，失败抛 `ResponseInvalidError` 走 recovery 层。eager fallback 仍待 Sprint 2 的 `FallbackChain`，目前走 `GenericBackoffStrategy` 通用重试 |
| 缺口 | invalid 触发的 eager fallback（依赖 Sprint 2 / P0-7） |
| 严重性 | P0 |
| 对应 Sprint | Sprint 1 ✅（基础）+ Sprint 2（eager fallback） |

---

## 阶段 10 — `finish_reason="length"` 处理

| 项 | 内容 |
|---|---|
| Hermes 行号 | 11632–11848 |
| Hermes 关键动作 | ① 检测 thinking-budget 耗尽（reasoning 占满 output budget）→ friendly-exit；② 续写 retry 最多 3 次，按 `_ephemeral_max_output_tokens = base × (n+1)` 渐进抬高输出预算（上限 32k）；③ 截断 tool_call 单独 1 次 retry（不进消息历史）；④ 失败时回滚到最后一个完整 assistant turn 再返回 partial=True |
| Aether 现状 | ✅ Sprint 1 / PR 1.2 + 1.3 已对齐：`_handle_length_finish_reason` 处理 thinking-budget friendly exit（→ `LENGTH_EXHAUSTED`） + 续写 retry（`_ephemeral_max_output_tokens` × (n+1)，上限 32k） + 回滚 `_get_messages_up_to_last_assistant`（→ `LENGTH_RECOVERED` / `LENGTH_EXHAUSTED`）；`_handle_length_with_tool_calls` 处理 length+tool_calls 的"不进历史 retry 一次"路径（→ `TOOL_CALL_TRUNCATED`） |
| 缺口 | — |
| 严重性 | P0 |
| 对应 Sprint | Sprint 1 ✅ |

---

## 阶段 11 — Token / cost 会话累计

| 项 | 内容 |
|---|---|
| Hermes 行号 | 11851–11964 |
| Hermes 关键动作 | `normalize_usage(...)` → 累加到 session 级计数器（input/output/cache_read/cache_write/reasoning）；`estimate_usage_cost`；`session_db.update_token_counts` 持久化；探测到的 context_length 持久化（仅在 parsed 自 provider 错误时） |
| Aether 现状 | ❌ 没有任何 token/cost 累计 |
| 缺口 | ① `EngineResult.metadata` 增加完整 token 字段；② 引入 `UsageTracker`；③ 接入 cost 估算（可选） |
| 严重性 | P1 |
| 对应 Sprint | Sprint 3 |

---

## 阶段 12 — API 异常分类与恢复（Hermes 最重的一段，~12 个分支）

| 项 | 内容 |
|---|---|
| Hermes 行号 | 12164–13063 |
| Hermes 关键动作 | `classify_api_error → FailoverReason` 把错误分成 ~12 种语义，再按类决定恢复路径： |
| 12.1 凭证池轮转 | 12185–12192 — `_recover_with_credential_pool` 切换下一把 key |
| 12.2 image_too_large | 12200–12216 — `_try_shrink_image_parts_in_messages` 就地缩图后 retry |
| 12.3 oauth_long_context_beta_forbidden | 12218–12246 — 关 1M-context beta 重建 client 后 retry |
| 12.4 401 自动 refresh | 12248–12323 — Codex/Anthropic/Nous/Copilot 各自 `_try_refresh_*_credentials` |
| 12.5 thinking_signature 失效 | 12325–12350 — 一次性剥掉所有 `reasoning_details` 后 retry |
| 12.6 llama.cpp grammar 拒绝 | 12352–12393 — 从 tools schema 剥 `pattern`/`format` 后 retry |
| 12.7 long_context_tier | 12472–12528 — 把 `context_length` 降到 200k 并触发压缩 |
| 12.8 rate_limit / billing | 12534–12554 — **eager fallback** 到下一家 provider，不等 retry 用完 |
| 12.9 Nous Portal 跨会话 rate guard | 12556–12619 — 写共享文件，其他 session 主动避让 |
| 12.10 payload_too_large (413) | 12621–12671 — 进入压缩循环最多 3 次 |
| 12.11 context_overflow | 12673–12830 — 区分"输入太长"vs"max_tokens 太大"，前者降 context_length 后压缩，后者只降 `_ephemeral_max_output_tokens` |
| 12.12 不可重试 client_error | 12832–12930 — 先尝试 fallback；落盘 `_dump_api_request_debug` 后失败退出 |
| 12.13 通用重试 | 13015–13063 — 尊重 `Retry-After`，否则 `jittered_backoff(base=2, cap=60)`；中途响应 interrupt |
| Aether 现状 | ❌ 整个错误处理就一条 `_handle_pipeline_error + _provider_error_retries += 1`，无分类、无分流、无 fallback |
| 缺口 | ① 引入 `ErrorClassifier → FailoverReason`；② 在 LLM_CALL 异常分支按 reason 分流；③ 引入 `FallbackChain` 抽象；④ provider 抽象增加 `refresh_credentials()` 与 `dump_request()` 钩子 |
| 严重性 | P0（12.7/12.8/12.10/12.11/12.13）+ P1（12.1/12.4/12.5）+ P2（12.2/12.3/12.6/12.9） |
| 对应 Sprint | Sprint 2（核心分类器 + fallback + 12.7–12.13）；Sprint 6（凭证池/OAuth/图像缩放等专用恢复） |

---

## 阶段 13 — Unicode 兜底

| 项 | 内容 |
|---|---|
| Hermes 行号 | 12011–12162 |
| Hermes 关键动作 | `UnicodeEncodeError` 分两种处理：① surrogate（lone U+D800–U+DFFF）— 全 payload sanitize 后 retry；② ASCII codec（LANG=C 环境）— 启用 `_force_ascii_payload` 全 payload strip 非 ASCII，连 API key 的非 ASCII 都清理；最多 2 次 |
| Aether 现状 | 仅在 turn 入口做一次 `_sanitize_text`；retry 时不会重清理；没有 ASCII codec 路径 |
| 缺口 | ① `UnicodeRecovery` 抽象；② 在 LLM_CALL 异常处理里加 `UnicodeEncodeError` 专项分支 |
| 严重性 | P2 |
| 对应 Sprint | Sprint 6 |

---

## 阶段 14 — 响应正常化

| 项 | 内容 |
|---|---|
| Hermes 行号 | 13098–13272 |
| Hermes 关键动作 | `transport.normalize_response()`；content 形状归一（dict/list 多模态 → str）；plugin `post_api_request` hook；校验 `<REASONING_SCRATCHPAD>` 是否完整；Codex `incomplete` 续写最多 3 次 |
| Aether 现状 | provider 在 `_parse_response` 里直接产出 `NormalizedResponse`；没有 scratchpad 完整性检查、没有 codex incomplete 续写 |
| 缺口 | ① 引擎层加"响应后置归一化"阶段（脱离 provider）；② scratchpad 完整性检测；③ codex incomplete 续写策略（也可放阶段 10） |
| 严重性 | P1 |
| 对应 Sprint | Sprint 4 |

---

## 阶段 15 — Tool call 路径（关键）

| 项 | 内容 |
|---|---|
| Hermes 行号 | 13274–13586 |
| 15.① 工具名校验与修复 | `_repair_tool_call` 模糊修复（`readFile → read_file`）；不存在工具时回写 `tool` 角色 error 让模型自纠（最多 3 次）|
| 15.② 参数 JSON 校验 + 截断检测 | 参数没以 `}/]` 结尾视为 length 截断 → 不执行；其余 JSON 错误 retry 3 次后回写 tool error 给模型自纠 |
| 15.③ 调用上限 | `_cap_delegate_task_calls` 限制递归 subagent 数 |
| 15.④ 去重 | `_deduplicate_tool_calls` 对相同 name+args 去重 |
| 15.⑤ housekeeping vs substantive | 仅 housekeeping（memory/todo/skill_manage）时 mute 后续输出；抓住"已说+顺手存 memory"的 fallback |
| 15.⑥ 实际执行 | `_execute_tool_calls`：含 per-tool steer drain + 每轮总预算 enforcement + tool guardrail halt |
| 15.⑦ post-tool 压缩 | 工具执行后用真实 `prompt_tokens` 估算，超阈值即压缩 |
| 15.⑧ 增量持久化 | 每轮 `_save_session_log` 让 Ctrl+C 能恢复进度 |
| Aether 现状 | ⚠️ 15.② ✅ Sprint 1 / PR 1.3：`_validate_tool_call_arguments` 做类型规整（dict/list pass-through，空串 → `{}`，非 str → `str()`） + JSON parse + 截断启发式（`_detect_truncated_tool_call`）→ 不进 dispatch 直接 retry/refuse；JSON 错误静默重试 `max_invalid_json_retries`(=3) 次后注入 assistant + role=tool 错误消息让模型自纠（保持角色轮替）。其余 15.①③④⑤⑥⑦⑧ ❌ |
| 缺口 | 15.①③④⑤⑥⑦⑧（共 7 个子步骤） |
| 严重性 | P0（15.①③④） + P1（15.⑤⑥⑦⑧） |
| 对应 Sprint | Sprint 1 ✅（②）+ Sprint 2（①③④）+ Sprint 5（⑦⑧）+ Sprint 6（⑤⑥） |

---

## 阶段 16 — 无 tool call → final response（9 步降级）

| 项 | 内容 |
|---|---|
| Hermes 行号 | 13588–13905 |
| 16.① partial stream recovery | 已经流出的内容直接当最终响应，避免连接死后 retry 浪费 |
| 16.② 上轮 housekeeping 内容 fallback | 当上一轮"内容 + 仅 housekeeping 工具"时直接拿上轮内容当最终响应 |
| 16.③ post-tool empty nudge | 模型刚执行完 tool 但回空 → 注入一条 user 引导让模型续 |
| 16.④ thinking prefill | 仅 thinking 块没有可见文本 → 把 reasoning 当 prefill 让模型自补，最多 2 次 |
| 16.⑤ 真空响应 retry 3 次 | 仍空才走下一步 |
| 16.⑥ fallback provider | 若 fallback 链可用，先切再说 |
| 16.⑦ 终态 `(empty)` | 写入历史 `(empty)`，break |
| 16.⑧ Codex intermediate ack | "好的，我马上去做" 检测，注入 system 指令继续推 2 次 |
| 16.⑨ 截断 prefix 拼接 | 把阶段 10 留下的 `truncated_response_prefix` 拼回最终响应；strip `<think>` 入历史 |
| Aether 现状 | 仅"response 为空 → `_empty_response_retries += 1` → `EMPTY_RESPONSE` 退出"；上述 9 条全 ❌ |
| 缺口 | 全部 9 个子步骤 |
| 严重性 | P0 |
| 对应 Sprint | Sprint 4 |

---

## 阶段 17 — Turn 出口收尾

| 项 | 内容 |
|---|---|
| Hermes 行号 | 13958–14167 |
| 17.① max_iterations 兜底 summary | `_handle_max_iterations`：单独发一次"裸调用（无工具）"让模型给 summary |
| 17.② trajectory 持久化 | 训练数据采集 |
| 17.③ task-scoped 资源清理 | VM/browser 等子资源 |
| 17.④ 结构化诊断 `Turn ended:` | ~20 种 `_turn_exit_reason`；`last_msg_role=tool` 时 WARNING（"agent 卡住"信号） |
| 17.⑤ plugin post_llm_call | 通知 + 持久化机会 |
| 17.⑥ 当前 turn reasoning 抽取 | 用于 reasoning UI；不跨 turn 边界 |
| 17.⑦ 丰富的 result dict | last_reasoning / token 计数 / cost / pending_steer / interrupt_message / response_previewed |
| Aether 现状 | 有 `_build_result` + `on_session_end` hook；其余全 ❌ |
| 缺口 | 全部 7 个子步骤 |
| 严重性 | P1（①④⑦） + P2（②③⑤⑥） |
| 对应 Sprint | Sprint 3（①⑦）+ Sprint 5（②③④⑤⑥） |

---

## 总结：缺口分布

| 严重性 | 阶段 | 数量 |
|---|---|---|
| P0 | 8 / 9 / 10 / 12.7 / 12.8 / 12.10 / 12.11 / 12.13 / 15.①②③④ / 16.全部 | ~16 个子项 |
| P1 | 2 / 3 / 5 / 7 / 11 / 12.1 / 12.4 / 12.5 / 14 / 15.⑤–⑧ / 17.① / 17.④ / 17.⑦ | ~14 个子项 |
| P2 | 1 / 4 / 6 / 12.2 / 12.3 / 12.6 / 12.9 / 13 / 17.②③⑤⑥ | ~11 个子项 |
| P3 | 4（memory）/ 11.x（账单）等专用功能 | 视产品决定 |

下一步：阅读 `02_p0_critical_gaps.md` 进入实际修复方案。
