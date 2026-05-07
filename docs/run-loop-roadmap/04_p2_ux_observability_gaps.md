# P2 缺口（UX / 可观测性）

> 范围：阶段 1、4、6、12.2、12.3、12.6、12.9、13、17.②③⑤⑥
>
> 这一档不影响"能不能跑通"，影响的是用户体感（卡死、看不见进度、不能纠偏）以及
> 调试 / 计费 / 训练数据采集这些"运营能力"。

---

## P2-1 Plugin 钩子协议升级（阶段 4）

### 现状
`EngineHooks` 是纯通知型，所有方法返回 None：
```13:24:backend/harness/aether/runtime/hooks.py
    def on_session_start(self, *, session_id: str, context_metadata: dict[str, Any]) -> None:
        return None

    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> None:
        return None
```

### 期望

1. **保留通知接口**，但允许 `pre_llm_call` 返回 `HookOutcome | None`：
   ```python
   @dataclass
   class HookOutcome:
       inject_user_context: str | None = None    # 注入到当前 user message 末尾
       inject_system_addendum: str | None = None # 注入到 system prompt 末尾（破缓存，慎用）
       short_circuit_response: NormalizedResponse | None = None  # 提前结束 LLM 调用
   ```
2. run_loop 在 `pre_llm_call` 后聚合 outcome：拼接 `inject_user_context` 到当前 user message。
3. 增加 `pre_api_request` / `post_api_request` 两个钩子（对应 Hermes 11327 / 13128 行），传入 `model / provider / api_call_count / approx_tokens` 等元信息，便于上层做埋点。

### 验收
- 单测：注册一个返回 `inject_user_context="Memory: foo"` 的 hook，断言 user message 末尾有 fenced `<memory>...</memory>` block。
- 单测：返回 `short_circuit_response` 的 hook，断言 provider.generate 未被调用。

---

## P2-2 `/steer` 异步用户引导（阶段 6）

### 现状
完全没有 steer 概念。

### 期望

1. **`AgentEngine.send_steer(session_id, text)`**：线程安全，往该 session 的 `_pending_steer` 队列追加。
2. **drain 时机**：在 PRE_LLM 阶段（每轮 LLM 调用之前），消费 `_pending_steer`：
   - 找最近一条 `role=tool` 消息，content 末尾追加 `\n\nUser guidance: ...`
   - 若没有 tool 消息（例如 turn 第一轮），把文本放回队列，等下一轮再尝试。
3. **收尾回传**：`run_loop` 结束时若仍有未消费的 steer，写入 `EngineResult.metadata["pending_steer"]`，让上层把它当作下一轮的 user message。

### 数据结构变更
- 把 pending_steer 存在 `interrupt_controller` 同级的新组件 `SteerInbox`（线程安全队列），按 session_id 分槽。

### 验收
- 多线程测试：A 线程跑 run_loop；B 线程在 LLM_CALL 中调 `send_steer`；断言下一轮 LLM 请求里的最后一条 tool message 末尾包含该文本。

---

## P2-3 Surrogate / ASCII codec 兜底（阶段 13）

### 现状
仅在 turn 入口做一次 `_sanitize_text`，retry 时不会重清。没有 ASCII codec 路径。

### 期望

1. 把 `_sanitize_text` 拆成：
   - `_strip_surrogates(text) -> str`
   - `_strip_non_ascii(text) -> str`
2. 在 LLM_CALL 异常分支识别 `UnicodeEncodeError`：
   - 若错误信息含 "surrogate" → 全 payload `_strip_surrogates`，retry 一次。
   - 若错误信息含 "ascii" / 系统 LANG=C → 启用 `_force_ascii_payload` 标记，全 payload 包括 `api_key` strip 非 ASCII，retry 一次。
3. 至多 2 次 unicode-recovery 尝试（`context.metadata["unicode_recovery_passes"]`）。

### 验收
- 单测：消息内容含 lone surrogate `\udcff`，第一次 raise，之后 retry 成功。
- 单测：API key 含 ʋ 而不是 v，断言被替换且警告日志输出。

---

## P2-4 image_too_large 缩图（阶段 12.2）

### 现状
没有图像通道，但未来 vision 工具一定会撞 5MB 限制。

### 期望

1. 新增 `runtime/image_shrink.py`：扫描 `messages[*].content` 里的 `image_url` / `source.data` 等节点，对超 5MB 的 base64 图像做 50% 缩小。
2. ErrorClassifier 识别 `image_too_large` reason；run_loop 调用一次 shrink，成功 retry，否则 fall through 到正常错误处理。

### 验收
- 单测：构造一条带 6MB base64 图片的消息，shrink 后 ≤ 3MB。

---

## P2-5 Anthropic OAuth 1M-context beta 关闭（阶段 12.3）

### 现状
未接 Anthropic provider；接入时会撞此问题。

### 期望

放在 `AnthropicProvider`（未来）的内部状态里：
- `_oauth_1m_beta_disabled: bool = False`
- 收到 401/403 + body 含 `"long context beta is not yet available"` → 置 True 并重建 client，retry 一次。
- 注意此状态**不持久化**，仅 in-memory（防止用户后来开通了 1M 但仍被本地状态阻塞）。

### 验收
- 留到接入 AnthropicProvider 时再写测试。

---

## P2-6 llama.cpp grammar 拒绝恢复（阶段 12.6）

### 现状
未对接 llama.cpp 后端；如果用户用本地推理服务会撞。

### 期望

1. 新增 `tools/schema_sanitizer.py` 的 `strip_pattern_and_format(tools)`：从 tool schema 里递归剥 `pattern` / `format` 字段。
2. ErrorClassifier 识别 `llama_cpp_grammar_pattern` reason；调一次 strip 后 retry。

### 验收
- 单测：tool schema 含 `{"pattern": "\\d+"}`，strip 后该字段消失。

---

## P2-7 跨会话 rate guard（阶段 12.9）

### 现状
单进程内多 session 并发时，所有 session 都会争抢同一个 provider 配额。

### 期望

1. 新增 `runtime/rate_guard.py`：
   - 文件锁 + JSON 文件：`{provider}_{base_url_hash}.json`，记录 `until_unix: float, reason: str`。
   - 在每轮 LLM_CALL 之前检查；如果存在未到期的 lockout，**直接走 fallback**而不是发请求。
2. 收到 429 + headers 含 `x-ratelimit-reset` 时写入此文件。
3. 成功响应时清除该 provider 的锁。

### 数据结构变更
- 锁存放路径默认 `{xdg_runtime_dir}/aether/rate_guard/`，可通过 `EngineConfig.rate_guard_dir` 覆盖。

### 验收
- 单测：模拟两个并发 session，第一个写入 60s lockout，第二个进 LLM_CALL 时跳过 provider 直接 fallback。

---

## P2-8 Trajectory 持久化（阶段 17.②）

### 现状
没有 trajectory 概念。

### 期望

1. `EngineConfig.save_trajectories: bool = False`、`trajectory_dir: Path = "./trajectories"`。
2. `_save_trajectory(messages, user_query, completed)`：
   - 写一行 JSON 到 `trajectories/sessions/{session_id}.jsonl`（成功）或 `failed/{session_id}.jsonl`（失败）。
   - 字段：`{ts, model, provider, completed, conversations: [...]}`。
3. `_convert_to_trajectory_format(messages, user_query, completed)`：标准化为 `[{from: "human" | "gpt", value: ...}]` 形式。

### 验收
- 单测：完成一个 turn 后断言对应 jsonl 文件存在且包含 user / assistant 两条记录。

---

## P2-9 Task-scoped 资源清理（阶段 17.③）

### 现状
未来子代理 / VM / 浏览器工具会需要按 task_id 清理。

### 期望

1. `EngineHooks.on_task_cleanup(task_id, completed)` 钩子。
2. `_cleanup_task_resources(task_id)` 在 `run_loop` 结束的 `finally` 里调用。
3. 工具可以注册 `tool.acquire_task_resource(task_id)` / `release_task_resource(task_id)`。

### 验收
- 单测：注册一个会跟踪 acquire/release 的假工具，run_loop 结束后断言 release 被调用。

---

## P2-10 当前 turn reasoning 抽取（阶段 17.⑥）

### 现状
`EngineResult` 里没有 reasoning 字段。

### 期望

`_extract_last_reasoning(messages, turn_start_idx)`：
- 从 `messages` 末尾向前走，遇到 `role=user` 立即 break（turn 边界）。
- 第一条非空 `reasoning` / `reasoning_content` 字段就是答案。

写入 `EngineResult.metadata["reasoning"]["last_reasoning"]`。

### 验收
- 单测：turn 内有 3 条 assistant 消息（前 2 个有 reasoning，最后一个没有），抽出来的应是倒数第二条的 reasoning。
- 单测：跨 user 消息的 reasoning 不应被抽到。

---

## P2-11 失败请求落盘（`_dump_api_request_debug`）

### 现状
provider 异常时只能靠 logger.exception；定位"为什么这次请求失败"很慢。

### 期望

1. `EngineConfig.dump_failed_requests: bool = False`、`request_dump_dir: Path = "./request_dumps"`。
2. `_dump_api_request_debug(api_kwargs, *, reason: str, error: Exception | None = None)`：
   - 写文件 `{ts}_{reason}.json`，含 `{model, provider, base_url, error: str, kwargs: {...}}`。
3. 触发点：max_retries_exhausted、non_retryable_client_error、invalid_response after fallback。

### 验收
- 单测：触发一次 PROVIDER_ERROR，断言 dump 文件存在且包含 model/provider 字段。

---

## P2-12 死连接清理（阶段 1 子项）

### 现状
没有连接池层面的健康检查。如果 provider 之前连接被 keep-alive 保留但已经死了，第一次请求会 hang 30s+ 才报错。

### 期望

1. `OpenAICompatibleModel.cleanup_dead_connections() -> bool`：调用前 socket 健康检查；如果有连接确认是死的，直接关闭 client 并重建。
2. run_loop turn 入口（仅在跨多个 turn 时）调一次。

### 验收
- 集成测：模拟 keep-alive 连接被防火墙断掉，重建后第一次请求能在正常时间内响应。

---

## P2 总览

P2 整体上不是"修单个 bug"，而是把工程化能力补齐：
- **可观测性**（P2-8、P2-10、P2-11）
- **可纠偏**（P2-1、P2-2）
- **环境适配**（P2-3、P2-6、P2-12）
- **多 session 安全**（P2-7、P2-9）

P2 全部放在 Sprint 5 / Sprint 6，且彼此之间没有强依赖，可以独立 PR 推进。

下一步：阅读 [05_p3_advanced_features.md](./05_p3_advanced_features.md)。
