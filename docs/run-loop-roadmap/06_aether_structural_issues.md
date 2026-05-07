# Aether 当前结构性问题（动手前必须先修的"地基"）

> 这些问题不在 17 阶段表里，但**它们是 P0/P1 所有功能正确实现的前提条件**。
>
> 如果不先修这些，再多新功能也会被这些"地基"问题拖到不正确的状态。

---

## 结构问题 1 — 引擎实例属性承载 turn 状态（多 session 不安全）

### 现象

```83:87:backend/harness/aether/agents/core/agent.py
        # Cross-turn counters used by nudge policies.
        self._memory_nudge_counter = 0
        self._skill_nudge_counter = 0

        # Turn-local counters reset by _prepare_turn_entry().
        self._empty_response_retries = 0
        self._provider_error_retries = 0
```

```411:413:backend/harness/aether/agents/core/agent.py
        # Reset per-turn counters so retries from a previous turn don't leak.
        self._empty_response_retries = 0
        self._provider_error_retries = 0
```

- `AgentEngine` 是设计为**单实例承多 session** 的（`run_loop` 入参里有 `session_id`，且 `_active_children` / `interrupt_controller` 都是按 session_id 分槽）。
- 但是 `_empty_response_retries` / `_provider_error_retries` / `_memory_nudge_counter` / `_skill_nudge_counter` 全是**实例属性**。
- 后果：两个 session 同时进 `run_loop`：A 在第 5 轮、B 刚进入；B 入口的 `_empty_response_retries = 0` 会**抹掉 A 的累计**；A 的 nudge 计数会被 B 的 turn 累加。

### 修复方案

把 turn 局部状态搬到 `TurnContext.metadata`：

```python
# 新约定：context.metadata 中保留以下标准字段（key 见下）
TURN_KEYS = {
    "empty_response_retries": int,
    "provider_error_retries": int,
    "thinking_prefill_retries": int,
    "invalid_tool_retries": int,
    "invalid_json_retries": int,
    "post_tool_empty_retried": bool,
    "length_continue_retries": int,
    "truncated_tool_call_retries": int,
    "compression_attempts": int,
    "last_content_with_tools": str,
    "last_content_tools_all_housekeeping": bool,
    "iteration_budget": IterationBudget,    # 见 P1-2
    "usage_accumulator": CanonicalUsage,    # 见 P1-4
    ...
}
```

而 nudge 计数器是真正的**跨 turn**状态，本来应该挂在"session" 层面。修法：

1. 引入 `SessionRuntimeState`（新文件 `runtime/session_runtime.py`）：
   ```python
   @dataclass
   class SessionRuntimeState:
       memory_nudge_counter: int = 0
       skill_nudge_counter: int = 0
       turns_since_memory: int = 0
       iters_since_skill: int = 0
       cached_system_prompt: str | None = None       # 见结构问题 4
       use_prompt_caching: bool = False
   ```
2. `AgentEngine` 持有 `dict[session_id, SessionRuntimeState]`，按 session 分槽。
3. `_apply_turn_nudges` 等方法接收 `state: SessionRuntimeState` 参数。

### 验收
- 多线程测试：两个 session 各跑一个 100-step 的 mock turn，断言两个 session 的计数器互不影响。
- `interrupt_controller` 的设计可作为参考（已经按 session_id 分槽）。

---

## 结构问题 2 — 死字段：`_empty_response_retries` 累加但无人读

### 现象

```365:370:backend/harness/aether/agents/core/agent.py
                    if final_response:
                        self._empty_response_retries = 0
                        exit_reason = ExitReason.TEXT_RESPONSE
                    else:
                        self._empty_response_retries += 1
                        exit_reason = ExitReason.EMPTY_RESPONSE
```

`_empty_response_retries` 累加了，但**没有任何代码读取它来决定"继续 retry vs 直接退出"**。
进入 else 分支后无条件 `state_machine.transition(LoopState.FINALIZE); break`。

### 修复方案

这是 P0-8（空响应 9 步降级）的副产物。修复 P0-8 时把这个字段挪到 `context.metadata["empty_response_retries"]` 并在 9 步降级里读它。

### 验收
- 完成 P0-8 的单测就同时验证了这个修复。

---

## 结构问题 3 — 重试控制权全在 provider 内部，引擎层不可干预

### 现象

```66:104:backend/harness/aether/models/provider/openai_compatible.py
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                with httpx.Client(timeout=self.request_timeout_sec) as client:
                    resp = client.post(...)
                    resp.raise_for_status()
                    return self._parse_response(resp.json(), stream_callback=stream_callback)
            except httpx.HTTPStatusError as exc:
                ...
                if status_code in {429, 500, 502, 503, 504, 529} and attempt < self.retry_max_attempts:
                    wait_ms = self._calc_backoff_ms(attempt)
                    time.sleep(wait_ms / 1000)
                    continue
                raise
```

- provider 内部独立做 3 次重试，不可中断、不感知 budget、不感知压缩、不感知 fallback。
- 一旦异常被 provider 抛上来，run_loop 只剩"middleware 一次救场"机会。

### 后果

按现状，下面这些 Hermes 路径在 Aether 是**结构上不可能实现**的（即使加再多 if）：
- 中断进入 retry 等待时立刻退出；
- retry 期间触发上下文压缩；
- retry 期间切到 fallback provider；
- retry 期间感知 token 用量并提前停。

### 修复方案

**把重试控制权上移到引擎层**，provider 只做"单次发包 + 解析"。

1. `OpenAICompatibleModel.generate` 改为单次 `httpx.post`，不再循环重试。
2. provider 抛出**结构化异常**（携带 status_code、Retry-After、body 摘要）：
   ```python
   class ProviderInvocationError(Exception):
       status_code: int | None
       retry_after_seconds: float | None
       body_summary: str | None
       raw: Exception
   ```
3. Engine 在 LLM_CALL 异常分支调 `ErrorClassifier`（P0-7）后**自己决定**是否：
   - 等 `Retry-After` 后重发（可中断）
   - 压缩
   - fallback
   - 失败退出

### 与 P0-7 的关系
P0-7 是依赖于这个修复的。先做这个再做 ErrorClassifier，否则分类器拿不到结构化异常。

### 验收
- 单测：mock provider 抛 429 一次，引擎应在 ≤ Retry-After 时长内调 fallback 而非重发。
- 中断测：retry 等待中 `interrupt()`，断言 200ms 内退出。

---

## 结构问题 4 — System prompt 写入策略破坏 prefix cache

### 现象

```445:489:backend/harness/aether/agents/core/agent.py
    def _prepare_session_and_system_prompt(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> tuple[List[Dict[str, Any]], str | None]:
        ...
        requested_prompt = self._sanitize_text(request.system_message) if request.system_message else None
        selected_prompt = requested_prompt or stored_prompt

        if selected_prompt:
            messages = self._inject_system_prompt(messages, selected_prompt)
        ...
        if selected_prompt and self._session_store and (is_new_session or requested_prompt is not None):
            self._session_store.update_system_prompt(request.session_id, selected_prompt)
```

**只要 `request.system_message` 非 None**，就会把它当作"显式传入"覆盖 SessionStore，
导致每轮 system prompt 实际上**字节级有可能不一致**（即便文本逻辑相同）。
这会让 Anthropic prefix cache 命中率跌到 0。

### 修复方案

1. 引入 `SessionRuntimeState.cached_system_prompt`（结构问题 1 已提到）。
2. 写入策略改为：
   - **新 session**（`session_store.get_session(...)` 返回 None）→ 用 `request.system_message`，写入 store + cache。
   - **续接 session** → **优先使用** `cached_system_prompt`（即使 request 也带了 system_message），保证字节一致。
   - **显式覆盖**：仅当 `request.metadata["force_system_prompt_replace"] == True` 时才覆盖。
3. 压缩 / 清空 session 时显式 `cached_system_prompt = None`。

### 验收
- 单测：续接 session 调 5 次 `run_loop`，断言每次发给 provider 的 system 字节完全相同。
- 单测：`force_system_prompt_replace=True` 时能正常覆盖。

---

## 结构问题 5 — Middleware 的 `on_error` 是单次救场，不是恢复链

### 现象

```202:209:backend/harness/aether/agents/core/agent.py
                    try:
                        prepared_messages = self.services.middleware_pipeline.run_before_llm(messages, context)
                    except Exception as exc:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break
```

```223:234:backend/harness/aether/agents/core/agent.py
                        except Exception as exc:
                            self._provider_error_retries += 1
                            # Give on_error middlewares one chance to convert provider failure
                            # into a normalized assistant response.
                            self._handle_pipeline_error(exc, state_machine.state, context)
                            recovered_response = self._pop_context_response(context, "llm_error_response")
                            if recovered_response is None:
                                error_text = str(exc)
                                exit_reason = ExitReason.PROVIDER_ERROR
                                state_machine.transition(LoopState.FAILED)
                                break
                            response = recovered_response
```

- `on_error` 中间件没有重试循环，没有 backoff，没有 retry 上限。
- 它只是把异常**变成一个 NormalizedResponse**或者放弃。

### 后果

这个抽象不适合做"按错误类型多策略恢复"。例如 P0-7 的 12 个分支在 middleware 里实现会非常蠢
（每个 middleware 自己判 reason + 决定是否能 retry + 改 messages…）。

### 修复方案

**保留 middleware** 用于"修改请求/响应/工具结果的横切关注点"（这是它的本职），
但**新增 `RecoveryStrategy`** 抽象专门处理错误恢复：

```python
class RecoveryStrategy(Protocol):
    def applies_to(self, classified: ClassifiedError) -> bool: ...
    def recover(
        self,
        classified: ClassifiedError,
        *,
        context: TurnContext,
        messages: list[dict],
        engine: "AgentEngine",
    ) -> RecoveryOutcome: ...

@dataclass
class RecoveryOutcome:
    action: Literal["retry", "compress_then_retry", "fallback", "abort"]
    modified_messages: list[dict] | None = None
    backoff_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

引擎按 strategy 列表顺序询问，第一个 `applies_to()` True 的就用它的 `recover`。

### 验收
- 单测：注册 3 个 strategy（thinking_signature / context_overflow / generic backoff），断言每种 mock 异常调到对应的 strategy。

---

## 结构问题 6 — `OpenAICompatibleModel` 一切硬编码：参数、URL、超时

### 现象

```51:101:backend/harness/aether/models/provider/openai_compatible.py
    def generate(
        self,
        ...
    ) -> NormalizedResponse:
        payload = self._build_payload(...)
        ...
        with httpx.Client(timeout=self.request_timeout_sec) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
```

- URL 硬编码 `/chat/completions`；
- 没有 stream payload；
- 没有 `extra_body` 透出；
- 没有 `provider_profile` 概念（不同 provider 接受的字段差异大）。

### 修复方案

1. 引入 `ProviderProfile`（新文件 `models/provider/profile.py`）：
   ```python
   @dataclass
   class ProviderProfile:
       name: str                           # "openai" / "openrouter" / "anthropic_messages" / "moonshot" / ...
       endpoint_path: str                  # "/chat/completions" / "/v1/messages" / ...
       supports_streaming: bool
       supports_prompt_cache: bool
       reasoning_field: Literal["reasoning_content", "reasoning_details", None]
       strict_unknown_fields: bool         # Mistral / Fireworks = True
       request_template: dict              # 静态 extra body
   ```
2. `OpenAICompatibleModel.__init__` 接收 `profile: ProviderProfile`；
3. `_build_payload` 按 profile 决定字段裁剪。

### 验收
- 单测：moonshot profile 输出 payload 含 `reasoning_content`，openai profile 输出 `reasoning`，strict profile 删除两者。

---

## 结构问题 7 — `EngineRequest` 缺少必要字段

### 现象

```82:91:backend/harness/aether/runtime/contracts.py
@dataclass(slots=True)
class EngineRequest:
    session_id: str
    user_message: str | None = None
    system_message: str | None = None
    stream_callback: StreamDeltaCallback | None = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model_config: ModelCallConfig = field(default_factory=ModelCallConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

后续要加的字段：
- `pending_steer: str | None`（P2-2）
- `force_system_prompt_replace: bool`（结构问题 4）
- `task_id: str | None`（现在塞在 `metadata` 里，应该提升为一等字段）
- `persist_user_message: str | None`（持久化 vs API-only 的对比，参考 Hermes 10589 行）

### 修复方案
扩 `EngineRequest`，保持向后兼容：所有新字段默认 None / False。

---

## 结构问题汇总

| # | 问题 | 严重性 | 必须先做的 Sprint |
|---|---|---|---|
| 1 | 引擎实例属性承载 turn 状态 | P0（多 session 安全） | Sprint 0（动 P0-* 之前） |
| 2 | `_empty_response_retries` 死字段 | P1 | Sprint 4（随 P0-8 一起修） |
| 3 | 重试控制权在 provider 内部 | P0 | Sprint 1（P0-7 之前） |
| 4 | System prompt 写入破缓存 | P1 | Sprint 5 |
| 5 | Middleware on_error 不是恢复链 | P0 | Sprint 2（P0-7 之前） |
| 6 | OpenAICompatibleModel 硬编码 | P1 | Sprint 5 |
| 7 | EngineRequest 字段不全 | P1 | 滚动添加 |

> **行动指南**：在动 P0-* 任何功能前，先在一个独立 PR 里完成结构问题 1 + 3 + 5
> 这"3 个地基修复"，为后续所有 P0/P1 工作扫清障碍。

下一步：阅读 [07_sprint_execution_plan.md](./07_sprint_execution_plan.md) 看每个 Sprint 的具体落地节奏。
