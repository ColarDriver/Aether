# PR 4.3 — Streaming 阶段 API 错误 Withholding

> **角色**：把"流式过程中遇到的 API 错误"从"立即 yield 出来 → 走 recovery"
> 改成"先 hold 在内部 → 流式结束后统一进 recovery"。这样 UI 不会闪烁
> "Error → Retry → OK"，历史也不会被半截 error 污染。
>
> 借鉴 [`open-claude-code/src/query.ts:788-825`](../../tmp/claude-code-references) 的
> `withheld` 机制（详见 00_overview.md § 1.2 设计 B）。

## 一、目标

1. 在 `agent.py::_invoke_provider_with_recovery` 周围新增一层 **`StreamingErrorBuffer`**，
   把以下三类错误"hold 住不立即 yield"：
   - `prompt_too_long` / `payload_too_large`（context_overflow / 413）
   - `max_output_tokens`（finish_reason="length" 但 tool_calls 又异常的情况）
   - `media_size_too_large`（image / pdf 太大）
2. 流式结束后**统一交给 ClassifiedRecoveryStrategy**（PR 2.2）做处理：
   - 优先调 CompactionPipeline.maybe_compress（PR 3.4 流水线）
   - 次选 strip thinking / strip media
   - 都不行才 surface 错误
3. **引擎层强制升级 fallback**（来自 Codex 反馈 #6）：当 cascade 已经尝试过一次
   `compress_context` 但下一次 attempt 仍然抛 withholdable 错误（同一类，例如连续 413），
   引擎**不再询问 strategy**，自行合成 `RecoveryDecision(activate_fallback=True)`
   并尝试旋转 fallback。前提：`fallback_chain.has_next()` 且 `fallback_chain_enabled=True`。
   这样不需要修改 `RecoveryDecision` 契约，也不要求 strategy 配合——是引擎层 cascade 的语义补强。
4. 与 7 步空响应降级（PR 4.2）**正交**：withholding 处理的是"API 返回的错误对象"，
   7 步处理的是"API 返回的正常响应但内容为空"。两者不重叠。
5. 与 stream_callback 兼容：withholding 不影响 callback 的 delta 流；
   `stream_callback_calls` 计数器照常推进。

## 二、现状分析

### 2.1 Aether 当前的错误处理路径

`agent.py::_invoke_provider_with_recovery` 流程（伪代码）：

```python
while True:
    try:
        response = self.services.provider.generate(messages, callback=stream_cb)
        return response
    except ProviderInvocationError as exc:
        classified = self.error_classifier.classify(exc)
        decision = self.recovery_strategy.first_match(classified).recover(...)
        if decision.activate_fallback:
            self.services.fallback_chain.activate_next(...)
            continue
        if decision.compress_context:
            messages = self.compaction_pipeline.maybe_compress(...)
            continue
        if decision.strip_thinking:
            messages = strip_thinking_signature(messages)
            continue
        # 没有 recovery 路径 → re-raise
        raise
```

**问题**：
- 错误抛出**立即**触发 recovery；如果在流式过程中（`callback` 已经 emit 过 token），
  UI 上会先看到部分 token，再看到 error，再看到 retry 后的新内容——视觉抖动。
- 错误对象本身的语义（context_overflow vs payload_too_large vs ...）有时通过
  HTTP 头/响应体传递，不一定能在抛错那一瞬完整提取，需要等流结束后再统一判定。
- claude-code 的 `withheld = true` 模式让流自然结束（包括 `message_delta` 的 stop_reason），
  然后再做"该不该 surface"的判定，比"边流边判"更稳。

### 2.2 claude-code 的 withholding 模式

`src/query.ts:788-825` 关键片段：

```typescript
let withheld = false
if (feature('CONTEXT_COLLAPSE')) {
  if (contextCollapse?.isWithheldPromptTooLong(message, ...)) {
    withheld = true
  }
}
if (reactiveCompact?.isWithheldPromptTooLong(message)) { withheld = true }
if (mediaRecoveryEnabled && reactiveCompact?.isWithheldMediaSizeError(message)) {
  withheld = true
}
if (isWithheldMaxOutputTokens(message)) { withheld = true }

if (!withheld) {
  yield yieldMessage  // ← 错误才会被 UI 看到
}
if (message.type === 'assistant') {
  assistantMessages.push(message)  // 错误对象仍进 buffer，供后续 recovery 读
  ...
}
```

之后在循环出口（`src/query.ts:1062-1175`）：

```typescript
if (!needsFollowUp) {
  const lastMessage = assistantMessages.at(-1)
  const isWithheld413 = lastMessage?.isApiErrorMessage && isPromptTooLongMessage(lastMessage)
  if (isWithheld413) {
    // 1. drain context collapse
    // 2. reactive compact
    // 3. surface
  }
  if (isWithheldMaxOutputTokens(lastMessage)) {
    // 1. 8k → 64k escalation
    // 2. inject "Resume directly" continuation user message
    // 3. surface
  }
}
```

**关键设计点**：
- 错误**入 buffer 但不出 UI**。
- 流自然走完（`message_delta` 的 stop_reason 才到位）。
- 流结束后**集中 dispatch** 多种 recovery（先便宜再昂贵）。
- recovery 全失败才 surface error。

## 三、设计

### 3.1 `StreamingErrorBuffer`（新概念）

在 Aether 里，"流式过程"被封装在 `provider.generate(messages, callback=...)` 里，
provider 直接 raise `ProviderInvocationError`。我们没有 claude-code 那种"流式过程
yield 多个 message" 的结构。所以 withholding 的实现位置不同：

**Aether 的实现位置**：在 `_invoke_provider_with_recovery` 内部，
当 catch 到 `ProviderInvocationError` 时**先判断是不是 withholdable**，
是 → 不立即进 recovery，而是把错误存进 `WithholdingBuffer`，
继续 sit in catch；流式 callback 自然结束（因为 `provider.generate` 已经返回了），
然后统一在 buffer flush 时进 recovery。

但这样会损失"先回退一次试试"的语义。所以我们采用更简单的设计：

**最终设计**：在 PR 2.2 的 `_invoke_provider_with_recovery` 周围加一层
**"withholding 上下文管理器"**，并不改变错误抛出/处理时机，只做两件事：

1. **延迟 stream callback 中关于错误的 visible 通知**。Aether 当前 stream callback
   只接收 `delta: str`（文本增量），并不直接接收错误——所以这一层基本是**no-op**，
   除非我们引入新的"error notification"通道。
2. **集中错误 dispatch**：把 PR 2.2 的 `decision.activate_fallback / compress_context /
   strip_thinking` 三轴正交分发**升级**为有先后顺序的级联 recovery（先 collapse 再 reactive
   compact 再 strip thinking 再 fallback 再 surface）。

> **结论**：在 Aether 的同步 / 单错路径里，"withholding" 的核心收益其实是
> **"recovery 级联升级"**而不是"延迟 yield"。我们把 PR 4.3 重新框定为
> **"recovery 级联 + 安静失败抑制"**，并在文档里保留"withholding"术语以对齐 claude-code。

### 3.2 实际改动：recovery 级联升级

#### 3.2.1 `WithholdingBuffer` dataclass

```python
@dataclass
class _WithholdingState:
    """Per-attempt state for an in-flight provider call.

    Sprint 4 / PR 4.3 — collects all "withhold" hints discovered while
    the call was in flight (or while subsequent recoveries were tried)
    so the run-loop can decide between cascading recovery vs surfacing
    the error.

    The buffer is created at the top of _invoke_provider_with_recovery
    and discarded when the call eventually succeeds OR surfaces.
    """

    pending_errors: list["ProviderInvocationError"] = field(default_factory=list)
    """All errors received during this call's recovery chain.  When the
    chain decides to surface, the LAST error is surfaced (it's the one
    that survived all the recovery attempts).  When recovery succeeds,
    the buffer is discarded silently — the user never sees the
    transient errors."""

    cascade_log: list[str] = field(default_factory=list)
    """Human-readable trail of recovery steps tried, e.g.
    ["payload_too_large", "compress_context(tier4)", "compress_context(tier5)",
     "strip_thinking", "fallback_chain.activate_next"]."""

    suppressed_callback_notifications: int = 0
    """Count of callback notifications about errors that were suppressed.
    For now always 0 — Aether's callback channel is text-only."""

    compression_attempted_for: set[str] = field(default_factory=set)
    """Codex 反馈 #6: track which classified_reason values have already
    been "fixed" with compress_context.  When the same reason fires
    again on the NEXT attempt, the cascade upgrades to a forced
    activate_fallback (without consulting the strategy again) on the
    assumption that more compression won't help.

    Keys are :class:`FailoverReason.value` strings, e.g.
    ``"context_overflow"``, ``"payload_too_large"``,
    ``"max_output_tokens"``."""
```

#### 3.2.2 升级 `_invoke_provider_with_recovery` 为级联 recovery

```python
def _invoke_provider_with_recovery(
    self,
    *,
    messages: list[dict],
    request: EngineRequest,
    context: TurnContext,
    state_machine: LoopStateMachine,
) -> NormalizedResponse:
    """Invoke the provider, escalating through recovery cascades.

    Sprint 4 / PR 4.3 — wraps the existing PR 2.2 recovery loop with
    a withholding buffer.  When an error fires, the recovery decision
    is consulted; if it offers ANY cascade option, we try them in order
    of cost (cheap-first) and only surface the error after all options
    are exhausted.  The user-visible callback chain sees zero error
    notifications during the cascade.

    Cascade order (cheap → expensive):
      1. compress_context (CompactionPipeline; tier 2/3/4 first, tier 5 last)
      2. strip_thinking (Sprint 0 - existing)
      3. activate_fallback (FallbackChain; PR 2.2)
    """
    state = _WithholdingState()
    attempts = 0
    max_attempts = max(1, int(getattr(self.config, "max_provider_recovery_attempts", 8)))

    while attempts < max_attempts:
        attempts += 1
        # PR 4.2: reset streamed_assistant_text before each attempt
        context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = ""
        try:
            response = self.services.provider.generate(
                messages,
                stream_callback=self._build_stream_callback(request, context),
                stream_silent_callback=self._build_stream_silent_callback(request, context),
            )
            # Success — discard buffered errors silently.
            self._observe_recovery_cascade(context, state, terminal="success")
            return response
        except ProviderInvocationError as exc:
            state.pending_errors.append(exc)
            classified = self.error_classifier.classify(exc, ...)
            decision = self.recovery_strategy.first_match(classified).recover(...)

            if not self.config.error_withholding_enabled:
                # User opted out of cascade — fall back to PR 2.2 behaviour:
                # surface immediately if no recovery, else single-step.
                applied = self._apply_recovery_decision_singleshot(
                    decision=decision, messages=messages, context=context, exc=exc,
                )
                if applied:
                    state.cascade_log.append(f"singleshot:{decision.classified_reason}")
                    continue
                self._observe_recovery_cascade(context, state, terminal="surface")
                raise

            # Codex 反馈 #6: forced fallback upgrade.
            # If we already compressed for the same classified_reason in
            # an earlier attempt and the SAME reason fires now, more
            # compression won't help — synthesise an activate_fallback
            # decision (only if a fallback is actually available).
            decision = self._maybe_upgrade_decision_for_repeat_withholding(
                decision=decision, classified=classified, state=state,
            )

            # Cascade recovery: try all relevant decisions in cheap → expensive order.
            applied_any = self._apply_recovery_decision_cascade(
                decision=decision, messages=messages, context=context,
                state=state, exc=exc,
            )
            if applied_any:
                continue

            # All cascade options exhausted — surface.
            self._observe_recovery_cascade(context, state, terminal="surface")
            raise

    # Outer attempt cap reached without success — surface the last error.
    self._observe_recovery_cascade(context, state, terminal="exhausted")
    raise state.pending_errors[-1] if state.pending_errors else (
        ProviderInvocationError("Recovery exhausted with no terminal error")
    )
```

#### 3.2.3 `_apply_recovery_decision_cascade`

```python
def _apply_recovery_decision_cascade(
    self,
    *,
    decision: RecoveryDecision,
    messages: list[dict],
    context: TurnContext,
    state: _WithholdingState,
    exc: ProviderInvocationError,
) -> bool:
    """Apply recovery hints in cascade order; return True if at least
    one was successfully applied (caller continues the loop)."""
    applied = False

    # Step 1 — compress_context (cheapest and most likely to help for 413/CTX).
    if decision.compress_context and self.config.compression_enabled:
        new_messages, freed = self.compaction_pipeline.maybe_compress(
            messages, ctx=self._make_compaction_ctx(context),
            turn_context=context, reason=decision.classified_reason or "withholding",
        )
        if freed > 0 or len(new_messages) < len(messages):
            messages[:] = new_messages
            state.cascade_log.append(
                f"compress_context(freed={freed},reason={decision.classified_reason})"
            )
            # Codex 反馈 #6: remember that compression has been tried for
            # this classified_reason — used by the forced-fallback upgrade
            # on the next attempt if the same reason fires again.
            if decision.classified_reason:
                state.compression_attempted_for.add(decision.classified_reason)
            applied = True

    # Step 2 — strip_thinking (also cheap; helps for thinking_signature errors).
    if decision.strip_thinking:
        new_messages = strip_thinking_signature(messages)
        if new_messages != messages:
            messages[:] = new_messages
            state.cascade_log.append(f"strip_thinking({decision.classified_reason})")
            applied = True

    # Step 3 — activate_fallback (most expensive: rotates to next provider).
    if decision.activate_fallback:
        chain = self.services.fallback_chain
        if chain is not None and chain.has_next():
            rotated = chain.activate_next(reason=decision.classified_reason or "withholding")
            if rotated:
                state.cascade_log.append(
                    f"fallback({chain.current_slot_name})"
                )
                # Reset attempt counter for fresh provider — same as PR 2.2.
                applied = True

    return applied
```

#### 3.2.4 `_maybe_upgrade_decision_for_repeat_withholding`（**Codex 反馈 #6 新增**）

```python
# Reasons we treat as "withholdable" for the cascade upgrade rule.
# These are the failure classes where compression is the natural first
# response; if it doesn't work the only structural fix left is to ask
# a different provider.
_WITHHOLDABLE_FAILOVER_REASONS: frozenset[str] = frozenset({
    "context_overflow",       # FailoverReason.CONTEXT_OVERFLOW
    "payload_too_large",      # FailoverReason.PAYLOAD_TOO_LARGE
    "max_output_tokens",      # FailoverReason.MAX_OUTPUT_TOKENS
    # NOTE: rate_limit / 429 is NOT in this set — backoff + retry is
    # the right shape there; PR 2.2 already handles it.
})


def _maybe_upgrade_decision_for_repeat_withholding(
    self,
    *,
    decision: RecoveryDecision,
    classified: ClassifiedFailure,   # whatever the classifier returns
    state: _WithholdingState,
) -> RecoveryDecision:
    """Engine-level upgrade: when the same withholdable reason fires
    twice in a row and we already compressed for it, add an
    activate_fallback hint to the decision.

    The strategy is not asked again — this is a structural conclusion
    the engine draws from the cascade trail.  The upgrade is a no-op
    when:
      - the reason is not withholdable
      - the strategy already set activate_fallback
      - the engine has no FallbackChain configured, no further slot,
        or fallback is disabled

    Returns either ``decision`` unchanged or a new decision with
    ``activate_fallback=True``.  ``compress_context`` is NOT cleared:
    if the caller still wants to try one more compression pass that's
    OK; the cascade applies all relevant hints in cheap → expensive
    order, so an extra (likely no-op) compress is harmless.
    """
    if not self.config.error_withholding_enabled:
        return decision
    reason = decision.classified_reason or ""
    if reason not in _WITHHOLDABLE_FAILOVER_REASONS:
        return decision
    if decision.activate_fallback:
        return decision  # strategy already requested it
    if reason not in state.compression_attempted_for:
        # First time this reason fires — let the cascade do its normal
        # compress pass; the upgrade only kicks in on the SECOND occurrence.
        return decision
    chain = self.services.fallback_chain
    if chain is None or not getattr(self.config, "fallback_chain_enabled", True):
        return decision
    if not chain.has_next():
        return decision
    state.cascade_log.append(
        f"force_fallback_upgrade(reason={reason},after_compression=true)"
    )
    return RecoveryDecision(
        retry=decision.retry,
        wait_seconds=decision.wait_seconds,
        reason=f"{decision.reason}|forced-fallback-after-compress-no-progress",
        activate_fallback=True,
        compress_context=decision.compress_context,
        strip_thinking=decision.strip_thinking,
        classified_reason=decision.classified_reason,
    )
```

#### 3.2.5 `_apply_recovery_decision_singleshot`（保留 PR 2.2 行为）

```python
def _apply_recovery_decision_singleshot(self, *, decision, messages, context, exc):
    """The PR 2.2 single-step apply path, kept for ``error_withholding_enabled=False``
    backwards-compat.  Returns True if any action was applied."""
    if decision.activate_fallback and self.services.fallback_chain:
        if self.services.fallback_chain.activate_next(reason=decision.classified_reason):
            return True
    if decision.compress_context and self.config.compression_enabled:
        new_messages, freed = self.compaction_pipeline.maybe_compress(...)
        if freed > 0 or len(new_messages) < len(messages):
            messages[:] = new_messages
            return True
    if decision.strip_thinking:
        new_messages = strip_thinking_signature(messages)
        if new_messages != messages:
            messages[:] = new_messages
            return True
    return False
```

#### 3.2.6 Observability

```python
def _observe_recovery_cascade(
    self, context: TurnContext, state: _WithholdingState, *, terminal: str
) -> None:
    """Surface the cascade trail into TurnContext.metadata for EngineResult."""
    md = context.metadata.setdefault("recovery", {})
    md["cascade_log"] = list(state.cascade_log)
    md["pending_errors_count"] = len(state.pending_errors)
    md["terminal"] = terminal
    md["suppressed_callback_notifications"] = state.suppressed_callback_notifications
    if state.cascade_log:
        self.services.logger.info(
            "recovery cascade terminal=%s steps=%s errors=%d",
            terminal, " → ".join(state.cascade_log), len(state.pending_errors),
        )
```

### 3.3 `EngineResult.metadata["recovery"]` 契约

PR 4.3 输出契约（`agent.py::_build_result` 收尾增加）：

```python
md["recovery"] = {
    "cascade_log": context.metadata.get("recovery", {}).get("cascade_log", []),
    "pending_errors_count": int(
        context.metadata.get("recovery", {}).get("pending_errors_count", 0)
    ),
    "terminal": context.metadata.get("recovery", {}).get("terminal", "n/a"),
    "withheld": int(
        context.metadata.get("recovery", {}).get("suppressed_callback_notifications", 0)
    ),
}
```

### 3.4 `EngineConfig` 新字段

```python
# Sprint 4 / PR 4.3: cascade-style provider recovery.
# When True, _invoke_provider_with_recovery tries multiple recovery
# hints (compress → strip_thinking → fallback) in cheap → expensive
# order before surfacing the error to the run loop.  When False, the
# PR 2.2 single-step apply path is used (back-compat).
error_withholding_enabled: bool = True

# Hard cap on cascade attempts per provider call.  Each attempt may
# apply ANY of compress / strip / fallback in one shot, so the
# practical maximum is small (3-5 in real workloads).  Set to 1 to
# disable cascading entirely (equivalent to error_withholding_enabled=False).
max_provider_recovery_attempts: int = 8
```

### 3.5 与 7 步空响应降级（PR 4.2）的边界

|  | API 错误（PR 4.3） | 空响应（PR 4.2） |
|---|---|---|
| 触发 | provider.generate 抛出 ProviderInvocationError | provider.generate 正常返回，但 content="" |
| 处理位置 | `_invoke_provider_with_recovery` 内（在 LLM_CALL state 内） | `_finalize_empty_response` → `_handle_empty_response`（7 步）；非空 content 进 `_maybe_continue_codex_intermediate_ack` finalise hook |
| 工具 | ClassifiedRecoveryStrategy + CompactionPipeline + FallbackChain（含 Codex 反馈 #6 的 forced upgrade）| 7 步状态机（truncated_prefix / partial stream / housekeeping / nudge / prefill / retry / fallback / terminal）+ Codex finalise hook |
| 失败 surface | `raise ProviderInvocationError` → run-loop 捕获 → exit_reason=PROVIDER_ERROR / CONTEXT_EXHAUSTED 等 | exit_reason=EMPTY_RESPONSE |
| 共享什么 | 都用 FallbackChain，但激活次数共享同一 budget（`max_fallback_activations_per_turn`） | 共享 |

**关键纪律**：PR 4.3 不进入"空响应"分支；PR 4.2 不进入"recovery cascade"分支。
两者通过 metadata key 命名空间分离（`metadata["recovery"]` vs `metadata["empty_recovery"]`）。

### 3.6 与 stream_callback 的关系

PR 4.3 **不改** `_build_stream_callback`（PR 4.2 已经在那里加了累计逻辑）。
withholding 的"延迟 yield"在 Aether 里目前是 no-op，因为 callback 只 emit text delta，
不 emit 错误对象。如果 Sprint 6 引入新的"error notification"通道，**那时再回来扩展**
本 PR 的 `suppressed_callback_notifications` 计数器与延迟逻辑。

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/agents/core/agent.py` | 修改 | `_WithholdingState` dataclass + `_invoke_provider_with_recovery` 重构 + `_apply_recovery_decision_cascade` + `_apply_recovery_decision_singleshot` + `_observe_recovery_cascade` + `_build_result` 加 recovery 子字典 | ~250 净增 |
| `backend/harness/aether/runtime/recovery.py` | 修改 | `RecoveryDecision` 加 `cascade_priority` 提示字段（可选）；ClassifiedRecoveryStrategy 不变 | ~10 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 `error_withholding_enabled` + `max_provider_recovery_attempts` | ~15 |
| `backend/harness/aether/tests/agents/core/test_recovery_cascade.py` | **新文件** | 见 § 五（约 25 case） | ~400 |
| `backend/harness/aether/tests/agents/core/test_recovery_cascade_integration.py` | **新文件** | 与 PR 2.2 / PR 3.4 集成（约 8 case） | ~250 |

## 五、测试用例（详细）

### 5.1 测试组 A：单步级联

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 第一次 413；compress 释放 5000 token；第二次 200 OK | recovery_cascade_log=["compress_context(freed=5000,reason=payload_too_large)"]；terminal=success；user 没看到 error |
| **T-A2** | 第一次 413；compress 释放 0（流水线全跳）；fallback chain 有下家 | cascade_log 含 compress + fallback；第二次成功 |
| **T-A3** | 413；compress 没用；fallback chain 没下家 | terminal=surface；raise 原 exception；UI 看到 PROVIDER_ERROR |
| **T-A4** | thinking_signature 错；strip_thinking 成功；下次 200 OK | cascade_log=["strip_thinking(thinking_signature)"]；terminal=success |

### 5.2 测试组 B：多步级联

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 413；compress 释放 1000（不够）；同样 413；fallback；下家 200 | cascade_log 含 `compress_context(...)`、`force_fallback_upgrade(reason=payload_too_large,after_compression=true)`、`fallback(provider_2)`；terminal=success |
| **T-B2** | 413 → compress → 413 again → strip_thinking → 413 again → fallback → 200 | 三步 cascade 全记录；含 forced upgrade；最终 success |
| **T-B3** | max_provider_recovery_attempts=2；3 次都 413 | terminal=exhausted；surface 第 2 次的 error |
| **T-B4** | 第 1 次 429 rate limit；fallback；第 2 次 413；compress；第 3 次 200 | 不同分类的混合 cascade；都正确触发；429 不触发 forced upgrade（不在 _WITHHOLDABLE_FAILOVER_REASONS） |
| **T-B5** | **Codex 反馈 #6 专项**：strategy 始终给 `compress_context=True / activate_fallback=False`；连续 413：第 1 次 compress 释放 100（无效）；第 2 次同样 413；引擎应**不询问 strategy**直接合成 `activate_fallback=True` | cascade_log 含 `force_fallback_upgrade(...)`；rotation 发生；下家 200 后 success |
| **T-B6** | 同 B5 但 `fallback_chain.has_next()=False` | 不升级；cascade exhaust → surface 原 413 |
| **T-B7** | 同 B5 但 `fallback_chain_enabled=False` | 不升级；同 B6 |
| **T-B8** | 同 B5 但第一次 reason="rate_limit"（不在 withholdable 集合） | 不升级；按 strategy 决定行为 |

### 5.3 测试组 C：回退路径（error_withholding_enabled=False）

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | error_withholding_enabled=False；413；fallback 一次；200 OK | 行为与 PR 2.2 一致（singleshot apply）；cascade_log=["singleshot:payload_too_large"] |
| **T-C2** | error_withholding_enabled=False；413；compress 释放；200 OK | singleshot apply；不级联到 fallback |
| **T-C3** | error_withholding_enabled=False；413；都失败 | surface；与 PR 2.2 行为完全一致 |

### 5.4 测试组 D：与 7 步空响应降级正交

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | provider 返回 200 OK + content=""（不是错误）| _invoke_provider_with_recovery 返回正常；进 _finalize_empty_response → _handle_empty_response；recovery_cascade_log=[]（空）|
| **T-D2** | provider 抛 413；recovery cascade 成功；返回 content="Hello" | recovery_cascade_log 有；empty_recovery.classification=NOT_EMPTY；两个 metadata 子字典各自独立 |
| **T-D3** | provider 抛 413；recovery 失败 surface；run-loop catch → exit_reason=CONTEXT_EXHAUSTED | 7 步降级**不**触发（exit_reason 不是 EMPTY_RESPONSE） |

### 5.5 测试组 E：observability

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | 完整成功 cascade，含 3 步 | logger.info 含 "recovery cascade terminal=success steps=A → B → C" |
| **T-E2** | 失败 surface | logger 输出 "terminal=surface"；EngineResult.metadata["recovery"]["terminal"]="surface" |
| **T-E3** | 0 步 cascade（第一次就成功）| cascade_log=[]；pending_errors_count=0；EngineResult metadata 干净 |
| **T-E4** | exhausted 路径 | metadata["recovery"]["terminal"]="exhausted"；surface 的是最后一个错误 |

### 5.6 测试组 F：边界

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | fallback_chain=None（未配置） | cascade 跳过 fallback step；只 compress + strip_thinking |
| **T-F2** | compression_enabled=False | cascade 跳过 compress step；走其它 |
| **T-F3** | decision 三个 hint 全 False（unknown error） | cascade_apply 返回 False；surface |
| **T-F4** | provider.generate 一次 raise，第二次 raise；类型不同 | 两个 error 都 push 进 pending_errors；surface 的是最后一个 |
| **T-F5** | callback 在 cascade 期间正常工作（流式 partial 累计） | streamed_assistant_text 在每次 attempt 入口被 reset；T-A1 验证最终 metadata 与最终 attempt 一致 |

### 5.7 测试组 G：集成（与 PR 2.2 / PR 3.4 共跑）

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 真实 ScriptedProvider：413 → compaction Tier 2/3 释放但不够 → Tier 5 释放 → 200 OK | 完整 compact pipeline 在 cascade 内运行；T-G1 跑全程 |
| **T-G2** | 真实 ScriptedProvider：429 rate_limit → fallback 切换 → 200 OK | PR 2.2 fallback 行为不回归 |
| **T-G3** | mock provider：连续 5 次 413 + compaction 永远只能释放 100 token | terminal=exhausted；surface CONTEXT_EXHAUSTED |
| **T-G4** | mock provider：413 → compress → 200，但 200 的 content="" + finish_reason="length" | recovery cascade 成功；进 7 步降级；step 6 retry → eventual EMPTY_RESPONSE |
| **T-G5** | mock provider：连续 thinking_signature → strip → 仍然 thinking_signature → strip 已是 no-op → 进入 fallback | cascade 正确退化（同步骤不重复无效 apply） |
| **T-G6** | full session 真实 anthropic mock + compaction + fallback chain | 端到端 result.metadata 完整 |
| **T-G7** | 用户传入 stream_callback；recovery cascade 期间 callback 不被错误污染 | callback 只看到成功 attempt 的 delta；errors 都 hidden |
| **T-G8** | recovery_cascade 与 invalid_json 路径不冲突 | 完整一轮：413 cascade → 200 OK with bad JSON → invalid-JSON inject → next turn OK |

## 六、验收门

- [ ] 所有新测试 green（约 37 case，含 Codex 反馈 #6 的 4 个新 case）
- [ ] PR 2.2 fallback 测试套件完全不回归（重要）
- [ ] PR 3.4 compaction 测试套件完全不回归
- [ ] PR 4.1/4.2 既有测试不回归
- [ ] `result.metadata["recovery"]["cascade_log"]` 在 cascade 触发时非空
- [ ] cascade_log 在 forced upgrade 命中时含 `force_fallback_upgrade(reason=...)` 条目
- [ ] 手工：mock provider 连续 413 → cascade 成功，UI 没看到错误闪烁
- [ ] 手工：error_withholding_enabled=False 时行为与 PR 2.2 完全一致

## 七、回滚开关

- `error_withholding_enabled=False`：退回 PR 2.2 单步 apply 行为（cascade 关闭，但 metadata.recovery 子字典仍输出，含 `terminal="singleshot"` 标记）。
- `max_provider_recovery_attempts=1`：等价于"不 cascade"（每次 attempt 后就 surface）。
- 完全 revert PR：删除 `_WithholdingState` + `_apply_recovery_decision_cascade` + `_observe_recovery_cascade`，把 `_invoke_provider_with_recovery` 改回 PR 2.2 的版本。

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. `_WithholdingState` dataclass（含 `compression_attempted_for`）+ `EngineConfig` 字段 | 30min | 编译通过 |
| 2. `_apply_recovery_decision_cascade`（含写 `compression_attempted_for`）+ `_apply_recovery_decision_singleshot` 抽取 | 2h | 两条路径并存 |
| 3. `_maybe_upgrade_decision_for_repeat_withholding` (Codex 反馈 #6) | 1h | helper + `_WITHHOLDABLE_FAILOVER_REASONS` 常量 |
| 4. `_invoke_provider_with_recovery` 重构（带 cascade loop + 升级 hook） | 2h | 主体改造 |
| 5. `_observe_recovery_cascade` + metadata["recovery"] 输出 | 30min | logger + result.metadata |
| 6. 测试组 A 单步 cascade | 1.5h | 4 case |
| 7. 测试组 B 多步 cascade（含 forced upgrade T-B5..B8）| 2.5h | 8 case |
| 8. 测试组 C 回退路径 | 1h | 3 case |
| 9. 测试组 D 与 7 步降级正交 | 1h | 3 case |
| 10. 测试组 E observability | 1h | 4 case |
| 11. 测试组 F 边界 | 1h | 5 case |
| 12. 测试组 G 集成 | 2h | 8 case |
| 13. PR 2.2 / PR 3.4 / PR 4.1/4.2 回归 | 1h | full suite green |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| cascade 顺序与 claude-code 不一致导致行为差异 | 中 | 严格遵守"cheap → expensive"（compress → strip → fallback）；T-B1/B2 测试覆盖 |
| 同一 attempt 内 cascade 多次 apply 同一种动作（如 compress 反复跑）导致死循环 | 高 | cascade 每个 step 内部基于"实际效果"判断（freed>0 / messages 改动 / chain.activate_next 返回 True）；外层 max_attempts 兜底 |
| recovery cascade 与 7 步降级 budget 互相吃 | 中 | 两者独立 metadata 命名空间；fallback 激活共享 chain 的 `max_fallback_activations_per_turn` 但分别记录 |
| `error_withholding_enabled=False` 时与 PR 2.2 行为不完全一致 | 高 | 严格 1-1 lift PR 2.2 的 apply 顺序；T-C 测试与 PR 2.2 测试集联动跑 |
| logger 输出大量 cascade 信息淹没主日志 | 低 | 只在 cascade_log 非空时 INFO；无 cascade 时不输出 |
| pending_errors 列表巨大（边界）| 低 | max_provider_recovery_attempts 限制；通常 ≤ 8 |
| compaction_pipeline 在 cascade 内 raise（罕见）| 中 | catch 后追加 cascade_log entry "compress_context(failed)"，继续后续 step |
| FallbackChain.activate_next 触发跨进程 race | 低 | PR 2.2 已用 RLock；cascade 在主线程串行，不引入新 race |
| **Codex 反馈 #6**：forced fallback upgrade 与 strategy 决策"打架" | 中 | 升级仅发生在 strategy 给 `activate_fallback=False` 且历史已 compress 的精确条件下；策略仍可通过设 `activate_fallback=True` 主动控制；T-B5/B6/B7/B8 覆盖 |
| forced fallback upgrade 在多 turn 间累计 compression_attempted_for 集合导致首次 attempt 就升级 | 高 | `_WithholdingState` 在每次 `_invoke_provider_with_recovery` 入口重新构造（PR 4.3 文档 § 3.2.2 已说明）；不跨 attempt 累计；T-B8 验证不同 reason 不互相污染 |

## 十、与后续 PR 的衔接

- **Sprint 5 / P1-7 MessageBuilder**：MessageBuilder 在 PR 4.3 的 `_apply_recovery_decision_cascade`
  之前会被调用，所以 strip_thinking 应该作用于 MessageBuilder 后的产物。Sprint 5 落地时
  把 strip_thinking 的实现迁到 MessageBuilder.strip_thinking_blocks，保持 cascade 接口不变。
- **Sprint 6 / P2-1 Plugin 钩子**：当时引入 `pre_api_request / post_api_request` 钩子，
  本 PR 的 cascade 状态可作为 `post_api_request` 的输入提供给 plugin。
- **PR 4.4 JSON 错误**：与本 PR 完全独立，可并行开发。
