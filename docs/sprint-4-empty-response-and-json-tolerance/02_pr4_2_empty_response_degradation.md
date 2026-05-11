# PR 4.2 — 空响应降级（核心，7 step pipeline + 1 finalise hook）

> **角色**：Sprint 4 的核心 PR。在 PR 4.1 留出的 `_finalize_empty_response()` 入口处插入完整的
> **7 步空响应降级状态机**，再单独挂一个 **finalise pre-hook 处理 Codex 中间 ack**
> （这是非空 content 的 case，不能放在空响应 pipeline 里），
> 让 reasoning 模型（DeepSeek-R1 / GLM-5 / QwQ）能稳定跑完一轮，
> 让流式断流的内容能被救活，让 Codex "好的我马上去做"能被继续推进。
>
> 实现 [`02_p0_critical_gaps.md` § P0-8](../run-loop-roadmap/02_p0_critical_gaps.md) 的核心 9 步语义，
> 并整合 [`03_p1_robustness_gaps.md` § P1-8](../run-loop-roadmap/03_p1_robustness_gaps.md) 的 Codex 续写。
>
> ## 与原计划的关键差异（来自 Codex 反馈，已采纳）
>
> | 反馈 | 修订 |
> |---|---|
> | **#3** Codex ack 是"非空 content"，但原计划把它放在 empty-response pipeline 的 step 8。当前 `_finalize_empty_response` 只在 `final_response==""` 时进，永远跑不到 step 8。 | **抽出来**：Codex ack 改成 finalise 阶段的独立 pre-hook（§ 3.7），不再属于"9 步降级"。empty-response pipeline 缩成 **7 步**。 |
> | **#4** 原表格说 step 7 terminal 在 8/9 之前，伪代码却把 terminal 放最后；同时 truncated_prefix + streamed partial 场景下 step 1 会先吃掉 streamed text，step 9 没机会拼接。 | **重排**：truncated_prefix concat 提到第 1 优先级（如果 prefix 存在，先拼），其余 step 顺序不变，terminal 仍是兜底。 |
> | **#5** Housekeeping fallback 跨 turn 状态。TurnContext.metadata 每 turn 由 `dict(request.metadata)` 重置，turn key 跨不到下一轮。 | **改放 `SessionRuntimeState`**（PR 4.1 已加字段）。step 2 reader 改为读 session runtime；end-of-turn writer 改为写 session runtime。 |
> | **#1** `NormalizedResponse` 没有 `reasoning` 字段。 | **改读 `metadata["reasoning_content"]` / `["reasoning_details"]`**。 |
> | **#2** Codex ack 触发条件原计划走 `provider.profile.codex_response_mode`，但 provider 当前没有 `profile` 抽象。 | **直接判 `provider_name=="codex" and api_mode=="responses"`**，封一个 `_is_codex_responses_provider(provider)` helper，Sprint 6 引入 `ProviderProfile` 时再迁实现，对外接口不变。 |

## 一、目标

1. 实现 `AgentEngine._handle_empty_response(...)`，在 PR 4.1 标记的"BUG_EMPTY"分支上落地 7 步：

   | 序号 | 名称 | 触发 | 副作用 |
   |---|---|---|---|
   | 1 | **truncated prefix concat** | `TURN_KEY_TRUNCATED_RESPONSE_PREFIX` 存在 | final_response = prefix + max(body, streamed)（清 prefix） |
   | 2 | partial stream recovery | streamed_assistant_text 非空（且 step 1 没接管） | final_response = 累计内容（去 `<think>`） |
   | 3 | 上轮 housekeeping fallback | `SessionRuntimeState.last_assistant_tools_all_housekeeping=True` | final_response = `state.last_assistant_text_with_tools` |
   | 4 | post-tool empty nudge | 上 5 条有 tool；每 turn 至多 1 次 | 注入 assistant `(empty)` + user 续写指令 → continue |
   | 5 | thinking prefill | 检测到 thinking（`metadata["reasoning_content"]` / `<think>`）；最多 N 次 | 把当前响应做 prefill 标记入历史 → continue |
   | 6 | 空响应 retry / fallback provider | retry budget 未耗尽 | 直接 continue / 切下家 fallback → continue |
   | 7 | 终态 EMPTY_RESPONSE | 前 6 步未命中 | exit_reason=EMPTY_RESPONSE，**不**写占位 |

2. **新增 finalise pre-hook**`_maybe_continue_codex_intermediate_ack(...)`（§ 3.7）：
   作用在**非空 content** 的 finalise 路径上（`final_response != ""` 且 `tool_calls=[]` 且 provider 是 codex+responses 且 content 像 ack），
   不属于上面 7 步 pipeline。
3. 实现 streaming 累计：在 `_build_stream_callback` 里把 delta 文本累加到
   `metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT]`，供 step 1/2 消费。
4. 实现 housekeeping 标记的写入：每轮成功完成 tool 后判断"本轮是否纯 housekeeping 工具串"
   并把内容快照写到 **`SessionRuntimeState`**，供下一轮 step 3 消费（**不是** TurnContext.metadata）。
5. 实现 thinking-prefill 历史清理：成功响应时调 `_pop_thinking_prefill_messages(messages)`。
6. 集中 metadata 更新到 `metadata["empty_recovery"]` 子字典，符合 PR 4.1 的契约。

## 二、现状分析

### 2.1 PR 4.1 留下的入口

PR 4.1 的 `_finalize_empty_response()` 末尾：

```python
if classification.is_success:
    context.metadata["empty_recovery"]["last_step"] = "legitimate_empty_passthrough"
    return "", ExitReason.TEXT_RESPONSE, None

context.metadata["empty_recovery"]["last_step"] = "no_recovery_yet"
return "", ExitReason.EMPTY_RESPONSE, None  # ← 本 PR 改为先调 _handle_empty_response
```

### 2.2 7 步降级 + Codex finalise hook 在 run_loop 中的整体位置

state machine 当前关键分支顺序（`agent.py` 大致行号）：

```
[L_840-880] phantom-tool 处理
[L_545-560] tool_call validation（含 invalid-JSON inject 路径）
[L_390-440] LLM_CALL（含 length finish_reason 分支）
[L_890-921] _finalize_empty_response（PR 4.1 抽出）
   ├─→ if final_response != "" and tool_calls == []:
   │        → _maybe_continue_codex_intermediate_ack(...)  ← finalise pre-hook (PR 4.2 新增)
   │              ├─ 命中 → 注入 system prod, continue 进下一轮
   │              └─ 不命中 → 正常 finalise 为 TEXT_RESPONSE
   └─→ if BUG_EMPTY: → _handle_empty_response（PR 4.2 新增）
              ├─ 步骤 1-7 按顺序判断（truncated_prefix → partial_stream → housekeeping
              │      → post_tool_nudge → thinking_prefill → retry/fallback → terminal）
              ├─ 命中 → 改 messages / final_response 后 RETURN_TO_LOOP 或 RETURN_FINAL
              └─ 全失败 → exit_reason=EMPTY_RESPONSE
```

**关键纪律 1**：7 步只在 phantom-tool / length-finish / invalid-JSON 都不命中后才进入，
避免双重 recovery。

**关键纪律 2**：Codex ack finalise hook 必须在 finalise 把 TEXT_RESPONSE 写出去之前
拦截，否则 `content="好的我马上读文件"` 会被当成正常完成结果直接终结。

### 2.3 已经能直接复用的 Sprint 1/2/3 设施

| Sprint | 设施 | PR 4.2 复用方式 |
|---|---|---|
| Sprint 1 / PR 1.1 | `streaming_enabled` + 流式 SSE | 步骤 1 的 streamed_assistant_text 来源 |
| Sprint 1 / PR 1.2 | `truncated_response_prefix` metadata | 步骤 9 直接读 |
| Sprint 1.5 | phantom-tool 合成 | 与步骤 4 thinking prefill 互斥；先 phantom 后 thinking |
| Sprint 2 / PR 2.2 | `FallbackChain.activate_next()` | 步骤 6 直接调 |
| Sprint 2 / PR 2.2 | `EngineServices.fallback_chain` 是否已配置 | 步骤 6 进入条件 |

## 三、设计

### 3.1 7 步降级状态机

新增 `agent.py::_handle_empty_response`：

```python
class _EmptyRecoveryStep(str, Enum):
    """7-step empty-response degradation steps.

    Names match metadata["empty_recovery"]["last_step"] for observability.

    NOTE (Codex 反馈 #3 + #4): "Codex intermediate ack" used to live
    here as step 8 but is now a finalise pre-hook (it operates on
    NON-EMPTY content), and "truncated prefix concat" was moved from
    step 9 to step 1 because it must run BEFORE partial-stream-recovery
    eats up streamed_assistant_text.
    """

    TRUNCATED_PREFIX_CONCAT = "truncated_prefix_concat"           # step 1 (was 9)
    PARTIAL_STREAM_RECOVERY = "partial_stream_recovery"           # step 2 (was 1)
    HOUSEKEEPING_FALLBACK = "housekeeping_fallback"               # step 3 (was 2)
    POST_TOOL_EMPTY_NUDGE = "post_tool_empty_nudge"               # step 4 (was 3)
    THINKING_PREFILL = "thinking_prefill"                          # step 5 (was 4)
    RETRY_OR_FALLBACK = "retry_or_fallback"                        # step 6 (was 5+6)
    TERMINAL_EMPTY = "terminal_empty"                              # step 7 (unchanged)


@dataclass(frozen=True)
class _EmptyRecoveryOutcome:
    """Outcome from :meth:`_handle_empty_response`."""

    step: _EmptyRecoveryStep
    """Which step ran (or terminated)."""

    final_response: Optional[str]
    """Non-None means: stop the loop, this IS the final response."""

    exit_reason: Optional[ExitReason]
    """When ``final_response`` is None and exit_reason is set, the loop
    should break with this exit reason (terminal failure)."""

    continue_loop: bool
    """When True the run-loop should continue iterating (the helper
    has injected messages / activated fallback / etc).  ``final_response``
    and ``exit_reason`` are both None in this case."""


def _handle_empty_response(
    self,
    *,
    response: NormalizedResponse,
    response_to_store: NormalizedResponse,
    messages: list[dict],
    context: TurnContext,
    request: EngineRequest,
    classification: ResponseClassification,
) -> _EmptyRecoveryOutcome:
    """Run the 7-step empty-response degradation.

    Steps execute strictly in order; the first one that "claims"
    the situation wins.  Step 7 is the terminal failure step.

    Step ordering rationale (Codex 反馈 #4):
      - Step 1 (truncated_prefix_concat) runs first because if a
        prior turn left a length-truncated prefix in metadata, it is
        the most informative answer-in-progress and must own both the
        body and the streamed partial.  Putting partial_stream_recovery
        first would leave the prefix dangling forever.
      - Step 2 (partial_stream_recovery) runs only when no prefix
        exists, so streamed text is just a half-emitted current turn.

    Returns: _EmptyRecoveryOutcome — caller branches on its three
    fields (final_response / exit_reason / continue_loop).
    """
```

#### 步骤 1：truncated prefix concat（**新顺序，原 step 9**）

```python
def _step1_truncated_prefix_concat(
    self, *, response_to_store, context, classification,
):
    """If a previous length-recovered turn left a prefix in metadata,
    glue it to whatever this turn produced (body or streamed partial).

    Runs BEFORE partial_stream_recovery (step 2) so streamed text can
    be folded into the prefix instead of being consumed standalone.
    """
    prefix = context.metadata.get(TURN_KEY_TRUNCATED_RESPONSE_PREFIX)
    if not prefix:
        return None
    body = response_to_store.content or ""
    body_visible = _strip_think_tags(body).strip()
    streamed = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
    streamed_clean = _strip_think_tags(streamed).strip()
    if not body_visible and not streamed_clean:
        return None  # nothing to glue
    payload = max([body_visible, streamed_clean], key=len)
    glued = f"{prefix}{payload}"
    # Consumed — clear so we don't double-glue on next turn.
    context.metadata[TURN_KEY_TRUNCATED_RESPONSE_PREFIX] = ""
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.TRUNCATED_PREFIX_CONCAT,
        final_response=glued,
        exit_reason=ExitReason.LENGTH_RECOVERED,
        continue_loop=False,
    )
```

#### 步骤 2：partial stream recovery（**原 step 1**）

```python
def _step2_partial_stream_recovery(self, *, context, classification):
    """Recover streamed assistant text when the call ended empty.

    Step 1 has priority: when a truncated_response_prefix exists, step
    1 already grabbed both body and streamed; this step won't re-fire
    because step 1 returned a non-None outcome above us.
    """
    if not self.config.empty_response_partial_stream_recovery_enabled:
        return None
    if not classification.has_streamed_partial:
        return None
    raw = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
    cleaned = _strip_think_tags(raw).strip()
    if not cleaned:
        return None
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.PARTIAL_STREAM_RECOVERY,
        final_response=cleaned,
        exit_reason=ExitReason.PARTIAL_STREAM_RECOVERY,
        continue_loop=False,
    )
```

新增 `ExitReason.PARTIAL_STREAM_RECOVERY`（见 § 3.6）。

#### 步骤 3：上轮 housekeeping fallback（**原 step 2，已改用 SessionRuntimeState**）

```python
def _step3_housekeeping_fallback(self, *, context):
    if not self.config.housekeeping_fallback_enabled:
        return None
    # Codex 反馈 #5: read from SessionRuntimeState (cross-turn) NOT
    # TurnContext.metadata (per-turn, reset by dict(request.metadata)).
    state = self.services.session_runtime_registry.get(context.session_id)
    if not state.last_assistant_tools_all_housekeeping:
        return None
    last = state.last_assistant_text_with_tools
    if not last or not str(last).strip():
        return None
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.HOUSEKEEPING_FALLBACK,
        final_response=str(last).strip(),
        exit_reason=ExitReason.FALLBACK_PRIOR_TURN_CONTENT,
        continue_loop=False,
    )
```

新增 `ExitReason.FALLBACK_PRIOR_TURN_CONTENT`。

**写入路径**（在 PR 4.2 同时实施）：每个 turn 完成后，
`_build_result` 之前调用——也写到 SessionRuntimeState：

```python
def _record_last_content_for_housekeeping_fallback(self, *, messages, context):
    """At end-of-turn, snapshot the latest assistant content + housekeeping
    flag onto SessionRuntimeState (cross-turn).

    Read by NEXT turn's step 3.  Called once per turn from _build_result
    (or from _finalize_empty_response when the turn ends in TEXT_RESPONSE
    with tool-only content).

    Codex 反馈 #5: writer targets SessionRuntimeState — TurnContext.metadata
    won't survive ``metadata = dict(request.metadata)`` at agent.py:1139.
    """
    last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    state = self.services.session_runtime_registry.get(context.session_id)
    if not last_assistant:
        # Reset both fields so we don't carry stale state.
        state.last_assistant_text_with_tools = ""
        state.last_assistant_tools_all_housekeeping = False
        return
    text = self._extract_assistant_visible_text(last_assistant)
    tool_calls = last_assistant.get("tool_calls") or []
    tool_names = {self._normalize_tool_name(c.get("name") or "") for c in tool_calls}
    housekeeping = self.config.housekeeping_tool_names
    all_housekeeping = bool(tool_names) and tool_names.issubset(
        {self._normalize_tool_name(n) for n in housekeeping}
    )
    state.last_assistant_text_with_tools = text or ""
    state.last_assistant_tools_all_housekeeping = all_housekeeping
```

#### 步骤 4：post-tool empty nudge（**原 step 3**）

```python
_NUDGE_USER_MESSAGE = (
    "[System: You just executed tool calls but returned an empty response. "
    "Please continue with your task using the tool results above, or explain "
    "what you intend to do next.]"
)

def _step4_post_tool_empty_nudge(self, *, messages, context):
    if not self.config.post_tool_empty_nudge_enabled:
        return None
    if context.metadata.get(TURN_KEY_POST_TOOL_EMPTY_RETRIED):
        return None  # one-shot per turn
    last_5 = messages[-5:]
    if not any(m.get("role") == "tool" for m in last_5):
        return None
    messages.append({
        "role": "assistant",
        "content": "(empty)",
        "_aether_meta": self._assistant_aether_meta(),
        "metadata": {"_post_tool_empty_nudge": "placeholder"},
    })
    messages.append({
        "role": "user",
        "content": _NUDGE_USER_MESSAGE,
        "metadata": {"_post_tool_empty_nudge": "user_nudge"},
    })
    context.metadata[TURN_KEY_POST_TOOL_EMPTY_RETRIED] = True
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.POST_TOOL_EMPTY_NUDGE,
        final_response=None,
        exit_reason=None,
        continue_loop=True,
    )
```

#### 步骤 5：thinking prefill（**原 step 4**）

```python
def _step5_thinking_prefill(
    self, *, response_to_store, messages, context, classification,
):
    if not self.config.thinking_prefill_enabled:
        return None
    if not classification.has_thinking:
        return None
    attempts = int(context.metadata.get(TURN_KEY_THINKING_PREFILL_RETRIES, 0))
    if attempts >= int(self.config.thinking_prefill_max_retries):
        return None  # exhausted, fall through
    # Append the assistant response (which contains thinking) marked
    # for prefill; the next LLM call will see this in history and treat
    # it as continuation context.
    prefill_msg = self._build_prefill_assistant_message(response_to_store)
    messages.append(prefill_msg)
    context.metadata[TURN_KEY_THINKING_PREFILL_RETRIES] = attempts + 1
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.THINKING_PREFILL,
        final_response=None,
        exit_reason=None,
        continue_loop=True,
    )

def _build_prefill_assistant_message(self, response_to_store):
    """Build the prefill message tag with `_thinking_prefill: True` so
    Sprint 5's MessageBuilder.pop_thinking_prefill_messages can clean
    it later (and so the next iteration can identify it).

    Codex 反馈 #1: read reasoning from metadata, NOT
    response_to_store.reasoning (NormalizedResponse has no such field).
    """
    response_md = getattr(response_to_store, "metadata", None) or {}
    reasoning_text = response_md.get("reasoning_content")
    reasoning_details = response_md.get("reasoning_details")
    return {
        "role": "assistant",
        "content": response_to_store.content or "",
        "_thinking_prefill": True,
        "_aether_meta": self._assistant_aether_meta(),
        "metadata": {
            "reasoning_content": reasoning_text,
            "reasoning_details": reasoning_details,
        },
    }

def _pop_thinking_prefill_messages(self, messages: list[dict]) -> int:
    """Remove all messages marked with _thinking_prefill: True.

    Called when a non-empty response arrives (recovery succeeded).
    Returns count removed."""
    removed = 0
    i = 0
    while i < len(messages):
        if messages[i].get("_thinking_prefill"):
            del messages[i]
            removed += 1
        else:
            i += 1
    return removed
```

集成点：在 `_finalize_empty_response` 的"final_response 非空"分支顶部调用：

```python
if final_response:
    # If we previously prefilled thinking messages and now have real
    # content, clean those prefills out of history before persisting.
    removed = self._pop_thinking_prefill_messages(messages)
    if removed > 0:
        context.metadata["empty_recovery"]["thinking_prefill_cleaned"] = removed
    context.metadata[TURN_KEY_THINKING_PREFILL_RETRIES] = 0
    ...
```

#### 步骤 6：retry / fallback provider（**原 step 5+6**）

```python
def _step6_retry_or_fallback(self, *, context):
    if not self.config.empty_response_recovery_enabled:
        return None
    attempts = int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0))
    max_attempts = int(self.config.empty_response_max_retries)
    if attempts <= max_attempts:
        # Sub-step "5": same provider, just continue.  attempts is
        # bumped by _finalize_empty_response BEFORE this step is reached,
        # so the comparison is "attempts <= max" not "<".
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.RETRY_OR_FALLBACK,
            final_response=None,
            exit_reason=None,
            continue_loop=True,
        )
    # Sub-step "6": same-provider budget exhausted; try to rotate fallback.
    chain = self.services.fallback_chain
    if chain is None:
        return None  # no fallback available; let terminal step fire
    if not chain.has_next():
        return None
    rotated = chain.activate_next(reason="empty_response_exhausted")
    if not rotated:
        return None
    # Reset the per-turn empty counter for the new provider so it
    # gets its own fresh budget — same pattern as PR 2.2's recovery.
    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
    context.metadata["empty_recovery"]["activated_fallback"] = chain.current_slot_name
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.RETRY_OR_FALLBACK,
        final_response=None,
        exit_reason=None,
        continue_loop=True,
    )
```

#### 步骤 7：terminal EMPTY_RESPONSE

```python
def _step7_terminal_empty(self):
    return _EmptyRecoveryOutcome(
        step=_EmptyRecoveryStep.TERMINAL_EMPTY,
        final_response=None,           # ← 不写 (empty) 占位（设计 D）
        exit_reason=ExitReason.EMPTY_RESPONSE,
        continue_loop=False,
    )
```

**关键纪律**：与原计划"终态 (empty)"的差异——我们**不再向 final_response 写
占位字符串**。展示层（CLI）需要时自行渲染 `(no content)`，引擎层只 surface
ExitReason.EMPTY_RESPONSE。

> **删除说明（Codex 反馈 #3）**：原 step 8 "Codex intermediate ack" 已经从这个 pipeline
> **移除**——它操作的是非空 content，不属于"空响应降级"，应当在 finalise 入口
> 拦截。完整实现见 § 3.7 "_maybe_continue_codex_intermediate_ack 的 finalise pre-hook"。
>
> **删除说明（Codex 反馈 #4）**：原 step 9 "truncated prefix concat" 已经迁到本文档
> **新的 step 1**（最高优先级），原因：当 prefix 存在时，streamed 文本归 prefix 拼接
> 所有，不能让 step 2 partial-stream-recovery 先把 streamed 用掉。

### 3.2 主调度器

```python
def _handle_empty_response(
    self, *, response, response_to_store, messages, context, request, classification,
) -> _EmptyRecoveryOutcome:
    if not self.config.empty_response_recovery_enabled:
        return self._step7_terminal_empty()

    # Steps execute in declared order.  First non-None outcome wins.
    # Order rationale (Codex 反馈 #4): step 1 (truncated_prefix_concat)
    # MUST run before step 2 (partial_stream_recovery), otherwise the
    # streamed text would be consumed standalone and prefix would dangle.
    pipeline: list[Callable[..., Optional[_EmptyRecoveryOutcome]]] = [
        # 1: cheap, no LLM — glue length-truncated prefix from previous turn
        lambda: self._step1_truncated_prefix_concat(
            response_to_store=response_to_store, context=context,
            classification=classification,
        ),
        # 2: cheap, no LLM — recover streamed assistant text
        lambda: self._step2_partial_stream_recovery(
            context=context, classification=classification,
        ),
        # 3: cheap, no LLM — substitute the previous turn's housekeeping content
        lambda: self._step3_housekeeping_fallback(context=context),
        # 4: one extra LLM round — nudge the model to continue after tools
        lambda: self._step4_post_tool_empty_nudge(messages=messages, context=context),
        # 5: one extra LLM round — thinking prefill for reasoning models
        lambda: self._step5_thinking_prefill(
            response_to_store=response_to_store, messages=messages,
            context=context, classification=classification,
        ),
        # 6: same-provider retry, or fallback rotation when budget exhausted
        lambda: self._step6_retry_or_fallback(context=context),
    ]

    for step_fn in pipeline:
        outcome = step_fn()
        if outcome is not None:
            context.metadata["empty_recovery"]["last_step"] = outcome.step.value
            self.services.logger.info(
                "empty_response: step=%s continue=%s exit=%s",
                outcome.step.value,
                outcome.continue_loop,
                outcome.exit_reason.value if outcome.exit_reason else None,
            )
            return outcome

    # Fall-through: terminal step 7.
    terminal = self._step7_terminal_empty()
    context.metadata["empty_recovery"]["last_step"] = terminal.step.value
    self.services.logger.info("empty_response: step=terminal_empty exit=EMPTY_RESPONSE")
    return terminal
```

> Codex intermediate ack 不在这里——它由 finalise pre-hook 处理（§ 3.7）。

### 3.3 与 `_finalize_empty_response` 的接合

PR 4.1 的 helper 末尾：

```python
# (pre-PR 4.2)
context.metadata["empty_recovery"]["last_step"] = "no_recovery_yet"
return "", ExitReason.EMPTY_RESPONSE, None
```

改为：

```python
# (PR 4.2)
outcome = self._handle_empty_response(
    response=response,
    response_to_store=response_to_store,
    messages=messages,
    context=context,
    request=request,
    classification=classification,
)
if outcome.continue_loop:
    return _CONTINUE_LOOP_SENTINEL, None, None  # caller transitions to PRE_LLM
if outcome.final_response is not None:
    return outcome.final_response, outcome.exit_reason, None
return "", outcome.exit_reason or ExitReason.EMPTY_RESPONSE, None
```

调用点（`agent.py:920` 附近）：

```python
final_response, exit_reason, error_text = self._finalize_empty_response(...)
if final_response is _CONTINUE_LOOP_SENTINEL:
    state_machine.transition(LoopState.CHECK_EXIT)
    if budget.exhausted:
        state_machine.transition(LoopState.FINALIZE)
        exit_reason = ExitReason.MAX_ITERATIONS
        break
    state_machine.transition(LoopState.PRE_LLM)
    continue
state_machine.transition(LoopState.FINALIZE)
break
```

### 3.4 Stream callback 累计逻辑

`agent.py::_build_stream_callback` 改造：

```python
def _build_stream_callback(self, request, context):
    callback = request.stream_callback
    if not getattr(self.config, "streaming_enabled", True):
        return None
    if callback is None and not self.config.empty_response_partial_stream_recovery_enabled:
        return None  # neither user callback nor our internal accumulator wants it

    def _wrapped(delta: str) -> None:
        if not isinstance(delta, str) or not delta:
            return
        # PR 4.2: always accumulate for partial-stream-recovery, even
        # if the user did not supply a callback.  The accumulation cost
        # is O(n) memory; cleared at end-of-turn.
        if self.config.empty_response_partial_stream_recovery_enabled:
            cur = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
            context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = cur + delta
        if callback is not None:
            try:
                callback(delta)
                context.metadata["streamed_output"] = True
                context.metadata["stream_callback_calls"] = (
                    int(context.metadata.get("stream_callback_calls", 0)) + 1
                )
            except Exception:
                self.services.logger.exception("stream callback failed")

    return _wrapped
```

清理：在每次 LLM_CALL 之前重置 `streamed_assistant_text`：

```python
# top of _invoke_provider_with_recovery (or right before provider.generate):
context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = ""
```

### 3.5 Housekeeping flag 的写入路径

实现已经在 § 3.1 step 3 章节给出（`_record_last_content_for_housekeeping_fallback`），
**目标对象是 `SessionRuntimeState`**，不是 `TurnContext.metadata`。

调用时机：
- **每次 turn 的 `_build_result()` 之前**（不论成功失败），保证 SessionRuntimeState 是
  "本轮结束的快照"，下一轮入口可以读到。
- 仅在 turn 末尾调一次（不是每次 tool 后）。

```python
def _build_result(self, *, context, messages, ...):
    self._record_last_content_for_housekeeping_fallback(
        messages=messages, context=context,
    )
    ...
```

> Codex 反馈 #5 修订：原方案打算用 `TURN_KEY_LAST_CONTENT_*` turn key 做跨 turn 传递，
> 但 `agent.py:1139` 的 `metadata = dict(request.metadata)` 会在每个 turn 入口重置 metadata
> （不依赖 `TURN_RETRY_COUNTER_KEYS` 集合），跨 turn 读不到。改放
> `SessionRuntimeState`（与 `memory_nudge_counter` / `skill_nudge_counter` 同级）后，
> 字段天然每 session 持久，跨 turn 自动可见，且不需要调用方在 `request.metadata` 里
> 手动回灌——这是 Aether 内部细节，不应该泄漏到公共 EngineRequest 契约。

### 3.6 `ExitReason` 增量

`runtime/contracts.py`：

```python
# Sprint 4 / PR 4.2: empty-response 7-step degradation surfaces three
# new exit reasons so observability can tell which recovery branch
# saved (or failed to save) the turn.

# step 2 — turn was rescued by partial-stream cumulative recovery
PARTIAL_STREAM_RECOVERY = "PARTIAL_STREAM_RECOVERY"

# step 3 — turn was rescued by fallback to prior turn's housekeeping content
FALLBACK_PRIOR_TURN_CONTENT = "FALLBACK_PRIOR_TURN_CONTENT"

# step 4 — the post-tool nudge was injected; this is a transient signal
# that should NOT appear in EngineResult (it always continues the loop)
# but is reserved for future "deep diagnostic" event types.
POST_TOOL_NUDGE = "POST_TOOL_NUDGE"
```

PR 4.2 实际只 surface 前两个；`POST_TOOL_NUDGE` 用于 logging 关联，不出现在 EngineResult.exit_reason。

> 注：step 1 truncated_prefix_concat 命中时复用既有 `ExitReason.LENGTH_RECOVERED`
> （PR 1.2 已有），不需要新加。

### 3.7 Codex intermediate ack — finalise pre-hook（**新章节，原 step 8**）

> **来源**：Codex 反馈 #3 修订点。Codex ack 操作的是非空 content，
> 不应当放在 empty-response pipeline 里——`_finalize_empty_response` 只在
> `final_response == ""` 时才进，永远跑不到。改成 finalise 阶段的独立 pre-hook，
> 在 finalise 把 `TEXT_RESPONSE` 写出去**之前**拦截。

#### 3.7.1 触发条件（来自 Codex 反馈 #2 的 helper）

```python
def _is_codex_responses_provider(provider) -> bool:
    """Detect a Codex provider running in the responses API mode.

    Sprint 4 / PR 4.2 — replaces the imagined ``provider.profile.codex_response_mode``
    attribute with a direct probe on the existing identity fields
    (``provider_name`` and ``api_mode``) defined on ``ModelProvider``
    (base.py:25-30) and concretised in ``CodexChatModel``
    (codex.py:36-37).  When Sprint 6 introduces a real ``ProviderProfile``
    abstraction this helper's body is the single point to swap; callers
    don't change.
    """
    return (
        getattr(provider, "provider_name", "") == "codex"
        and getattr(provider, "api_mode", "") == "responses"
    )
```

#### 3.7.2 ack 检测（保留原来的语料启发式）

```python
_CODEX_ACK_PHRASES_ZH = (
    "好的", "好的，", "好的我", "好，", "我来", "我会", "马上", "稍等",
)
_CODEX_ACK_PHRASES_EN = (
    "okay", "ok,", "sure,", "alright", "i'll", "let me", "one moment",
)

def _looks_like_codex_intermediate_ack(content: str) -> bool:
    if not content:
        return False
    head = content.strip().lower()[:64]
    return any(p in head for p in _CODEX_ACK_PHRASES_EN) or any(
        p in head for p in _CODEX_ACK_PHRASES_ZH
    )
```

#### 3.7.3 finalise pre-hook 主体

```python
def _maybe_continue_codex_intermediate_ack(
    self, *, response_to_store, messages, context,
) -> bool:
    """Finalise pre-hook.

    Returns True if the helper consumed this turn (caller must continue
    the run-loop instead of finalising as TEXT_RESPONSE).  Returns False
    when the response should finalise normally.

    Triggers when ALL of:
      - codex_intermediate_ack_enabled
      - provider is codex+responses (Codex 反馈 #2)
      - response.tool_calls is empty (otherwise the model already
        emitted the action — no need to nudge)
      - response.content stripped is non-empty AND looks like an ack
      - per-turn ack budget not exhausted
    """
    if not self.config.codex_intermediate_ack_enabled:
        return False
    if not _is_codex_responses_provider(self.services.provider):
        return False
    if response_to_store.tool_calls:
        return False
    content = (response_to_store.content or "").strip()
    if not content:
        return False  # empty-response pipeline takes care of this
    if not _looks_like_codex_intermediate_ack(content):
        return False
    attempts = int(context.metadata.get(TURN_KEY_CODEX_ACK_RETRIES, 0))
    if attempts >= int(self.config.codex_intermediate_ack_max_retries):
        return False  # budget exhausted; let it finalise as TEXT_RESPONSE

    # Inject the system prod and bump the counter; caller will continue.
    messages.append({
        "role": "system",
        "content": (
            "[System: You said you would perform an action but did not "
            "emit any tool_calls.  Please proceed by issuing the actual "
            "tool call now.]"
        ),
        "metadata": {"_codex_intermediate_ack": True},
    })
    context.metadata[TURN_KEY_CODEX_ACK_RETRIES] = attempts + 1
    context.metadata.setdefault("empty_recovery", {})["codex_ack_finalise_hook"] = (
        f"attempt={attempts + 1}"
    )
    self.services.logger.info(
        "codex_intermediate_ack: continue (attempt=%d/%d, content=%s)",
        attempts + 1,
        int(self.config.codex_intermediate_ack_max_retries),
        content[:32],
    )
    return True
```

#### 3.7.4 接合点 — 改 `_finalize_empty_response` 的 `final_response != ""` 分支

```python
if final_response:
    # PR 4.2 (finalise pre-hook): Codex intermediate ack interception.
    # Operates on NON-EMPTY content where the model only said "I'll do X"
    # without emitting a tool_call.  When triggered, we don't finalise;
    # we tell the caller to continue the run-loop with the prod injected.
    if self._maybe_continue_codex_intermediate_ack(
        response_to_store=response_to_store,
        messages=messages,
        context=context,
    ):
        # Sentinel: caller transitions to PRE_LLM and re-enters the loop.
        return _CONTINUE_LOOP_SENTINEL, None, None

    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
    if phantom_outcome == "exhausted":
        ...
    else:
        prefix = context.metadata.get(TURN_KEY_TRUNCATED_RESPONSE_PREFIX)
        exit_reason = ExitReason.LENGTH_RECOVERED if prefix else ExitReason.TEXT_RESPONSE
    return final_response, exit_reason, None
```

### 3.8 与 phantom-tool / length-finish 的优先级

run_loop 决策树（PR 4.2 后）：

```
LLM_CALL 完成
  ↓
finish_reason == "length"?
  ├─ Y → _handle_length_finish_reason (PR 1.2)
  │       ├─ 续写成功 → continue
  │       ├─ thinking-budget exit → exit
  │       └─ 续写失败 → exit
  ↓ N
phantom-tool 检测?
  ├─ Y → _maybe_recover_phantom_tool_intent (PR 1.5)
  │       ├─ "synthesized" → 走合成路径
  │       ├─ "retry" → continue
  │       └─ "exhausted" → 进 PHANTOM_TOOL_INTENT
  ↓ N
tool_calls 校验?
  ├─ Y → _validate_tool_call_arguments (PR 1.3 + PR 4.4)
  │       ├─ "ok" → dispatch tools
  │       ├─ "retry" → continue
  │       ├─ "inject_error" → 注入 + continue
  │       └─ "truncated" → exit TOOL_CALL_TRUNCATED
  ↓ N
_finalize_empty_response (PR 4.1)
  ├─ final_response != "" → 先尝试 Codex finalise pre-hook (§ 3.7)
  │     ├─ ack 命中 → continue（不 finalise）
  │     └─ 不命中 → exit TEXT_RESPONSE / LENGTH_RECOVERED
  ├─ NOT_EMPTY / LEGITIMATE / THINKING_ONLY → exit TEXT_RESPONSE
  └─ BUG_EMPTY → _handle_empty_response (PR 4.2)
                  ├─ step 1-6 一一尝试
                  └─ terminal step 7 → exit EMPTY_RESPONSE
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/agents/core/agent.py` | 修改 | `_handle_empty_response` 主调度 + 7 个 step 方法 + `_maybe_continue_codex_intermediate_ack` finalise pre-hook + `_is_codex_responses_provider` helper + `_record_last_content_for_housekeeping_fallback` + `_build_stream_callback` 累加 + 接入 `_finalize_empty_response`（含 `_CONTINUE_LOOP_SENTINEL` 路径） | ~600 净增 |
| `backend/harness/aether/runtime/contracts.py` | 修改 | `ExitReason` 加 3 项 | ~15 |
| `backend/harness/aether/runtime/session_runtime.py` | 修改 | 把 PR 4.1 占位的 4 个 `TURN_KEY_*` 加入 `TURN_RETRY_COUNTER_KEYS` 的 reset 路径；`SessionRuntimeState` 字段已经在 PR 4.1 加好，本 PR 只是 wire reader/writer | ~10 |
| `backend/harness/aether/agents/core/empty_recovery.py` | **新文件**（可选） | `_EmptyRecoveryStep` enum（7 项）+ `_EmptyRecoveryOutcome` dataclass + `_NUDGE_USER_MESSAGE` / `_CODEX_ACK_PHRASES_*` 等常量 + `_is_codex_responses_provider` / `_looks_like_codex_intermediate_ack` 辅助函数 | ~120 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step1_truncated_prefix.py` | **新文件** | 测试组 A：truncated_prefix_concat（原 step 9，现在是首位）（5 case） | ~150 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step2_partial_stream.py` | **新文件** | 测试组 B：partial_stream_recovery（5 case） | ~150 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step3_housekeeping_fallback.py` | **新文件** | 测试组 C：housekeeping_fallback + SessionRuntimeState 写入路径（6 case） | ~180 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step4_post_tool_nudge.py` | **新文件** | 测试组 D（5 case） | ~150 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step5_thinking_prefill.py` | **新文件** | 测试组 E（7 case） | ~200 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step6_retry_fallback.py` | **新文件** | 测试组 F（8 case） | ~220 |
| `backend/harness/aether/tests/agents/core/test_empty_response_step7_terminal.py` | **新文件** | 测试组 G（4 case） | ~120 |
| `backend/harness/aether/tests/agents/core/test_empty_response_codex_ack_finalise_hook.py` | **新文件** | 测试组 H：finalise pre-hook（不在 7 步 pipeline 内）（6 case） | ~200 |
| `backend/harness/aether/tests/agents/core/test_empty_response_full_pipeline.py` | **新文件** | 测试组 I（端到端 8 case） | ~280 |

## 五、测试用例（详细）

### 5.1 步骤 1：truncated prefix concat（测试组 A，**原 step 9**）

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | TURN_KEY_TRUNCATED_RESPONSE_PREFIX="prev..."；本轮 content="next." | final_response="prev...next."；exit_reason=LENGTH_RECOVERED；prefix 清空 |
| **T-A2** | prefix 存在；content=""；streamed_text="partial" | final_response="prev...partial"；step 2 不再 fire（同 turn） |
| **T-A3** | prefix 存在；content=""；streamed_text="" | 不命中（无可拼接），fall through 到 step 2 |
| **T-A4** | prefix=""；content="X" | 不命中（无 prefix），让步给后续 step |
| **T-A5** | prefix 存在；body 与 streamed 都有内容 | 选 longer 的拼接 |

### 5.2 步骤 2：partial stream recovery（测试组 B）

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | streamed_text="Hello world"；response.content="" | final_response="Hello world"；exit_reason=PARTIAL_STREAM_RECOVERY |
| **T-B2** | streamed_text="<think>x</think>Hello"；content="" | strip 后 final_response="Hello" |
| **T-B3** | streamed_text="<think>only think</think>"；content="" | 全是 think → 步骤 2 不命中，进步骤 3 |
| **T-B4** | streamed_text=""；content="" | 步骤 2 不命中 |
| **T-B5** | empty_response_partial_stream_recovery_enabled=False；streamed_text="X" | 跳过步骤 2 |

### 5.3 步骤 3：housekeeping fallback（测试组 C，**SessionRuntimeState 路径**）

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | `state.last_assistant_text_with_tools="Updated todo"`；`state.last_assistant_tools_all_housekeeping=True`；本轮 content="" | final_response="Updated todo"；exit_reason=FALLBACK_PRIOR_TURN_CONTENT |
| **T-C2** | state.last_assistant_text_with_tools=""；all_housekeeping=True | 不命中（last 为空） |
| **T-C3** | last 非空；all_housekeeping=False | 不命中 |
| **T-C4** | housekeeping_fallback_enabled=False | 跳过 |
| **T-C5** | last 含 trailing whitespace | strip 后 final_response 干净 |
| **T-C6** | 写入路径：上一轮 tool=memory + update_todo → all_housekeeping=True；切到下一轮（新 TurnContext）后 step 3 命中 | 验证 SessionRuntimeState 跨 turn 持久（不依赖 request.metadata 回灌） |

### 5.4 步骤 4：post-tool empty nudge（测试组 D）

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | 上 5 条含 tool；本轮空 | continue_loop=True；messages 末尾追加 (empty) + nudge user 消息；POST_TOOL_EMPTY_RETRIED=True |
| **T-D2** | 上 5 条无 tool（纯对话） | 不命中 |
| **T-D3** | 已 retried 一次（POST_TOOL_EMPTY_RETRIED=True） | 不重复（一次 only） |
| **T-D4** | post_tool_empty_nudge_enabled=False | 跳过 |
| **T-D5** | 注入消息 metadata 含 `_post_tool_empty_nudge` 键 | 标记正确 |

### 5.5 步骤 5：thinking prefill（测试组 E，**Codex 反馈 #1：只读 metadata**）

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | response.metadata["reasoning_content"]="..."；content="" | continue_loop=True；messages 末尾追加 `_thinking_prefill: True` 消息；THINKING_PREFILL_RETRIES=1 |
| **T-E2** | response.metadata["reasoning_details"]=[{...}]；content="" | 同 E1 |
| **T-E3** | content="<think>foo</think>" | has_thinking=True 路径触发 |
| **T-E4** | 已 retried 2 次（max=2）；本次 thinking-only | 不再 prefill，进步骤 6 |
| **T-E5** | thinking_prefill_enabled=False | 跳过 |
| **T-E6** | 下一轮回来响应非空 | _pop_thinking_prefill_messages 清掉之前的 prefill；THINKING_PREFILL_RETRIES 重置 0 |
| **T-E7** | classification.has_thinking=False | 不命中 |

### 5.6 步骤 6：retry / fallback（测试组 F）

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | EMPTY_RESPONSE_RETRIES=1；max=3；fallback_chain=None | continue_loop=True（同 provider retry） |
| **T-F2** | EMPTY_RESPONSE_RETRIES=4；max=3；fallback_chain.has_next()=True | activate_next 调用；EMPTY_RESPONSE_RETRIES 重置为 0；continue_loop=True |
| **T-F3** | EMPTY_RESPONSE_RETRIES=4；fallback_chain=None | 不命中（进步骤 7） |
| **T-F4** | EMPTY_RESPONSE_RETRIES=4；fallback_chain.has_next()=False | 同 F3 |
| **T-F5** | activate_next 返回 False | 不命中 |
| **T-F6** | empty_response_recovery_enabled=False | 整个 _handle_empty_response 短路到 terminal |
| **T-F7** | metadata["empty_recovery"]["activated_fallback"] 写入 | F2 后该字段为新 slot 名 |
| **T-F8** | retry budget 边界：max=0 | 不 retry，进 fallback；都不行进步骤 7 |

### 5.7 步骤 7：terminal（测试组 G）

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 1-6 都不命中 | exit_reason=EMPTY_RESPONSE；final_response="" |
| **T-G2** | final_response **不**写 "(empty)" / "(no content)" 占位 | 严格 == "" |
| **T-G3** | metadata["empty_recovery"]["last_step"]="terminal_empty" | 标记正确 |
| **T-G4** | 既有 stuck_after_tool 启发式（agent.py:2842）能读到 last_step | 兼容性测试 |

### 5.8 Codex intermediate ack — finalise pre-hook（测试组 H，**不在 7 步 pipeline 内**）

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | provider_name="codex" + api_mode="responses"；content="好的，我马上读文件"；tool_calls=[] | finalise hook 命中；注入 system 消息；CODEX_ACK_RETRIES=1；run-loop 继续（不 finalise） |
| **T-H2** | 同 H1 但英文 "Sure, let me check that file" | 命中 |
| **T-H3** | provider_name="openai" + api_mode="chat"；content="好的我马上去读" | 不命中（gate 关；非 codex+responses） |
| **T-H4** | provider 是 codex+responses；content="好的"（边界）| 命中 |
| **T-H5** | content="实际工作内容..."（不像 ack） | 不命中；正常 finalise=TEXT_RESPONSE |
| **T-H6** | 已 retried 2 次（max=2）；codex 又给 ack | 不再 retry；正常 finalise=TEXT_RESPONSE，最终 final_response=ack 文本（接受） |
| **T-H7** | provider 是 codex+responses；content="好的"；tool_calls=[shell(...)]（非空） | 不命中（已 emit 实际 action） |
| **T-H8** | 验证 _is_codex_responses_provider 单元 | provider_name/api_mode 组合 4 种全覆盖 |

### 5.9 端到端（测试组 I）

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | DeepSeek-R1 mock：3 次空 + reasoning，第 4 次拿到 "Done"；max=2 prefill | 前 2 次走 thinking_prefill（step 5）；第 3 次 step 6 retry；第 4 次成功；exit=TEXT_RESPONSE |
| **T-I2** | mock 流式回调累计 100 字符然后断流；response empty；no prefix | exit=PARTIAL_STREAM_RECOVERY（step 2）；final_response 是 100 字符（去 think 后） |
| **T-I3** | 上一轮 update_todo 成功（写入 SessionRuntimeState）；本轮新 TurnContext，空响应 | exit=FALLBACK_PRIOR_TURN_CONTENT（step 3）；final_response=上一轮文本 |
| **T-I4** | mock provider 一直返回空 + 无 fallback；max_retries=3 | 经过 step 6 三次 retry 后 exit=EMPTY_RESPONSE |
| **T-I5** | mock provider 配 fallback；前 3 次主 provider 空；fallback 第 1 次返回 "OK" | exit=TEXT_RESPONSE；activated_fallback="fallback_1" |
| **T-I6** | codex+responses；mock 第 1 次返回 "好的我读文件"，第 2 次真的 emit tool_call | finalise hook 注入 → tool dispatch → exit=TEXT_RESPONSE |
| **T-I7** | length 续写产生 prefix；下一轮 content="next."；streamed_text="" | step 1 拼接 → exit=LENGTH_RECOVERED；prefix 清空 |
| **T-I8** | length 续写产生 prefix；下一轮 content=""；streamed_text="partial" | step 1 命中（含 streamed），exit=LENGTH_RECOVERED；step 2 不 fire（验证 #4 排序正确） |
| **T-I9** | EngineResult.metadata["empty_recovery"] 完整 schema | classification / last_step / retries / post_tool_empty_retried 全部就位 |

## 六、验收门

- [ ] 所有新测试 green（约 54 case，分布到 8 个 step 测试文件 + 1 个端到端文件）
- [ ] PR 4.1 既有测试不回归
- [ ] Sprint 1/2/3 全套测试不回归
- [ ] `result.metadata["empty_recovery"]["last_step"]` 在 7 个 step 分支下都有正确取值
- [ ] Codex finalise hook 触发时 metadata 含 `empty_recovery.codex_ack_finalise_hook`
- [ ] DeepSeek-R1 mock fixture：连续 3 次空 + reasoning，第 4 次成功 → 整轮 exit=TEXT_RESPONSE（非 EMPTY_RESPONSE）
- [ ] Codex mock fixture：第 1 次 emit "好的我马上去做"后下一轮成功 emit tool_call

> 真实 DeepSeek-R1 / Codex fixture（联网）作为 P2 / nice-to-have，不阻塞 merge——
> 见 `99_acceptance_matrix.md` § 七 修订后的判据。

## 七、回滚开关

- `empty_response_recovery_enabled=False`：整个 7 步降级关闭，退回 PR 4.1 行为
- `empty_response_partial_stream_recovery_enabled=False`：仅关步骤 2（partial_stream）
- `housekeeping_fallback_enabled=False`：仅关步骤 3（housekeeping_fallback）
- `post_tool_empty_nudge_enabled=False`：仅关步骤 4（post_tool_empty_nudge）
- `thinking_prefill_enabled=False`：仅关步骤 5（thinking_prefill）
- `codex_intermediate_ack_enabled=False`：仅关 finalise pre-hook（不在 7 步 pipeline 内）
- 步骤 1/6/7 没有独立开关——
  - step 1 truncated_prefix 由 `TURN_KEY_TRUNCATED_RESPONSE_PREFIX` 是否非空决定，
    本质由 PR 1.2 `length_continuation_enabled` 控；
  - step 6 retry/fallback 由 `empty_response_max_retries=0` 关闭 retry；fallback 由
    `EngineServices.fallback_chain=None` 或 `fallback_chain_enabled=False` 关闭；
  - step 7 terminal 是兜底，不能关。

## 八、实施顺序（建议 3 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. `agents/core/empty_recovery.py` 常量与 dataclass + helper | 30min | enum（7 项）+ outcome + nudge text + `_is_codex_responses_provider` + `_looks_like_codex_intermediate_ack` |
| 2. `_handle_empty_response` 主调度框架 | 1h | logger + dispatch loop（先全部 step return None） |
| 3. step 1 truncated prefix + 测试组 A | 2h | 5 case |
| 4. step 2 partial stream + 测试组 B + stream callback 累加改造 | 3h | 5 case + reset 时机 |
| 5. SessionRuntimeState reader/writer + step 3 housekeeping + 测试组 C | 3h | 6 case + 跨 turn 验证 |
| 6. step 4 post-tool nudge + 测试组 D | 1.5h | 5 case |
| 7. step 5 thinking prefill（read metadata，不读 .reasoning）+ pop helper + 测试组 E | 3h | 7 case |
| 8. step 6 retry/fallback + 测试组 F | 2h | 8 case |
| 9. step 7 terminal + 测试组 G | 1h | 4 case |
| 10. Codex finalise pre-hook（_maybe_continue_codex_intermediate_ack）+ 接合点 + 测试组 H | 3h | 8 case |
| 11. _finalize_empty_response 接入（含 _CONTINUE_LOOP_SENTINEL）+ ExitReason 扩展 | 2h | 主 run-loop 调用点替换 |
| 12. 端到端测试组 I | 3h | 9 case |
| 13. Sprint 1/2/3 回归 | 1h | full suite green |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 7 步顺序依赖出错（如步骤 5 先于步骤 2）| 中 | pipeline 数组形态固化顺序；端到端测试 T-I 系列严格覆盖典型组合（含 T-I7/I8 验证 step 1 优先于 step 2） |
| streamed_assistant_text 累计后忘清，跨 turn 污染 | 高 | LLM_CALL 入口强制 reset；T-B 测试覆盖；`TURN_KEY_STREAMED_ASSISTANT_TEXT` 列入 turn-reset 集合（见 § 3.4） |
| thinking prefill 不清理导致历史无限膨胀 | 高 | 任何 final_response 非空路径必调 _pop；T-E6 严格验证 |
| step 6 与 PR 2.2 fallback chain 行为不一致 | 中 | 复用 `chain.activate_next()` 而不是新写；T-F2 与 PR 2.2 的 fallback 测试套件并跑 |
| Codex ack finalise hook 误触发普通对话场景 | 中 | gate by `_is_codex_responses_provider` (provider_name+api_mode 双条件)；`tool_calls=[]` gate；`codex_intermediate_ack_max_retries=2` 限制蔓延 |
| Codex ack hook 错误地把"已经在做事"的非 ack 内容当 ack（content="好的，下面是..."）| 中 | head 64 字符前缀匹配；max_retries=2 即便误判 2 次后接受 finalise；T-H5/H6 覆盖 |
| `_thinking_prefill: True` 标记被 provider 拒绝（部分严格 schema） | 中 | 标记放在 `metadata` 子字典或自定义 `_aether_meta`；provider 按惯例剥离未知 key（与 `_aether_meta` 同规则） |
| step 1 重复拼接（多轮 truncated）| 中 | 命中后立即清空 prefix；T-A1 验证 |
| housekeeping 状态在 session 销毁时未清理 | 低 | `SessionRuntimeRegistry.discard(session_id)` 已存在；新增字段随 SessionRuntimeState 整体销毁 |
| EngineResult.metadata 字段过多冲淡核心信号 | 低 | 全部嵌套在 `empty_recovery` 子字典；外层只暴露 last_step 摘要 |
| 与既有 stuck_after_tool 启发式（agent.py:2842）耦合断裂 | 中 | T-G4 专测；保留 `EMPTY_RESPONSE` 一类 exit_reason 在 stuck 启发式集合内 |
| `_CONTINUE_LOOP_SENTINEL` 在 finalise hook 与 empty pipeline 共用导致 caller 区分不清 | 中 | 同一 sentinel 含义统一是 "请 caller 转 PRE_LLM"；caller 不需要区分来源；T-H1 + T-I6 共同验证 |

## 十、与后续 PR 的衔接

- **PR 4.3** 在 LLM_CALL 阶段做 withholding 时不会与 7 步降级冲突——withholding 的目标是
  withhold **API 错误**消息（让 recovery 接管）；7 步降级处理的是 **legitimate empty** 响应。
  两者在不同分支，不重叠。
- **PR 4.4** 的结构化 JSON 错误是 7 步降级**之前**的分支（`_validate_tool_call_arguments`），
  不影响本 PR。
- **Sprint 5** 的 MessageBuilder 接管 `_pop_thinking_prefill_messages` 实现：本 PR 在 agent.py
  内部实现一个简单版；Sprint 5 把它迁到 `runtime/message_builder.py` 并扩展为
  `pop_thinking_prefill_messages(messages: list[dict]) -> list[dict]`。
- **Sprint 6** 引入 `ProviderProfile`：`_is_codex_responses_provider` helper 内部
  会迁到 `provider.profile.codex_response_mode`；本 PR 的 helper 是单一替换点，
  caller 不变。
