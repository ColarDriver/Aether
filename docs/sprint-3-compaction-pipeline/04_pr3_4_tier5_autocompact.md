# PR 3.4 — services/compact/ 骨架 + Tier 5 Autocompact（LLM Fork）

> **角色**：流水线骨架 PR + 第 5 层最高成本压缩。
> 本 PR 是整个 Sprint 3 最重的一块——引入 `services/compact/` 目录、
> `CompactionPipeline` 编排器、`AutoCompactor` 五条件门控、`LLMForkSummarizer`
> 重武器实现。后续 PR 3.5/3.6/3.7 在此骨架上补 Tier 3/2/4 即可。

## 一、目标

1. 引入 `backend/harness/aether/services/compact/` 目录，建立**横切关注点**的命名空间。
2. 实现 `CompactionPipeline` 编排器：按 Tier 顺序触发、每级后重新评估、熔断。
3. 实现 `AutoCompactor`（Tier 5）：5 条件门控 + 3 次失败熔断。
4. 实现 `LLMForkSummarizer`：fork 一次"无 tools / 限制输出 token"的 provider 请求生成对话 summary。
5. 与 Sprint 2 的 `ClassifiedRecoveryStrategy.compress_context=True` 信号接合：
   把"早退 CONTEXT_EXHAUSTED"改造成"流水线入口"。
6. 留好 Tier 2/3/4 的接入位（PR 3.5-3.7 实现）。

## 二、现状分析

### 2.1 当前压缩相关代码

无。`runtime/contracts.py` 里只有一个 `ContextCompressor` Protocol 的注释引用
（在 [`03_p1_robustness_gaps.md`](../run-loop-roadmap/03_p1_robustness_gaps.md) 计划中提到，
但代码里还没实现）。

### 2.2 Sprint 2 留下的接合点

[`backend/harness/aether/runtime/recovery.py`](../../backend/harness/aether/runtime/recovery.py)
里 `ClassifiedRecoveryStrategy` 已经会发：

```python
@dataclass
class RecoveryDecision:
    retry: bool
    wait_seconds: float = 0.0
    reason: str = ""
    activate_fallback: bool = False
    compress_context: bool = False        # ← Sprint 3 接管
    strip_thinking: bool = False
    classified_reason: Optional[str] = None
```

[`agents/core/agent.py`](../../backend/harness/aether/agents/core/agent.py)
的 `_invoke_provider_with_recovery` 现在的处理：

```python
if decision.compress_context:
    context.metadata["recovery_compress_required"] = True
    if decision.classified_reason == FailoverReason.payload_too_large.value:
        context.metadata["recovery_terminal_exit_reason"] = (
            ExitReason.PAYLOAD_TOO_LARGE.value
        )
    else:
        context.metadata["recovery_terminal_exit_reason"] = (
            ExitReason.CONTEXT_EXHAUSTED.value
        )
    return AgentEngine._ProviderInvocationOutcome(error=exc)
```

也就是**直接早退**。本 PR 把这段改造成"丢给流水线压缩 → 成功则 retry，失败 N 次再退"。

## 三、设计

### 3.1 目录结构

```
backend/harness/aether/services/
└── compact/
    ├── __init__.py             # 公共导出
    ├── compactor.py            # CompactionPipeline + CompactionContext
    ├── autocompact.py          # AutoCompactor (Tier 5 门控)
    ├── llm_fork.py             # LLMForkSummarizer (Tier 5 实现)
    ├── microcompact.py         # PR 3.5 实现 Tier 3
    ├── snip.py                 # PR 3.6 实现 Tier 2
    └── collapse.py             # PR 3.7 实现 Tier 4
```

PR 3.4 只创建 `__init__.py / compactor.py / autocompact.py / llm_fork.py`，
其他 3 个文件留 stub class（NoOp 实现），让后续 PR 替换。

### 3.2 `CompactionContext` — 流水线运行时状态

新文件 `services/compact/compactor.py`，第一段：

```python
"""Five-tier context compaction pipeline orchestrator.

Modeled on open-claude-code's services/compact/* organisation.
Pipeline order (cheap → expensive):

    Tier 1: per-tool result spill (PR 3.3, applied at tool dispatch
            time, not in this pipeline)
    Tier 2: snip — local message redundancy removal (PR 3.6)
    Tier 3: microcompact — time-based old tool-result clearing (PR 3.5)
    Tier 4: context collapse — projection-based folding (PR 3.7)
    Tier 5: autocompact — LLM-fork summarisation (this PR)

Each tier:
  * Receives (messages, ctx) and returns (messages', ctx', tokens_freed).
  * Pipeline re-estimates tokens after each tier; stops as soon as the
    threshold is met.
  * Each tier records its outcome in ctx.tier_outcomes for observability
    and into context.metadata["compaction"]["tierN_*"] for EngineResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from aether.runtime.contracts import TurnContext


@dataclass
class CompactionContext:
    """Snapshot of the runtime state passed through the pipeline."""

    session_id: str
    model: str
    model_window: int                      # tokens, e.g. 200_000 for Claude
    pre_compaction_tokens: int             # estimated before pipeline start
    target_pct: float                      # e.g. 0.85 — compress until below this
    trigger_reason: str                    # 'preflight' / 'context_overflow' / 'payload_too_large' / 'post_tool'
    consecutive_failures: int = 0          # for the autocompact circuit breaker
    tier_outcomes: list[dict] = field(default_factory=list)


class CompactorTier(Protocol):
    """Interface every Tier implements."""

    name: str   # 'tier1' / 'tier2' / ...

    def maybe_run(
        self,
        messages: list[dict],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict], int]:
        """Run if applicable; return (new_messages, tokens_freed).

        If the tier doesn't apply (disabled, threshold not met, etc),
        return (messages, 0) unchanged. Pipeline tolerates 0 freed.
        """
        ...
```

### 3.3 `CompactionPipeline` 主循环

```python
class CompactionPipeline:
    """Run tiers in order, re-evaluating after each."""

    def __init__(
        self,
        *,
        tiers: list[CompactorTier],
        token_estimator,                    # callable: messages → int
        config,                             # EngineConfig
        logger,
    ) -> None:
        self.tiers = tiers
        self._estimate = token_estimator
        self.config = config
        self.logger = logger

    def maybe_compress(
        self,
        messages: list[dict],
        *,
        turn_context: TurnContext,
        model: str,
        model_window: int,
        trigger_reason: str,
    ) -> "CompactionResult":
        """Run pipeline until threshold met or all tiers exhausted.

        Returns CompactionResult with:
          * compressed_messages: final message list
          * tokens_before / tokens_after
          * tiers_run: ordered list of which tiers fired
          * exhausted: True if pipeline ran all tiers and still > target
        """
        if not self.config.compression_enabled:
            return CompactionResult(
                compressed_messages=messages,
                tokens_before=self._estimate(messages),
                tokens_after=self._estimate(messages),
                tiers_run=[],
                exhausted=False,
            )

        target_pct = self.config.compression_pre_llm_pct
        tokens_before = self._estimate(messages)
        target_tokens = int(model_window * target_pct)

        if tokens_before <= target_tokens and trigger_reason == "preflight":
            # Preflight, well below threshold — no work to do.
            return CompactionResult(
                compressed_messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tiers_run=[],
                exhausted=False,
            )

        ctx = CompactionContext(
            session_id=turn_context.session_id,
            model=model,
            model_window=model_window,
            pre_compaction_tokens=tokens_before,
            target_pct=target_pct,
            trigger_reason=trigger_reason,
            consecutive_failures=int(
                turn_context.metadata.get("compaction_consecutive_failures", 0)
            ),
        )

        current = messages
        tiers_run: list[str] = []

        for tier in self.tiers:
            tier_before = self._estimate(current)
            new_msgs, freed = tier.maybe_run(current, ctx, turn_context)
            tier_after = self._estimate(new_msgs)
            tier_actual_freed = tier_before - tier_after  # truth from estimator
            ctx.tier_outcomes.append({
                "name": tier.name,
                "tokens_before": tier_before,
                "tokens_after": tier_after,
                "freed_reported": freed,
                "freed_actual": tier_actual_freed,
            })
            self.logger.info(
                "compact[%s] %s: %d → %d tokens (-%d)",
                trigger_reason, tier.name, tier_before, tier_after,
                tier_actual_freed,
            )
            current = new_msgs
            tiers_run.append(tier.name)
            if tier_after <= target_tokens:
                break

        tokens_after = self._estimate(current)
        exhausted = tokens_after > target_tokens

        # observability
        turn_context.metadata.setdefault("tier_outcomes", []).extend(ctx.tier_outcomes)

        return CompactionResult(
            compressed_messages=current,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tiers_run=tiers_run,
            exhausted=exhausted,
        )


@dataclass
class CompactionResult:
    compressed_messages: list[dict]
    tokens_before: int
    tokens_after: int
    tiers_run: list[str]
    exhausted: bool
```

### 3.4 `AutoCompactor`（Tier 5 门控）

新文件 `services/compact/autocompact.py`：

```python
"""Tier 5 autocompact — five-condition gate + circuit breaker.

Mirrors open-claude-code's autoCompact.ts decision logic with our own
LLM-fork summariser as the underlying mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.llm_fork import LLMForkSummarizer


@dataclass
class AutoCompactor:
    """Tier 5 — fork a sub-agent to summarise the conversation.

    Five conditions (all must be true to fire):
      1. Recursion guard — caller not itself a compaction sub-agent
         (querySource not in {'compaction', 'session_memory'}).
      2. Master compression switch enabled.
      3. autocompact-specific switch enabled.
      4. context-collapse not active (Tier 4 owns headroom when on).
      5. Token threshold met.

    Plus a circuit breaker: after consecutive_failures >= max_failures
    in this turn, refuse to try again — burning tokens on doomed
    summaries is worse than admitting defeat.
    """

    name: str = "tier5_autocompact"

    def __init__(
        self,
        *,
        config,                 # EngineConfig
        summarizer: LLMForkSummarizer,
        logger,
    ) -> None:
        self.config = config
        self.summarizer = summarizer
        self.logger = logger

    def maybe_run(
        self,
        messages: list[dict],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict], int]:
        # Condition 1: recursion guard — handled by caller setting
        # turn_context.metadata['_compaction_in_progress']. Re-entry returns
        # unchanged.
        if turn_context.metadata.get("_compaction_in_progress"):
            return messages, 0

        # Condition 2: master switch
        if not self.config.compression_enabled:
            return messages, 0

        # Condition 3: autocompact toggle
        if not getattr(self.config, "autocompact_enabled", True):
            return messages, 0

        # Condition 4: context-collapse takes priority (Tier 4)
        if turn_context.metadata.get("collapse_owns_headroom"):
            return messages, 0

        # Condition 5: threshold (already checked by pipeline, but
        # also enforce locally so direct callers see the same gate)
        threshold_tokens = int(
            ctx.model_window * self.config.compression_autocompact_pct
        )
        if ctx.pre_compaction_tokens < threshold_tokens:
            return messages, 0

        # Circuit breaker
        max_failures = self.config.compression_max_failures
        if ctx.consecutive_failures >= max_failures:
            self.logger.warning(
                "tier5: circuit breaker tripped (%d failures); skipping",
                ctx.consecutive_failures,
            )
            return messages, 0

        # Mark recursion guard before forking
        turn_context.metadata["_compaction_in_progress"] = True
        try:
            new_messages = self.summarizer.summarise(
                messages,
                model=ctx.model,
                turn_context=turn_context,
            )
        except Exception as exc:  # noqa: BLE001 — summarise must never crash
            self.logger.warning("tier5: summarise failed: %s", exc)
            turn_context.metadata["compaction_consecutive_failures"] = (
                ctx.consecutive_failures + 1
            )
            return messages, 0
        finally:
            turn_context.metadata.pop("_compaction_in_progress", None)

        # Reset failure counter on success
        turn_context.metadata["compaction_consecutive_failures"] = 0
        # observability counter
        turn_context.metadata["tier5_summaries_generated"] = (
            int(turn_context.metadata.get("tier5_summaries_generated", 0)) + 1
        )

        # Return new_messages with rough freed estimate (estimator will recompute)
        return new_messages, max(0, len(messages) - len(new_messages))
```

### 3.5 `LLMForkSummarizer`（Tier 5 实现）

新文件 `services/compact/llm_fork.py`：

```python
"""Fork a stateless summarisation call to compress the conversation.

Strategy:
  * Always preserve protect_first_n + protect_last_n messages verbatim
    (system + first user + most recent context).
  * Send the middle slice to the model with an explicit summarisation
    instruction; tools=[]; max_output_tokens limited.
  * Replace middle slice with a synthetic user message containing the
    summary text.
  * Boundary marker (compact_boundary system message) inserted so we
    can locate this compaction in future operations (e.g. /context).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import TurnContext


COMPACT_PROMPT = (
    "You are a conversation summariser. Read the conversation excerpt above "
    "and produce a compact summary that preserves:\n"
    "  1. The user's original task and any sub-tasks discovered.\n"
    "  2. Files / commands / data the assistant has touched (with paths).\n"
    "  3. Decisions made and their reasoning.\n"
    "  4. Any open questions or pending sub-tasks.\n"
    "Format as a Markdown bulleted list under the heading "
    "'## Earlier conversation summary'. Do NOT call any tools — output "
    "summary text only. Be specific; reference actual file paths and "
    "commands; aim for {target_tokens} tokens or fewer."
)


@dataclass
class LLMForkSummarizer:
    """Stateless conversation summariser via a one-shot provider call."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        config,                          # EngineConfig
        logger,
    ) -> None:
        self.provider = provider
        self.config = config
        self.logger = logger

    def summarise(
        self,
        messages: list[dict],
        *,
        model: str,
        turn_context: TurnContext,
    ) -> list[dict]:
        """Return a compressed message list with the middle replaced by summary."""
        protect_first = self.config.compression_protect_first_n
        protect_last = self.config.compression_protect_last_n
        target_tokens = self.config.compression_target_summary_tokens

        if len(messages) <= protect_first + protect_last:
            # Too few messages to compress — bail.
            return messages

        head = messages[:protect_first]
        tail = messages[-protect_last:]
        middle = messages[protect_first:-protect_last]

        # Build a prompt for the summariser fork.
        excerpt = "\n\n".join(_format_message_for_summary(m) for m in middle)
        instruction = COMPACT_PROMPT.format(target_tokens=target_tokens)
        summariser_messages = [
            {
                "role": "user",
                "content": f"{excerpt}\n\n---\n\n{instruction}",
            },
        ]

        response = self.provider.generate(
            messages=summariser_messages,
            tools=[],
            stream_callback=None,
            model_call_config=self.config.model_call_config.replace(
                max_tokens=target_tokens + 1000,  # buffer
                temperature=0.0,
            ),
        )
        summary_text = (response.text or "").strip()
        if not summary_text:
            raise RuntimeError("Summariser returned empty text")

        # Build the boundary + summary stub
        boundary = {
            "role": "system",
            "content": (
                "[compact_boundary] "
                f"Compacted {len(middle)} messages between protect_first={protect_first} "
                f"and protect_last={protect_last} via Tier 5 LLM fork."
            ),
            "_aether_meta": {"compact_boundary": True},
        }
        summary_msg = {
            "role": "user",  # 'system' wouldn't be honoured mid-conversation
            "content": summary_text,
            "_aether_meta": {"compact_summary": True},
        }

        return head + [boundary, summary_msg] + tail


def _format_message_for_summary(msg: dict) -> str:
    """Compact one message to fit into the summariser's input."""
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, str):
        body = content
    elif isinstance(content, list):
        # Multimodal / tool-call blocks — flatten to text where possible
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use: {block.get('name')}({block.get('input')})]")
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    parts.append(f"[tool_result: {inner if isinstance(inner, str) else '...'}]")
            else:
                parts.append(str(block))
        body = "\n".join(parts)
    else:
        body = str(content)
    return f"### {role.upper()}\n{body}"
```

### 3.6 与 `agent.py` 接合

入口 PREFLIGHT 阶段（`run_loop` 主循环开始前）：

```python
# === PR 3.4 新增 ===
from aether.services.compact.compactor import CompactionPipeline
from aether.services.compact.autocompact import AutoCompactor
from aether.services.compact.llm_fork import LLMForkSummarizer
from aether.services.compact.microcompact import NoOpMicrocompactor  # PR 3.5 替换
from aether.services.compact.snip import NoOpSnipper                # PR 3.6 替换
from aether.services.compact.collapse import NoOpCollapseTier       # PR 3.7 替换
from aether.runtime.usage import sum_usages
from aether.services.compact.token_estimation import estimate_messages_tokens

if not hasattr(self, "_compaction_pipeline"):
    self._compaction_pipeline = CompactionPipeline(
        tiers=[
            NoOpSnipper(),                                          # PR 3.6
            NoOpMicrocompactor(),                                   # PR 3.5
            NoOpCollapseTier(),                                     # PR 3.7
            AutoCompactor(
                config=self.config,
                summarizer=LLMForkSummarizer(
                    provider=self.services.provider,
                    config=self.config,
                    logger=self.services.logger,
                ),
                logger=self.services.logger,
            ),
        ],
        token_estimator=estimate_messages_tokens,
        config=self.config,
        logger=self.services.logger,
    )

# Preflight check
result = self._compaction_pipeline.maybe_compress(
    messages,
    turn_context=context,
    model=getattr(self.services.provider, "model", "unknown"),
    model_window=self._resolve_model_window(),
    trigger_reason="preflight",
)
messages = result.compressed_messages
```

`_invoke_provider_with_recovery` 段，把"早退"改造成"流水线压缩 + retry"：

```python
# === PR 3.4 改造 ===
# OLD: 直接 set context.metadata["recovery_terminal_exit_reason"] 然后 return error
# NEW:
if decision.compress_context:
    context.metadata["recovery_compress_required"] = True
    trigger = (
        "payload_too_large"
        if decision.classified_reason == FailoverReason.payload_too_large.value
        else "context_overflow"
    )
    result = self._compaction_pipeline.maybe_compress(
        messages,
        turn_context=context,
        model=getattr(self.services.provider, "model", "unknown"),
        model_window=self._resolve_model_window(),
        trigger_reason=trigger,
    )
    if result.exhausted:
        # Pipeline ran out of options. Fall back to legacy terminal.
        context.metadata["recovery_terminal_exit_reason"] = (
            ExitReason.PAYLOAD_TOO_LARGE.value if trigger == "payload_too_large"
            else ExitReason.COMPRESSION_EXHAUSTED.value
        )
        return AgentEngine._ProviderInvocationOutcome(error=exc)
    # Compression worked — caller should retry with the compressed messages.
    # We mutate the messages list in place so the next iteration sees them.
    messages.clear()
    messages.extend(result.compressed_messages)
    continue  # retry the provider call
```

辅助方法：

```python
def _resolve_model_window(self) -> int:
    """Best-effort resolve current model's context window."""
    # Stage 1: check provider attribute
    window = getattr(self.services.provider, "context_window", None)
    if isinstance(window, int) and window > 0:
        return window
    # Stage 2: known defaults (will be replaced by ProviderProfile in Sprint 5)
    name = (getattr(self.services.provider, "model", "") or "").lower()
    if "claude" in name:
        return 200_000
    if "kimi" in name or "moonshot" in name:
        return 200_000
    if "gpt-4o" in name or "gpt-4-turbo" in name:
        return 128_000
    return 32_000  # safe fallback
```

### 3.7 token 估算工具

新文件 `services/compact/token_estimation.py`（小巧、独立）：

```python
"""Lightweight token estimation for compaction decisions.

Intentionally a rough estimate (not a real tokeniser) — we just need
"is this above 85% of the window" granularity. Pipeline rerun after
each tier validates the actual saved tokens via the same estimator,
so consistency matters more than absolute accuracy.

Heuristic: 1 token ≈ 4 characters (English-heavy) or 1.6 characters
(CJK-heavy). We use 4 as the baseline because models typically charge
in BPE tokens and we want a CONSERVATIVE upper bound (slightly
overestimate, trigger compaction slightly early — safer than the other
direction). Pad by 4/3 like claude-code does.
"""

from __future__ import annotations

import json
from typing import Any


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for a message list.

    Strategy:
      * Each message overhead ≈ 4 tokens (role + delimiters).
      * Body content: len(text) // 4, then × 4/3 (claude-code's padding).
    """
    total_chars = 0
    for msg in messages:
        total_chars += 4  # overhead
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                total_chars += _estimate_block_chars(block)
        elif content is not None:
            total_chars += len(json.dumps(content, ensure_ascii=False))
    return int((total_chars / 4) * (4 / 3))


def _estimate_block_chars(block: Any) -> int:
    if not isinstance(block, dict):
        return len(str(block))
    block_type = block.get("type")
    if block_type == "text":
        return len(block.get("text", ""))
    if block_type == "tool_use":
        return 20 + len(json.dumps(block.get("input") or {}, ensure_ascii=False))
    if block_type == "tool_result":
        inner = block.get("content")
        if isinstance(inner, str):
            return 20 + len(inner)
        if isinstance(inner, list):
            return 20 + sum(_estimate_block_chars(b) for b in inner)
        return 20
    if block_type in ("image", "document"):
        return 8000  # ≈ 2000 tokens, claude-code's heuristic
    if block_type == "thinking":
        return 20 + len(block.get("thinking", ""))
    return len(json.dumps(block, ensure_ascii=False))
```

### 3.8 NoOp 占位类（让骨架闭环，后续 PR 替换）

`services/compact/microcompact.py`：

```python
class NoOpMicrocompactor:
    """Placeholder for PR 3.5 implementation."""
    name = "tier3_microcompact"
    def maybe_run(self, messages, ctx, turn_context):
        return messages, 0
```

同样的 NoOp 在 `snip.py` 和 `collapse.py` 各放一个。

### 3.9 `EngineConfig` 新字段

```python
# Sprint 3 / PR 3.4: master switch for the entire compaction pipeline.
# When False, ALL tiers (1-5) are inert and the engine behaves as
# pre-Sprint-3 — useful as an emergency rollback if any tier
# misbehaves in production. Default False (opt-in) per the sprint plan.
compression_enabled: bool = False

# Sprint 3 / PR 3.4: token threshold (as fraction of context window)
# for triggering preflight compression in the pipeline. Below this,
# preflight is a no-op. 0.85 mirrors claude-code's autocompact target
# (context_window - 13k buffer ≈ 93% of window, but ours runs slightly
# earlier because we have less downstream cleanup).
compression_pre_llm_pct: float = 0.85

# Sprint 3 / PR 3.4: separate threshold for Tier 5 autocompact
# specifically. In claude-code these can diverge (see calculateTokenWarningState
# / WARNING_THRESHOLD_BUFFER_TOKENS) but we start with the same value.
compression_autocompact_pct: float = 0.85

# Sprint 3 / PR 3.4: per-turn limit on consecutive Tier 5 (autocompact)
# failures before we trip the circuit breaker for the rest of the turn.
# Mirrors claude-code's MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3.
compression_max_failures: int = 3

# Sprint 3 / PR 3.4: when True (default) Tier 5 autocompact is allowed
# to fire. Separate from compression_enabled so operators can disable
# Tier 5 specifically and still keep cheaper tiers running.
autocompact_enabled: bool = True

# Sprint 3 / PR 3.4: head/tail messages always preserved verbatim
# during Tier 5 summarisation. Default 2/6: protect system + first
# user message at head; protect 6 most recent messages at tail
# (≈3 user-assistant exchanges).
compression_protect_first_n: int = 2
compression_protect_last_n: int = 6

# Sprint 3 / PR 3.4: target token budget for the summary text Tier 5
# generates. Caps the output of the LLM fork so even a runaway
# summariser doesn't replace 100k of conversation with 100k of
# summary. 4_000 ≈ a few thousand words.
compression_target_summary_tokens: int = 4_000
```

### 3.10 `ExitReason` 新增

[`runtime/contracts.py`](../../backend/harness/aether/runtime/contracts.py)：

```python
# Sprint 3 / PR 3.4: pipeline ran every tier and still couldn't
# compress messages below the model's blocking limit. Distinct from
# CONTEXT_EXHAUSTED (which is the legacy single-shot terminal) so
# operators can tell "we tried everything" apart from "we noticed
# context was over".
COMPRESSION_EXHAUSTED = "COMPRESSION_EXHAUSTED"
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/services/__init__.py` | **新文件** | 空 | 0 |
| `backend/harness/aether/services/compact/__init__.py` | **新文件** | 公共导出 | ~20 |
| `backend/harness/aether/services/compact/compactor.py` | **新文件** | `CompactionContext` + `CompactionPipeline` + `CompactionResult` | ~180 |
| `backend/harness/aether/services/compact/autocompact.py` | **新文件** | `AutoCompactor` + 5 条件门控 + 熔断 | ~120 |
| `backend/harness/aether/services/compact/llm_fork.py` | **新文件** | `LLMForkSummarizer` + `_format_message_for_summary` | ~150 |
| `backend/harness/aether/services/compact/token_estimation.py` | **新文件** | `estimate_messages_tokens` + `_estimate_block_chars` | ~70 |
| `backend/harness/aether/services/compact/microcompact.py` | **新文件** | `NoOpMicrocompactor` 占位 | ~10 |
| `backend/harness/aether/services/compact/snip.py` | **新文件** | `NoOpSnipper` 占位 | ~10 |
| `backend/harness/aether/services/compact/collapse.py` | **新文件** | `NoOpCollapseTier` 占位 | ~10 |
| `backend/harness/aether/runtime/contracts.py` | 修改 | 加 `ExitReason.COMPRESSION_EXHAUSTED` | +5 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 7 个 compression_* 字段 | ~50（含详细注释） |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 流水线初始化 + preflight 调用 + `_invoke_provider_with_recovery` 改造 + `_resolve_model_window` | ~80 |
| `backend/harness/aether/tests/test_compaction_pipeline.py` | **新文件** | 见 § 五.1 | ~250 |
| `backend/harness/aether/tests/test_autocompact_gate.py` | **新文件** | 见 § 五.2 | ~250 |
| `backend/harness/aether/tests/test_llm_fork_summarizer.py` | **新文件** | 见 § 五.3 | ~200 |
| `backend/harness/aether/tests/test_token_estimation.py` | **新文件** | 见 § 五.4 | ~120 |
| `backend/harness/aether/tests/test_compaction_integration.py` | **新文件** | 见 § 五.5 端到端 | ~250 |

## 五、测试用例（详细）

### 5.1 `test_compaction_pipeline.py`

**测试组 A：禁用与早退**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | `compression_enabled=False` | maybe_compress 直接返回原消息；`tiers_run == []` |
| **T-A2** | preflight + 当前 token < 85% 阈值 | 直接早退；`tiers_run == []` |
| **T-A3** | `trigger_reason="context_overflow"` 且 token < 阈值 | 仍运行流水线（强制触发，不看预 estimate） |

**测试组 B：流水线编排**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 所有 4 tier 都是 NoOp | tiers_run = 4 个名字；exhausted=True；messages 不变 |
| **T-B2** | Tier 2 freed 50% token，已达阈值 | 只跑 tier2；tiers_run = ['tier2_snip']；不再跑 tier3-5 |
| **T-B3** | 4 tier 依次释放 10% / 20% / 0% / 80% | tiers_run = 4 个名字（每次都没达到阈值，跑完所有） |
| **T-B4** | 某 tier 抛异常（mock） | 异常上抛，不被静默吞掉（pipeline 不该 try/except） |
| **T-B5** | tier_outcomes 记录每级 before/after | 每级 dict 含 `name / tokens_before / tokens_after / freed_reported / freed_actual` |

**测试组 C：observability**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | 跑完一次 maybe_compress | `turn_context.metadata["tier_outcomes"]` 含此次所有级别的记录 |
| **T-C2** | 跑两次 maybe_compress（同一 turn） | tier_outcomes 累加（不替换） |

### 5.2 `test_autocompact_gate.py`

**测试组 D：5 条件门控**

| ID | 条件失败的那一条 | 验证 |
|---|---|---|
| **T-D1** | `_compaction_in_progress=True` | maybe_run 返回 (messages, 0)；不调 summariser |
| **T-D2** | `compression_enabled=False` | 同上 |
| **T-D3** | `autocompact_enabled=False` | 同上 |
| **T-D4** | `collapse_owns_headroom=True` | 同上 |
| **T-D5** | `pre_compaction_tokens < threshold` | 同上 |
| **T-D6** | 所有条件都满足 | 调 summariser 一次；返回新 messages |

**测试组 E：熔断**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | `consecutive_failures=2`, max=3 | 仍运行 |
| **T-E2** | `consecutive_failures=3`, max=3 | 跳过；返回 (messages, 0) |
| **T-E3** | summariser 抛异常 | `consecutive_failures` 在 metadata 里 +1 |
| **T-E4** | summariser 成功 | `consecutive_failures` 重置为 0 |
| **T-E5** | 第 3 次失败后 metadata 累积到 3 | 第 4 次调用直接被熔断 |

**测试组 F：observability**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | summariser 成功 | `metadata["tier5_summaries_generated"]` +1 |
| **T-F2** | `_compaction_in_progress` 在 finally 中 pop | 即使 summariser 抛异常也清理 |

### 5.3 `test_llm_fork_summarizer.py`

**测试组 G：基本生成**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 10 条消息（system + user + 8 中间） | 返回 head(2) + boundary + summary + tail(6)；总数 10 → 11 但每条更短 |
| **T-G2** | 太少消息（≤ protect_first + protect_last） | 返回原消息不动 |
| **T-G3** | summariser 返回空字符串 | 抛 RuntimeError |
| **T-G4** | summariser 返回长文本（> target） | 仍接受（target 是 prompt hint，不是硬截断） |
| **T-G5** | 检查 summariser 调用的 `tools=[]` | 强制不带工具 |
| **T-G6** | 检查 summariser 调用的 `temperature=0.0` | 确定性输出 |

**测试组 H：消息格式化**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | `_format_message_for_summary` 处理 string content | 输出 `### USER\n<text>` |
| **T-H2** | 处理 list content（含 text + tool_use + tool_result） | 各 block flatten 成 `[tool_use: name(input)]` 等可读形式 |
| **T-H3** | 处理 None content | 输出 `### ROLE\nNone` |

**测试组 I：boundary marker**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | summarise 后 boundary 消息存在 | `_aether_meta == {"compact_boundary": True}` |
| **T-I2** | summary 消息标记 | `_aether_meta == {"compact_summary": True}` |

### 5.4 `test_token_estimation.py`

**测试组 J：基础估算**

| ID | 场景 | 验证 |
|---|---|---|
| **T-J1** | 空 messages | 返回 0 |
| **T-J2** | 一条 user "hello" (5 字符) | ≈ (4+5)/4 * 4/3 ≈ 3 |
| **T-J3** | 一条 user 含 1000 字符英文 | ≈ 1000/4 * 4/3 ≈ 333 |
| **T-J4** | 一条含 image block | ≈ 8000/4 * 4/3 ≈ 2666 |
| **T-J5** | 含 tool_use block | overhead 20 + json(input) 长度 |
| **T-J6** | 含 thinking block | overhead 20 + thinking 长度 |
| **T-J7** | 复杂 mixed list content | 各 block 长度求和 |

**测试组 K：连续运行一致性**

| ID | 场景 | 验证 |
|---|---|---|
| **T-K1** | 连续 10 次 estimate 同一 messages | 完全相同结果（确定性） |
| **T-K2** | estimate(messages) > estimate(messages[:-1]) | 删消息后 token 数下降 |

### 5.5 `test_compaction_integration.py`（端到端）

**测试组 L：preflight 触发**

| ID | 场景 | 验证 |
|---|---|---|
| **T-L1** | mock provider，第一次发出 50 条长消息（> 85% of 200k）后调 run_loop | preflight 触发 Tier 5 fork；fork 调用一次 provider；返回压缩后消息；正式调用第二次 provider；turn 完成 |
| **T-L2** | 同上但 `compression_enabled=False` | preflight 不触发；直接发原消息；如果上下文超限就退到 `CONTEXT_EXHAUSTED`（沿用 Sprint 2 行为） |

**测试组 M：context_overflow 触发**

| ID | 场景 | 验证 |
|---|---|---|
| **T-M1** | mock provider 第一次返 400 + classified=context_overflow | recovery 发 compress_context=True；流水线触发 Tier 5；fork 调用一次；retry provider；成功返回 |
| **T-M2** | mock provider 5 次都返 400 | 流水线 5 次触发，每次熔断递增；3 次后熔断；最终 `ExitReason.COMPRESSION_EXHAUSTED` |
| **T-M3** | mock 是 payload_too_large 而不是 context_overflow | 流水线触发但 trigger_reason=`payload_too_large`；exhausted 时 ExitReason 是 `PAYLOAD_TOO_LARGE`（不是 COMPRESSION_EXHAUSTED） |

**测试组 N：与 Sprint 2 fallback 共存**

| ID | 场景 | 验证 |
|---|---|---|
| **T-N1** | mock provider 1 返 429（rate_limit）→ fallback 到 provider 2，provider 2 返 400 + context_overflow | provider 1 触发 fallback；provider 2 触发流水线压缩；最终成功 |
| **T-N2** | 流水线压缩成功后 retry provider 1（不应该）vs provider 2（应该） | 验证 fallback chain 状态在压缩后不重置 |

**测试组 O：跨 session 隔离**

| ID | 场景 | 验证 |
|---|---|---|
| **T-O1** | session A 流水线触发熔断（3 失败）后，session B 第一次触发 | session B 不受影响（`compaction_consecutive_failures` 在 metadata 里 per-turn-context） |

## 六、验收门

- [ ] 所有新测试 green
- [ ] 既有 357 个测试无回归
- [ ] 手工验证：mock 50 条长消息触发 preflight，CLI 看到 `compact[preflight] tier5_autocompact: ... → ... tokens` 日志
- [ ] 手工验证：触发 context_overflow（构造一个超长 prompt），看到 retry 后成功
- [ ] 手工验证：连续 3 次 context_overflow，最终退到 `COMPRESSION_EXHAUSTED` 而非死循环
- [ ] `result.metadata["compaction"]["tier5_summaries_generated"]` 与实际 fork 次数一致

## 七、回滚开关

- `compression_enabled=False`（**默认**）：整个流水线 inert，行为完全等同 Sprint 2
- `autocompact_enabled=False`：仅关 Tier 5，其他 tier 仍可跑
- 完全 revert PR：删除 `services/` 目录 + 回滚 `agent.py` 的两段改造

## 八、实施顺序（建议 3 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 创建 `services/compact/` 目录骨架 + NoOp 占位 | 1h | 4 个 stub 文件 |
| 2. `services/compact/token_estimation.py` + 测试 | 1.5h | 完整工具 + T-J/K |
| 3. `services/compact/compactor.py` (Pipeline + Context + Result) | 2h | 主编排 |
| 4. `services/compact/llm_fork.py` (Summarizer) | 2h | 重武器实现 |
| 5. `services/compact/autocompact.py` (5 条件 + 熔断) | 1.5h | 门控 |
| 6. `tests/test_compaction_pipeline.py` (T-A/B/C) | 2h | ~10 case |
| 7. `tests/test_autocompact_gate.py` (T-D/E/F) | 2h | ~10 case |
| 8. `tests/test_llm_fork_summarizer.py` (T-G/H/I) | 2h | ~10 case |
| 9. `agent.py` 流水线初始化 + 接合 | 2h | 入口 + recovery 改造 |
| 10. `config/schema.py` + `runtime/contracts.py` | 1h | 7 字段 + 1 ExitReason |
| 11. `tests/test_compaction_integration.py` (T-L/M/N/O) | 4h | 端到端 |
| 12. 既有测试回归 | 1h | unittest discover |
| 13. 手工验证 | 1h | 真实长 prompt + 模拟 413 |

总计 ≈ 23h ≈ 3 工程日。

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| LLM fork 调用本身又触发 context_overflow（递归） | 中 | `_compaction_in_progress` 递归守卫 + `protect_first_n / protect_last_n` 限制 fork 输入 |
| token estimator 误差大导致频繁触发 | 中 | 4/3 padding 偏保守；后续可换真 tokeniser；当前测试覆盖一致性而非绝对精度 |
| 流水线在不该触发时触发（preflight 阈值低） | 低 | `compression_enabled` 默认 False，必须显式开 |
| Tier 5 输出的 summary 丢失关键上下文导致后续轮决策错误 | 中 | `protect_last_n=6` 保留近 6 条原消息；prompt 明确要求保留"用户原始任务、文件路径、未决问题" |
| `_resolve_model_window` 对未知模型 fallback 太小 | 低 | 32k 是非常保守的下限；用户可在 ProviderProfile（Sprint 5）覆盖 |
| 流水线 + fallback chain 互相干扰 | 中 | T-N1/N2 端到端测试；逻辑上：fallback 切换发生在 provider 层，压缩发生在 message 层，两者解耦 |

## 十、与后续 PR 的接合

- **PR 3.5** 把 `NoOpMicrocompactor` 替换为 `TimeBasedMicrocompactor`，骨架不变。
- **PR 3.6** 把 `NoOpSnipper` 替换为 `Snipper`，骨架不变。
- **PR 3.7** 把 `NoOpCollapseTier` 替换为 `ContextCollapseTier`；同时让 collapse 启用时
  `turn_context.metadata["collapse_owns_headroom"] = True` 触发 `AutoCompactor` 条件 4。
- **Sprint 5** prompt cache 落地后，`Microcompact` 加 `CachedMicrocompactor` 子分支；
  `LLMForkSummarizer` 的 prompt 可以用 cache 命中节约 token。
- **Sprint 5** ProviderProfile 落地后，`_resolve_model_window` 改为读 profile。
