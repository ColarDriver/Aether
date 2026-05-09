# PR 3.7 — Tier 4：Context Collapse（自研投影式折叠）

> **角色**：流水线第 4 层。在 Tier 5 重武器之前再做一次"高粒度但仍可逆"的折叠。
> claude-code 的 `contextCollapse/` 是 stub，所以这是自研实现。
> 默认关闭（`context_collapse_enabled=False`），需要在生产数据观察 1-2 周后再决定是否默认开。

## 一、目标

1. 实现"投影式上下文折叠"：**不修改原 messages**，只创建 `CollapseStore` 存折叠摘要 + 原消息引用。
2. 在 PRE_LLM 阶段把"原 messages → 视图 messages"的转换应用到发出去的 payload。
3. 与 Tier 5 互斥：Tier 4 启用且活跃时，AutoCompactor 的条件 4（`collapse_owns_headroom`）失败 → 不跑。
4. 在 commit_pct（默认 90%）时提交折叠；blocking_pct（默认 95%）时强制阻塞继续折叠。

## 二、现状分析

### 2.1 PR 3.4 留下的 NoOp

```python
class NoOpCollapseTier:
    name = "tier4_collapse"
    def maybe_run(self, messages, ctx, turn_context):
        return messages, 0
```

### 2.2 claude-code 的设计描述

`contextCollapse/operations.ts` / `index.ts` 都是 `@generated stub`，但 `autoCompact.ts:215`
和 `MAX_TOKENS_RECOVERY` 的注释明确：

> Context collapse uses a "projection view": original messages stay
> intact (so resume / replay still works), but the view sent to the
> model is a folded summary. Two-stage trigger: commit-start at ~90%
> utilisation, blocking at ~95%. Autocompact (~93%) gets squeezed
> in between, which is why we explicitly suppress autocompact when
> collapse owns the headroom.

也就是：

- 折叠是"视图"——database VIEW 的语义，不修改底层 table（messages）。
- 折叠存在独立的 store（`CollapseStore`），按 segment 索引。
- 与 autocompact 的优先级竞争通过 `collapse_owns_headroom` 标志解决。

### 2.3 我们的差异化

| claude-code | Aether 实现 |
|---|---|
| GrowthBook 配 commit/block 阈值 | EngineConfig 静态配 |
| 大量 ant-internal SessionMemory 联动 | 不引入；纯本地 |
| 投影视图通过 React state | 通过 Engine `_apply_collapse_view(messages)` 函数 |
| `notifyCollapseEvent` 前端通知 | 写日志 + metadata 计数 |

## 三、设计

### 3.1 数据模型 — `CollapseSegment`

新文件 `services/compact/collapse.py`：

```python
"""Tier 4 context collapse — projection-based folding.

Self-developed (claude-code's contextCollapse/* is stub-only).

Key properties:
  * Original messages list is NEVER mutated. The original is what
    session_record / replay sees.
  * A CollapseStore (per-session, lives on TurnContext.metadata) holds
    one or more CollapseSegments — each a (start_idx, end_idx) range
    plus the LLM-generated summary text replacing it in the projection.
  * A "view" function applied at PRE_LLM time replaces the segment
    indices in the original list with the summary; everything outside
    the segment is untouched.

Trigger:
  * tier4 fires only between commit_pct (default 0.90) and the model
    blocking limit (default 0.95). Before commit_pct: we have headroom,
    no need. After blocking_pct: too late, autocompact already would
    have triggered.
  * Two-stage commit means: at commit_pct, we PROPOSE a segment
    (sumamrise it but DON'T finalise yet); at blocking_pct, we
    finalise (any subsequent maybe_run uses the projection).
  * In Sprint 3 we collapse the ONE oldest non-protected segment per
    fire. Future iterations may multi-segment.

Mutual exclusion with Tier 5:
  * When Tier 4 is enabled AND has at least one committed segment in
    this session, we set turn_context.metadata['collapse_owns_headroom']=True.
  * AutoCompactor checks this flag (condition 4) and bails out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.token_estimation import estimate_messages_tokens
from aether.services.compact.llm_fork import LLMForkSummarizer


@dataclass
class CollapseSegment:
    """One folded slice of the original messages."""

    start_idx: int                    # inclusive
    end_idx: int                      # inclusive
    summary_text: str                 # LLM-generated
    tokens_before: int                # estimator-counted, original
    tokens_after: int                 # estimator-counted, summary
    committed: bool = False           # True after blocking_pct stage

    @property
    def freed(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class CollapseStore:
    """All collapse state for one session."""

    segments: list[CollapseSegment] = field(default_factory=list)

    @property
    def has_committed(self) -> bool:
        return any(s.committed for s in self.segments)

    def total_freed(self) -> int:
        return sum(s.freed for s in self.segments if s.committed)

    def as_view(self, messages: list[dict]) -> list[dict]:
        """Apply committed segments as a projection over messages.

        Original messages list is not mutated — returns a NEW list
        with each committed segment replaced by a single
        synthetic 'collapsed_segment' system message.
        """
        if not self.has_committed:
            return messages
        # Sort committed segments by start_idx ascending; non-overlapping
        committed = sorted(
            (s for s in self.segments if s.committed),
            key=lambda s: s.start_idx,
        )
        out: list[dict] = []
        cursor = 0
        for seg in committed:
            if seg.start_idx > cursor:
                out.extend(messages[cursor:seg.start_idx])
            out.append({
                "role": "system",
                "content": (
                    f"[collapsed_segment idx={seg.start_idx}-{seg.end_idx} "
                    f"freed≈{seg.freed} tokens]\n{seg.summary_text}"
                ),
                "_aether_meta": {"collapsed_segment": True},
            })
            cursor = seg.end_idx + 1
        if cursor < len(messages):
            out.extend(messages[cursor:])
        return out


@dataclass
class ContextCollapseTier:
    """Tier 4 — projection-based collapse."""

    name: str = "tier4_collapse"

    def __init__(
        self,
        *,
        config,
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
        if not self.config.compression_enabled:
            return messages, 0
        if not self.config.context_collapse_enabled:
            return messages, 0

        store: CollapseStore = turn_context.metadata.setdefault(
            "_collapse_store", CollapseStore()
        )

        # Apply existing committed segments to current messages first
        # (so the rest of the analysis sees the projection).
        view = store.as_view(messages)
        view_tokens = estimate_messages_tokens(view)

        commit_threshold = int(
            ctx.model_window * self.config.context_collapse_commit_pct
        )
        blocking_threshold = int(
            ctx.model_window * self.config.context_collapse_blocking_pct
        )

        if view_tokens < commit_threshold:
            # Not enough pressure yet — just return the existing view.
            self._update_collapse_owns_flag(turn_context, store)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        # Pressure! Find the next segment to collapse.
        segment_range = self._select_next_segment(messages, store)
        if segment_range is None:
            # Nothing more to collapse — pipeline will fall through to Tier 5.
            self._update_collapse_owns_flag(turn_context, store)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        start_idx, end_idx = segment_range
        segment_msgs = messages[start_idx:end_idx + 1]

        # Summarise the segment via LLM fork.
        try:
            # Reuse summariser but feed only the segment.
            new_segment_view = self.summarizer.summarise(
                segment_msgs,
                model=ctx.model,
                turn_context=turn_context,
            )
            # The summariser returns head + boundary + summary + tail;
            # for a small segment we just take the summary text.
            summary_text = self._extract_summary_text(new_segment_view)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("tier4: summarise failed: %s", exc)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        tokens_before = estimate_messages_tokens(segment_msgs)
        tokens_after = estimate_messages_tokens([
            {"role": "system", "content": summary_text}
        ])
        seg = CollapseSegment(
            start_idx=start_idx,
            end_idx=end_idx,
            summary_text=summary_text,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            # Two-stage commit: only commit when blocking_threshold reached
            committed=(view_tokens >= blocking_threshold),
        )
        store.segments.append(seg)

        # observability
        turn_context.metadata["tier4_collapse_segments"] = (
            int(turn_context.metadata.get("tier4_collapse_segments", 0)) + 1
        )

        new_view = store.as_view(messages)
        self._update_collapse_owns_flag(turn_context, store)
        self.logger.info(
            "tier4: %s segment[%d-%d], view %d→%d tokens",
            "committed" if seg.committed else "proposed",
            start_idx, end_idx, view_tokens, estimate_messages_tokens(new_view),
        )
        return new_view, max(0, view_tokens - estimate_messages_tokens(new_view))

    # --------------------------- helpers ---------------------------

    def _select_next_segment(
        self,
        messages: list[dict],
        store: CollapseStore,
    ) -> Optional[tuple[int, int]]:
        """Pick the next contiguous range to collapse.

        Strategy (Sprint 3 simplest version):
          * Skip protected head (config.compression_protect_first_n).
          * Skip already-collapsed ranges (in any segment).
          * Skip protected tail (config.compression_protect_last_n).
          * From what remains, take a contiguous range of up to
            collapse_segment_max_messages (default 20).
          * If fewer than 4 remain in the candidate window, bail (nothing
            worth collapsing — Tier 5 will handle it).
        """
        protect_head = self.config.compression_protect_first_n
        protect_tail = self.config.compression_protect_last_n
        max_seg = self.config.context_collapse_segment_max_messages

        n = len(messages)
        if n <= protect_head + protect_tail + 4:
            return None

        # Build the set of indices already covered by any segment
        covered: set[int] = set()
        for s in store.segments:
            for i in range(s.start_idx, s.end_idx + 1):
                covered.add(i)

        # Scan for first uncollapsed index after protect_head
        start = None
        for i in range(protect_head, n - protect_tail):
            if i not in covered:
                start = i
                break
        if start is None:
            return None

        # Extend until we hit covered or protect_tail or hit max_seg
        end = start
        while (
            end + 1 < n - protect_tail
            and (end + 1) not in covered
            and (end - start + 1) < max_seg
        ):
            end += 1

        if end - start + 1 < 4:
            return None  # too small to bother

        return (start, end)

    @staticmethod
    def _extract_summary_text(summarised_messages: list[dict]) -> str:
        """Pull the summary text out of LLMForkSummarizer's output."""
        for msg in summarised_messages:
            meta = msg.get("_aether_meta") or {}
            if meta.get("compact_summary"):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        return ""

    @staticmethod
    def _update_collapse_owns_flag(
        turn_context: TurnContext,
        store: CollapseStore,
    ) -> None:
        if store.has_committed:
            turn_context.metadata["collapse_owns_headroom"] = True
        else:
            turn_context.metadata.pop("collapse_owns_headroom", None)
```

### 3.2 流水线接入

`agent.py`：

```python
from aether.services.compact.collapse import ContextCollapseTier

self._compaction_pipeline = CompactionPipeline(
    tiers=[
        Snipper(config=self.config, logger=self.services.logger),
        TimeBasedMicrocompactor(config=self.config, logger=self.services.logger),
        ContextCollapseTier(                                # ← PR 3.7
            config=self.config,
            summarizer=LLMForkSummarizer(
                provider=self.services.provider,
                config=self.config,
                logger=self.services.logger,
            ),
            logger=self.services.logger,
        ),
        AutoCompactor(...),
    ],
    ...
)
```

### 3.3 视图应用 — PRE_LLM hook

`agent.py` 在每次 LLM 调用前（流水线已跑完）应用一次最终视图：

```python
# Right before provider.generate(messages=...)
# === PR 3.7 新增 ===
store = context.metadata.get("_collapse_store")
if store is not None:
    request_messages = store.as_view(messages)
else:
    request_messages = messages

response = self.services.provider.generate(messages=request_messages, ...)
```

注意：**只有发给 provider 的 payload 用 view，本地 messages 列表保持原样**。
所以 session_record / replay / steer 看到的都是完整原文，符合"投影"语义。

### 3.4 `EngineConfig` 新字段

```python
# Sprint 3 / PR 3.7: master switch for Tier 4 context collapse.
# Default OFF — context collapse mutates the projection view sent to
# the model and competes with Tier 5; we want production observation
# first before flipping default. Operators enable explicitly when
# they see Tier 5 firing too aggressively on long sessions.
context_collapse_enabled: bool = False

# Sprint 3 / PR 3.7: token utilisation (as fraction of model window)
# at which Tier 4 starts proposing collapse segments. 0.90 mirrors
# claude-code's commit-start threshold. Below this, Tier 4 only
# applies any existing committed segments (read-only path).
context_collapse_commit_pct: float = 0.90

# Sprint 3 / PR 3.7: token utilisation at which proposed collapse
# segments become committed (i.e. take effect in subsequent
# projections). 0.95 mirrors claude-code's blocking threshold.
# Between commit_pct and blocking_pct: proposed but not committed,
# pipeline can re-evaluate.
context_collapse_blocking_pct: float = 0.95

# Sprint 3 / PR 3.7: maximum messages in one collapse segment.
# Larger = fewer summary calls but each one harder for the LLM
# to summarise well. Default 20 is a sweet spot in our profiling.
context_collapse_segment_max_messages: int = 20
```

### 3.5 与 AutoCompactor 的接合

PR 3.4 已经在 `AutoCompactor.maybe_run` 加了：

```python
if turn_context.metadata.get("collapse_owns_headroom"):
    return messages, 0
```

`ContextCollapseTier._update_collapse_owns_flag` 会维护这个 flag。结果：
**只要本 session 有任意一段 committed collapse，autocompact 就被屏蔽**。
这与 claude-code "Autocompact 通常赢，把 collapse 正要保存的细粒度上下文炸掉" 的设计相反——
我们选 collapse 优先（用户更可能想保留细粒度）。

如果 Tier 4 完全失败（一段都没 commit），autocompact 仍会接管。

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/services/compact/collapse.py` | **重写** | `CollapseSegment` + `CollapseStore` + `ContextCollapseTier` | ~280 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 流水线初始化替换 NoOp + PRE_LLM apply view | ~15 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 4 个 `context_collapse_*` 字段 | ~30 |
| `backend/harness/aether/tests/test_collapse_store.py` | **新文件** | 见 § 五.1 | ~200 |
| `backend/harness/aether/tests/test_collapse_tier.py` | **新文件** | 见 § 五.2 | ~280 |
| `backend/harness/aether/tests/test_collapse_integration.py` | **新文件** | 见 § 五.3 | ~200 |

## 五、测试用例（详细）

### 5.1 `test_collapse_store.py`

**测试组 A：CollapseStore.as_view**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 空 store + 5 条原 messages | view == 原 messages（同对象引用 OK） |
| **T-A2** | 1 个未 committed segment | view == 原 messages（未 committed 不进视图） |
| **T-A3** | 1 个 committed segment[2-5] in 8 条 messages | view == [m0, m1, summary_msg, m6, m7]（5 条） |
| **T-A4** | 2 个不重叠 committed segments[2-3] + [5-6] in 8 条 | view == [m0, m1, sum1, m4, sum2, m7] |
| **T-A5** | committed segment 覆盖 head（idx 0-3） | view == [sum, m4, m5, ...] |
| **T-A6** | committed segment 覆盖 tail（idx N-3 to N-1） | view == [m0, ..., sum] |
| **T-A7** | committed segment 是单条（start==end） | view 替换 1 条为 1 条 summary |
| **T-A8** | as_view 不修改原 messages | id 不变；元素引用还能在原列表找到 |

**测试组 B：CollapseSegment.freed**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | tokens_before=1000, tokens_after=200 | freed == 800 |
| **T-B2** | tokens_after > tokens_before（异常但容错） | freed == 0（max(0,...)）|

**测试组 C：CollapseStore observability**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | 2 committed + 1 proposed | total_freed == 两个 committed 的 freed 之和 |
| **T-C2** | has_committed = True 仅当任意 committed 存在 | 验证标记逻辑 |

### 5.2 `test_collapse_tier.py`

**测试组 D：禁用 / 早退**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | `compression_enabled=False` | 返回 (messages, 0) |
| **T-D2** | `context_collapse_enabled=False` | 同上 |
| **T-D3** | view_tokens < commit_threshold | 不提案；但若有已 committed segment 仍返回 view（顺便消费已存在的） |
| **T-D4** | messages 太少（<= protect_head + protect_tail + 4） | _select_next_segment 返回 None；不提案 |

**测试组 E：分段选择**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | 30 条 messages, protect_head=2, protect_tail=6, store 空 | 选 idx=2 起，end ≤ 21（max_seg=20） |
| **T-E2** | 30 条 messages, 已有 segment[2-10] committed | 下一段从 idx=11 起 |
| **T-E3** | 30 条 messages, 已有 segment[2-25] committed | 候选区域 [26-23]（无效）→ 返回 None |
| **T-E4** | max_seg=5 + 30 条 messages | 段长度恰好 5 |
| **T-E5** | 候选段 < 4 条 | 返回 None |

**测试组 F：commit/propose 状态机**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | view_tokens 在 [commit_pct, blocking_pct) 之间 | seg.committed=False（仅 proposed） |
| **T-F2** | view_tokens >= blocking_pct | seg.committed=True |
| **T-F3** | proposed segment 在下一次 maybe_run 时被 commit | 当 view_tokens 跨 blocking_pct 时，新加的 segment 是 committed；proposed 状态不被自动升级（每次 propose 都新建 segment） |

**测试组 G：collapse_owns_headroom flag**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | store 有 committed | maybe_run 后 `metadata["collapse_owns_headroom"] == True` |
| **T-G2** | store 仅 proposed（无 committed） | 标记被 pop（不 own headroom） |
| **T-G3** | store 空 | 标记被 pop |

**测试组 H：summariser 失败**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | summariser 抛异常 | 不 crash；不 add segment；返回 view 不变 |
| **T-H2** | summariser 返回空 summary_text | 不 add segment（_extract_summary_text 返回 ""，跳过）|

**测试组 I：observability**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | 一次成功 propose | `metadata["tier4_collapse_segments"] +1` |
| **T-I2** | 日志包含 `proposed/committed segment[a-b], view X→Y tokens` | 文本匹配 |

### 5.3 `test_collapse_integration.py`（端到端）

**测试组 J：与 Tier 5 互斥**

| ID | 场景 | 验证 |
|---|---|---|
| **T-J1** | Tier 4 enabled + 已有 committed segment + view_tokens 足够低 | Tier 5 不触发 |
| **T-J2** | Tier 4 enabled + 但还没 commit（所有 proposed） | Tier 5 仍可触发（Tier 4 没"拥有"headroom） |
| **T-J3** | Tier 4 disabled | Tier 5 完全按原逻辑触发 |

**测试组 K：投影正确性**

| ID | 场景 | 验证 |
|---|---|---|
| **T-K1** | turn 内触发 Tier 4 commit；下一 turn 进 PRE_LLM | provider 收到的 messages 是 view（含 collapsed_segment system msg）；session_record 看到的是原 messages |
| **T-K2** | resume 旧 session（CollapseStore 未持久化情况下） | view == 原 messages（store 重置）；模型上下文是完整原文 |

**测试组 L：与流水线协同**

| ID | 场景 | 验证 |
|---|---|---|
| **T-L1** | Snipper + Microcompact + Collapse 都启用 | 流水线按 2→3→4→5 顺序跑；每级日志可见 |
| **T-L2** | Snipper 已经把 token 降到 commit_pct 之下 | Tier 4 只 apply view 不 propose；不触发 LLM fork |
| **T-L3** | Microcompact 释放后 view_tokens 降到 commit_pct 之下 | 同上 |

**测试组 M：CollapseStore 多 session 隔离**

| ID | 场景 | 验证 |
|---|---|---|
| **T-M1** | session A 有 committed segment | session B 的 metadata 里没有 collapse_owns_headroom |
| **T-M2** | session A 跑完 turn 1 + turn 2 | turn 2 的 store 含 turn 1 的 segments（_collapse_store 在 metadata 跨 turn 持久化 via session_record? 待确认）|

> **注**：T-M2 需要确认 `TurnContext.metadata` 是否跨 turn 持久化。若不是，需要把
> `CollapseStore` 的序列化加到 session_record。这个判断在实施 PR 3.7 时根据
> `runtime/contracts.py` + `session_record.py` 现状决定。Sprint 3 的最小可用版本可以
> 让 store **per-turn 重建**——每个新 turn 重新评估，不跨 turn 复用 segments。这样设计
> 简单且仍能工作：实际上同一长会话内 collapse 是在 turn 进行中触发的（preflight 或
> recovery 阶段），跨 turn 的"已 committed"信号通过 messages 里的 collapsed_segment
> meta block 体现，下一 turn 重新看到原 messages 时其实并不损失（最多就是多跑一次
> propose）。

## 六、验收门

- [ ] 所有新测试 green（约 28 case）
- [ ] PR 3.4/3.5/3.6 的既有测试不回归
- [ ] 手工：构造一个 200-message 长 session（每条平均 500 字符 ≈ 38k token）
       → 设 model_window=40k → 应触发 Tier 4 commit（90% threshold = 36k）
- [ ] 手工：committed segment 后 next turn provider 调用日志显示压缩后 messages 长度
- [ ] 手工：同 session 多 turn 后 `metadata["collapse_owns_headroom"] == True`，autocompact 被屏蔽

## 七、回滚开关

- `context_collapse_enabled=False`（**默认**）：完全不工作
- 如发现 view 错误：临时手工清空 `metadata["_collapse_store"]` → 退到原 messages
- 完全 revert PR：恢复 `services/compact/collapse.py` 为 NoOp

## 八、实施顺序（建议 2.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 重写 `services/compact/collapse.py` 数据模型部分 | 1.5h | CollapseSegment + CollapseStore |
| 2. `tests/test_collapse_store.py` (T-A 至 T-C) | 2h | ~12 case，纯单元 |
| 3. ContextCollapseTier 主类 | 3h | maybe_run + _select_next_segment + _update_flag |
| 4. `tests/test_collapse_tier.py` (T-D 至 T-I) | 4h | ~14 case |
| 5. agent.py 流水线接入 + PRE_LLM apply view | 2h | 替换 NoOp + view hook |
| 6. config/schema.py | 30min | 4 字段 |
| 7. `tests/test_collapse_integration.py` (T-J 至 T-M) | 4h | ~12 case 端到端 |
| 8. 既有回归 + 手工验证 | 2h | 200-message session 模拟 |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 投影 view 与原 messages 在 tool_use/tool_result 配对上断裂（segment 切在配对中间） | 高 | `_select_next_segment` 应该按"消息边界"而非"tool 配对边界"切——但要避免半个工具配对。**实施时强制：如果 segment 边界正好在某 tool_use 之后但 tool_result 之前，end_idx 后退到 tool_result 之后**。需在测试 T-K1 中显式覆盖。 |
| Summariser 把段总结成一句完全无用的话 | 中 | 复用 PR 3.4 的 `LLMForkSummarizer`（已有 prompt 强调保留路径/决策/未决问题）；commit_pct 高（0.90）才触发，损失粒度对换头部空间是合理交易 |
| collapse 提交后模型在后续轮里"不知道"被折叠了 | 中 | view 中的 collapsed_segment system 消息明确标注 `freed≈N tokens` + 摘要文本，模型可见 |
| 多 turn 的 `_collapse_store` 持久化语义不清 | 中 | 见 § 五.3 T-M2 注释；Sprint 3 最小版本 per-turn 重建，留 TODO 在 `08_compaction_pipeline.md` 跨 turn 持久化作为后续工作 |
| Tier 4 commit 与 Tier 5 都想动手时谁赢 | 已设计 | `collapse_owns_headroom` flag；测试 T-J1/J2/J3 严格覆盖 |
| 默认开后用户看到模型回答有"我之前说过的 X 我忘了"现象 | 中 | 默认关；启用前在生产数据观察 1-2 周 |

## 十、与后续 PR 的接合

- **Sprint 3 收尾**：8 个文档（00-07 + 99）落地完成后，run-loop-roadmap 同步更新 P1-1 状态。
- **Sprint 5 prompt cache**：当 view 命中 cache 时，cached_microcompact + collapse 配合可以做到
  "前缀全部命中 cache，只重新算 collapsed_segment 之后的部分"，是 Sprint 5 的核心收益场景。
- **未来跨 turn 持久化**：把 `CollapseStore` 序列化到 session_record，让 resume 时也能复用历史 collapse。
  这是 Sprint 3 的明确后续工作。
- **CLI `/context` 命令**（Sprint 6）：从 `_collapse_store` 渲染压缩历史给用户看
  （"段 1：原 5400 token，折叠为 480 token，覆盖第 4-15 条消息"）。
