# PR 3.5 — Tier 3：Microcompact 时间触发分支

> **角色**：流水线第 3 层。本地、低成本，专门处理"用户离开一段时间回来"的场景。
> 依赖 PR 3.4 的 `services/compact/` 骨架，把 `NoOpMicrocompactor` 替换为
> `TimeBasedMicrocompactor`。

## 一、目标

1. 实现 `TimeBasedMicrocompactor` 复刻 [`microCompact.ts:253-292`](../../tmp/claude-code-references)
   的时间触发分支（cache-edit 分支等 Sprint 5 prompt cache 落地后再补）。
2. 检测最后一条 assistant 消息时间，超过 `gap_threshold_minutes`（默认 5）后触发。
3. 保留最近 N 条工具结果（默认 5），其余替换为 `[Old tool result content cleared]`。
4. 留 `CachedMicrocompactor` 占位类（cache-edit 分支），结构与 claude-code 对齐。

## 二、现状分析

### 2.1 PR 3.4 留下的 NoOp

```python
# services/compact/microcompact.py
class NoOpMicrocompactor:
    name = "tier3_microcompact"
    def maybe_run(self, messages, ctx, turn_context):
        return messages, 0
```

本 PR 把这个文件整个重写。

### 2.2 claude-code 的核心逻辑

[`microCompact.ts:253`](../../tmp/claude-code-references) 的 `microcompactMessages`：

```typescript
export async function microcompactMessages(
  messages: Message[],
  toolUseContext?: ToolUseContext,
  querySource?: QuerySource,
): Promise<MicrocompactResult> {
  // 时间触发优先
  const timeBasedResult = maybeTimeBasedMicrocompact(messages, querySource)
  if (timeBasedResult) {
    return timeBasedResult
  }
  // ... cache-edit 路径（feature gated, ant-internal）
}
```

`maybeTimeBasedMicrocompact` 的关键步骤：

1. 取最后一条 assistant 消息的时间戳，计算 gap。
2. 如果 gap < threshold（默认服务端配 5min），返回 null（不触发）。
3. 否则收集所有"可压缩工具"的 tool_use_id，按出现顺序排列。
4. 保留最后 `keep_recent` 个（默认 5），其余的 tool_result 替换为 `[Old tool result content cleared]`。
5. 触发 `notifyCacheDeletion`（claude-code 的缓存追踪）。

### 2.3 我们的差异化

| claude-code | Aether 实现 |
|---|---|
| `Date.now() - lastAssistant.timestamp` | 我们的 messages 不一定带 timestamp。需要在 PRE_LLM 阶段往最后一条 assistant 消息塞 `_aether_meta.timestamp`。或者复用 session_record 的 updated_at。 |
| `feature('CACHED_MICROCOMPACT')` ant-internal | 我们没有，cache-edit 路径只放 stub class |
| `notifyCacheDeletion` 通知前端 | 我们没有 cache 追踪系统，留 hook 但 no-op |
| GrowthBook 配置 | 用 EngineConfig 静态字段 |

## 三、设计

### 3.1 重写 `services/compact/microcompact.py`

```python
"""Tier 3 microcompact — clear stale tool results when cache is cold.

Two intended paths (mirroring claude-code):

  * TimeBasedMicrocompactor (this file, ALWAYS active when Tier 3 enabled)
    Detects "user came back after a long break" via gap since the last
    assistant message; replaces all-but-the-most-recent compactable tool
    results with a placeholder. Mutates messages directly because the
    cache is assumed cold anyway.

  * CachedMicrocompactor (stub here, real impl after Sprint 5 prompt cache)
    For warm-cache scenarios — uses provider-specific cache_edits API to
    delete tool results server-side without invalidating the cached prefix.
    Does NOT mutate local messages.

Two paths are mutually exclusive: time-based wins when the cache is cold
(rewriting the prefix is unavoidable anyway), cached path otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext


# Mirrors microCompact.ts:36 — keep the literal string identical so any
# downstream consumer that pattern-matches gets the same behaviour.
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"


# Mirrors microCompact.ts COMPACTABLE_TOOLS (line 41-50). Names listed
# here are the AETHER builtin tool names; we'll normalize callers' tool
# names through phantom_tool's _normalize_name so casing / namespace
# variants match. The default set covers all 6 builtins (except
# write_file which produces tiny output and isn't worth clearing).
DEFAULT_COMPACTABLE_TOOLS: tuple[str, ...] = (
    "read_file",
    "shell",
    "grep",
    "glob",
    "list_dir",
    # write_file intentionally omitted: result is just a "wrote N bytes"
    # confirmation; clearing adds noise without saving meaningful tokens.
    # Add web_search / web_fetch when those builtins land.
)


@dataclass
class TimeBasedMicrocompactor:
    """Tier 3 — clear old tool results when the model's been idle.

    Triggered ONLY when:
      * the master compression switch is on,
      * the gap since the last assistant message exceeds the threshold,
      * there is more than `keep_recent` compactable tool calls in history.

    Does NOT consult cache state — assumes cache is cold by the time the
    gap exceeds threshold. Future CachedMicrocompactor handles warm-cache.
    """

    name: str = "tier3_microcompact"

    def __init__(self, *, config, logger) -> None:
        self.config = config
        self.logger = logger

    def maybe_run(
        self,
        messages: list[dict],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict], int]:
        if not self.config.compression_enabled:
            return messages, 0

        gap_minutes = self._compute_gap_minutes(messages, turn_context)
        threshold = self.config.microcompact_gap_threshold_minutes
        if gap_minutes is None or gap_minutes < threshold:
            return messages, 0

        keep_recent = max(1, self.config.microcompact_keep_recent)
        compactable_tool_ids = self._collect_compactable_tool_ids(messages)
        if len(compactable_tool_ids) <= keep_recent:
            return messages, 0  # nothing to clear

        keep_set = set(compactable_tool_ids[-keep_recent:])
        clear_set = set(compactable_tool_ids[:-keep_recent])

        new_messages, tokens_freed = self._clear_tool_results(
            messages, clear_set
        )
        if tokens_freed == 0:
            return messages, 0

        # observability
        turn_context.metadata["tier3_cleared_count"] = (
            int(turn_context.metadata.get("tier3_cleared_count", 0))
            + len(clear_set)
        )
        self.logger.info(
            "tier3: gap=%.1fmin > %.1fmin; cleared %d tool results, kept last %d, "
            "freed ≈%d tokens",
            gap_minutes, threshold, len(clear_set), keep_recent, tokens_freed,
        )

        return new_messages, tokens_freed

    # --------------------------- helpers ---------------------------

    def _compute_gap_minutes(
        self,
        messages: list[dict],
        turn_context: TurnContext,
    ) -> Optional[float]:
        """Minutes since the last assistant message; None if can't tell."""
        # Prefer per-message timestamp written by AgentEngine
        # (see § 3.3 below for the engine-side change).
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            meta = msg.get("_aether_meta") or {}
            ts = meta.get("timestamp")
            if isinstance(ts, (int, float)):
                return max(0.0, (time.time() - ts) / 60.0)
            break
        # Fallback: session_record updated_at if available
        sess_meta = turn_context.metadata.get("session_record_meta") or {}
        ts = sess_meta.get("last_assistant_timestamp")
        if isinstance(ts, (int, float)):
            return max(0.0, (time.time() - ts) / 60.0)
        return None

    def _collect_compactable_tool_ids(self, messages: list[dict]) -> list[str]:
        """Walk messages, return tool_use IDs in encounter order."""
        from aether.agents.core.phantom_tool import _normalize_name

        compactable_set = {
            _normalize_name(n)
            for n in self.config.microcompact_compactable_tools
        }

        ids: list[str] = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                tool_name = _normalize_name(block.get("name") or "")
                if tool_name in compactable_set:
                    tool_id = block.get("id")
                    if tool_id:
                        ids.append(tool_id)
        return ids

    def _clear_tool_results(
        self,
        messages: list[dict],
        clear_set: set[str],
    ) -> tuple[list[dict], int]:
        """Replace cleared tool_results with placeholder; return (new, tokens_freed_estimate)."""
        from aether.services.compact.token_estimation import (
            estimate_messages_tokens,
        )

        before_tokens = estimate_messages_tokens(messages)
        new_messages: list[dict] = []
        for msg in messages:
            if msg.get("role") != "user":
                new_messages.append(msg)
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                new_messages.append(msg)
                continue
            new_content = []
            touched = False
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") in clear_set
                ):
                    inner = block.get("content")
                    if (
                        isinstance(inner, str)
                        and inner == TIME_BASED_MC_CLEARED_MESSAGE
                    ):
                        # Already cleared on a previous pass — leave alone
                        new_content.append(block)
                        continue
                    new_block = dict(block)
                    new_block["content"] = TIME_BASED_MC_CLEARED_MESSAGE
                    new_content.append(new_block)
                    touched = True
                else:
                    new_content.append(block)
            if touched:
                new_msg = dict(msg)
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        after_tokens = estimate_messages_tokens(new_messages)
        return new_messages, max(0, before_tokens - after_tokens)


@dataclass
class CachedMicrocompactor:
    """Stub for the cache-edit path (Sprint 5+ implementation).

    When prompt cache lands (Sprint 5), this class will:
      * Track tool_use_ids that have been "registered" with the cache.
      * On warm cache: emit a `cache_edits` block on the next request
        instructing the server to delete specific tool_results, without
        invalidating the cached prefix.

    For now: maybe_run is a no-op so callers can already wire it into
    the pipeline without breaking.
    """

    name: str = "tier3_microcompact_cached"

    def maybe_run(self, messages, ctx, turn_context):
        return messages, 0
```

### 3.2 timestamp 写入 — `agent.py` 改动

每次 LLM 调用成功后，在 assistant 消息上盖 timestamp：

```python
# agents/core/agent.py — 约在 _append_assistant_message / equivalent 段
import time

assistant_msg = {
    "role": "assistant",
    "content": ...,
    # === PR 3.5 新增 ===
    "_aether_meta": {
        "timestamp": time.time(),
    },
}
messages.append(assistant_msg)
```

如果当前代码用了别的方式构造 assistant 消息，需要找到所有写入点（约 3-5 处），统一加 `_aether_meta`。

### 3.3 流水线接入

`agent.py` 流水线初始化处替换 NoOp：

```python
# === PR 3.5 替换 ===
from aether.services.compact.microcompact import TimeBasedMicrocompactor

self._compaction_pipeline = CompactionPipeline(
    tiers=[
        NoOpSnipper(),                                      # PR 3.6 替换
        TimeBasedMicrocompactor(                            # ← PR 3.5
            config=self.config,
            logger=self.services.logger,
        ),
        NoOpCollapseTier(),                                 # PR 3.7 替换
        AutoCompactor(...),
    ],
    ...
)
```

### 3.4 `EngineConfig` 新字段

```python
# Sprint 3 / PR 3.5: minutes since the last assistant message that must
# elapse before Tier 3 considers the cache "cold" enough to clear old
# tool results. Default 5 minutes (mirrors claude-code's default
# tengu_time_based_microcompact_gap_threshold_minutes growthbook config).
# Set to inf or a very large value to effectively disable Tier 3.
microcompact_gap_threshold_minutes: float = 5.0

# Sprint 3 / PR 3.5: how many of the most recent compactable tool
# results to PRESERVE (in original form) when Tier 3 fires. Default 5
# matches claude-code. Floor at 1: zero would leave the model with no
# working tool context at all.
microcompact_keep_recent: int = 5

# Sprint 3 / PR 3.5: tool names whose results Tier 3 will replace with
# the placeholder. Compared via the same normalisation tool_hardening
# uses (case-fold + dash↔underscore + namespace strip). Default covers
# the 5 read-class builtins; write_file omitted because its result is
# already tiny.
microcompact_compactable_tools: tuple[str, ...] = (
    "read_file",
    "shell",
    "grep",
    "glob",
    "list_dir",
)
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/services/compact/microcompact.py` | **重写** | 删除 NoOp，加 `TimeBasedMicrocompactor` + `CachedMicrocompactor` stub + `TIME_BASED_MC_CLEARED_MESSAGE` 常量 + `DEFAULT_COMPACTABLE_TOOLS` | ~200（净增约 190） |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 流水线初始化处替换 NoOp；assistant 消息构造处加 `_aether_meta.timestamp` | ~15 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 3 个 `microcompact_*` 字段 | ~25（含详细注释） |
| `backend/harness/aether/tests/test_microcompact_time_based.py` | **新文件** | 见 § 五 | ~280 |

## 五、测试用例（详细）

### 5.1 `test_microcompact_time_based.py`

**测试组 A：禁用与早退**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | `compression_enabled=False` | maybe_run 返回原消息；不读 timestamp |
| **T-A2** | gap 计算返回 None（无 timestamp） | 返回原消息；不修改 |
| **T-A3** | gap < threshold（4 分钟 < 5） | 返回原消息 |
| **T-A4** | gap 充分但只有 3 个 compactable tool（< keep_recent=5） | 返回原消息 |

**测试组 B：基本清理行为**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 10 个 compactable tool，gap=10min，keep_recent=5 | 前 5 个被替换为 `[Old tool result content cleared]`，后 5 个完整保留 |
| **T-B2** | 检查替换后的 tool_result block 结构 | content 字段被替换；其他字段（type / tool_use_id）保持 |
| **T-B3** | 已被清理过的 tool（content == placeholder）| 不再次替换（幂等） |
| **T-B4** | tokens_freed > 0 | 返回值正确反映释放量 |

**测试组 C：tool 名匹配**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | tool_use 用 `ReadFile`（驼峰） | 仍被识别为 compactable |
| **T-C2** | tool_use 用 `mcp__router__shell` | 仍被识别为 compactable |
| **T-C3** | tool_use 用 `update_todo`（不在名单） | **不**被清理 |
| **T-C4** | `microcompact_compactable_tools=()`（空配置） | 不清理任何工具 |
| **T-C5** | mixed batch：5 个 read_file + 5 个 update_todo | 只清理前 0 个 read_file（保留 5 个 read_file 已经是全部，update_todo 不在名单） |

**测试组 D：timestamp 解析**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | last assistant msg `_aether_meta.timestamp` 是 5min ago | gap ≈ 5.0 |
| **T-D2** | last assistant msg 没有 _aether_meta 但 session_record_meta.last_assistant_timestamp 有 | 用 fallback |
| **T-D3** | 都没有 | 返回 None；不触发 |
| **T-D4** | timestamp 是 string（异常类型） | 返回 None；不抛 |
| **T-D5** | 多条 assistant，只有第一条有 timestamp，最后一条没有 | 用最后一条（reverse 找到第一个 assistant 就停） |

**测试组 E：keep_recent 边界**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | `keep_recent=0` | 强制 floor 到 1，至少保留 1 个 |
| **T-E2** | `keep_recent=1` + 6 个 compactable | 清理 5 个，保留最后 1 个 |
| **T-E3** | `keep_recent=100` + 6 个 compactable | 都保留，不触发 |

**测试组 F：消息结构容错**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | content 不是 list（string） | 跳过该消息，不抛 |
| **T-F2** | tool_result block 缺 tool_use_id | 跳过 |
| **T-F3** | tool_use block 缺 id | 跳过该 block |
| **T-F4** | 空 messages | 返回 ([], 0) |

**测试组 G：observability**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 跑一次清理 5 个 | `metadata["tier3_cleared_count"] == 5` |
| **T-G2** | 第二次跑清理 3 个 | tier3_cleared_count 累加到 8 |
| **T-G3** | 日志输出 | 包含 gap / threshold / cleared / kept / freed_tokens |

**测试组 H：与 Tier 5 协同（集成）**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | 流水线先跑 Tier 3 释放 50% token，达到阈值 | 不再跑 Tier 5 |
| **T-H2** | 流水线 Tier 3 释放 10%，未达阈值 | 继续跑 Tier 5 |
| **T-H3** | gap < threshold（Tier 3 不触发）+ token > 阈值 | 直接跑到 Tier 5 |

**测试组 I：CachedMicrocompactor stub**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | CachedMicrocompactor.maybe_run | 返回 (messages, 0) 不变 |
| **T-I2** | name 字段 | == `tier3_microcompact_cached`（与时间触发的不同，便于日志区分） |

## 六、验收门

- [ ] 所有新测试 green（约 25 case）
- [ ] PR 3.4 的既有测试不回归（特别是 `test_compaction_pipeline.py`）
- [ ] 手工：模拟 sleep 6 分钟（或 mock time.time），看到日志 `tier3: gap=6.0min > 5.0min; cleared N tool results`
- [ ] 手工：跑一个含 8 次 read_file 的 turn → idle 6min → 下一 turn 的 messages 里前 3 次 read_file 的 tool_result 已被清理

## 七、回滚开关

- `microcompact_gap_threshold_minutes=inf`：永不触发
- `microcompact_compactable_tools=()`：永不清理任何工具
- 完全 revert PR：把 `services/compact/microcompact.py` 恢复为 NoOp 版本

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 重写 `services/compact/microcompact.py` | 2.5h | `TimeBasedMicrocompactor` + Cached stub + 常量 |
| 2. `tests/test_microcompact_time_based.py` (T-A 至 T-G) | 4h | 约 22 case |
| 3. `agent.py` 流水线初始化替换 + assistant timestamp 注入 | 1.5h | 找全 assistant 写入点 |
| 4. `config/schema.py` 加新字段 | 30min | 含详细注释 |
| 5. 集成测试 (T-H + T-I) | 1.5h | 与 Tier 5 协同 |
| 6. 既有回归 + 手工验证 | 1h | unittest discover + 真实 idle 模拟 |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| timestamp 不准（系统时间被改） | 低 | gap 算出负数时 max(0.0)，不触发 |
| keep_recent 设太小导致模型失去工作上下文 | 中 | floor 到 1；默认 5；测试组 E 覆盖边界 |
| 替换文本与 Sprint 5 cache_edit 冲突 | 低 | 已用与 claude-code 完全一致的字符串 `[Old tool result content cleared]`，测试可断言 |
| 用户主动清空 _aether_meta 后 Tier 3 永不触发 | 低 | 是预期行为（用户显式控制） |
| 一个 turn 内多次进 pipeline（被 PR 3.4 调多次）→ 重复清理 | 已防御 | T-B3 幂等测试覆盖 |

## 十、与后续 PR 的接合

- **PR 3.6** Snipper 在流水线里跑在 Tier 3 之前。Snip 删除的消息减少 Tier 3 要扫描的数量，
  但语义无冲突（Snip 删整个消息，Tier 3 只改单个 tool_result 内容）。
- **PR 3.7** Context Collapse 启用时**不影响**Tier 3（两者关心不同尺度：collapse 折叠大段历史，
  Tier 3 修剪 stale tool 结果）。但 collapse 提交后历史变短，Tier 3 触发概率自然降低。
- **Sprint 5** prompt cache 落地后，把 `CachedMicrocompactor` 的 stub 实现真正的 cache_edits 路径；
  根据 `is_warm_cache` 决定走哪条分支（与 claude-code 的 `microCompact.ts:283-285` 一致）。
