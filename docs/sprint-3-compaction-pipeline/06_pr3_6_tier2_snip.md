# PR 3.6 — Tier 2：Snip（自研版冗余消息修剪）

> **角色**：流水线第 2 层。本地、零成本，专门拿掉"模型反复读同一文件 / 失败 tool 调用 /
> 空 thinking 消息"这类显然冗余的历史。完成后 `tokens_freed` 直接抵扣 Tier 5 触发阈值。
> claude-code 的 `snipCompact.ts` 是 stub，所以这是自研实现。

## 一、目标

1. 实现 `Snipper`（替换 PR 3.4 的 `NoOpSnipper`），把以下三类冗余从历史里删掉：
   - **重复读文件**：同一 turn 内 `read_file <same_path>` 多次出现，只保留最后一次。
   - **失败 tool 调用**：`is_error=True` 的 tool_result 已被后续成功调用替代时删除该对。
   - **空 assistant 消息**：assistant 消息只有空 thinking / 空 text 且无 tool_call。
2. 返回 `tokens_freed` 数值，让 `CompactionPipeline` 在重新评估 token 时正确扣减。
3. 严格保留 tool_use ↔ tool_result 配对（删除 user 的 tool_result 时也删对应 assistant 的 tool_use）。

## 二、现状分析

### 2.1 PR 3.4 留下的 NoOp

```python
class NoOpSnipper:
    name = "tier2_snip"
    def maybe_run(self, messages, ctx, turn_context):
        return messages, 0
```

### 2.2 claude-code 的对应

[`snipCompact.ts`](../../tmp/claude-code-references) 是 `@generated stub`：

```typescript
// AUTO-GENERATED STUB. The actual implementation is internal to Anthropic.
export async function snipCompact(...): Promise<SnipCompactResult> {
  return { messages, tokensFreed: 0 }
}
```

但接口契约清晰：

- 输入：messages + context
- 输出：`{ messages: 修剪后的消息列表, tokensFreed: 释放的 token 数 }`
- 调用点：`autoCompact.ts:212` 在所有其他 tier 之前调用，结果传给后续阈值判断。

### 2.3 Aether 的优势

我们能看到：
- 每个 tool_use 的 `name` 和 `input`（用来做"重复"判定）。
- 每个 tool_result 的 `is_error` 标记。
- assistant 消息的 thinking / text 完整内容。

这些信息让我们能做出比 claude-code stub 更聚焦的修剪。

## 三、设计

### 3.1 三类修剪规则

#### Rule 1：重复 read 同一资源

定义"同一资源"：tool_use 的 `(name, normalized_input)` 完全相同。
对每组重复调用，**只保留最后一次**——前几次的 tool_use 和对应 tool_result 都删除。

适用工具（保守白名单）：

```python
SNIP_DUPE_RULE_TOOLS: tuple[str, ...] = (
    "read_file",   # 读同文件
    "list_dir",    # 列同目录
    "glob",        # 同 pattern
    "grep",        # 同 pattern + path
)
```

不包括 `shell` / `write_file`：
- shell 即使命令一样，副作用可能不同（时间戳、外部状态）。
- write_file 第二次写入语义不是"读同样东西"。

判定算法：

```python
def _key_for_tool_use(block: dict) -> str | None:
    name = _normalize_name(block.get("name") or "")
    if name not in compactable:
        return None
    input_ = block.get("input") or {}
    # Sort keys for deterministic hashing
    return json.dumps([name, input_], sort_keys=True, ensure_ascii=False)
```

收集所有 (key, position) → 同 key 出现 ≥ 2 次时，标记前 N-1 次为"待删"。

#### Rule 2：失败 tool 调用被后续成功的同 key 调用替代

```
[assistant] tool_use(read_file path=/x.txt) id=A
[user]      tool_result tool_use_id=A is_error=True content="ENOENT"
[assistant] tool_use(read_file path=/x.txt) id=B
[user]      tool_result tool_use_id=B is_error=False content="<file content>"
```

→ 第一对 (A, A's result) 完全可删。它已经被第二次成功调用 supersede。

适用范围：**所有**工具（不限于读类）。判定 key 同 Rule 1 但加 `is_error` 标记：

```python
# (key, is_error) → list of (tool_use_position, tool_result_position)
```

对每个 key，如果存在"先 error 后 success"的序列，删除所有先于第一次 success 的 error 对。

#### Rule 3：空 assistant 消息

定义"空 assistant 消息"：

```python
def _is_empty_assistant(msg: dict) -> bool:
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                return False  # 不可识别 → 保守不删
            t = block.get("type")
            if t == "text" and (block.get("text") or "").strip():
                return False
            if t == "thinking" and (block.get("thinking") or "").strip():
                return False
            if t == "tool_use":
                return False  # 有 tool_use 就不是空
        return True  # 全是空 text / 空 thinking
    return False
```

直接删除整条消息——但**不能**删 `role=user` 后紧跟的下一条 `role=assistant` 是 tool 流的中间帧。

### 3.2 配对完整性保证

任何删除都要确保配对：

- 删 assistant 的 tool_use（block 级）→ 必须同步删 user 消息里对应 tool_use_id 的 tool_result。
- 反之亦然：删 user 的 tool_result → 同步删 assistant 里对应的 tool_use。
- 如果 assistant 消息里**所有** tool_use 都被删了，且没有其他 content 块（text/thinking），
  整个消息也删除（避免留空消息）。
- 如果 user 消息里所有 content 都是被删的 tool_result，整个消息也删除。

实现策略：**先收集所有要删的 tool_use_id 集合**，再做一遍消息重写：

```python
class _SnipPlan:
    delete_tool_use_ids: set[str]   # 同步删 use + result
    delete_msg_indices: set[int]    # 整条 assistant empty msg 删除
```

### 3.3 `Snipper` 主类

新建（重写） `services/compact/snip.py`：

```python
"""Tier 2 snip — local redundancy removal.

Self-developed implementation (claude-code's snipCompact.ts is a stub).
Three rules apply, in order:

  Rule 1 (DUPE): for SNIP_DUPE_RULE_TOOLS, when the same (name, input)
                 tool_use appears ≥2 times, keep only the last one.
                 Tool_use_ids of earlier calls (and their tool_results)
                 are deleted.

  Rule 2 (FAIL): for any tool, when an is_error=True tool_result is
                 followed later by an is_error=False call with the same
                 (name, input), the earlier failed pair is deleted.

  Rule 3 (EMPTY): assistant messages containing only empty text /
                  empty thinking blocks (no tool_use, no real text)
                  are deleted whole.

Pairing invariant: every assistant tool_use[id=X] MUST have a matching
user tool_result[tool_use_id=X] (or vice versa). The Snipper deletes
both halves together to preserve this invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.token_estimation import estimate_messages_tokens


SNIP_DUPE_RULE_TOOLS: tuple[str, ...] = (
    "read_file",
    "list_dir",
    "glob",
    "grep",
)


@dataclass
class _SnipPlan:
    """Set of deletions to apply in one pass."""

    delete_tool_use_ids: set[str] = field(default_factory=set)
    delete_msg_indices: set[int] = field(default_factory=set)

    @property
    def has_work(self) -> bool:
        return bool(self.delete_tool_use_ids or self.delete_msg_indices)


@dataclass
class Snipper:
    """Tier 2 redundancy snipper."""

    name: str = "tier2_snip"

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
        if not self.config.snip_enabled:
            return messages, 0
        if not messages:
            return messages, 0

        plan = _SnipPlan()
        if self.config.snip_dupe_enabled:
            self._plan_rule1_dupe(messages, plan)
        if self.config.snip_fail_enabled:
            self._plan_rule2_failed(messages, plan)
        if self.config.snip_empty_enabled:
            self._plan_rule3_empty_assistant(messages, plan)

        if not plan.has_work:
            return messages, 0

        before = estimate_messages_tokens(messages)
        new_messages = self._apply_plan(messages, plan)
        after = estimate_messages_tokens(new_messages)
        freed = max(0, before - after)

        deleted_pairs = len(plan.delete_tool_use_ids)
        deleted_msgs = len(plan.delete_msg_indices)

        # observability
        turn_context.metadata["tier2_snipped_count"] = (
            int(turn_context.metadata.get("tier2_snipped_count", 0))
            + deleted_pairs + deleted_msgs
        )

        self.logger.info(
            "tier2: snipped %d tool-use/result pairs + %d empty assistants, "
            "freed ≈%d tokens",
            deleted_pairs, deleted_msgs, freed,
        )

        return new_messages, freed

    # --------------------------- Rule 1 ---------------------------

    def _plan_rule1_dupe(self, messages: list[dict], plan: _SnipPlan) -> None:
        """Mark all-but-last duplicates of (name, input) for deletion."""
        from aether.agents.core.phantom_tool import _normalize_name

        compactable = {_normalize_name(n) for n in SNIP_DUPE_RULE_TOOLS}
        # key → list of tool_use_ids in encounter order
        groups: dict[str, list[str]] = {}
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
                name = _normalize_name(block.get("name") or "")
                if name not in compactable:
                    continue
                tool_id = block.get("id")
                if not tool_id:
                    continue
                key = json.dumps(
                    [name, block.get("input") or {}],
                    sort_keys=True, ensure_ascii=False,
                )
                groups.setdefault(key, []).append(tool_id)

        for ids in groups.values():
            if len(ids) >= 2:
                plan.delete_tool_use_ids.update(ids[:-1])

    # --------------------------- Rule 2 ---------------------------

    def _plan_rule2_failed(self, messages: list[dict], plan: _SnipPlan) -> None:
        """Mark failed pairs that are superseded by a later successful pair."""
        from aether.agents.core.phantom_tool import _normalize_name

        # Build: (key) → list of (tool_use_id, is_error_of_result)
        # Walk both assistant tool_use and user tool_result to fill in.
        tool_use_meta: dict[str, dict] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_id = block.get("id")
                if not tool_id:
                    continue
                name = _normalize_name(block.get("name") or "")
                key = json.dumps(
                    [name, block.get("input") or {}],
                    sort_keys=True, ensure_ascii=False,
                )
                tool_use_meta[tool_id] = {"key": key, "is_error": None}

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id")
                if tool_id and tool_id in tool_use_meta:
                    tool_use_meta[tool_id]["is_error"] = bool(block.get("is_error", False))

        # Group by key, ordered by encounter
        groups: dict[str, list[tuple[str, bool]]] = {}
        for tid, meta in tool_use_meta.items():
            if meta["is_error"] is None:
                continue  # missing result, don't touch
            groups.setdefault(meta["key"], []).append((tid, meta["is_error"]))

        # For each group: find first success; delete all preceding failures
        for entries in groups.values():
            first_success_idx = None
            for i, (_, is_err) in enumerate(entries):
                if not is_err:
                    first_success_idx = i
                    break
            if first_success_idx is None or first_success_idx == 0:
                continue
            for tid, is_err in entries[:first_success_idx]:
                if is_err:
                    plan.delete_tool_use_ids.add(tid)

    # --------------------------- Rule 3 ---------------------------

    def _plan_rule3_empty_assistant(
        self,
        messages: list[dict],
        plan: _SnipPlan,
    ) -> None:
        for idx, msg in enumerate(messages):
            if self._is_empty_assistant(msg):
                plan.delete_msg_indices.add(idx)

    @staticmethod
    def _is_empty_assistant(msg: dict) -> bool:
        if msg.get("role") != "assistant":
            return False
        content = msg.get("content")
        if content is None:
            return True
        if isinstance(content, str):
            return not content.strip()
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    return False
                t = block.get("type")
                if t == "text" and (block.get("text") or "").strip():
                    return False
                if t == "thinking" and (block.get("thinking") or "").strip():
                    return False
                if t == "tool_use":
                    return False
            return True
        return False

    # --------------------------- Apply ---------------------------

    def _apply_plan(
        self,
        messages: list[dict],
        plan: _SnipPlan,
    ) -> list[dict]:
        new_messages: list[dict] = []
        for idx, msg in enumerate(messages):
            if idx in plan.delete_msg_indices:
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                new_messages.append(msg)
                continue
            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue
                t = block.get("type")
                if t == "tool_use" and block.get("id") in plan.delete_tool_use_ids:
                    continue
                if (
                    t == "tool_result"
                    and block.get("tool_use_id") in plan.delete_tool_use_ids
                ):
                    continue
                new_content.append(block)
            # Drop the message whole if all content removed
            if not new_content:
                continue
            # Drop assistant messages that ended up only thinking-empty
            if msg.get("role") == "assistant" and all(
                isinstance(b, dict)
                and b.get("type") == "thinking"
                and not (b.get("thinking") or "").strip()
                for b in new_content
            ):
                continue
            new_msg = dict(msg)
            new_msg["content"] = new_content
            new_messages.append(new_msg)
        return new_messages
```

### 3.4 流水线接入

`agent.py` 流水线初始化处替换 `NoOpSnipper`：

```python
from aether.services.compact.snip import Snipper

self._compaction_pipeline = CompactionPipeline(
    tiers=[
        Snipper(config=self.config, logger=self.services.logger),  # ← PR 3.6
        TimeBasedMicrocompactor(config=self.config, logger=self.services.logger),
        NoOpCollapseTier(),
        AutoCompactor(...),
    ],
    ...
)
```

### 3.5 `EngineConfig` 新字段

```python
# Sprint 3 / PR 3.6: master switch for the Tier 2 snipper. Can be
# disabled even when the rest of the pipeline is active (e.g. if a
# specific snip rule causes unexpected behaviour and we want to ship
# a hotfix without reverting the whole pipeline).
snip_enabled: bool = True

# Sprint 3 / PR 3.6: per-rule switches. All on by default; flip
# individual rules off if one misbehaves on real workloads.
snip_dupe_enabled: bool = True   # Rule 1: keep only last of (name, input) duplicates
snip_fail_enabled: bool = True   # Rule 2: drop failed pairs superseded by success
snip_empty_enabled: bool = True  # Rule 3: drop assistant messages with only empty text/thinking
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/services/compact/snip.py` | **重写** | 整个 `Snipper` + `_SnipPlan` + 三个 `_plan_*` + `_apply_plan` + `SNIP_DUPE_RULE_TOOLS` | ~280（净增 ~270） |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 流水线初始化处替换 NoOp | ~3 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 4 个 `snip_*` 字段 | ~25 |
| `backend/harness/aether/tests/test_snipper.py` | **新文件** | 见 § 五 | ~400 |

## 五、测试用例（详细）

### 5.1 `test_snipper.py`

**测试组 A：禁用 / 早退**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | `compression_enabled=False` | maybe_run 返回原消息；不规划 |
| **T-A2** | `snip_enabled=False` | 同上 |
| **T-A3** | 空 messages | 返回 ([], 0) |
| **T-A4** | messages 全是 user 文本（无 tool） | 不修改 |

**测试组 B：Rule 1 — DUPE**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 3 次 read_file path=/x.txt（相同 input） | 删除前 2 次的 tool_use + 对应 tool_result，保留第 3 次 |
| **T-B2** | 2 次 read_file path=/x.txt + 2 次 read_file path=/y.txt | 各删 1 次（保留各自最后一次） |
| **T-B3** | 2 次 read_file path=/x.txt（input 字段顺序不同的 dict）| key 应稳定（json.dumps sort_keys），仍识别为重复 |
| **T-B4** | 2 次 shell command="ls" | **不**删（shell 不在 SNIP_DUPE_RULE_TOOLS） |
| **T-B5** | 2 次 write_file（同 path 同内容） | 不删（write_file 不在白名单） |
| **T-B6** | grep 同 pattern 但不同 path → 不视为重复 | 不删 |
| **T-B7** | `snip_dupe_enabled=False` | 不应用 Rule 1，所有调用保留 |

**测试组 C：Rule 2 — FAIL**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | read_file path=/x failed → read_file path=/x success | 删除第一对（A + A's result） |
| **T-C2** | read_file failed → 不再调（永远没 success） | 不删（保留失败信号） |
| **T-C3** | 3 次 read_file: error → error → success | 前两个 error 都删 |
| **T-C4** | shell cmd="x" failed → shell cmd="x" success | 也删（Rule 2 不限工具） |
| **T-C5** | tool_use 没有对应 tool_result（中途打断） | 不动该 tool_use（meta 不完整） |
| **T-C6** | `snip_fail_enabled=False` | 不应用 Rule 2 |

**测试组 D：Rule 3 — EMPTY ASSISTANT**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | assistant 消息 content="" | 删 |
| **T-D2** | assistant 消息 content="  \n\t" | 删 |
| **T-D3** | assistant 消息 content=None | 删 |
| **T-D4** | assistant content=[{"type":"text","text":""}, {"type":"thinking","thinking":""}] | 删（全空） |
| **T-D5** | assistant content=[{"type":"thinking","thinking":"some thought"}] | **不**删（有真实 thinking） |
| **T-D6** | assistant content=[{"type":"text","text":"hi"}] | **不**删 |
| **T-D7** | assistant content=[{"type":"tool_use","name":"x","id":"a","input":{}}] | **不**删（有 tool_use） |
| **T-D8** | assistant content=[{"type":"unknown","data":"..."}]（不可识别） | 不删（保守） |
| **T-D9** | `snip_empty_enabled=False` | 不应用 Rule 3 |

**测试组 E：配对完整性**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | Rule 1 触发删某 assistant 内的某 tool_use | 同一 tool_use_id 的 tool_result 也被删（在另一条 user 消息里） |
| **T-E2** | assistant 消息含 1 个 text + 2 个 tool_use；Rule 1 删 1 个 tool_use | text + 剩 1 个 tool_use 保留；该 user tool_result 也部分保留 |
| **T-E3** | assistant 消息原本只有 2 个 tool_use；Rule 1 删两个 | 整个 assistant 消息被删（content 空） |
| **T-E4** | user 消息原本只有 2 个 tool_result，对应都被删 | 整个 user 消息被删 |
| **T-E5** | tool_use 在 assistant，tool_result 在距离 5 条之后的 user 消息 | 仍正确配对删除 |

**测试组 F：组合场景**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | Rule 1 + Rule 2 同时触发同一对 tool | 不重复删（set 自然去重） |
| **T-F2** | Rule 3 删了一个 assistant 后，下一条 user 还在 | 不影响（Rule 3 不参与配对计算） |
| **T-F3** | 巨大消息列表（100 条，10 组重复 + 5 个失败 + 3 个空）| 一次跑完，新消息列表正确 |

**测试组 G：observability**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 触发 Rule 1 删 3 对 + Rule 3 删 1 个 | `metadata["tier2_snipped_count"] == 4` |
| **T-G2** | 第二次跑触发 Rule 2 删 1 对 | tier2_snipped_count 累加到 5 |
| **T-G3** | tokens_freed > 0 | 与 estimator 计算的 before-after 一致 |
| **T-G4** | 日志 | 含 `snipped N pairs + M empty assistants, freed ≈X tokens` |

**测试组 H：与 Tier 5 协同（集成）**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | Snipper 释放 50% token 达到阈值 | 不再跑 Tier 3-5 |
| **T-H2** | Snipper 释放 5% 未达阈值 | 继续跑 Tier 3 |
| **T-H3** | Snipper + Microcompact 配合 | 都能正常跑，互不干扰 |

**测试组 I：边界 / 容错**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | tool_use 没 id 字段 | 跳过该 block，不抛 |
| **T-I2** | content 不是 list（是 string） | 跳过该消息 |
| **T-I3** | tool_use 的 input 是非 JSON-friendly 类型（set / 自定义 class） | json.dumps 失败时 fallback 到 str() 比较 |

## 六、验收门

- [ ] 所有新测试 green（约 32 case）
- [ ] PR 3.4/3.5 既有测试不回归
- [ ] 手工：构造 5 次重复 read_file 的 turn → 看 `tier2: snipped 4 tool-use/result pairs ...` 日志
- [ ] 手工：构造一个失败 + 重试成功的 read_file → 看到失败对被删
- [ ] `result.metadata["compaction"]["tier2_snipped_count"]` 与实际删除数一致

## 七、回滚开关

- `snip_enabled=False`：整层关闭
- `snip_dupe_enabled / snip_fail_enabled / snip_empty_enabled`：单独关闭某规则
- 完全 revert PR：恢复 `services/compact/snip.py` 为 NoOp

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 重写 `services/compact/snip.py`（Rule 1 + apply） | 2h | DUPE 路径完整 |
| 2. Rule 2 (FAIL) 实现 | 1.5h | 失败对扫描 |
| 3. Rule 3 (EMPTY) 实现 | 1h | 空消息检测 |
| 4. 配对完整性的 `_apply_plan` 完善 | 1.5h | 双向删除 + 空消息清理 |
| 5. `tests/test_snipper.py` (T-A 至 T-D) | 3h | 单规则测试约 22 case |
| 6. 配对 + 组合测试 (T-E + T-F) | 2h | 8 case |
| 7. observability + 集成 (T-G + T-H + T-I) | 2h | 10 case |
| 8. agent.py 接入 + config | 30min | 替换 NoOp |
| 9. 既有回归 + 手工 | 1h | unittest discover |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| Rule 1 把"重要的中间状态读"也删了（用户期望保留） | 中 | 只对 read_file/list_dir/glob/grep 应用；shell/write_file 不参与；可在 input 含特殊参数时白名单豁免（后续优化） |
| Rule 2 把"失败本身就是有用信号"的对删了 | 低 | 只在**后续有同 key 成功**时才删；纯失败保留 |
| 配对删除遗漏导致 provider 报错 `tool_use_id without tool_result` | 中 | 测试组 E 严格覆盖 5 个配对场景；任何 tool_use 删除时强制 set 同步 |
| input dict 顺序导致 key 不稳定 | 低 | json.dumps `sort_keys=True` |
| 三规则同时触发产生意料外组合 | 中 | T-F1/F2/F3 组合测试；先收集 plan 再统一 apply 避免中间状态 |
| Snipper 释放 token 太多导致 Tier 5 永远不触发，模型失去早期上下文 | 低 | snipped 内容本就是冗余/失败/空，删除后语义无损；`tokens_freed` 准确反映让 autocompact 阈值正确 |

## 十、与后续 PR 的接合

- **PR 3.7** Context Collapse 在 Snipper 之后跑。Snipper 已经把冗余删完，Collapse 只要折叠"真正的有效上下文"。
- **CLI** 可以从 `metadata["compaction"]["tier2_snipped_count"]` 显示 `↻ snipped: 4 pairs`。
- **未来的 SnipperRule 扩展**（不在 Sprint 3 范围）：
  - Rule 4：连续多次失败的相同失败模式（已知 broken 的 path），删除最早 N-1 次。
  - Rule 5：被后续 write_file 完全覆盖的 read_file（读了文件后又写了，旧 read 内容已陈旧）。
