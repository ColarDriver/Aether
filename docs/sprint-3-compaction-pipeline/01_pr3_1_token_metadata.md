# PR 3.1 — Token / Result 基础（P1-4 + P1-11）

> **角色**：地基 PR。Sprint 3 的所有压缩触发判断都依赖 token 数；
> 所有外部消费者（CLI footer、SDK、计费）都依赖标准化的 metadata 字段。
> 本 PR 不引入任何流水线行为，纯加字段，零风险。

## 一、目标

1. 提供 `CanonicalUsage` 数据结构，统一 anthropic / openai / codex 三种 usage shape。
2. 在 `AgentEngine.run_loop` 内累计每次 LLM 调用的 usage 到 `context.metadata["usage_accumulator"]`。
3. 把累计值写入 `EngineResult.metadata["usage"]`。
4. 标准化 `EngineResult.metadata` 的 key 命名空间（约定 schema），杜绝后续 PR 继续叠 ad-hoc。

## 二、现状分析

### 2.1 当前 token 处理的问题

[`backend/harness/aether/agents/middlewares/token_usage.py`](../../backend/harness/aether/agents/middlewares/token_usage.py)
现在只 **log** 不 **累计**也不**对外暴露**：

```python
def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
    usage = self._extract_usage(response.metadata)
    if usage:
        self.logger.info(
            "LLM token usage: input=%s output=%s total=%s",
            usage.get("input_tokens", usage.get("prompt_tokens", "?")),
            ...
        )
    return response
```

问题：
- **三家 provider 字段名不同**：OpenAI 用 `prompt_tokens / completion_tokens`，
  Anthropic 用 `input_tokens / output_tokens`，Codex 又有 `reasoning_tokens`。
- **没有累计**：每次只 log 当次，turn 内 5 次 LLM 调用就丢失了 4 次的数据。
- **没暴露**：CLI footer / SDK 消费者拿不到。

### 2.2 当前 metadata 的问题

`EngineResult.metadata` 现在散落了：

| key | 来源 | 说明 |
|---|---|---|
| `phantom_synth_count` | Sprint 1.5 | phantom 工具合成次数 |
| `phantom_synth_notes` | Sprint 1.5 | phantom 合成详情 |
| `tool_names_repaired` | Sprint 2 | 工具名修复次数 |
| `tool_calls_deduped` | Sprint 2 | 工具调用去重次数 |
| `tool_calls_capped` | Sprint 2 | 委派调用上限次数 |
| `recovery_decisions` | Sprint 2 | 恢复决策列表 |
| `recovery_terminal_exit_reason` | Sprint 2 | 终态原因（用于映射 ExitReason）|
| `fallback_activations_this_turn` | Sprint 2 | 本轮 fallback 激活次数 |
| `partial` | Sprint 1 | 是否部分输出 |
| `_active_tool_call` | runtime | 当前激活的 tool call（前缀 `_` 表内部） |
| ... | | |

问题：
- **没分组**：所有 key 都在顶层，CLI / SDK 不知道哪些是稳定字段、哪些是内部状态。
- **没文档**：没有任何文档约定哪些 key 是公开 API 的一部分。
- **再加新 key 会乱**：Sprint 3-7 还有 token / budget / reasoning / steer 等大量字段要加。

## 三、设计

### 3.1 `CanonicalUsage` 数据结构

新文件 [`backend/harness/aether/runtime/usage.py`](../../backend/harness/aether/runtime/usage.py)：

```python
"""Canonical token usage representation, shared across providers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class CanonicalUsage:
    """Provider-neutral token usage.

    All fields default to 0 so partial provider responses don't crash
    downstream code. The semantic meaning of each field follows
    Anthropic + OpenAI conventions:

    * input_tokens       — non-cached prompt tokens billed at full rate
    * output_tokens      — completion tokens (always billed)
    * cache_read_tokens  — prompt tokens served from cache (cheap rate)
    * cache_write_tokens — prompt tokens being inserted into cache (often premium)
    * reasoning_tokens   — model-internal reasoning tokens (Codex / o1-style)

    Aliases (preserved for back-compat with legacy logging):
    * prompt_tokens      = input_tokens + cache_read_tokens + cache_write_tokens
    * completion_tokens  = output_tokens
    * total_tokens       = prompt_tokens + completion_tokens + reasoning_tokens
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def add(self, other: "CanonicalUsage") -> "CanonicalUsage":
        """Return a new CanonicalUsage with field-wise sum."""
        return CanonicalUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly dict including derived totals (for serialization)."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
```

### 3.2 `normalize_usage()` 三种 provider 转换规则

```python
def normalize_usage(
    raw: Any,
    *,
    provider: str,
    api_mode: str = "chat",
) -> CanonicalUsage:
    """Convert a provider-native usage dict to CanonicalUsage.

    Tolerant: returns CanonicalUsage() (all zeros) when raw is None / missing /
    malformed — never raises. Logging is the caller's responsibility.

    provider/api_mode currently accepted:
      * provider="openai", api_mode in {"chat", "responses"}
      * provider="anthropic", api_mode in {"messages", "text"}
      * provider="codex", api_mode in {"responses"}
      * provider="kimi" / "moonshot" → openai-compatible
      * provider="zhipu" / "glm" → openai-compatible
      * provider="*" or unknown → openai-compatible best-effort
    """
    if not isinstance(raw, dict):
        return CanonicalUsage()

    if provider == "anthropic" and api_mode == "messages":
        return _normalize_anthropic_messages(raw)
    if provider == "codex":
        return _normalize_codex(raw)
    return _normalize_openai_compatible(raw)


def _normalize_openai_compatible(raw: dict) -> CanonicalUsage:
    """OpenAI Chat Completions / Responses API, plus all openai-compatible.

    Schema (post-2024 with cache + reasoning details):
        {
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "prompt_tokens_details": {
            "cached_tokens": int,        # ← cache_read
            "audio_tokens": int           # ignored
          },
          "completion_tokens_details": {
            "reasoning_tokens": int,     # ← reasoning (o1)
            "audio_tokens": int           # ignored
          }
        }
    """
    cache_read = 0
    details = raw.get("prompt_tokens_details")
    if isinstance(details, dict):
        cache_read = int(details.get("cached_tokens", 0) or 0)

    reasoning = 0
    cd = raw.get("completion_tokens_details")
    if isinstance(cd, dict):
        reasoning = int(cd.get("reasoning_tokens", 0) or 0)

    prompt = int(raw.get("prompt_tokens", 0) or 0)
    completion = int(raw.get("completion_tokens", 0) or 0)
    # Subtract cache_read because OpenAI's prompt_tokens already includes
    # cached tokens — splitting here gives accurate billing math.
    input_tokens = max(0, prompt - cache_read)

    return CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=completion,
        cache_read_tokens=cache_read,
        cache_write_tokens=0,  # OpenAI doesn't expose write separately
        reasoning_tokens=reasoning,
    )


def _normalize_anthropic_messages(raw: dict) -> CanonicalUsage:
    """Anthropic Messages API.

    Schema:
        {
          "input_tokens": int,
          "output_tokens": int,
          "cache_read_input_tokens": int,
          "cache_creation_input_tokens": int   # ← cache_write
        }
    """
    return CanonicalUsage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
        reasoning_tokens=0,  # Anthropic returns thinking inline, not in usage
    )


def _normalize_codex(raw: dict) -> CanonicalUsage:
    """Codex / OpenAI Responses API.

    Schema (Responses API):
        {
          "input_tokens": int,
          "output_tokens": int,
          "input_tokens_details": { "cached_tokens": int },
          "output_tokens_details": { "reasoning_tokens": int }
        }
    """
    cache_read = 0
    itd = raw.get("input_tokens_details")
    if isinstance(itd, dict):
        cache_read = int(itd.get("cached_tokens", 0) or 0)

    reasoning = 0
    otd = raw.get("output_tokens_details")
    if isinstance(otd, dict):
        reasoning = int(otd.get("reasoning_tokens", 0) or 0)

    input_t = int(raw.get("input_tokens", 0) or 0)
    output_t = int(raw.get("output_tokens", 0) or 0)
    # Codex Responses API: input_tokens already excludes cached_tokens (verified
    # via fixture). Don't subtract.

    return CanonicalUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        cache_read_tokens=cache_read,
        cache_write_tokens=0,
        reasoning_tokens=reasoning,
    )


def sum_usages(usages: Iterable[CanonicalUsage]) -> CanonicalUsage:
    """Convenience: fold a sequence of CanonicalUsage into one."""
    total = CanonicalUsage()
    for u in usages:
        total = total.add(u)
    return total
```

### 3.3 `EngineResult.metadata` 标准 schema

约定以下顶层 key 是**稳定 API**，下游可依赖：

```python
{
    # Stable v1 schema (PR 3.1 引入)
    "usage": CanonicalUsage,                # 累计 usage
    "iteration_budget": {                    # PR 3.2 填充
        "used": int,
        "max": int,
        "remaining": int,
        "grace_consumed": bool,
    },
    "exit": {                                # PR 3.1 引入
        "reason": str,                       # = ExitReason.value
        "last_msg_role": str,                # "assistant"/"tool"/"user"
        "stuck_after_tool": bool,            # 上一条是 tool 但没续推理
    },
    "reasoning": {                           # PR 3.1 引入（Sprint 5 完善）
        "last_reasoning": str | None,
    },
    "compaction": {                          # PR 3.4+ 填充
        "tier1_spilled_count": int,
        "tier2_snipped_count": int,
        "tier3_cleared_count": int,
        "tier4_collapse_segments": int,
        "tier5_summaries_generated": int,
    },
    "pending_steer": str | None,             # 留给 Sprint 6
    "interrupt_message": str | None,
    "partial": bool,                         # 已有，迁入 schema

    # Sprint 1.5 / Sprint 2 既有 key（保持原位，作为非稳定 ad-hoc 字段）
    "phantom_synth_count": int,
    "tool_names_repaired": int,
    "tool_calls_deduped": int,
    "tool_calls_capped": int,
    "recovery_decisions": list,
    ...
}
```

**约定**：
- 顶层 `usage / iteration_budget / exit / reasoning / compaction / pending_steer / interrupt_message / partial` 是**稳定 v1 schema**，PR 3.1 引入后不再删除/改名（只允许追加）。
- 其他既有 key 保留原位，标记为"非稳定 ad-hoc"。CLI / SDK 消费者**只应**依赖稳定字段。
- 前缀 `_` 的 key（如 `_active_tool_call`）保持"内部状态"语义，外部禁止依赖。

### 3.4 累计与暴露

`AgentEngine.run_loop`（约 `agents/core/agent.py:265-285` 处的 LLM_CALL 段）：

```python
state_machine.transition(LoopState.LLM_CALL)
response = ...  # 原有调用

# === PR 3.1 新增 ===
from aether.runtime.usage import normalize_usage, CanonicalUsage

raw_usage = (response.metadata or {}).get("usage") if response else None
this_call_usage = normalize_usage(
    raw_usage,
    provider=getattr(self.services.provider, "provider_name", "openai"),
    api_mode=getattr(self.services.provider, "api_mode", "chat"),
)
acc = context.metadata.setdefault("usage_accumulator", CanonicalUsage())
context.metadata["usage_accumulator"] = acc.add(this_call_usage)
context.metadata.setdefault("api_calls", 0)
context.metadata["api_calls"] += 1
```

`_build_result`（约 `agents/core/agent.py:1900-2000` 段）：

```python
# === PR 3.1 新增 ===
acc = context.metadata.get("usage_accumulator", CanonicalUsage())
metadata["usage"] = acc.to_dict()  # 序列化为 dict 而非 dataclass，保证 JSON-friendly

# Exit 元信息
last_msg = messages[-1] if messages else None
metadata["exit"] = {
    "reason": exit_reason.value,
    "last_msg_role": last_msg.get("role", "unknown") if last_msg else "unknown",
    "stuck_after_tool": (
        last_msg is not None
        and last_msg.get("role") == "tool"
        and exit_reason in {ExitReason.MAX_ITERATIONS, ExitReason.EMPTY_RESPONSE}
    ),
}

# Reasoning（Sprint 5 完善，这里只占位）
metadata["reasoning"] = {
    "last_reasoning": context.metadata.get("last_reasoning_text"),
}

# Compaction 计数器（默认全 0；PR 3.4+ 填充实际值）
metadata["compaction"] = {
    "tier1_spilled_count": context.metadata.get("tier1_spilled_count", 0),
    "tier2_snipped_count": context.metadata.get("tier2_snipped_count", 0),
    "tier3_cleared_count": context.metadata.get("tier3_cleared_count", 0),
    "tier4_collapse_segments": context.metadata.get("tier4_collapse_segments", 0),
    "tier5_summaries_generated": context.metadata.get("tier5_summaries_generated", 0),
}
```

### 3.5 Provider 侧改动

需要确认三家 provider 的 `_parse_response` 都把 raw usage 透出到 `NormalizedResponse.metadata["usage"]`：

| provider 文件 | 检查点 |
|---|---|
| [`models/provider/openai_compatible.py`](../../backend/harness/aether/models/provider/openai_compatible.py) | `_parse_chat_completion_response` 已经透出 `usage`（已验证） |
| [`models/provider/claude.py`](../../backend/harness/aether/models/provider/claude.py) | 需要确认 `usage` 字段名（Anthropic 用 `usage`，应该已经透出） |
| [`models/provider/codex.py`](../../backend/harness/aether/models/provider/codex.py) | 需要确认 Codex Responses API 的 usage 透出 |

**实施步骤**：先读这三个文件确认现状，缺失的补 1-2 行 `metadata["usage"] = raw_response.get("usage")`。

### 3.6 复用 `TokenUsageMiddleware`

[`agents/middlewares/token_usage.py`](../../backend/harness/aether/agents/middlewares/token_usage.py)
原本自己解析 raw usage dict。改为复用 `normalize_usage`：

```python
from aether.runtime.usage import normalize_usage

class TokenUsageMiddleware(RuntimeMiddlewareBase):
    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        usage = normalize_usage(
            (response.metadata or {}).get("usage"),
            provider=context.metadata.get("active_provider_name") or "openai",
        )
        if usage.total_tokens > 0:
            self.logger.info(
                "LLM token usage: input=%d output=%d cache_read=%d reasoning=%d total=%d",
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.reasoning_tokens,
                usage.total_tokens,
            )
        return response
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/runtime/usage.py` | **新文件** | 完整 `CanonicalUsage` + `normalize_usage` + 三个 `_normalize_*` + `sum_usages` | ~200 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | LLM_CALL 段累计 usage；`_build_result` 写入标准 metadata schema | ~40 |
| `backend/harness/aether/agents/middlewares/token_usage.py` | 修改 | 复用 `normalize_usage`，删除内部解析逻辑 | -20 / +15 |
| `backend/harness/aether/runtime/contracts.py` | 修改 | 在 `EngineResult` 类的 docstring 加 metadata schema 注释（非代码改动） | +30 注释 |
| `backend/harness/aether/models/provider/openai_compatible.py` | 检查 | 确认 `usage` 已透出；缺则补 | 0-3 |
| `backend/harness/aether/models/provider/claude.py` | 检查 | 同上 | 0-3 |
| `backend/harness/aether/models/provider/codex.py` | 检查 | 同上 | 0-3 |
| `backend/harness/aether/tests/test_usage_normalize.py` | **新文件** | 见 § 五.1 | ~280 |
| `backend/harness/aether/tests/test_engine_result_metadata.py` | **新文件** | 见 § 五.2 | ~180 |

## 五、测试用例（详细）

### 5.1 `test_usage_normalize.py`

**测试组 A：OpenAI Chat Completions 兼容**

| ID | 输入 | 期望输出 | 备注 |
|---|---|---|---|
| **T-A1** | `{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}` | `input=100, output=50, cache_read=0, reasoning=0` | 最基础形态 |
| **T-A2** | `{"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 80}}` | `input=20, output=50, cache_read=80, reasoning=0` | cache_read 从 prompt_tokens 减掉 |
| **T-A3** | `{"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 30}}` | `input=100, output=50, reasoning=30` | o1 reasoning |
| **T-A4** | `{}` | 全 0 | 容错：空 usage |
| **T-A5** | `None` | 全 0 | 容错：None |
| **T-A6** | `"not a dict"` | 全 0 | 容错：错误类型 |
| **T-A7** | `{"prompt_tokens": "100"}` | `input=100, ...` | 容错：字符串数字 |
| **T-A8** | `{"prompt_tokens_details": {"cached_tokens": None}}` | 全 0 | 容错：None 值 |

**测试组 B：Anthropic Messages**

| ID | 输入 | 期望输出 |
|---|---|---|
| **T-B1** | `{"input_tokens": 100, "output_tokens": 50}` | `input=100, output=50, cache_read=0, cache_write=0` |
| **T-B2** | `{"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 20}` | `input=100, output=50, cache_read=80, cache_write=20` |
| **T-B3** | provider 为 anthropic 但 schema 是 OpenAI 形态 → 应 fallback 到 openai_compatible 路径 | input/output 不为 0 | 验证 api_mode 不匹配时的行为 |

**测试组 C：Codex Responses**

| ID | 输入 | 期望输出 |
|---|---|---|
| **T-C1** | `{"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 30}}` | `input=100, output=50, reasoning=30` |
| **T-C2** | `{"input_tokens": 100, "input_tokens_details": {"cached_tokens": 60}, "output_tokens": 50}` | `input=100, cache_read=60, output=50` — 注意 input_tokens **不**减 cache_read（Codex schema 已分离）|

**测试组 D：CanonicalUsage 数学**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | `CanonicalUsage(input=10, output=20, cache_read=5, reasoning=3)` | `prompt_tokens == 15`, `completion_tokens == 20`, `total_tokens == 38` |
| **T-D2** | `add(CanonicalUsage(input=10), CanonicalUsage(input=20, output=5))` | 字段相加正确 |
| **T-D3** | `sum_usages([])` | 全 0 |
| **T-D4** | `sum_usages([u1, u2, u3])` | 等价于 `u1.add(u2).add(u3)` |
| **T-D5** | `CanonicalUsage()` 实例不可变（frozen） | `setattr(u, 'input_tokens', 100)` 抛 `FrozenInstanceError` |
| **T-D6** | `to_dict()` 返回的字段含 `prompt_tokens / completion_tokens / total_tokens` 派生值 | 验证 dict 结构 |

**测试组 E：未知 provider 兼容**

| ID | 输入 | 期望 |
|---|---|---|
| **T-E1** | `provider="kimi"` + OpenAI 形态 usage | 同 OpenAI 解析路径 |
| **T-E2** | `provider="zhipu"` + OpenAI 形态 usage | 同上 |
| **T-E3** | `provider="unknown_xyz"` + OpenAI 形态 usage | 同上（best-effort） |
| **T-E4** | `provider="anthropic"` + `api_mode="text"` | 走 openai_compatible 路径 |

### 5.2 `test_engine_result_metadata.py`

**测试组 F：累计行为**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | mock provider 单次调用返回 usage=`{prompt_tokens: 100, completion_tokens: 50}` | `result.metadata["usage"]["total_tokens"] == 150`，`result.metadata["api_calls"] == 1` |
| **T-F2** | mock provider 5 次连续调用，每次 usage=100/50 | `result.metadata["usage"]["total_tokens"] == 750`，`api_calls == 5` |
| **T-F3** | mock provider 一次成功 + 一次 ProviderInvocationError + 一次成功 | 失败那次不累计；total = 两次成功之和 |
| **T-F4** | provider 不返回 usage 字段 | `result.metadata["usage"]["total_tokens"] == 0`，不抛错 |

**测试组 G：metadata schema**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 任意一次成功的 turn | `result.metadata` 包含全部稳定 key：`usage / iteration_budget / exit / reasoning / compaction / partial` |
| **T-G2** | exit_reason = MAX_ITERATIONS | `result.metadata["exit"]["reason"] == "MAX_ITERATIONS"` |
| **T-G3** | last_msg = `{"role": "tool", ...}` 且 exit=MAX_ITERATIONS | `exit["stuck_after_tool"] == True` |
| **T-G4** | last_msg = `{"role": "assistant", ...}` | `exit["stuck_after_tool"] == False` |
| **T-G5** | `compaction` 在 PR 3.1 阶段所有计数器 == 0 | 字典存在但全 0 |
| **T-G6** | `result.metadata["usage"]` 是 dict 而非 CanonicalUsage 实例 | 保证 JSON-serializable（用 `json.dumps(result.metadata)` 不抛） |

**测试组 H：跨 session 隔离**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | 两个 session 并发 run_loop，各有 3 次 LLM 调用 | session A 的 `result.metadata["usage"]` 不含 session B 的 token；反之亦然 |

**测试组 I：与 Sprint 2 字段共存**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | 一次有 phantom synthesis 的 turn | `metadata["phantom_synth_count"] > 0` 且 `metadata["usage"]` 也存在 |
| **T-I2** | 一次有 fallback 切换的 turn | `metadata["fallback_activations_this_turn"] > 0` 且新字段全部正常 |

## 六、验收门

实施完成后必须全部通过：

- [ ] `tests/test_usage_normalize.py` 全部 case green
- [ ] `tests/test_engine_result_metadata.py` 全部 case green
- [ ] 既有测试套件全绿（无回归）
- [ ] 真实跑一次 GPT-4 调用，看 CLI 日志能输出 cache_read/reasoning 字段
- [ ] 真实跑一次 Claude 调用，看 CLI 日志能区分 cache_read 和 cache_write
- [ ] `result.metadata` 用 `json.dumps` 能成功序列化（不能含非 JSON 类型）

## 七、回滚开关

无开关（纯加字段）。如发现问题：

1. revert 该 PR 即可。
2. CLI / SDK 消费者如果已经依赖 `result.metadata["usage"]`，需要同步处理。
   建议在 PR 落地后再合 PR 3.4 之前 1 周内观察。

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 新文件 `runtime/usage.py` | 2h | `CanonicalUsage` + `normalize_usage` 三种 provider |
| 2. `tests/test_usage_normalize.py` | 2h | 测试组 A-E（约 30 个 case） |
| 3. 检查 3 个 provider 的 `_parse_response` 是否透出 usage | 1h | 确认或补 1-2 行 |
| 4. `agents/core/agent.py` 累计逻辑 | 1.5h | LLM_CALL 段 + `_build_result` |
| 5. `agents/middlewares/token_usage.py` 复用 normalize | 30min | 删除内部解析 |
| 6. `runtime/contracts.py` 加 schema docstring | 30min | 文档化 |
| 7. `tests/test_engine_result_metadata.py` | 3h | 测试组 F-I |
| 8. 真实 provider 调用 sanity check | 1h | 跑 GPT-4 + Claude 各一次 |
| 9. 既有测试回归 | 30min | `unittest discover` |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 某家 provider 的 usage shape 在文档之外 | 低 | `normalize_usage` 全部 try/except 容错，最差返回全 0 |
| `CanonicalUsage(frozen=True)` 与某些序列化库不兼容 | 低 | `to_dict()` 提供逃生口；`metadata["usage"]` 永远是 dict |
| 测试组 F 的 mock provider 不发 usage 导致测试假绿 | 中 | 至少一个真实 provider sanity check（步骤 8） |

## 十、与后续 PR 的接合

- **PR 3.2** 的 `IterationBudget` 也写入 `metadata["iteration_budget"]`，schema 在本 PR 已约定。
- **PR 3.4** 的流水线触发判断**直接消费**累计 usage：
  ```python
  current_tokens = context.metadata["usage_accumulator"].prompt_tokens
  if current_tokens >= self.config.compression_pre_llm_pct * model_window:
      pipeline.run(...)
  ```
- **PR 3.4-3.7** 各自填 `compaction.tierN_*` 计数器，schema 已预留。
