# PR 3.2 — IterationBudget + max_iterations Summary 兜底（P1-2 + P1-3）

> **角色**：地基 PR。把"硬上限 break"换成"结构化预算 + 用尽时给 summary"。
> 与 PR 3.1 平行落地（不互相阻塞），但 PR 3.4 的 LLM Fork 需要复用本 PR 的
> `grace_call` 机制和 cheap_tool_names 配置。

## 一、目标

1. 引入 `IterationBudget` 数据结构，挂在 `context.metadata["iteration_budget"]`（多 session 安全）。
2. cheap tool（todo / memory / skill_manage / session_search）回合 `refund()`，不扣预算。
3. 用尽预算后通过 `grace_call()` 多发一次**不带工具**的请求生成 summary，写入 `final_response`。
4. 替换 `agents/core/agent.py` 里 9 处散落的 `iterations >= self.config.max_iterations` 判断。

## 二、现状分析

### 2.1 散落的 max_iterations 判断

[`backend/harness/aether/agents/core/agent.py`](../../backend/harness/aether/agents/core/agent.py)
里有 9 处 `iterations >= self.config.max_iterations`：

| 位置（约） | 上下文 |
|---|---|
| 231 | 主循环 while 条件 |
| 356, 387, 418, 517, 592, 601 | 各种异常分支后的"是否还能继续"判断 |
| 644 | finalize 段的兜底设置 ExitReason |
| 851 | docstring 引用 |

问题：
- **散落**导致逻辑分裂——任何 budget 相关调整（如"剩余 ≤ 2 时优先收尾"提示）要改 9 处。
- **不区分"重活"vs"记账"**——cheap tool（如 update_todo）也算一轮预算。
- **耗尽时直接 break**——`final_response` 是空字符串，用户看到 `done · 0.0s` 没任何说明。

### 2.2 当前用户感知

- CLI 模式：max_iterations 触发后 footer 显示 `done` 但没内容，用户滚 scrollback 才能拼凑发生了什么。
- 自动化模式（`-p` / SDK）：脚本拿到的 `final_response = ""`，下游写日志直接是空白。
- 没有任何信号告诉调用方"我做完了 38/52 个文件，剩 14 个"。

### 2.3 cheap_tool_names 的设计前提

不同工具对预算的"占用感"不同：

| 工具类别 | 例子 | 占用预算？ | 理由 |
|---|---|---|---|
| 重活 | shell, write_file, web_fetch | 是 | 真实 IO + token |
| 中间产物记账 | update_todo, memory.write | 否 | 几乎无 token、无 IO 副作用 |
| 检索 | grep, glob, read_file | 是 | 占 token，影响后续 prompt |
| 会话搜索 | session_search | 否 | 内部状态查询，不进对话 |

`cheap_tool_names` 配置一组"调用后 refund 预算"的工具名，让模型有更多额度做实际工作。

## 三、设计

### 3.1 `IterationBudget` 数据结构

新文件 [`backend/harness/aether/runtime/iteration_budget.py`](../../backend/harness/aether/runtime/iteration_budget.py)：

```python
"""Per-turn iteration budget with cheap-tool refund + grace call."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class IterationBudget:
    """Track remaining iterations for one turn.

    Lifecycle:
      * Created at run_loop entry with max_total = EngineConfig.max_iterations
      * consume() at top of each iteration; returns False when exhausted
      * refund() after a cheap-tool-only iteration (cancels the consume)
      * grace_call() once after exhaustion, for the summary fallback
      * to_dict() for serialization to EngineResult.metadata

    Counters:
      * used: total consume() calls minus refund() calls
      * grace_consumed: True once grace_call() has been used
      * refund_count / consume_count: for observability
    """

    max_total: int
    used: int = 0
    grace_consumed: bool = False
    consume_count: int = 0
    refund_count: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_total - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_total

    def consume(self) -> bool:
        """Try to consume one iteration. Returns False if exhausted."""
        if self.exhausted:
            return False
        self.used += 1
        self.consume_count += 1
        return True

    def refund(self) -> None:
        """Cancel the most recent consume(). No-op if used == 0."""
        if self.used > 0:
            self.used -= 1
            self.refund_count += 1

    def grace_call(self) -> bool:
        """Allow one extra iteration after exhaustion (summary fallback).

        Returns True if grace was granted, False if already used.
        Idempotent: only grants once per budget.
        """
        if self.grace_consumed:
            return False
        self.grace_consumed = True
        # Grace doesn't consume from max_total — it's a one-time bonus
        return True

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "used": self.used,
            "max": self.max_total,
            "remaining": self.remaining,
            "grace_consumed": self.grace_consumed,
            "consume_count": self.consume_count,
            "refund_count": self.refund_count,
        }
```

### 3.2 cheap tool 判定

`AgentEngine.run_loop` 在每次 tool dispatch 后判断：

```python
# In tool execution loop, after all tools dispatched
all_calls_were_cheap = all(
    self._is_cheap_tool(call.name) for call in response.tool_calls
)
if all_calls_were_cheap and budget.consume_count > 0:
    budget.refund()
    context.metadata["iteration_budget"] = budget.to_dict()
```

辅助函数：

```python
def _is_cheap_tool(self, tool_name: str) -> bool:
    """Whether tool_name is in the cheap-tool refund list.

    Match uses the same normalisation as tool_hardening (case-fold + dash↔underscore +
    namespace strip), so 'mcp__foo__update_todo' matches 'update_todo'.
    """
    from aether.agents.core.phantom_tool import _normalize_name

    cheap_set = {_normalize_name(n) for n in self.config.cheap_tool_names}
    return _normalize_name(tool_name) in cheap_set
```

### 3.3 `_handle_max_iterations` summary 兜底

新方法（约 1500-1600 行处）：

```python
def _handle_max_iterations(
    self,
    request: EngineRequest,
    messages: list[dict],
    context: TurnContext,
) -> str | None:
    """Generate a summary when iteration budget is exhausted.

    Steps:
      1. Check EngineConfig.summary_on_budget_exhausted; bail if False.
      2. Use IterationBudget.grace_call(); bail if already consumed.
      3. Build a one-shot prompt: copy messages, append user [System: …]
         instruction, drop tools=[] from the API call.
      4. Issue provider.generate(); on success return the text.
      5. On any error: log and return None (caller falls back to empty
         final_response — same behaviour as before this PR).

    Caller writes the returned text into final_response; status stays
    MAX_ITERATIONS so observability is unchanged from the user's POV
    except for the now-meaningful summary text.
    """
    if not self.config.summary_on_budget_exhausted:
        return None
    budget = context.metadata.get("iteration_budget")
    if isinstance(budget, dict):
        # Reconstruct lightweight: we only need grace_consumed semantics.
        # The actual budget object lives in context.metadata as the dataclass
        # itself (we serialize to dict for EngineResult only).
        pass
    actual_budget = context.metadata.get("_iteration_budget_obj")
    if actual_budget is None or not actual_budget.grace_call():
        return None

    # Build summary prompt.
    summary_messages = list(messages)
    summary_messages.append({
        "role": "user",
        "content": (
            "[System: You've used your iteration budget. "
            "Please provide a clean summary of:\n"
            "  1. What you accomplished in this turn,\n"
            "  2. What remains to be done,\n"
            "  3. Any important findings the user should know.\n"
            "Be concise but specific. Reference actual files / actions taken.]"
        ),
    })

    try:
        # Bypass all middleware and tool registration — this is a one-shot
        # text completion, not part of the iterative loop. Pass an empty
        # tool registry so the model has no choice but to write text.
        summary_response = self.services.provider.generate(
            messages=summary_messages,
            tools=[],
            stream_callback=None,
            model_call_config=self.config.model_call_config,
        )
        text = (summary_response.text or "").strip()
        if text:
            context.metadata["summary_provided"] = True
            return text
    except Exception as exc:  # noqa: BLE001 — summary failure must not crash turn
        self.services.logger.warning(
            "max_iterations summary generation failed: %s", exc
        )
    return None
```

### 3.4 `EngineConfig` 新字段

[`backend/harness/aether/config/schema.py`](../../backend/harness/aether/config/schema.py) 追加：

```python
# Sprint 3 / PR 3.2: cheap-tool refund list. Tools whose call doesn't
# consume real iteration budget (e.g. todo bookkeeping, memory writes,
# session search) — when an iteration consists ONLY of these calls,
# the budget is refunded so the model has more headroom for real work.
# Names compared with the same normalisation tool_hardening uses
# (case-fold + dash↔underscore + namespace strip).
cheap_tool_names: tuple[str, ...] = (
    "update_todo",
    "todo_write",
    "memory",
    "memory_write",
    "memory_read",
    "skill_manage",
    "session_search",
)

# Sprint 3 / PR 3.2: when True, exhausting max_iterations triggers a
# one-shot non-tool LLM call to generate a summary of what happened.
# The summary becomes EngineResult.final_response (status stays
# MAX_ITERATIONS) so users / scripts see meaningful output instead of
# an empty string.  Default True because the cost is bounded (one
# extra LLM call per max-iterations event) and the UX win is large.
summary_on_budget_exhausted: bool = True
```

### 3.5 `agent.py` 关键改动

入口（约 211 行处）：

```python
# === PR 3.2 替换 ===
# OLD: while iterations < self.config.max_iterations:
budget = IterationBudget(max_total=self.config.max_iterations)
context.metadata["_iteration_budget_obj"] = budget  # 内部状态
context.metadata["iteration_budget"] = budget.to_dict()  # 外部可见

while True:
    if not budget.consume():
        break
    iterations = budget.used
    context.iteration = iterations
    # ... rest of loop unchanged ...
```

工具循环结束处（CHECK_EXIT 之前）：

```python
# === PR 3.2 新增 ===
all_calls_were_cheap = bool(response.tool_calls) and all(
    self._is_cheap_tool(call.name) for call in response.tool_calls
)
if all_calls_were_cheap:
    budget.refund()
context.metadata["iteration_budget"] = budget.to_dict()
```

替换所有 `iterations >= self.config.max_iterations`：

```python
# === PR 3.2 替换 ===
# OLD:  if iterations >= self.config.max_iterations:
#           exit_reason = ExitReason.MAX_ITERATIONS
#           break
# NEW:
if budget.exhausted:
    summary_text = self._handle_max_iterations(request, messages, context)
    if summary_text:
        final_response = summary_text
    exit_reason = ExitReason.MAX_ITERATIONS
    break
```

`_build_result` 段：

```python
# === PR 3.2 新增 ===
budget = context.metadata.get("_iteration_budget_obj")
if budget is not None:
    metadata["iteration_budget"] = budget.to_dict()
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/runtime/iteration_budget.py` | **新文件** | `IterationBudget` 完整实现 | ~80 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 替换主循环；加 `_is_cheap_tool` / `_handle_max_iterations`；填充 `metadata["iteration_budget"]` | ~80 净增 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 `cheap_tool_names` + `summary_on_budget_exhausted` | ~30（含注释） |
| `backend/harness/aether/tests/test_iteration_budget.py` | **新文件** | 见 § 五.1 | ~250 |
| `backend/harness/aether/tests/test_max_iterations_summary.py` | **新文件** | 见 § 五.2 | ~200 |
| `backend/harness/aether/tests/test_cheap_tool_refund.py` | **新文件** | 见 § 五.3 | ~180 |

## 五、测试用例（详细）

### 5.1 `test_iteration_budget.py`

**测试组 A：基础语义**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | `IterationBudget(max_total=5)` | `remaining == 5`, `exhausted == False`, `used == 0` |
| **T-A2** | consume 5 次 | 每次返回 True；第 6 次返回 False |
| **T-A3** | consume 1 + refund 1 | `used == 0`, `consume_count == 1`, `refund_count == 1` |
| **T-A4** | consume 5 + refund 1 + consume 1 | `used == 5`, 第 7 次 consume 返回 False |
| **T-A5** | refund() in fresh budget（未 consume） | 静默 no-op，`used == 0` |
| **T-A6** | exhausted 状态下再 refund() | `used == max_total - 1`，可继续 consume |

**测试组 B：grace_call**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | grace_call() 第一次 | 返回 True, `grace_consumed == True` |
| **T-B2** | grace_call() 第二次 | 返回 False（已用） |
| **T-B3** | grace_call() 不影响 used 计数 | 调用后 `used` 不变 |
| **T-B4** | exhausted 状态下 grace_call() 后再 consume | consume 仍返回 False（grace 是一次性 bonus，不解锁 consume） |

**测试组 C：序列化**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | `to_dict()` 字段完整 | 含 `used / max / remaining / grace_consumed / consume_count / refund_count` |
| **T-C2** | `to_dict()` 是 JSON-friendly | `json.dumps(budget.to_dict())` 不抛 |
| **T-C3** | 多次操作后 `to_dict()` 反映最新状态 | 等同直接读 properties |

**测试组 D：边界**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | `max_total=0` | `consume()` 立即返回 False；`grace_call()` 仍可用一次 |
| **T-D2** | `max_total=1` | consume 1 次成功；refund 后再 consume 仍可 |
| **T-D3** | 极大 `max_total=10**6` | 循环 1M 次 consume 性能可接受（< 0.5s） |

### 5.2 `test_max_iterations_summary.py`

**测试组 E：基本生成**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | mock provider，max_iterations=3，每次返回 `tool_calls=[ToolCall("noop")]`，第 4 次（grace）返回文本 "Summary text" | `result.final_response == "Summary text"`, `result.exit_reason == MAX_ITERATIONS`, `metadata["summary_provided"] == True` |
| **T-E2** | summary_on_budget_exhausted=False | `final_response == ""`（兼容旧行为），`metadata.get("summary_provided") in (None, False)` |
| **T-E3** | grace 调用本身抛 ProviderInvocationError | summary 失败但不 crash；`final_response == ""`；warning 日志包含异常信息 |
| **T-E4** | grace 返回空字符串 | `metadata.get("summary_provided") in (None, False)`；不写入 `final_response` |

**测试组 F：prompt 内容**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | 检查 grace 调用的 messages 列表 | 最后一条 user message 含 "[System: You've used your iteration budget"；前序 messages 完整保留 |
| **T-F2** | 检查 grace 调用传给 provider 的 tools 参数 | `tools=[]`（不带任何工具，强制纯文本回复） |
| **T-F3** | provider 收到的 model_call_config 与正常调用一致 | 同 EngineConfig.model_call_config |

**测试组 G：grace 单次性**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 同一 turn 内 _handle_max_iterations 被调用两次（理论上不应该，但防御性测试） | 第二次返回 None；不发第二次 LLM 请求 |

### 5.3 `test_cheap_tool_refund.py`

**测试组 H：refund 触发**

| ID | 场景 | 验证 |
|---|---|---|
| **T-H1** | 一轮中只调用 `update_todo` | budget.used == 0（refund 抵消 consume） |
| **T-H2** | 一轮中调用 `shell` + `update_todo` | 不 refund（混合调用），`used == 1` |
| **T-H3** | 5 轮全是 cheap tool | `used == 0`, `refund_count == 5` |
| **T-H4** | 5 轮全是 cheap tool 后开始正常调用 5 轮 | 总 used == 5（前 5 轮 refunded） |

**测试组 I：cheap tool 名匹配**

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | 调 `UpdateTodo`（驼峰）| 仍触发 refund（复用 `_normalize_name`） |
| **T-I2** | 调 `update-todo`（dash）| 仍触发 refund |
| **T-I3** | 调 `mcp__router__update_todo`（namespace 前缀） | 仍触发 refund |
| **T-I4** | 调 `mcp__router__shell` | 不 refund（shell 不在 cheap 名单） |
| **T-I5** | 调 `update_todo_v2`（不在名单的相似名） | 不 refund |
| **T-I6** | `cheap_tool_names = ()`（空配置） | 任何调用都不 refund |

**测试组 J：与 P1-3 summary 协同**

| ID | 场景 | 验证 |
|---|---|---|
| **T-J1** | max_iterations=3，3 轮 cheap tool 后 max_iterations 仍未用尽（refund 抵消），继续 3 轮真正工作 | summary 不触发（未 exhausted） |
| **T-J2** | max_iterations=3，3 轮真正工作后 exhausted，触发 summary | summary 文本进 final_response |

**测试组 K：metadata 暴露**

| ID | 场景 | 验证 |
|---|---|---|
| **T-K1** | 任意一次 turn 完成 | `result.metadata["iteration_budget"]` 含全部字段 |
| **T-K2** | refund 发生过 | `iteration_budget["refund_count"] > 0` |
| **T-K3** | grace 触发过 | `iteration_budget["grace_consumed"] == True` |

## 六、验收门

- [ ] 所有新测试 green
- [ ] 既有测试无回归
- [ ] 真实跑一次 max_iterations 触发场景，CLI 看到 summary 文本（手工验证）
- [ ] 真实跑一次有 update_todo 的场景，看 footer 显示 `used=N` 比 `consume_count=M` 小

## 七、回滚开关

- `summary_on_budget_exhausted=False`：退回原"沉默 break"行为
- `cheap_tool_names=()`：退回"所有工具都消耗预算"行为
- 完全 revert PR：上述两个开关为默认即可

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. 新文件 `runtime/iteration_budget.py` | 1.5h | `IterationBudget` 完整实现 |
| 2. `tests/test_iteration_budget.py` | 1.5h | 测试组 A-D（约 16 case） |
| 3. `config/schema.py` 加新字段 | 30min | 含详细注释 |
| 4. `agents/core/agent.py` 替换主循环 | 2h | 替换 9 处判断 + budget 注入 |
| 5. `agents/core/agent.py` 加 `_handle_max_iterations` + `_is_cheap_tool` | 1.5h | 见 § 三.2/3.3 |
| 6. `tests/test_max_iterations_summary.py` | 2h | 测试组 E-G |
| 7. `tests/test_cheap_tool_refund.py` | 2h | 测试组 H-K |
| 8. 真实场景手工验证 | 1h | summary + cheap tool |
| 9. 既有测试回归 | 30min | unittest discover |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 替换 9 处 max_iterations 时漏了某处 | 中 | grep `iterations >=` 全文确认；测试组 H-K 间接覆盖 |
| `_handle_max_iterations` 调 provider 失败导致 turn 卡住 | 低 | try/except 包住，失败时返回 None，行为退回原 break |
| cheap tool 列表遗漏导致用户实际工具被错误 refund | 低 | 默认列表保守（仅明显的记账类）；用户可 override |
| `_is_cheap_tool` 用 `_normalize_name` 引入 phantom_tool 依赖循环 | 低 | 已确认 phantom_tool 无反向依赖 agent.py |

## 十、与后续 PR 的接合

- **PR 3.4** 的 `LLMForkSummarizer` 复用本 PR 的 `summary_on_budget_exhausted` 路径模式：
  也是发一次"无 tools / 限制输出 token"的 provider 请求。
  实现可以直接抄 `_handle_max_iterations` 的骨架。
- **PR 3.4** 的流水线触发判断里 `compression_max_failures` 复用本 PR 的"熔断"思路：
  Tier 5 连续失败 N 次后整轮终止。
- **CLI footer**（独立小 PR 或 Sprint 5）可以从 `metadata["iteration_budget"]`
  渲染 `↻ 12/15 (3 cheap)` 信息。
