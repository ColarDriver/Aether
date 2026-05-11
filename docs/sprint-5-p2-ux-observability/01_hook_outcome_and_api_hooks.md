# PR 5.1 — HookOutcome + API Hooks

## 目标

把 `EngineHooks` 从纯通知接口升级为轻量控制面：仍保持现有通知能力，但允许 `pre_llm_call` 返回结构化 outcome，并新增 API-call 前后观测 hook。这个 PR 是 Sprint 5 的地基。

## 当前问题

- `backend/harness/aether/runtime/hooks.py` 中所有 hook 都返回 `None`。
- `AgentEngine._safe_call_hook` 当前只调用 hook 并吞掉返回值。
- 插件或上层系统无法在当前 user message 后追加 transient context，也不能 short-circuit LLM call。
- API request 级别缺少统一观测点，无法记录 model/provider/api_call_count/approx_tokens。

## 设计

新增 dataclass：

```python
@dataclass(slots=True)
class HookOutcome:
    inject_user_context: str | None = None
    inject_system_addendum: str | None = None
    short_circuit_response: NormalizedResponse | None = None
```

行为规则：

- `pre_llm_call` 可以返回 `HookOutcome | None`；旧 hook 返回 `None` 仍合法。
- 多个 hook outcome 聚合时，`inject_user_context` 按调用顺序拼接到当前 turn 的 user message 末尾。
- `inject_user_context` 必须以 fenced block 或明确 marker 包裹，避免和用户原文混淆。
- `inject_system_addendum` 追加到本次 API-call copy 的 system prompt，不写回 session history；需要 metadata 标记它破坏 prompt cache。
- `short_circuit_response` 一旦出现，跳过 provider.generate，直接按正常 assistant response 路径落入结果构建。
- hook 异常继续 logger.exception 并吞掉，不能影响 run loop。

新增 hook：

```python
def pre_api_request(
    self,
    *,
    session_id: str,
    iteration: int,
    model: str,
    provider: str,
    api_call_count: int,
    message_count: int,
    tool_count: int,
    approx_input_tokens: int,
    request_char_count: int,
    context_metadata: dict[str, Any],
) -> None: ...

def post_api_request(
    self,
    *,
    session_id: str,
    iteration: int,
    model: str,
    provider: str,
    api_call_count: int,
    elapsed_ms: float,
    response_finish_reason: str | None,
    error: Exception | None,
    context_metadata: dict[str, Any],
) -> None: ...
```

## 文件改动

- `runtime/hooks.py`：新增 `HookOutcome` 和 API hooks，导出 public symbol。
- `runtime/__init__.py`：导出 `HookOutcome`。
- `agents/core/agent.py`：让 `_safe_call_hook` 返回 hook result；新增 `_collect_pre_llm_hook_outcome` 或等价 helper。
- `runtime/contracts.py`：仅在类型注解需要时引用 `NormalizedResponse`，避免循环 import。

## 参考实现

- Hermes 在 API-call 前后调用 `pre_api_request` / `post_api_request`，传递 provider、model、token、message_count 等元信息。
- open-claude-code 的 hook response schema 支持 `continue`、`stopReason`、`hookSpecificOutput`，说明 hook 是结构化控制面，不只是生命周期通知。Aether 只实现最小必要集，避免引入权限系统复杂度。

## 测试

- `tests/engine/test_hook_outcome.py`
- hook 返回 `inject_user_context="Memory: foo"`，断言 provider 收到的当前 user message 末尾包含 fenced context，原始 `request.messages` 未被污染。
- hook 返回 `inject_system_addendum`，断言 API-call copy 的 system prompt 包含 addendum，session history 不包含。
- hook 返回 `short_circuit_response`，断言 provider.generate 未被调用，`EngineResult.final_response` 来自 short-circuit。
- hook raise exception，断言 run loop 正常继续且 logger 记录异常。
- pre/post API hooks 收到 model/provider/api_call_count/approx_input_tokens；provider 抛错时 post hook 的 `error` 非空。

## 验收门

- 旧的 `EngineHooks` subclass 不需要改代码即可继续工作。
- `EngineResult.metadata["runtime"]` 和现有 streaming metadata 不回归。
- 所有 hook outcome 都只影响当前 API call，不写入持久 session，除非明确是最终 assistant response。
