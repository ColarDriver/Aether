# PR 11.2 — `post_tool_use` Hook 上 Engine

## 目标 / Goal

把 `open-claude-code/src/types/hooks.ts:101,109` 的 `PostToolUse` / `PostToolUseFailure` 语义对齐到 Aether：在 `EngineHooks` 上**新增两个**回调，并在主循环里 `run_after_tool` 之后**保证**触发。

这是后续 PR 11.3 / 11.4 / 11.5 的地基：
- PR 11.3 的 `DiagnosticTracker` 要在 *tool 真的成功之后* 才采集诊断。
- PR 11.4 的 `file_edit` / `write_file` 需要一个明确的"工具完成时刻"挂钩点，不污染工具本身的实现。
- PR 11.5 的 attachment 注入需要根据 *本轮发生过哪些 tool call* 决定要不要拉诊断。

把这些都塞进 `middleware.after_tool` 是错的：middleware 是用来 *改 ToolResult 内容* 的，不该承载"副作用回路"。

## 当前问题 / Current Problem

### 1. `EngineHooks` 没有 tool 边界

```python
# aether/runtime/core/hooks.py:27-107（实际方法列表，省略 body）
class EngineHooks:
    def on_session_start(...) -> None: ...
    def pre_llm_call(...) -> HookOutcome | None: ...
    def pre_api_request(...) -> None: ...
    def post_api_request(...) -> None: ...
    def post_llm_call(...) -> None: ...
    def on_session_end(...) -> None: ...
    def on_task_cleanup(...) -> None: ...
```

—— **没有任何 tool 相关钩子**。模型决定调一个 `file_edit`、Engine 把它派发出去、tool 执行完毕、结果回到主循环—— 这一连串事件里 EngineHooks 一个都不参与。

### 2. `middleware.after_tool` 不是合适的替代品

```python
# aether/agents/middlewares/base.py:24
def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
    return result
```

签名说明问题：返回值必须是 `ToolResult`。这意味着：
- 实现方"必须"返回（或转发）result —— 想 fire-and-forget 触发一个异步诊断采集，要么 hack 一个全局变量，要么 monkey-patch context。
- 实现方 **不知道**这次 tool call 的 `tool_call_id`、`tool_name`、`args`（这些信息要从 `result.metadata` 里捞，并不所有 tool 都写齐）。
- 一旦中间任何一个 middleware 抛异常，整条 `run_after_tool` 链就断了 —— 用作 *记录* 用途的钩子最忌"一个挂全挂"。

### 3. 现实下一步：`DefaultSubagentBuilder.hooks` 透传链已经存在

`default_builder.py:65` 已经把 `parent._hooks` 透传给 child engine，所以一旦 `EngineHooks` 长出 `post_tool_use`，subagent 自动同步生效——不需要再改 builder 的代码。

## 改动 / Changes

### 1. 扩展 `EngineHooks`

`aether/runtime/core/hooks.py`：

```python
# 头部新增 import（type only）
from aether.runtime.core.contracts import ToolResult

# 在 EngineHooks 类里新增两个方法
@dataclass(slots=True)
class EngineHooks:
    # ...(existing methods unchanged)...

    def post_tool_use(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolResult,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        """Fire-and-forget hook after a tool completes successfully.

        Parity with open-claude-code ``PostToolUse``.  Implementations
        MUST NOT raise — they may log, schedule background work, or
        mutate ``context_metadata``.  They MUST NOT modify ``result``;
        use middleware ``after_tool`` for ToolResult transformation.
        """
        return None

    def post_tool_use_failure(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        """Fire-and-forget hook after a tool raises.

        Parity with open-claude-code ``PostToolUseFailure``.  Same
        contract as :meth:`post_tool_use`.
        """
        return None
```

`Any` 来自现有 `typing` import，无需新增。

### 2. 在主循环里触发

`aether/agents/core/agent.py` 里找到 *tool 执行落地点*。当前结构（简化）：

```python
# agent.py:1004 附近
result = await self._dispatch_tool(...)
result = self._middleware.run_after_tool(result, context)
# (no hook fired here)
```

改为：

```python
import time

# 包裹 dispatch
t0 = time.perf_counter()
try:
    result = await self._dispatch_tool(call, context)
except Exception as exc:
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    self._safe_post_tool_use_failure(
        session_id=session_id,
        iteration=iteration,
        call=call,
        error=exc,
        elapsed_ms=elapsed_ms,
        context_metadata=context.metadata,
    )
    raise
elapsed_ms = (time.perf_counter() - t0) * 1000.0

result = self._middleware.run_after_tool(result, context)

self._safe_post_tool_use(
    session_id=session_id,
    iteration=iteration,
    call=call,
    result=result,
    elapsed_ms=elapsed_ms,
    context_metadata=context.metadata,
)
```

新增两个私有 helper（在同一文件靠近现有 `_safe_*` helper 的位置）：

```python
def _safe_post_tool_use(
    self,
    *,
    session_id: str,
    iteration: int,
    call: "ToolCall",
    result: ToolResult,
    elapsed_ms: float,
    context_metadata: dict[str, Any],
) -> None:
    hooks = self._hooks
    if hooks is None:
        return
    try:
        hooks.post_tool_use(
            session_id=session_id,
            iteration=iteration,
            tool_call_id=call.id,
            tool_name=call.name,
            tool_args=call.args or {},
            result=result,
            elapsed_ms=elapsed_ms,
            context_metadata=context_metadata,
        )
    except Exception:
        self.services.logger.exception(
            "hook.post_tool_use_raised", tool=call.name
        )

def _safe_post_tool_use_failure(
    self,
    *,
    session_id: str,
    iteration: int,
    call: "ToolCall",
    error: Exception,
    elapsed_ms: float,
    context_metadata: dict[str, Any],
) -> None:
    hooks = self._hooks
    if hooks is None:
        return
    try:
        hooks.post_tool_use_failure(
            session_id=session_id,
            iteration=iteration,
            tool_call_id=call.id,
            tool_name=call.name,
            tool_args=call.args or {},
            error=error,
            elapsed_ms=elapsed_ms,
            context_metadata=context_metadata,
        )
    except Exception:
        self.services.logger.exception(
            "hook.post_tool_use_failure_raised", tool=call.name
        )
```

注意：
- hook 异常**全部吞掉并记日志**（参考 `open-claude-code/src/Tool.ts:476` 那里 hook 也是 fire-and-forget）。
- middleware 在前、hook 在后—— hook 看到的是 *middleware 链处理过的最终 ToolResult*，与 OCC `PostToolUse` 一致。
- failure path 在 `_dispatch_tool` 抛出时也触发；middleware 抛出本身**不**当成 failure（middleware 自己负责自己的 try/except，错了说明 middleware 实现 bug，不是 tool failure）。

### 3. Subagent 路径自动同步

`aether/subagents/default_builder.py:65` 已传 `hooks=parent._hooks`，本 PR 零额外改动。child engine 触发 hook 时 `session_id` 自然是 child 的 session id —— hook 实现方据此区分父子。

### 4. 文档 / 公开 API

在 `aether/runtime/core/__init__.py` 公开 `EngineHooks` 的位置上确认 `post_tool_use` / `post_tool_use_failure` 都暴露给外部使用方（CLI custom hooks、gateway plugin、test fixture）。

## 测试 / Tests

新建 `aether/tests/agents/test_post_tool_use_hook.py`：

- `test_post_tool_use_fires_on_success` —— 自定义 hook 子类记录调用；scripted provider 让模型调一次成功的 tool → hook 被调用一次，参数完整。
- `test_post_tool_use_carries_middleware_result` —— middleware `after_tool` 把 result.text 改成 `"OK"` → hook 看到的 result.text == `"OK"`（顺序正确）。
- `test_post_tool_use_failure_fires_on_exception` —— tool dispatch 抛 `ToolNotFoundError` → `post_tool_use_failure` 被调一次，`error` 是该异常实例。
- `test_post_tool_use_hook_exception_does_not_propagate` —— hook 实现自身抛错 → 主循环继续推进；日志含 `hook.post_tool_use_raised`。
- `test_hook_elapsed_ms_is_measured` —— scripted provider 让 tool sleep 50ms → hook 收到的 `elapsed_ms >= 40`（带宽容）。
- `test_subagent_inherits_parent_hooks` —— 父 engine 装 hook → 通过 `task` 工具 spawn 一个 child → child 调 tool 时父的 hook 同样被触发；hook 收到的 `session_id` 是 *child* 的。

## 验收 / Acceptance

- `uv run pytest aether/tests/agents/test_post_tool_use_hook.py` 全绿。
- `uv run pytest aether/tests/agents/` 既有 274 条仍全绿（无回归）。
- `uv run pyright` 零新增告警。
- 已有的 `middleware.after_tool` 行为完全不变（顺序 / 签名 / 错误传播）。
- 日志验证：故意写一个抛 RuntimeError 的 hook 实现，跑一次后 log 中应能 grep 到 `hook.post_tool_use_raised`。

## 不在本 PR / Deferred to other PRs

- **PR 11.3** 用此 hook 安装 `DiagnosticTracker`。
- **PR 11.4** 在 `file_edit` / `write_file` 之后 *通过 hook* 触发 LSP didChange/didSave —— 工具本身不直接持有 LSPManager 引用，保持 tool 单一职责。
- "异步 hook"（hook 想 `await`）不在本 PR；OCC 的对应 hook 也都是 fire-and-forget。如果未来要支持，加 `async_post_tool_use` 同名方法即可，不影响本 PR 的同步签名。
