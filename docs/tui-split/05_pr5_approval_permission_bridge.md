# PR 5 · Approval / permission bridge

## 摘要

把 `Prompter`（plan / questions）与 `ToolPermissionPrompter`（工具执行前确认）从"进程内 future"模式改造成"跨进程 server-initiated request"模式。Engine 代码不变；交互调用走 PR 4 的 dispatcher，反向方向发 request 给 TS 客户端，等待客户端回 response。

## Scope

In scope:

- 新建 `aether/gateway/reverse_rpc.py`：server → client request 的发起 + pending response map + 超时
- 新建 `aether/gateway/handlers/prompter_bridge.py`：实现 `Prompter` 协议 + `ToolPermissionPrompter` 协议
- 在 `agent.run` 的 EngineRequest 注入这两个 prompter（替换 PR 4 的 stub）
- 新增 RPC 方法（客户端 → 服务端）：`approval.response`、`permission.response`
- 新增反向事件（服务端 → 客户端）：`approval.request`、`permission.request`
- 整合 sprint-7 已落地的 `ToolPermissionRequest` / `ToolPermissionDecision`（`aether/runtime/tools/tool_permissions.py`）
- 测试：accept / reject / timeout / 客户端断开

Out of scope:

- TS 端的 modal UI（PR 8）
- 命令行 non-interactive 行为（保持 sprint-7 的默认 deny；gateway 只是搬运）

## Contracts

### Reverse-direction request envelope

JSON-RPC 没有区分方向；本协议**约定**：

- `id` 为字符串、且以 `srv_` 开头 → 来自 server。
- response 写回时保留 `id` 原样。
- 客户端必须在 `request_timeout` 内回 response，否则 server 端 future 被超时 reject。

```jsonc
// server → client (approval.request)
{
  "jsonrpc": "2.0",
  "id": "srv_app_1",
  "method": "approval.request",
  "params": {
    "kind": "plan" | "questions",
    "session_id": "ses_abc",
    "run_id": "r1",
    "tool_call_id": "call_xyz",            // null for plan-mode confirmation
    "plan_text": "...",                    // when kind="plan"
    "questions": [                          // when kind="questions"
      {"id": "q1", "text": "...", "kind": "open" | "select", "options": ["a","b"]}
    ],
    "deadline_ms": 60000
  }
}

// client → server (approval.response)
{
  "jsonrpc": "2.0",
  "id": "srv_app_1",
  "result": {
    "kind": "plan" | "questions",
    "confirmed": true,                       // when kind="plan"
    "answers": {"q1": "..."}                 // when kind="questions"
  }
}
```

```jsonc
// server → client (permission.request)
{
  "jsonrpc": "2.0",
  "id": "srv_perm_2",
  "method": "permission.request",
  "params": {
    "session_id": "ses_abc",
    "run_id": "r1",
    "request": {
      // mirror of ToolPermissionRequest
      "tool_call_id": "call_xyz",
      "tool_name": "file_edit",
      "arguments": { ... },
      "category": "write",
      "risk": "medium",
      "preview": {
        "title": "Edit file",
        "subtitle": "src/foo.py",
        "diff": "...",
        "path": "src/foo.py"
      },
      "reason": "model wants to fix the typo"
    },
    "deadline_ms": 120000
  }
}

// client → server (permission.response)
{
  "jsonrpc": "2.0",
  "id": "srv_perm_2",
  "result": {
    "type": "allow_once" | "allow_session" | "deny" | "abort",
    "updated_arguments": null | { ... },
    "feedback": null | "...",
    "rule": null | { "tool_name": "file_edit", "behavior": "allow", "scope": "session", "path_prefix": "src/" }
  }
}
```

### Python 侧适配

```python
# aether/gateway/handlers/prompter_bridge.py
from aether.cli.approval_prompter import Prompter
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionRequest,
    ToolPermissionPrompter,
)

class GatewayPrompter(Prompter):
    def is_interactive(self) -> bool: return True
    def confirm_plan(self, plan: str, *, context=None) -> bool: ...
    def ask_questions(self, questions, *, context=None) -> dict[str, str]: ...

class GatewayToolPermissionPrompter(ToolPermissionPrompter):
    def is_interactive(self) -> bool: return True
    def request_tool_permission(
        self,
        request: ToolPermissionRequest,
        *,
        timeout: float | None = None,
    ) -> ToolPermissionDecision: ...
```

实现都走同一个 `reverse_rpc.call()`：

```python
def call(method: str, params: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    """Send a server-initiated request and block until response (or timeout)."""
```

`call` 内部：

1. 生成 id（`srv_<method>_<n>`，全局原子计数器）
2. 在 `pending: dict[str, Future]` 注册 future
3. `transport.write({...request envelope...})`
4. `future.result(timeout)` 阻塞
5. response 到来时 `approval.response` / `permission.response` 的 handler 把 future 翻为 set

### 客户端 → 服务端的两个新 method

| Method | params | result | long |
|---|---|---|---|
| `approval.response` | 与上面 envelope 的 `result` 一致 | `{ok: true}` | no |
| `permission.response` | 同上 | `{ok: true}` | no |

未知 id 的 response → -32602 + `unknown_pending`，但不影响其它流。

## 设计要点

**为什么用 server-initiated request 而不是 push event。** approval 需要"问问题 - 收答案"严格一对一，且 engine 必须**阻塞**直到拿到答案。push event 没有 id 关联，客户端如果在两个 request 间断开重连，server 会卡死。带 id 的 reverse request + response 是清晰的契约。

**Future 桥的形状。** Aether 现在已经有同样的模式（`aether/cli/tool_permission_prompter.py:19-52`），它用 `asyncio.Future` 让 engine 线程等 prompt_toolkit 异步循环回应。本 PR 不再依赖 asyncio：使用 `concurrent.futures.Future`，gateway 自己的 worker 线程阻塞在 `future.result(timeout)` 上，dispatcher 主线程在收到 `approval.response` 时 set future。简化一层。

**为什么不复用现有 `ApprovalPrompter`。** 它继承自 `prompt_toolkit` 的 dialog 机制；要在 gateway 进程里实现 `Prompter` 协议，从零写更干净。Engine 端注入哪个 Prompter 实现是 `EngineRequest` 的字段，本 PR 只是再加一个实现。

**接住 sprint-7 的契约。** PR 7.1 已经把 `ToolPermissionRequest` / `ToolPermissionDecision` 落进 `aether/runtime/tools/tool_permissions.py`。本 PR 不动这些类型，只在 wire schema 层做镜像。`ToolPermissionRule` 也直接序列化。

**Deadline 与 timeout 双轨。** Server 端 future 有自己的 timeout（`call(timeout=...)`），protocol 还携带 `deadline_ms` 给客户端用于显示倒计时。两者不绑定：客户端 UI 倒计时到 0 时应该自发起 reject 并回 response；server timeout 是兜底（比如客户端崩了）。

**客户端断开怎么办。** transport 检测到 peer-gone → 把所有 pending future 用 `OSError("peer gone")` 拒绝，engine 线程会收到异常。`agent.run` handler 顶层捕获，返回 `exit_reason="error"`。客户端重连不接续 future。

**幂等性。** 同一 id 重复 response → 第二次 -32602。同一 id 在 response 之后又来一次 → 同样 -32602。

**Engine 路径无需改动验证。** `aether/agents/core/agent.py` 的 `before_tool` gate 仍然调用 `EngineRequest.tool_permission_prompter`；只要本 PR 把那个 prompter 设为 `GatewayToolPermissionPrompter`，engine 不需要任何 patch。

## Files touched

- new: `aether/gateway/reverse_rpc.py`
- new: `aether/gateway/handlers/prompter_bridge.py`
- new: `aether/gateway/handlers/response_methods.py`（`approval.response`, `permission.response`）
- modified: `aether/gateway/protocol.py`（追加 approval/permission schema）
- modified: `aether/gateway/handlers/agent_methods.py`（PR 4 里 stub 的 prompter 换成 GatewayPrompter / GatewayToolPermissionPrompter）
- new: `aether/tests/gateway/test_reverse_rpc.py`
- new: `aether/tests/gateway/test_prompter_bridge.py`
- new: `aether/tests/gateway/test_permission_bridge.py`

## Dependencies

PR 4（agent.run 已就绪），PR 3（session.current），PR 2（dispatcher）。

## Acceptance criteria

- 调一次会触发 plan-mode 确认的 `agent.run`：server 写出 `approval.request`，客户端 mock 回 `approval.response confirmed=true`，engine 继续，最终 response 正常。
- 上面同一流程，客户端回 `confirmed=false` → engine 收到拒绝，最终 response 含拒绝信息。
- 工具 permission：拒绝路径下 engine 收到 `ToolPermissionDecision(type=deny)`，工具不被执行，返回的 synthetic tool result 不含 mutation 痕迹（沿用 sprint-7 测试用例）。
- 客户端在 server 等 future 时主动断开 → server 端 future 在 transport 死链 50 ms 内被 reject；`agent.run` 返回 `exit_reason="error"`，message 含 `peer disconnected`。
- 客户端发未知 id 的 `approval.response` → -32602，进程继续工作。
- 同一 id 重复 response → 第二次 -32602。
- `tests/runtime/test_tool_permissions.py`、`tests/engine/test_tool_permission_gate.py`（sprint-7 已有）继续通过。

## Manual verification

```bash
python aether/tests/gateway/_manual_approval_flow.py \
  --scenario plan-confirm-yes
python aether/tests/gateway/_manual_approval_flow.py \
  --scenario file-edit-allow-once
python aether/tests/gateway/_manual_approval_flow.py \
  --scenario client-disconnect-mid-prompt
```
