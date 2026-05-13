# PR 4 · Run-loop streaming

## 摘要

新增 `agent.run` 与 `agent.cancel` 两个 RPC 方法，把 `AgentEngine.run_loop()` 接到 dispatcher 上。run loop 跑在 long-handler 线程池里；token deltas、tool calls、tool results、status 与生命周期事件全部以 JSON-RPC notification 形式流回客户端。

## Scope

In scope:

- 新建 `aether/gateway/handlers/agent_methods.py`
- 在 `aether/gateway/protocol.py` 追加事件 schema：`TextDelta`, `ToolCall`, `ToolResult`, `Reasoning`, `Status`, `IterationStart`, `IterationEnd`, `LoopStateChanged`, `Done`, `Cancelled`, `Error`
- 把 `EngineRequest.stream_callback` / `EngineHooks` 适配到 `notify("event", ...)`
- `agent.cancel` 通过 `threading.Event` 通知正在执行的 run loop 退出
- 单元测试 + integration test（mock provider，校验 event 序列）

Out of scope:

- approval / permission 反向 request（PR 5；本 PR 假定 `Prompter` 为 stub，不会触发交互）
- TS 端事件订阅（PR 6 起）

## Contracts

### `agent.run`

```jsonc
// request
{
  "jsonrpc": "2.0",
  "id": "r1",
  "method": "agent.run",
  "params": {
    "session_id": "ses_abc123",
    "user_message": "...",            // 单条 user message；多模态先支持 text
    "max_iterations": 8,              // optional, fallback to EngineConfig
    "temperature": null,              // optional
    "system_override": null           // optional, single-turn system override
  }
}
```

handler 标记为 `long=True`。

```jsonc
// final response (after the loop finishes)
{
  "jsonrpc": "2.0",
  "id": "r1",
  "result": {
    "final_text": "...",
    "exit_reason": "done" | "cancelled" | "max_iterations" | "error",
    "usage": { "input_tokens": 123, "output_tokens": 45, ... },
    "metadata": { ... }
  }
}
```

### `agent.cancel`

```jsonc
// request
{ "jsonrpc": "2.0", "id": "c1", "method": "agent.cancel", "params": { "session_id": "ses_abc123" } }

// response
{ "jsonrpc": "2.0", "id": "c1", "result": { "ok": true } }
```

session_id 不在 running map 中 → 仍然返回 `{ok: true}`，幂等。

### 事件 schema（追加到 `protocol.py`）

```python
class TextDelta(EventBase):
    type: Literal["text.delta"]
    text: str
    sequence: int                     # monotonic per run

class Reasoning(EventBase):
    type: Literal["reasoning.delta"]
    text: str
    sequence: int

class ToolCall(EventBase):
    type: Literal["tool.call"]
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    iteration: int

class ToolResult(EventBase):
    type: Literal["tool.result"]
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    iteration: int

class IterationStart(EventBase):
    type: Literal["iteration.start"]
    iteration: int

class IterationEnd(EventBase):
    type: Literal["iteration.end"]
    iteration: int

class LoopStateChanged(EventBase):
    type: Literal["loop.state"]
    state: str                         # 直接镜像 LoopState 字符串

class Status(EventBase):
    type: Literal["status"]
    kind: Literal["thinking", "responding", "tool_use", "idle"]
    detail: str | None = None

class TokenUsage(EventBase):
    type: Literal["usage"]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
```

事件外层包装：

```jsonc
{ "jsonrpc": "2.0", "method": "event", "params": { "type": "text.delta", "text": "...", "sequence": 7, "session_id": "ses_abc123", "run_id": "r1" } }
```

所有事件都额外携带 `session_id` 与 `run_id`，让客户端能正确路由（未来 web 端多 session 并发）。

### 与现有契约的对接

| 现有结构 | 映射 |
|---|---|
| `EngineRequest.stream_callback: StreamDeltaCallback` | gateway 注入一个 callback，对每个 chunk 调 `notify("event", TextDelta(...))` |
| `EngineRequest.silent_stream_callback: StreamSilentCallback` | tool argument JSON 流不上送；只用于内部计数 |
| `EngineHooks.before_llm` / `after_llm` | 翻成 `iteration.start` / `iteration.end` |
| `EngineHooks.before_tool` / `after_tool` | 翻成 `tool.call` / `tool.result` |
| `EngineHooks.on_session_start` / `on_session_end` | 翻成 `status` 事件 |
| `LoopState` 转换 | 翻成 `loop.state` 事件（PR 8 的 activity bar 主要数据源） |

> 引用：`aether/runtime/core/contracts.py`、`aether/runtime/core/hooks.py`、`aether/agents/core/agent.py:448`。

## 设计要点

**为什么不重写 engine。** engine 是同步阻塞的，并发性靠 `asyncio.to_thread` + provider 内部的 streaming callback 解决。本 PR 完全保留这个模式：handler 在线程池里调 `engine.run_loop(request)`，stream callback 由 gateway 提供成包装函数，写出 RPC notification。Hermes 也是同等做法。

**cancel 怎么生效。** 维护一个 `running_runs: dict[session_id, RunHandle]`，`RunHandle` 持有 `threading.Event`。`agent.cancel` 把 event 翻为 set，engine 的循环顶端检查 event，从下一次 iteration 进入 `EXITING`。这是现有 sprint-6 interrupt 机制的复用，本 PR 只是把 trigger 从 prompt_toolkit signal handler 换成 RPC。

**sequence 字段。** 每次 `agent.run` 内部维护一个原子计数器，每条 `text.delta` / `reasoning.delta` 递增。TS 端用它检测乱序或丢帧，并合并断流（连接恢复后从 `sequence > last_seen` 继续）。本 PR 不实现重连，但字段先铺好。

**线程池里的 contextvars。** 由 PR 1 决定的 `current_transport` contextvar 必须在 worker 里被正确读到。通过 `contextvars.copy_context()` 在 `submit` 时绑定一次。Hermes 的等价代码在 `tui_gateway/server.py:dispatch` 周围。

**back-pressure 与 TextDelta 速率。** 高吞吐时（Claude streaming 每个 chunk 几 ms），可能产生上千个 notification。stdio 上不会阻塞（pipe 有 buffer），但 TS 端如果不主动消费会让 Node OS pipe 堆积。本 PR 不做合并 / 节流，验证一次粗略量后视情况在后续 PR 加。

**why `usage` is its own event.** Provider 在每条 chunk 上只能给到部分用量；最终用量来自最后一帧。让 `usage` 独立事件可以发"中途估算"也可以发"最终值"，TS 端按 `iteration` 累加。

**Multi-session safety.** 当前 TUI 是单 session，但 server 不假设单 session。`running_runs` 用 session_id 做 key；同一个 session 同时只能有一个 active run，第二次 `agent.run` 进来 → -32000 application error `RUN_ALREADY_ACTIVE`。

## Files touched

- new: `aether/gateway/handlers/agent_methods.py`
- new: `aether/gateway/run_handle.py`（运行中状态结构 + cancel event）
- modified: `aether/gateway/protocol.py`（追加 event schema）
- modified: `aether/gateway/entry.py`（import agent_methods）
- new: `aether/tests/gateway/test_agent_run_streaming.py`
- new: `aether/tests/gateway/test_agent_cancel.py`

## Dependencies

PR 1（transport），PR 2（dispatcher + protocol）。

可以与 PR 3 并行。PR 5 / PR 7 等都依赖本 PR 的事件模型。

## Acceptance criteria

- 用 mock provider 调一次 `agent.run`，能按顺序收到：`status(thinking)` → `iteration.start(0)` → 多条 `text.delta` → `iteration.end(0)` → `status(idle)` → 最终 response。
- 含工具的 mock turn：`tool.call` → `tool.result`，两条事件携带相同 `tool_call_id`。
- `agent.cancel` 在 run loop 处于 LLM streaming 中时，能让最终 response 的 `exit_reason="cancelled"`。
- 两次并发 `agent.run`（同 session）→ 第二次得到 `RUN_ALREADY_ACTIVE`。
- 现有 engine 测试套件不回归（`pytest aether/tests/agents`、`pytest aether/tests/runtime`、`pytest aether/tests/engine`）。
- 没有任何来自 gateway 的事件遗漏 `session_id` 与 `run_id`（snapshot 测试）。

## Manual verification

```bash
# 单 turn，mock provider（开发用）
python aether/tests/gateway/_manual_agent_run.py --provider mock --turn "say hello"

# 真实 provider 长输出 + 中途 cancel
python aether/tests/gateway/_manual_agent_run.py --provider anthropic --model claude-haiku-4-5-20251001 \
  --turn "write a 500-word essay" --cancel-after 1.0
```
