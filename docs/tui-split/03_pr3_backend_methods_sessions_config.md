# PR 3 · Backend methods — sessions / prefs / providers / commands

## 摘要

把现有 Python 实现里的 sessions、prefs、providers、slash 命令目录暴露成 RPC 方法。每个 method 都是现有函数的薄包装，不重写逻辑。完成本 PR 后，TS 端（PR 6+）就有所有非 agent 的请求面可调。

## Scope

In scope:

- 新建 `aether/gateway/handlers/__init__.py`
- 新建 `aether/gateway/handlers/session_methods.py` — 包装 `aether/cli/sessions.py`
- 新建 `aether/gateway/handlers/prefs_methods.py` — 包装 `aether/cli/prefs.py`
- 新建 `aether/gateway/handlers/providers_methods.py` — 包装 `aether/cli/providers.py`
- 新建 `aether/gateway/handlers/commands_methods.py` — 暴露 slash registry，但**不**执行 `_cmd_*` handler
- 修改 `aether/gateway/entry.py`：import handlers 包以触发 `@method` 注册
- 单元测试

Out of scope:

- 不动 `aether/cli/sessions.py` 等源文件的实现（PR 9 才迁移）
- 不暴露 `agent.run`（PR 4）
- 不暴露 approval / permission 反向请求（PR 5）
- 不执行 slash 命令的 server-side 逻辑（slash 命令的"动作"将由 TS 端组合多个 RPC 调用来完成，server 只暴露 catalog 与原子操作）

## Contracts

### Method catalog

| Method | params | result | long |
|---|---|---|---|
| `session.create` | `{provider, model, system?, ...}` | `{session_id, info}` | no |
| `session.list` | `{limit?: int}` | `{sessions: SessionInfo[]}` | yes |
| `session.resume` | `{session_id}` | `{info, messages: TranscriptMessage[]}` | yes |
| `session.delete` | `{session_id}` | `{deleted: bool}` | no |
| `session.current` | `{}` | `{session_id: str \| null, info?}` | no |
| `prefs.get` | `{key: str}` | `{value}` | no |
| `prefs.set` | `{key: str, value: Any}` | `{ok: true}` | no |
| `prefs.all` | `{}` | `{prefs: dict}` | no |
| `providers.list` | `{}` | `{providers: ProviderInfo[]}` | no |
| `providers.models` | `{provider: str}` | `{models: ModelInfo[]}` | yes |
| `commands.catalog` | `{}` | `{commands: SlashCommandInfo[]}` | no |

### Schema 节选

```python
class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    updated_at: float
    provider: str
    model: str
    system_prompt: str | None = None
    message_count: int = 0
    summary: str | None = None

class TranscriptMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    text: str | None = None
    name: str | None = None       # tool name when role="tool"
    tool_call_id: str | None = None
    metadata: dict[str, Any] | None = None

class ProviderInfo(BaseModel):
    name: str
    display_name: str
    requires_api_key: bool
    default_base_url: str | None

class ModelInfo(BaseModel):
    id: str
    display_name: str
    context_window: int | None = None

class SlashCommandInfo(BaseModel):
    name: str                       # "/help"
    description: str
    category: str | None = None     # for grouping in TS UI
```

### handler 实现模式

```python
# aether/gateway/handlers/session_methods.py
from aether.cli import sessions as _sessions
from aether.gateway.dispatcher import method

@method("session.list", long=True)
def _list(params):
    limit = (params or {}).get("limit")
    rows = _sessions.list_sessions(limit=limit)
    return {"sessions": [_to_info(r).model_dump() for r in rows]}
```

> 关键：handler 只做"调函数 + 转 schema + 返回"，不写任何业务逻辑。

### Slash 命令暴露策略

`aether/cli/commands.py` 的 `REGISTRY` 当前 14 条（`/help`、`/exit`、`/new`、`/clear`、`/refresh`、`/session`、`/sessions`、`/system`、`/tools`、`/verbose`、`/model`、`/interrupt`、`/resume`、`/stats`），handler 签名是 `(state: ReplState, args) -> CommandResult`，依赖 `prompt_toolkit` 状态。

本 PR **只暴露 catalog**，不让 server 跑 `_cmd_*`。原因：

- `_cmd_*` 读写 `ReplState`（Python 端 UI 状态），跨进程没有意义。
- TS 端拿到 catalog 后，对每条命令自己决定走哪种 RPC：
  - `/help`、`/exit`、`/clear` → 纯前端
  - `/model`、`/new`、`/session`、`/sessions`、`/system`、`/resume` → 拼装 `session.*` / `prefs.*` / `providers.*` 调用
  - `/interrupt` → `agent.cancel`（PR 4 提供）
  - `/stats`、`/tools` → 新增 `gateway.stats` / `tools.catalog`（如果 TS 需要，可在 PR 8 补）

`commands.catalog` 返回的 `SlashCommandInfo` 是"展示用"的，TS 自己掌控 dispatch。

## 设计要点

**为什么不把 `_cmd_*` 直接暴露成 RPC。** 这些 handler 是 Python REPL 内嵌的状态机操作，依赖 `ReplState`、`prompt_toolkit` buffer、`asyncio` 任务。跨进程暴露它们等于把 UI 状态压进协议，会让 PR 9 迁移时被迫先解开这层耦合。让 TS 重新组装命令更干净。

**为什么 `session.list` / `session.resume` / `providers.models` 是 long。** 这些会读磁盘 / 网络（providers.models 需要列 API 模型清单）。即便单次 ms 级，并发也会卡 dispatcher 主线程，按 Hermes 的经验直接挂线程池。

**Schema 与现有 dataclass 的关系。** `aether/cli/sessions.py` 已有自己的 `SessionInfo` dataclass。本 PR 不复用它，因为：

1. 它依赖 `dataclasses` 而 wire schema 用 Pydantic，序列化策略可能不一致。
2. 长远来看 wire schema 是合同，内部 dataclass 是实现，把两者解耦更安全。

`_to_info` helper 做转换。代价是 14 行 boilerplate，收益是合同稳定。

**`prefs.set` 的写入语义。** 现有 `aether/cli/prefs.py` 写入 `~/.aether/prefs.json`。本 PR 保持原子写入（temp file + rename），不引入新存储。

**为什么单独有 `session.current`。** TS 启动后需要知道"我现在跟哪个 session 说话"。server 维护一个进程级"current session"（用 contextvar 或简单全局），第一次 `session.create` 或 `session.resume` 后自动设置。

## Files touched

- new: `aether/gateway/handlers/__init__.py`
- new: `aether/gateway/handlers/session_methods.py`
- new: `aether/gateway/handlers/prefs_methods.py`
- new: `aether/gateway/handlers/providers_methods.py`
- new: `aether/gateway/handlers/commands_methods.py`
- modified: `aether/gateway/entry.py`（import handlers 触发注册）
- new: `aether/tests/gateway/test_session_methods.py`
- new: `aether/tests/gateway/test_prefs_methods.py`
- new: `aether/tests/gateway/test_providers_methods.py`
- new: `aether/tests/gateway/test_commands_methods.py`

## Dependencies

PR 1（transport），PR 2（dispatcher + protocol）。

可以与 PR 4 并行进行：两者都只依赖 PR 2。

## Acceptance criteria

- 启动 gateway 后调用 `commands.catalog` 能拿到当前 `REGISTRY` 中所有命令的 name + description（数量与 `aether/cli/commands.py:443` 起的 `REGISTRY` dict 一致）。
- 不存在的 session_id 调用 `session.resume` 返回 -32000 + 可读 message。
- `session.create` 后 `session.current` 返回新 session_id。
- `prefs.set` 后 `prefs.get` 取到一致 value；崩溃中断不会留下半写入文件（atomic rename 行为已被现有实现保证，本 PR 只验证未回归）。
- `providers.list` 至少返回 anthropic / openai 等当前 `aether/cli/providers.py` 已支持的 provider。
- 所有 handler 都不 import `prompt_toolkit` 或 `rich`（用 grep 校验）。

## Manual verification

```bash
# 列命令清单
echo '{"jsonrpc":"2.0","id":"1","method":"commands.catalog"}' \
  | aether-gateway 2>/dev/null | tail -1 | jq '.result.commands | length'

# 创建一个临时 session
echo '{"jsonrpc":"2.0","id":"1","method":"session.create","params":{"provider":"anthropic","model":"claude-sonnet-4-6"}}' \
  | aether-gateway 2>/dev/null | tail -1 | jq .

# 列 prefs
echo '{"jsonrpc":"2.0","id":"1","method":"prefs.all"}' \
  | aether-gateway 2>/dev/null | tail -1 | jq .

# 校验：gateway 包不 import prompt_toolkit / rich
grep -rn "prompt_toolkit\|^from rich\|^import rich" aether/gateway/ || echo "clean"
```
