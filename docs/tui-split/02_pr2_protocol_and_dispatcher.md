# PR 2 · JSON-RPC protocol + dispatcher

## 摘要

在 PR 1 的 transport 之上铺 RPC 层：定义 JSON-RPC envelope、所有事件 / 请求的 Pydantic schema、`@method("name")` 注册表、长任务线程池。本 PR 之后，dispatcher 能正确收发，但没有任何真实业务 handler 注册（PR 3 起开始注册）。

## Scope

In scope:

- 新建 `aether/gateway/protocol.py`：所有 wire schema（Pydantic v2 models）
- 新建 `aether/gateway/dispatcher.py`：`@method` 装饰器 + 注册表 + 同步线程池
- 修改 `aether/gateway/entry.py`：把 dispatcher 接到 transport，跑请求循环
- 注册一个内置方法 `gateway.ping`（仅用于联通性测试，永不删除）
- 单元测试：schema round-trip、注册表、long-handler 走线程池、错误码、并发安全

Out of scope:

- 任何业务 method（PR 3+）
- approval / permission 的 server-initiated request（PR 5）
- TS 端类型镜像（PR 6）

## Contracts

### JSON-RPC envelope

```python
# aether/gateway/protocol.py
from pydantic import BaseModel, ConfigDict
from typing import Annotated, Any, Literal

JSONRPC_VERSION = "2.0"

class RpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: str | int
    method: str
    params: dict[str, Any] | None = None

class RpcNotification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    method: str
    params: dict[str, Any] | None = None

class RpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None

class RpcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: str | int | None
    result: Any | None = None
    error: RpcError | None = None
```

错误码集合：

| 代码 | 含义 |
|---|---|
| -32700 | parse error（非 JSON 或非 UTF-8） |
| -32600 | invalid request（envelope 不合法） |
| -32601 | method not found |
| -32602 | invalid params（schema 校验失败） |
| -32603 | internal error |
| -32000 | application error（handler 抛 `GatewayError`） |
| -32001 | handler timed out |
| -32002 | handler cancelled |

### 事件 schema（dispatcher 自己用 + 后续 PR 复用）

```python
class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str

class GatewayReady(EventBase):
    type: Literal["gateway.ready"]
    version: str
    capabilities: list[str]

class GatewayError(EventBase):
    type: Literal["gateway.error"]
    message: str
    where: str | None = None
```

> 业务事件（`text_delta`、`tool_call`、`tool_result`、`approval_request` 等）由 PR 4 / PR 5 在自己的文件里定义并在 dispatcher 的 event union 上注册。

### `@method` 注册表

```python
# aether/gateway/dispatcher.py
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

_METHODS: dict[str, Callable[..., Any]] = {}
_LONG_METHODS: set[str] = set()

def method(name: str, *, long: bool = False) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register an RPC handler. `long=True` routes to the worker pool."""

def dispatch_request(req: RpcRequest, transport: Transport | None = None) -> RpcResponse | None:
    """Validate, route, return response. None means the handler is async-replied."""

def notify(method: str, params: dict[str, Any]) -> None:
    """Emit an outbound notification on the current transport."""
```

线程池配置：

| 参数 | 默认 | 备注 |
|---|---|---|
| `max_workers` | `min(8, (os.cpu_count() or 4) + 2)` | 与 Hermes 对齐 |
| `thread_name_prefix` | `aether-gw-worker` | 便于 `py-spy` 定位 |
| `atexit` | `shutdown(wait=False, cancel_futures=True)` | shutdown grace 内不阻塞 |

handler 函数签名：

```python
@method("gateway.ping", long=False)
def _ping(params: dict[str, Any] | None) -> dict[str, Any]:
    return {"pong": True}
```

`long=True` handler 在线程池里跑；short handler 在 dispatcher 线程内同步跑。

## 设计要点

**为什么手写 Pydantic 而不是 jsonrpc 库。** 想要的不是"通用 JSON-RPC"，而是 Aether 自己的协议。库会把字段命名、错误码、batch 行为锁死，反而限制后续 server-initiated request 的设计。Hermes 也是手写 dispatcher。

**`long=True` 标记的语义。** 长 handler（`agent.run`、`session.list` 中文件 IO 多的、未来 web 端的批量操作）会阻塞 dispatcher 主线程。短 handler（ping、prefs.get、config 读取）走主线程直接返回。线程池只服务 long，避免无脑全池化让短请求等被批的回应。

**请求 id 命名空间。** 客户端发的 request id 与未来 server-initiated request id 共用一个池子（int 或 string），但走相反方向的 pending map。这样不会冲突，也不需要前缀。Hermes 用同一种处理方式。

**handler 返回值的语义。** 同步返回 dict → dispatcher 包成 `RpcResponse`。返回 `None` → 表示 handler 会异步在自己时机调用 `respond(id, ...)`（server-initiated request 的回路也走这条）。抛 `GatewayError` → -32000 application error。抛其它异常 → -32603 + crash log。

**Schema 漂移防线。** 所有 wire model 启用 `extra="forbid"`，新增字段必须显式加。TS 侧手写镜像；PR 9 上线前增量加 `tests/gateway/test_schema_snapshot.py`，对 wire schema 做 snapshot 测试，CI 跑 diff。

**Batch 不支持。** JSON-RPC 2.0 支持 batch，本协议显式不支持：每帧一个 request / notification / response。这是 Hermes 的实践，简化 GatewayClient 解析逻辑。

## Files touched

- new: `aether/gateway/protocol.py`
- new: `aether/gateway/dispatcher.py`
- modified: `aether/gateway/entry.py`（接入 dispatcher，跑请求循环）
- new: `aether/tests/gateway/test_protocol_roundtrip.py`
- new: `aether/tests/gateway/test_dispatcher.py`
- new: `aether/tests/gateway/test_long_method_pool.py`

## Dependencies

PR 1（用 transport 写帧、用 entry 进入请求循环）。

## Acceptance criteria

- `aether-gateway` 启动后先发出一帧 `gateway.ready` notification，再进入请求循环。
- 客户端写一帧 `{"jsonrpc":"2.0","id":1,"method":"gateway.ping"}`，能收到 `{"jsonrpc":"2.0","id":1,"result":{"pong":true}}`。
- 未知 method 返回 -32601。
- params 校验失败返回 -32602 并附带详细字段路径。
- 长 method（设置 `long=True` 的 dummy handler）跑在 worker pool 上，并发 N 个不会阻塞同时进来的 short request。
- handler 抛 `GatewayError("...", code=-32000)` → 客户端收到 -32000 + message + optional data。
- handler 抛其它 exception → -32603，stderr 一行 + crash log 全栈，进程不退出。
- `extra="forbid"` 在 envelope 与所有 event model 上生效；未声明字段会被 -32600 拒绝。

## Manual verification

```bash
# 1. 联通性
( echo '{"jsonrpc":"2.0","id":"1","method":"gateway.ping"}'; sleep 0.2 ) \
  | aether-gateway 2>/dev/null \
  | head -2

# 期望：第一行 gateway.ready，第二行 {"jsonrpc":"2.0","id":"1","result":{"pong":true}}

# 2. 未知方法
echo '{"jsonrpc":"2.0","id":"x","method":"nope"}' | aether-gateway 2>/dev/null | head -2

# 3. 并发 long vs short（用 helper 脚本）
python aether/tests/gateway/_manual_concurrency.py
```
