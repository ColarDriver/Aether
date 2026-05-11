# PR 6.7 — HTTP / WebFetch 事件驱动取消

## 目标

让所有走 HTTP / 网络 IO 的 tool（`WebFetch` / `WebSearch` / HTTP-based MCP /
任何用 `httpx` / `requests` 的代码）在按打断键后**立刻断开连接**，不等远端响应
或 TCP timeout。

## 当前问题

按 ESC 时一个 30s 的远端 API 调用：
- `httpx.get(url, timeout=...)` 是阻塞调用，timeout 不到不返回。
- 我们的 `InterruptController` flag 设上了，但 `httpx.get` 看不到。
- 用户得等 timeout（默认可能 30-60s）或远端响应才能继续。

参考 `open-claude-code/src/tools/WebFetchTool/utils.ts:417-421`：

```typescript
const response = await getWithPermittedRedirects(
    upgradedUrl,
    abortController.signal,     // ← signal 直接传给 axios
    isPermittedRedirect,
)
```

```typescript
// 内部
const response = await axios.get(url, { signal })
// signal.aborted → axios 立即扔 CanceledError
```

JavaScript 的 `fetch` / `axios` 原生支持 AbortSignal。**Python `httpx` 不支持**
原生 signal，但支持手动 `client.close()` —— 我们用 PR 6.3 的 listener 接事件，
listener 里 close client。

## 设计

### Pattern: `add_listener` + `client.close()`

```python
# tools/builtins/web_fetch.py
class WebFetchTool(ToolExecutor):
    interrupt_behavior = "cancel"  # PR 6.5 声明

    def run(self, call: ToolCall, ctx: ToolDispatchContext) -> ToolResult:
        sig: InterruptSignal | None = getattr(ctx, "interrupt_signal", None)

        # 已经 aborted → 立即返回
        if sig is not None and sig.is_aborted():
            return _make_aborted_result(call, reason=sig.reason)

        client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
            follow_redirects=True,
        )

        interrupted = {"flag": False}

        def _on_abort(reason: str | None) -> None:
            interrupted["flag"] = True
            try:
                client.close()  # 关掉 connection pool — 正在 in-flight 的请求立刻抛 ReadError
            except Exception:
                pass

        unregister = sig.add_listener(_on_abort, once=True) if sig else (lambda: None)

        try:
            response = client.get(url)
            body = response.text
        except httpx.ReadError as exc:
            # 这是 abort 触发 close 后 in-flight 请求抛的异常
            if interrupted["flag"]:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"[fetch interrupted by user before response]",
                    is_error=True,
                    metadata={"interrupted": True, "url": url},
                )
            # 不是中断引起，正常错误处理
            return _make_error_result(call, str(exc))
        except httpx.HTTPError as exc:
            return _make_error_result(call, str(exc))
        finally:
            unregister()
            client.close()

        return ToolResult(content=body, ...)
```

### 为什么用 `client.close()` 而不是 cancel-context

Python 的 `httpx` 不像 JS `fetch` 支持 `signal` 参数。`httpx` 有两种"取消"方式：

1. **`client.close()`** —— 关 connection pool，in-flight 请求会抛 `ReadError`。
   这是我们用的方式。
2. **`asyncio.CancelledError`** —— 只对 `httpx.AsyncClient` 有效，需要把整个
   tool 改成 async。Aether 当前 tool 是 sync 的，不走这条路。

`close()` 的副作用：connection 在 server 端可能要一段时间才释放，但**对 client
线程**是立即返回的，符合我们的"按 ESC 立刻停"目标。

### 通用 helper

把这套模板抽成 `runtime/interrupt_helpers.py:run_interruptible_http`：

```python
# runtime/interrupt_helpers.py
from contextlib import contextmanager

@contextmanager
def httpx_client_aware_of(signal: InterruptSignal | None, **client_kwargs):
    """Context manager that yields an httpx.Client wired to close on signal.abort.

    用法：

        with httpx_client_aware_of(ctx.interrupt_signal, timeout=30) as client:
            response = client.get(url)
            # 中断 → client.close() → 这里抛 httpx.ReadError，caller 用 try/except 拿

    退出 context 时会同时 close client 和 remove listener。
    """
    client = httpx.Client(**client_kwargs)
    unregister = lambda: None
    interrupted = {"flag": False}

    if signal is not None:
        def _on_abort(_reason):
            interrupted["flag"] = True
            try:
                client.close()
            except Exception:
                pass
        if signal.is_aborted():
            _on_abort(signal.reason)
            try:
                yield client
            finally:
                client.close()
            return
        unregister = signal.add_listener(_on_abort, once=True)

    try:
        yield client
    finally:
        unregister()
        try:
            client.close()
        except Exception:
            pass
```

Tool 用法：

```python
class WebFetchTool(ToolExecutor):
    interrupt_behavior = "cancel"

    def run(self, call, ctx):
        try:
            with httpx_client_aware_of(ctx.interrupt_signal, timeout=30) as client:
                response = client.get(url)
        except httpx.ReadError:
            if ctx.interrupt_signal and ctx.interrupt_signal.is_aborted():
                return _make_aborted_result(call)
            raise
        ...
```

### MCP HTTP transport

MCP 有 HTTP / stdio / WebSocket 三种 transport。同 pattern：

| Transport | 取消方式 |
|---|---|
| HTTP | `httpx_client_aware_of` |
| stdio | 进程组 kill（同 shell tool） |
| WebSocket | `add_listener(ws.close)` |

JSON-RPC `notifications/cancelled` 是另一层（应用层取消，告诉远端"我不要这个 request 的结果了"），跟 transport 层 close 不冲突，可以同时做：

```python
def _on_abort(reason):
    # 1. 应用层：发 cancel notification（best-effort，server 可能不支持）
    try:
        mcp_client.send_cancel_notification(request_id, reason)
    except Exception:
        pass
    # 2. transport 层：close client（保底）
    try:
        client.close()
    except Exception:
        pass
```

但 PR 6.7 **只做 transport 层**。应用层 cancel notification 留给后续 PR
（取决于 MCP server 生态对 `notifications/cancelled` 的支持率）。

## 影响范围

需要改造的 tool（grep 一下 `httpx.get` / `httpx.post` / `requests.` 用法）：

```
tools/builtins/web_fetch.py        ← 主要
tools/builtins/web_search.py       ← 如果走 HTTP
tools/mcp/http_transport.py        ← MCP HTTP transport（如果存在）
... 其他用 httpx 的 tool
```

让我们先用 `grep` 在 PR 实施时确定准确范围。所有改造点都用同一个 `httpx_client_aware_of` helper，diff 大小取决于有多少个 tool 用 httpx，预估 ~200 行（含测试）。

## 文件改动

```
runtime/interrupt_helpers.py  [新文件]
  + httpx_client_aware_of(signal, **kwargs) context manager
  + 后续 PR 可以加 websocket_aware_of / 其他

tools/builtins/web_fetch.py
  ~ 用 httpx_client_aware_of 替换裸 httpx.Client
  ~ 加 interrupt_behavior = "cancel"
  ~ 中断返回 metadata["interrupted"]=True

tools/builtins/web_search.py
  ~ 同上

tools/mcp/http_transport.py（如果存在）
  ~ 同上

tests/runtime/test_interrupt_helpers.py  [新文件]
  + httpx_client_aware_of 单元测试

tests/tools/test_web_fetch_interrupt.py  [新文件]
  + WebFetch 集成测试（要起一个 local HTTP server 模拟慢响应）
```

## 实现细节

### Local test server

测试需要一个"故意慢响应"的 HTTP server 来模拟"按 ESC 前远端没回"的场景：

```python
# tests/conftest.py 或 tests/tools/_helpers.py
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class SlowHandler(BaseHTTPRequestHandler):
    delay_sec = 5.0
    def do_GET(self):
        time.sleep(self.delay_sec)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"finally")
    def log_message(self, *args, **kw):
        pass  # 安静

@contextmanager
def slow_local_server(delay_sec: float = 5.0):
    SlowHandler.delay_sec = delay_sec
    server = HTTPServer(("127.0.0.1", 0), SlowHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
```

### 测试

```python
class HttpxClientAwareOfTests(unittest.TestCase):
    def test_abort_closes_client_and_raises_read_error(self):
        sig = InterruptSignal()
        with slow_local_server(delay_sec=3.0) as url:
            threading.Timer(0.1, lambda: sig.abort("user-interrupt")).start()
            start = time.monotonic()
            with self.assertRaises(httpx.ReadError):
                with httpx_client_aware_of(sig, timeout=10.0) as client:
                    client.get(url)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.3)  # 中断 100ms + close 同步开销

    def test_no_abort_completes_normally(self):
        sig = InterruptSignal()
        with slow_local_server(delay_sec=0.05) as url:
            with httpx_client_aware_of(sig, timeout=10.0) as client:
                r = client.get(url)
            self.assertEqual(r.status_code, 200)

    def test_signal_already_aborted_short_circuits(self):
        sig = InterruptSignal()
        sig.abort("user-interrupt")
        with slow_local_server(delay_sec=10.0) as url:
            start = time.monotonic()
            with self.assertRaises(httpx.ReadError):
                with httpx_client_aware_of(sig, timeout=10.0) as client:
                    client.get(url)
            self.assertLess(time.monotonic() - start, 0.1)

    def test_listener_removed_after_normal_exit(self):
        sig = InterruptSignal()
        with slow_local_server(delay_sec=0.05) as url:
            with httpx_client_aware_of(sig, timeout=10.0) as client:
                client.get(url)
        # 后续 abort 应该不调任何东西 — 不抛即过
        sig.abort("user-interrupt")


class WebFetchInterruptTests(unittest.TestCase):
    def test_web_fetch_interrupt_returns_aborted_result(self):
        # 完整 tool run() 路径
        ...

    def test_web_fetch_normal_completion_no_interrupted_metadata(self):
        ...
```

## 验收门

- `test_interrupt_helpers.py` 全过（5+ 测试）。
- `test_web_fetch_interrupt.py` 全过。
- 现有 WebFetch / WebSearch / MCP 相关测试不回归。
- 手测：让模型 `WebFetch` 一个故意慢的 endpoint，3s 内按 ESC，立刻停（< 300ms）。

## 不在本 PR 内（后续 PR）

- MCP `notifications/cancelled` 应用层 cancel —— 视 MCP server 生态做。
- WebSocket transport 的 signal-aware close —— Aether 当前没有 WS-based tool，未来要时再加。
- `asyncio` 路径（如果未来 tool 改成 async）—— 现在不需要。
