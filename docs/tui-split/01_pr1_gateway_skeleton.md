# PR 1 · Gateway skeleton + Transport

## 摘要

新建 `aether/gateway/` 包，落地最薄的进程骨架：通过 `pyproject.toml` 的 `aether-gateway` 控制台脚本启动；定义可插拔的 `Transport` 协议；实现一个 stdio transport；铺好崩溃与信号处理。不挂任何 RPC 方法，不引入新 runtime 依赖。

## Scope

In scope:

- 新建 `aether/gateway/__init__.py`
- 新建 `aether/gateway/transport.py`：`Transport` Protocol + `StdioTransport` + `contextvars` 当前 transport
- 新建 `aether/gateway/entry.py`：进程入口（panic hook、SIGPIPE、shutdown grace、idle loop）
- 修改 `pyproject.toml`：新增 `aether-gateway = "aether.gateway.entry:main"` 控制台脚本
- 测试：transport 单元测试 + entry boot/exit 烟测

Out of scope:

- JSON-RPC dispatcher（PR 2）
- 任何 `@method` handler（PR 3+）
- WebSocket transport（设计中预留接口，不实现）
- TS 侧任何变动（PR 6 起）

## Contracts

```python
# aether/gateway/transport.py
from typing import Any, Callable, Optional, Protocol, runtime_checkable
import contextvars

@runtime_checkable
class Transport(Protocol):
    def write(self, frame: dict[str, Any]) -> None:
        """Serialize `frame` as one JSON line and send to the peer."""

    def close(self) -> None: ...

class StdioTransport:
    def __init__(self, stream_getter: Callable[[], Any] | None = None) -> None: ...
    def write(self, frame: dict[str, Any]) -> None: ...
    def close(self) -> None: ...

_current: contextvars.ContextVar[Optional[Transport]] = contextvars.ContextVar(
    "aether_gateway_transport", default=None
)

def bind_transport(t: Transport) -> contextvars.Token: ...
def reset_transport(token: contextvars.Token) -> None: ...
def current_transport() -> Transport: ...   # falls back to module-level StdioTransport
```

Framing 规则：

- 一帧 = 一行 UTF-8 JSON + `\n`。
- 仅允许 ASCII-safe（`ensure_ascii=False` + `\n` 作分隔，禁止裸 `\r`）。
- stderr 不在 transport 管辖范围内，留给日志与崩溃信息。

入口与生命周期：

```python
# aether/gateway/entry.py
def main() -> int: ...

def _panic_hook(exc_type, exc_value, exc_tb) -> None: ...
def _thread_panic_hook(args) -> None: ...
def _log_signal(signum: int, frame) -> None: ...
```

- crash log 路径：`<HOME>/.aether/logs/gateway_crash.log`
- 默认 shutdown grace：`1.0s`，可被 env `AETHER_GATEWAY_SHUTDOWN_GRACE_S` 覆盖
- SIGPIPE 安装 handler，不让后台线程写已关闭管道时把进程闪杀

## 设计要点

**为什么先做 transport 抽象。** 这个 PR 单看没有用户价值，但它把 web/IPC 形态都封死成"换 transport"。Hermes 的 `tui_gateway/transport.py` 在 v0.6→v0.9 完整跑过了 stdio / WS / Tee 三种实现，证明这个分层不假，所以直接借用。

**为什么用 `contextvars` 而不是全局。** dispatcher 在线程池里跑多个 long handler，每个 handler 的"对侧"可能是不同 WebSocket 连接（未来）。`contextvars` 是唯一能跨线程池任务正确传播的方案。在当前 stdio-only 阶段它退化成全局，但接口不需要变。

**为什么不用 asyncio。** engine 是同步阻塞 + `asyncio.to_thread` 跑的。整个 gateway 在 PR 4 也要走同样路径。引入 asyncio 会强迫每个 handler 是 coroutine，得不偿失。Hermes 也选了同步 + 线程池。

**Process bootstrap 的几个坑（已在 Hermes 趟过）：**

- `sys.path` 加固：父进程通过 env `AETHER_PYTHON_SRC_ROOT` 传入工程根，子进程 insert 到 path 首位，再剔除空字符串与 `.`，避免被 CWD 中同名包 shadow。
- SIGPIPE → 安装 handler 记录"哪个线程在哪里被信号打中"，避免后台 TTS / 通知线程让进程静默退出。
- crash log + stderr 一行摘要双轨：log 文件留全栈，stderr 让父进程把消息抛进 TUI 的活动条。
- shutdown grace：`sys.exit(0)` 会与持 stdout lock 的工作线程互锁；先 join、超时后 `os._exit`。

**Web 预留位。** `transport.py` 文档注释里直接画清楚未来 `WebSocketTransport` 与 `TeeTransport` 的形状，但不写代码；这是 PR 设计文档而非 stub。

## Files touched

- new: `aether/gateway/__init__.py`
- new: `aether/gateway/entry.py`
- modified: `pyproject.toml`（追加 `aether-gateway` 控制台脚本）
- new: `aether/gateway/transport.py`
- new: `aether/tests/gateway/__init__.py`
- new: `aether/tests/gateway/test_transport.py`
- new: `aether/tests/gateway/test_entry_boot.py`

## Dependencies

无。这是 sprint 的第一个 PR。

## Acceptance criteria

- `pip install -e .` 后 `aether-gateway` 启动不退出、不向 stdout 写任何内容（dispatcher 还没接，handler 还没注册）。
- 收到 SIGTERM / SIGINT 后在 shutdown grace 内退出，crash log 路径不存在多余写入。
- `StdioTransport.write({"hello": "world"})` 写出一行 JSON + `\n`，且并发写不交叉（线程锁正确）。
- `bind_transport` / `reset_transport` 在子线程中可见同一 transport（contextvar 传播验证）。
- transport 单测覆盖：peer-gone（EPIPE）静默吞掉、非 peer-gone IO 错误向上抛、`close()` 幂等。
- 进程异常退出时 `~/.aether/logs/gateway_crash.log` 有 traceback 段落，stderr 有一行 `[gateway-crash] ...`。

## Manual verification

```bash
# 0. 一次性安装（dev 环境）
pip install -e .

# 1. 进程可启动
( sleep 5 | aether-gateway ) &
GPID=$!
sleep 0.5
ps -p $GPID && kill $GPID

# 2. SIGPIPE 不秒杀
python -c "
import subprocess, signal, time
p = subprocess.Popen(['aether-gateway'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
p.stdout.close()  # close pipe under it
time.sleep(0.5)
p.send_signal(signal.SIGTERM); p.wait(timeout=2)
"

# 3. crash log 写入
python -c "
import aether.gateway.entry as e
try:
    raise RuntimeError('boom')
except Exception:
    import sys; e._panic_hook(*sys.exc_info())
"
ls ~/.aether/logs/gateway_crash.log
```
