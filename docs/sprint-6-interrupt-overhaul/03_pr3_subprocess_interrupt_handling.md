# PR 6.3 — Shell tool subprocess 中断

## 目标

让 shell tool（以及任何用 `subprocess.run` 跑外部命令的 tool）在用户按打断键后能**立即终止子进程**，不必等到 `timeout` 自然到期。这是除 stream 之外另一个高频"打断没用"场景：用户跑了个 `pytest` / `npm install`，等了 30s 才意识到要中断。

## 当前问题

`tools/builtins/shell.py` 用的是 `subprocess.run(command, shell=True, ..., timeout=timeout, check=False)`：

- `subprocess.run` 是阻塞调用，子进程不结束、不超时就一直 hang 在那。
- 调用线程被卡，`InterruptController` 即使在 main thread 设了 flag，shell tool 看不到。
- 等 `timeout` 触发 `TimeoutExpired` 后才能退出 —— 默认 `_DEFAULT_TIMEOUT_SEC` 看代码是 60s，最长上限 600s。
- 用户按 Ctrl-C → 主线程 / asyncio 线程的 `_on_interrupt` 调用 `engine.interrupt()` → flag 设上 → 但 shell tool worker thread 完全不感知。

## 设计

### 用 Popen + poll 替代 run

把 `subprocess.run(..., timeout=...)` 改成手动 Popen + 周期性 poll 中断标志：

```python
proc = subprocess.Popen(
    command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=str(cwd) if cwd else None,
    # 把子进程放到独立 process group，方便一次性 kill 整组
    # （不然 shell=True 起的 subshell 自己跑了 child，单 kill subshell
    #  会留下孤儿）。Windows 用 CREATE_NEW_PROCESS_GROUP。
    start_new_session=True if os.name != "nt" else False,
    creationflags=(
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    ),
)

POLL_INTERVAL_SEC = 0.2
GRACE_AFTER_SIGTERM_SEC = 2.0

deadline = time.monotonic() + timeout
interrupted = False

try:
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=POLL_INTERVAL_SEC)
            break  # 子进程正常退出
        except subprocess.TimeoutExpired:
            # 200ms 内没完成 → 检查中断标志
            if interrupt_check is not None and interrupt_check():
                interrupted = True
                _terminate_process_group(proc)
                # 给 SIGTERM 一个 grace period 让子进程做清理
                try:
                    stdout, stderr = proc.communicate(timeout=GRACE_AFTER_SIGTERM_SEC)
                except subprocess.TimeoutExpired:
                    # 还不死 → SIGKILL
                    _kill_process_group(proc)
                    stdout, stderr = proc.communicate(timeout=1.0)
                break
            # 检查总超时
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(command, timeout)
finally:
    # 防御性 cleanup：万一上面走了别的异常路径，也保证 fd 释放
    if proc.poll() is None:
        _kill_process_group(proc)
```

### Process group 管理

关键 helper：

```python
def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the entire process group (POSIX) or process tree (Windows)."""
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        # 子进程已经退出了 — 不算错
        pass


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL fallback when SIGTERM didn't work in grace period."""
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
```

为什么用 process group：`shell=True` 启动会跑一层 sh/bash，sh 再 fork 真正的命令。直接 `proc.terminate()` 只 kill sh，不 kill grandchild —— 留下孤儿进程继续占资源。所以要 kill 整个 process group。

### Tool 接收 interrupt_check 的接口

shell tool 不应该直接访问 `services.interrupt_controller`（耦合太深）。改用 `EngineRequest` 透传一个 callback：

```python
# runtime/contracts.py
@dataclass(slots=True)
class EngineRequest:
    ...
    interrupt_check: Callable[[], bool] | None = None
```

`AgentEngine.run_loop` 在每个 tool dispatch 前注入 closure：

```python
# agents/core/agent.py 内部
def _make_interrupt_check(self, session_id: str) -> Callable[[], bool]:
    """Closure that lets long-running tools poll the interrupt flag."""
    def _check() -> bool:
        return self._is_interrupted(session_id)
    return _check
```

然后在 dispatch 阶段把它放进 tool context（具体 plumbing 跟现有 `ToolDispatchContext` 模式一致）：

```python
context.interrupt_check = self._make_interrupt_check(request.session_id)
```

shell tool 从 `context.interrupt_check` 拿到这个 callback，不需要 import engine。

### 中断后的返回值

中断的 shell tool 应该返回**带标记的 ToolResult**，而不是 raise（不要让一个 tool 失败拖垮整个 run_loop）：

```python
if interrupted:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"[command interrupted by user after {duration_ms}ms]\n"
                f"--- partial stdout ---\n{stdout_text[:1024]}",
        is_error=True,
        metadata={
            "exit_code": -signal.SIGTERM,
            "interrupted": True,
            "duration_ms": duration_ms,
            "command": command,
        },
    )
```

为什么要返回 ToolResult 而不是 raise `EngineInterrupted`：

- run_loop 的中断检查在 tool dispatch 之间已经做了；tool 自己只需要尽快退出，把"我被打断了"的事实通过 ToolResult metadata 传出来。
- 让 partial stdout 进入 message 历史，模型下一轮能看到"我刚才跑这个命令被打断，看到了这些输出"。
- 如果 tool raise EngineInterrupted，会跳过 after_tool middleware，partial 数据丢失。

但还要 raise 一个**轻量 sentinel**让 run_loop 知道"这次 dispatch 是被打断触发的，应该走 INTERRUPTED 退出而不是继续下一个 iteration"。最简方案：在 ToolResult.metadata 里设 `"interrupted": True`，run_loop 在 after_tool 之后检查：

```python
if result.metadata.get("interrupted"):
    state_machine.transition(LoopState.INTERRUPTED)
    exit_reason = ExitReason.INTERRUPTED
    break
```

## 文件改动

```
runtime/contracts.py
  ~ EngineRequest: + interrupt_check: Callable[[], bool] | None = None

agents/core/agent.py
  + _make_interrupt_check(session_id) helper
  ~ run_loop: 把 closure 注入 context / EngineRequest（具体位置看 dispatch
    context 的初始化点）
  ~ tool dispatch loop: after_tool 之后检查 result.metadata["interrupted"]，
    true 则转到 INTERRUPTED + break

tools/builtins/shell.py
  ~ run(): 用 Popen + poll 替代 subprocess.run
  + _terminate_process_group / _kill_process_group helpers
  + start_new_session=True (POSIX) / CREATE_NEW_PROCESS_GROUP (Windows)
  ~ 中断时返回带 metadata["interrupted"]=True 的 ToolResult

tests/tools/test_shell_interrupt.py  [新文件]
```

## 实现细节

### POSIX vs Windows process group

- POSIX：`os.setsid()`（通过 `start_new_session=True`）让子进程成为新 process group leader，`os.killpg(pid, sig)` 一次 kill 整组。
- Windows：`CREATE_NEW_PROCESS_GROUP` flag 让子进程能接收 `CTRL_BREAK_EVENT`。`proc.send_signal(signal.CTRL_BREAK_EVENT)` 是 Windows 上的 SIGTERM 等价。`proc.kill()` 直接走 TerminateProcess 是 SIGKILL 等价。
- 测试需要分别覆盖（Windows runner 用 skip-on-nt 标记，至少在 CI 跑 Linux 路径）。

### Polling 节奏

`POLL_INTERVAL_SEC = 0.2` 是延迟 vs CPU 开销的折中：

- 200ms：用户感知是"按了几乎立刻就停"，能接受。
- 100ms：感知更快但一秒钟 10 次 syscall + dict lookup，对 thousands-of-tools 场景累计成本可见。
- 500ms：明显感觉"按了之后还要再等一下"。

200ms 跟 PR 6.2 的 stream callback polling cadence（per chunk，通常 ≤100ms）协调，整体响应延迟 < 500ms 可达。

### 跟 timeout 的交互

旧逻辑是 `subprocess.run(timeout=N)`，N 到了抛 `TimeoutExpired`。新逻辑里我们手动维护 `deadline = time.monotonic() + timeout`，每次 poll 既检查中断又检查总超时，超时后 raise `subprocess.TimeoutExpired` 让外层既有的处理逻辑（已经在 shell.py 里）继续工作 —— 不动 timeout 路径的 user-visible 行为。

### 不在本 PR 范围内的 tool

只改 `tools/builtins/shell.py`。其他 tool（read_file / Edit / Grep 等）通常 < 1s 完成，不需要中断；如果未来加新的 long-running tool，按同样的 pattern：

1. 接 `context.interrupt_check`（或从 `request.interrupt_check` 拿）。
2. 阻塞循环里每 200ms poll 一次。
3. 中断时返回 `metadata["interrupted"]=True` 的 ToolResult。

把这个约定写进 `docs/run-loop-roadmap/` 的 tool 开发指南（不在本 PR 改）。

## 参考实现

- `open-claude-code/src/tools/BashTool/BashTool.tsx` 行 283：`interrupted: z.boolean()` schema 字段，把"是否被打断"作为标准 ToolResult 元数据。
- `open-claude-code/src/tools/PowerShellTool/PowerShellTool.tsx` 行 248：相同。
- `open-claude-code/src/utils/toolErrors.ts` 行 28：`error.interrupted ? INTERRUPT_MESSAGE_FOR_TOOL_USE : ''` —— 中断的 tool error 转成专用 marker（PR 6.4 会用到）。

## 测试

`tests/tools/test_shell_interrupt.py`

```python
import threading
import time
import pytest

from aether.tools.builtins.shell import ShellTool
from aether.runtime.contracts import ToolCall, ToolDispatchContext


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only process group test")
class ShellInterruptPosixTests(unittest.TestCase):
    def test_long_sleep_interrupted_within_500ms(self):
        # sleep 30 → expect to die within 500ms after interrupt flag set
        tool = ShellTool()
        flag = threading.Event()

        def interrupt_check():
            return flag.is_set()

        ctx = ToolDispatchContext(..., interrupt_check=interrupt_check)
        call = ToolCall(name="shell", arguments={"command": "sleep 30"}, ...)

        # Trip the flag from another thread after 200ms
        threading.Timer(0.2, flag.set).start()

        start = time.monotonic()
        result = tool.run(call, ctx)
        elapsed = time.monotonic() - start

        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata["interrupted"])
        self.assertLess(elapsed, 0.7)  # 200ms 触发 + 200ms poll + 200ms grace

    def test_normal_completion_returns_no_interrupted_flag(self):
        # echo hello → completes naturally, metadata["interrupted"] not set
        tool = ShellTool()
        ctx = ToolDispatchContext(..., interrupt_check=lambda: False)
        call = ToolCall(name="shell", arguments={"command": "echo hello"}, ...)

        result = tool.run(call, ctx)
        self.assertFalse(result.is_error)
        self.assertNotEqual(result.metadata.get("interrupted"), True)
        self.assertIn("hello", result.content)

    def test_partial_stdout_preserved_on_interrupt(self):
        # Bash command that prints stuff then sleeps
        tool = ShellTool()
        flag = threading.Event()
        threading.Timer(0.5, flag.set).start()

        cmd = "for i in 1 2 3 4 5; do echo line$i; sleep 0.2; done"
        ctx = ToolDispatchContext(..., interrupt_check=flag.is_set)
        call = ToolCall(name="shell", arguments={"command": cmd}, ...)

        result = tool.run(call, ctx)
        self.assertTrue(result.metadata["interrupted"])
        # 应该至少看到前几行输出
        self.assertIn("line1", result.content)
        self.assertIn("line2", result.content)

    def test_subshell_grandchildren_killed_too(self):
        # bash -c 'sleep 30 & wait' — grandchild 必须也死
        tool = ShellTool()
        flag = threading.Event()
        threading.Timer(0.2, flag.set).start()

        cmd = "sleep 30 & echo $! ; wait"
        ctx = ToolDispatchContext(..., interrupt_check=flag.is_set)
        call = ToolCall(name="shell", arguments={"command": cmd}, ...)

        start = time.monotonic()
        result = tool.run(call, ctx)
        elapsed = time.monotonic() - start

        # 抓出 grandchild PID
        import re
        m = re.search(r"\d+", result.content)
        self.assertIsNotNone(m)
        gc_pid = int(m.group())

        # 等 0.5s 让 SIGKILL 落地
        time.sleep(0.5)

        # ps 不到这个 PID 就说明 process group kill 成功
        with self.assertRaises(ProcessLookupError):
            os.kill(gc_pid, 0)

        self.assertLess(elapsed, 1.0)


class ShellInterruptIntegrationTests(unittest.TestCase):
    def test_run_loop_marks_interrupted_when_tool_returns_interrupted_metadata(self):
        # 1. Build engine with a stub provider that asks shell to run sleep 30
        # 2. Spawn engine.run_loop in worker thread
        # 3. After 300ms, set interrupt flag via engine.interrupt(...)
        # 4. run_loop returns; assert result.exit_reason == ExitReason.INTERRUPTED
        # 5. Assert result.metadata["interrupt"]["was_in_tool_call"] == True
        ...
```

## 验收门

- POSIX 平台 4 个 shell-interrupt 测试通过。
- Windows runner 上至少基础 termination 测试通过（grandchild 测试 skip）。
- 整体打断响应延迟（按下 ESC → shell tool 返回）≤ 700ms。
- 现有 `test_shell.py` 等已有 shell tool 测试不回归。
- 手测：跑 `pytest`、`npm install` 等长命令，按 ESC 立刻停。

## 不在本 PR 内（留给后续 PR）

- `INTERRUPT_MESSAGE_FOR_TOOL_USE` 注入对话历史 → PR 6.4
- 其他 tool 的中断改造 → 按需追加，本 sprint 不强制
- subagent task 的中断（已通过 `_interrupt_active_children` 部分支持）→ 后续 audit
