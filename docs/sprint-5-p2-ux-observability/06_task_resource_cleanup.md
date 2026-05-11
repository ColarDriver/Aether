# PR 5.6 — Task-Scoped Resource Cleanup

## 目标

为子代理、浏览器、VM、长生命周期工具提供 task-scoped 资源清理协议，确保 run loop 正常结束、失败、interrupt 时都能释放资源。

## 设计

新增 hook：

```python
def on_task_cleanup(
    self,
    *,
    task_id: str,
    session_id: str,
    completed: bool,
    interrupted: bool,
    context_metadata: dict[str, Any],
) -> None:
    ...
```

工具可选协议：

```python
class TaskResourceAware(Protocol):
    def acquire_task_resource(self, task_id: str) -> None: ...
    def release_task_resource(self, task_id: str) -> None: ...
```

清理规则：

- `task_id` 存在时，在 run_loop `finally` 中调用 cleanup。
- cleanup 顺序：工具 release -> hook `on_task_cleanup` -> runtime 临时资源。
- 所有 cleanup 都 best-effort，异常记录日志和 metadata，不覆盖原始 run_loop 结果。
- 同一个 `(task_id, resource)` release 必须幂等。

## 文件改动

- `runtime/hooks.py`：新增 `on_task_cleanup`。
- 新增 `runtime/task_cleanup.py`，集中实现 cleanup helper 和 metadata aggregation。
- `tools/base.py` 或 registry 层：定义可选 duck-typed 协议，不强迫所有工具继承。
- `agents/core/agent.py`：run_loop `finally` 调用 cleanup helper。

## 资源范围

本 PR 只建立协议和调用点，不要求立即改造所有工具。建议首批接入：

- browser manager session/page。
- subagent task handles。
- tool result spill 临时目录中 task-scoped 的临时文件。

## 测试

- `tests/engine/test_task_resource_cleanup.py`
- 假工具实现 acquire/release，run_loop 成功结束后 release 被调用一次。
- provider 抛错，release 仍被调用。
- interrupt 后 release 仍被调用，metadata 标记 `interrupted=True`。
- release 抛异常，run_loop 原结果不被覆盖，metadata 记录 cleanup error。
- 无 task_id 时不调用 task cleanup。

## 验收门

- cleanup 不能引入新的 hard dependency。
- cleanup failure 不影响原始 error reporting。
- `on_session_end` 仍按原语义执行；`on_task_cleanup` 是更细粒度的新增 hook。
