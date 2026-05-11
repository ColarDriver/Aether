# PR 7.3 - CLI Permission Overlay

## 目标

在 `AetherApp` 底部区域实现 permission queue 和确认 overlay。确认 UI 必须显示在输入框上方的 bottom region，不能写入 Rich scrollback，也不能进入模型 transcript。

## 当前 CLI 结构

`backend/harness/aether/cli/app.py` 已经是长驻 `prompt_toolkit.Application`。底部区域包含：

| 区域 | 当前用途 |
|---|---|
| active group | 展示正在 coalesce 的工具组 |
| activity bar | spinner、elapsed、tokens、thinking/responding/tool-use |
| reasoning line | 摘要 reasoning |
| input frame | 用户输入 |
| footer | 快捷键、queued、running hint |

permission overlay 应该插入在 input frame 上方，并在存在 pending permission 时压过 active group/activity bar 的普通状态展示。

## 不要复用阻塞式 ApprovalPrompter

`ApprovalPrompter.confirm_plan()` 当前会调用 `prompt_toolkit.prompt()` 或 dialog 的 `.run()`。这适合工具内部一次性提问，但不适合长驻 `AetherApp`。

原因：

| 问题 | 结果 |
|---|---|
| 运行中的 `Application` 再启动 prompt | 两个 prompt_toolkit app 抢 stdin/stdout |
| 确认内容通过 stdout 打印 | 会进入输出框/scrollback |
| 无法复用 AetherApp keybinding | ESC、queued input、footer 状态会错乱 |

正确做法是实现新的 `AetherToolPermissionPrompter`，通过 thread-safe bridge 把 request 放进 `AetherApp` 的 queue。

## Thread-safe Bridge

engine 在 worker thread 中同步运行，`AetherApp` 在 asyncio/prompt_toolkit event loop 中运行。

推荐桥接：

```python
class AetherToolPermissionPrompter:
    def __init__(self, app: AetherApp, loop: asyncio.AbstractEventLoop) -> None: ...

    def request_tool_permission(self, request, *, timeout=None):
        future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(
            self._app.enqueue_permission_request,
            request,
            future,
        )
        return future.result(timeout=timeout)
```

`AetherApp` 内部状态：

```python
@dataclass(slots=True)
class PendingPermissionPrompt:
    request: ToolPermissionRequest
    future: concurrent.futures.Future[ToolPermissionDecision]
    selected_index: int = 0
```

队列行为：

| 操作 | 行为 |
|---|---|
| enqueue | append 到 `_permission_queue`，invalidate |
| render | 只渲染 `_permission_queue[0]` |
| resolve | set_result decision，pop 队首，invalidate |
| abort | set_result deny/abort，pop 队首 |
| shutdown | 对所有未决 future set_result abort |

## Overlay 内容

首版 options：

| 选项 | Decision |
|---|---|
| `Yes, allow this time` | `ALLOW_ONCE` |
| `Yes, allow edits in this session` | `ALLOW_SESSION` |
| `No` | `DENY` |

文案按 tool 调整：

| tool | title | question |
|---|---|---|
| `file_edit` | `Edit file` | `Do you want to make this edit to <basename>?` |
| `write_file` | `Write file` | `Do you want to write <basename>?` |
| `notebook_edit` | `Edit notebook` | `Do you want to edit <basename>?` |
| `shell` | `Run command` | `Do you want to run this command?` |
| fallback | `Use tool` | `Do you want to allow <tool_name>?` |

内容展示：

| request 字段 | 渲染 |
|---|---|
| `preview.diff` | 等宽 diff block，限制高度，长 diff 截断 |
| `preview.command` | 等宽命令 block |
| `preview.path` | subtitle 或 dim line |
| `preview.body` | 普通说明 |

## Keybindings

确认 overlay 激活时，普通输入框不应该消费这些键。

| 键 | 行为 |
|---|---|
| Up / Down | 切换选项 |
| Enter | 确认当前选项 |
| Esc | deny/abort 当前 request |
| Ctrl-C | 同 Esc，或沿用 Sprint 6 busy interrupt 策略 |
| Tab | 后续再做反馈输入，首版可不实现 |

建议 `AetherApp._handle_esc()` 第一优先级新增：

```python
if self._has_permission_prompt():
    self._reject_active_permission()
    return
```

这样确认 overlay 的 ESC 不会误触发 turn interrupt 或清输入框。

## UI Middleware 调整

`backend/harness/aether/cli/ui_middleware.py` 当前 `before_tool()` 会马上：

- 结束 stream。
- 增加 `tool_calls` 统计。
- 设置状态为 tool-use。
- 对 read/edit/write 进入 tool group 或直接渲染 tool call。

引入 permission 后需要避免“未批准工具像已运行一样出现在输出框”。

建议：

| 阶段 | UI 行为 |
|---|---|
| permission pending | 只设置 bottom status，如 `Awaiting approval: edit file ...` |
| user approved | 再调用现有 `render_tool_call` / tool group `start_call` |
| user denied | 不渲染真实 tool call header，最多渲染一条非 transcript 状态或 footer |
| tool result synthetic denied | `after_tool` 不输出完整 denied content 到 scrollback，避免把确认反馈刷屏 |

实现上可以在 `ToolPermissionPrompter` enqueue 时直接让 `AetherApp` 管 overlay，不通过 `CLIUIMiddleware.before_tool()` 展示。

## 测试

新增或扩展 `tests/cli/test_cli_app.py`：

```python
async def test_permission_request_renders_above_input_not_scrollback(): ...
async def test_permission_enter_accept_once_resolves_future(): ...
async def test_permission_down_enter_accept_session_resolves_future(): ...
async def test_permission_esc_rejects_active_request(): ...
async def test_permission_queue_renders_one_prompt_at_a_time(): ...
async def test_permission_prompt_blocks_regular_enter_submit(): ...
async def test_footer_mentions_approval_when_permission_pending(): ...
```

新增 `tests/cli/test_permission_prompter_bridge.py`：

```python
def test_bridge_uses_call_soon_threadsafe(): ...
def test_bridge_timeout_returns_deny_decision(): ...
def test_shutdown_aborts_pending_permission_futures(): ...
```

## 验收门

- 确认 overlay 不使用 `console.print()`。
- 确认 overlay 不进入 transcript、stream buffer 或 Rich scrollback。
- engine worker thread 等待 decision 时 UI 仍可重绘。
- ESC 在确认 overlay 激活时只取消确认，不清历史，不误触发其他优先级。
- 多个 permission request 按 queue 顺序处理。

