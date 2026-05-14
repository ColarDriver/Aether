# PR 9.1 — Forward `ToolResult.metadata` over the Wire

## 目标 / Goal

让工具运行后产生的结构化元数据（行数变化、文件大小、exit code、耗时…）能够从 Python runtime 一路送达 TUI，使 PR 9.2 / 9.4 能基于这些字段渲染 `● Edited X (+N −M)` 与 `[exit 0 · 12ms]` 等 Claude Code 风格的展示。

不改变现有 wire 形状的兼容性：纯 additive 扩展。

## 当前问题 / Current Problem

### 1. Wire `ToolResult` 缺 `metadata`

`aether/gateway/protocol.py:216-222`：

```python
class ToolResult(AgentEventBase):
    type: Literal["tool.result"] = "tool.result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    iteration: int
```

### 2. 中间件丢弃 `result.metadata`

`aether/gateway/handlers/agent_methods.py:226-239`：

```python
def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
    self._sink.emit(
        ToolResultEvent(
            session_id=self._sink.session_id,
            run_id=self._sink.run_id,
            tool_call_id=result.tool_call_id,
            tool_name=result.name,
            content=result.content,
            is_error=bool(result.is_error),
            iteration=_wire_iteration(context),
            # result.metadata 在这里被丢掉
        )
    )
```

### 3. 工具内部的元数据其实已经齐全

- `aether/tools/builtins/write_file.py:211-216` 写入 `metadata={"path", "size_bytes", "sha256", "existed"}`
- `aether/tools/builtins/file_edit.py`（搜索 `metadata=`）写入 `{"change_count", "replace_all", "bytes_before", "bytes_after"}`
- `aether/tools/builtins/shell.py:232-258` `_format_output` 持有 `exit_code` / `duration_ms` / `truncated`，但只把它格式化进 `content` 文本

—— 这些信息全部已知，只缺一条出门的路。

## 改动 / Changes

### 1. 扩展 wire schema

**`aether/gateway/protocol.py:216-222`**：

```python
class ToolResult(AgentEventBase):
    type: Literal["tool.result"] = "tool.result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    iteration: int
    metadata: dict[str, Any] = Field(default_factory=dict)
```

`Field(default_factory=dict)` 保证旧客户端解码不需要变化；新字段默认空。

### 2. 中间件转发

**`aether/gateway/handlers/agent_methods.py:226-239`**：

```python
def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
    self._sink.emit(
        ToolResultEvent(
            session_id=self._sink.session_id,
            run_id=self._sink.run_id,
            tool_call_id=result.tool_call_id,
            tool_name=result.name,
            content=result.content,
            is_error=bool(result.is_error),
            iteration=_wire_iteration(context),
            metadata=_safe_metadata(result.metadata),
        )
    )
```

新增 helper（同文件内，靠近其他 `_wire_*` helper）：

```python
def _safe_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop non-JSON-serialisable values so the wire never fails on encode."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[str(k)] = v
    return out
```

### 3. 统一标准元数据键 / Standard Keys

为了让 TUI 端不必区分工具来写适配代码，所有内置工具应当填入以下命名（缺则不填，TUI 端按 optional 处理）：

| 适用工具 | Key | 类型 | 含义 |
|---|---|---|---|
| `write_file`, `file_edit` | `path` | str | 操作的绝对路径 |
| `write_file`, `file_edit` | `bytes_before` | int | 修改前字节数（新建时 0） |
| `write_file`, `file_edit` | `bytes_after` | int | 修改后字节数 |
| `write_file`, `file_edit` | `lines_added` | int | diff 中 `+` 行计数 |
| `write_file`, `file_edit` | `lines_removed` | int | diff 中 `−` 行计数 |
| `write_file`, `file_edit` | `hunks` | int | hunk 数量 |
| `write_file`, `file_edit` | `diff` | str（可选） | 完整 unified diff，长度 < 4 KiB 时填入；超长则省略，TUI 按 `path` 重读 |
| `file_edit` | `change_count` | int | 已存在；保留 |
| `file_edit` | `replace_all` | bool | 已存在；保留 |
| `write_file` | `existed` | bool | 已存在；保留 |
| `write_file` | `sha256` | str | 已存在；保留 |
| `shell` | `exit_code` | int | 进程退出码 |
| `shell` | `duration_ms` | int | 墙钟耗时 |
| `shell` | `cwd` | str | 实际工作目录 |
| `shell` | `truncated` | bool | 输出是否被截断 |
| `shell` | `timed_out` | bool | 是否超时 |
| `shell` | `stderr_lines` | int | stderr 行数（用于 footer） |

### 4. 工具内部填字段

**`aether/tools/builtins/write_file.py`** —— 在 `_apply_plan` 返回 `ToolResult` 处（约 206-217 行）扩展 metadata：

```python
diff_text, lines_added, lines_removed, hunks = self._build_diff_with_stats(plan)
return ToolResult(
    tool_call_id=call.id,
    name=call.name,
    content=body,
    is_error=False,
    metadata={
        "path": str(path),
        "bytes_before": len(plan.old_content.encode("utf-8")),
        "bytes_after": len(encoded),
        "size_bytes": len(encoded),
        "sha256": digest,
        "existed": plan.existed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "hunks": hunks,
        **({"diff": diff_text} if diff_text and len(diff_text) < 4096 else {}),
    },
)
```

把现有 `_build_diff` 重命名为 `_build_diff_text` 并新增 `_build_diff_with_stats`：

```python
@staticmethod
def _build_diff_with_stats(plan: WriteFilePlan) -> tuple[str, int, int, int]:
    raw_lines = list(difflib.unified_diff(
        plan.old_content.splitlines(keepends=True),
        plan.new_content.splitlines(keepends=True),
        fromfile=str(plan.path) if plan.existed else "/dev/null",
        tofile=str(plan.path),
        n=2,
    ))
    added = sum(1 for line in raw_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in raw_lines if line.startswith("-") and not line.startswith("---"))
    hunks = sum(1 for line in raw_lines if line.startswith("@@"))
    # 同 _build_diff 现有逻辑：超过 _DIFF_PREVIEW_LINES 时截断
    if len(raw_lines) > _DIFF_PREVIEW_LINES:
        elided = len(raw_lines) - _DIFF_PREVIEW_LINES
        raw_lines = raw_lines[:_DIFF_PREVIEW_LINES] + [
            f"\n... ({elided} more diff lines elided) ...\n"
        ]
    return "".join(raw_lines), added, removed, hunks
```

**`aether/tools/builtins/file_edit.py`** —— 同样的 `_build_diff_with_stats` 模式，并在 execute 返回的 `ToolResult.metadata` 中加 `path` / `bytes_before` / `bytes_after` / `lines_added` / `lines_removed` / `hunks` / 可选 `diff`。

**`aether/tools/builtins/shell.py`** —— 在 `execute` 返回 `ToolResult` 处把已有的 `exit_code` / `duration_ms` / `cwd` / `truncated` / `timed_out` 写进 metadata（这些值在 `_format_output` 周边已存在，无需新算）。同时把 stderr 行数从 `_format_output` 头部解析下来塞进去。

### 5. TS wire 类型

**`tui/src/gatewayTypes.ts`** —— 找到 `tool-result` / `ToolResultEvent` 的定义（或 `JsonObject` 复用的事件 union），扩展：

```typescript
export interface ToolResultEvent {
  type: 'tool.result'
  session_id: string
  run_id: string
  tool_call_id: string
  tool_name: string
  content: string
  is_error: boolean
  iteration: number
  metadata?: ToolResultMetadata
}

export interface ToolResultMetadata {
  // file tools
  path?: string
  bytes_before?: number
  bytes_after?: number
  lines_added?: number
  lines_removed?: number
  hunks?: number
  diff?: string
  change_count?: number
  // shell
  exit_code?: number
  duration_ms?: number
  cwd?: string
  truncated?: boolean
  timed_out?: boolean
  stderr_lines?: number
  // 其它工具的自由字段
  [key: string]: unknown
}
```

### 6. ChatStore 透传

**`tui/src/store/chatStore.ts`** —— `ChatItem.tool-call.result` 现状为 `{ text, isError }`，扩展为 `{ text, isError, metadata?: ToolResultMetadata }`。`addToolResult`（约 215-245 行）把 metadata 一并附上。**注意只动 store；UI 行为变化留给 PR 9.2 / 9.4。**

## 测试 / Tests

### Python

新建 `aether/tests/gateway/test_tool_result_wire.py`：

- `test_metadata_round_trips_through_event` — 构造一个带 `metadata={"path": "/x", "exit_code": 0}` 的 `runtime.ToolResult`，断言 `_GatewayAgentHooks.after_tool` 触发的 `ToolResultEvent.metadata` 等值。
- `test_non_serialisable_values_dropped` — metadata 含 `set()` 或 lambda 时不抛错，结果中没有该键。
- `test_empty_metadata_defaults_to_dict` — `runtime.ToolResult` 不传 metadata 时 event metadata 为 `{}`。

扩展现有 `aether/tests/tools/test_file_permission_preview.py`（或新建 `test_file_tool_metadata.py`）：

- 写入新内容到不存在的文件 → metadata 含 `lines_added>0`, `lines_removed==0`, `existed==False`。
- 覆写 → `lines_added` + `lines_removed` 与 difflib 行计数一致。
- 大文件（>4 KiB diff）→ metadata 不含 `diff` 键。

### TS

**`tui/src/__tests__/gatewayClient.test.ts`** —— 已经在 working tree 里修改过；补一个 case：

- 收到带 `metadata: { exit_code: 0, duration_ms: 12 }` 的 `tool.result` 事件后，对应 `ChatItem.tool-call.result.metadata` 等值。

## 验收 / Acceptance

- `uv run pytest aether/tests/gateway/test_tool_result_wire.py aether/tests/tools` 全绿
- `npm test --prefix tui` 全绿
- `uv run pyright` / `npm run typecheck --prefix tui` 无新增告警
- 手动：跑一次 `uv run aether`，让模型读写一个文件并跑一条 shell 命令，开 verbose mode（`/help` 找到 `verboseMode`），确认 chat 中 tool-result 携带 metadata（透过 dev log 或 PR 9.2 的渲染验证）

## 不在本 PR / Deferred to other PRs

- 任何 UI 上对 metadata 的可视渲染 → PR 9.2（`● Edited X (+N −M)`）与 PR 9.4（`[exit 0 · 12ms]`）
- `write_file` 弹框的 fallback 修复 → PR 9.3
