# PR 7.4 - File Edit / Write Diff Preview

## 目标

让 `file_edit`、`write_file`、`notebook_edit` 在执行前生成安全 preview，并在 permission overlay 展示 diff。拒绝时必须证明没有修改磁盘。

## 当前问题

`FileEditTool.execute()` 当前流程是：解析参数、读取文件、替换、写回、构建 diff summary、返回结果。这个流程不能直接用于确认 UI，因为调用 `execute()` 预览会提前修改文件。

`WriteFileTool.execute()` 同理，会立刻创建目录并原子写入。确认前不能调用。

## Preview Builder Interface

建议新增 runtime helper 或工具 mixin：

```python
class ToolPreviewProvider(Protocol):
    def build_permission_preview(
        self,
        call: ToolCall,
        context: TurnContext,
    ) -> ToolPermissionPreview | ToolResult: ...
```

返回 `ToolPermissionPreview` 表示预览成功，返回 `ToolResult` 表示参数或文件校验失败。校验失败时不需要问用户，直接走正常 error result。

更保守的做法是在 `runtime/tool_permissions.py` 写 registry 外部 preview builder：

```python
def build_tool_permission_preview(
    executor: ToolExecutor,
    call: ToolCall,
    context: TurnContext,
) -> ToolPermissionPreview | ToolResult | None: ...
```

这样不要求所有工具继承新基类。

## file_edit Preview

把 `FileEditTool.execute()` 拆出纯函数：

```python
@dataclass(slots=True, frozen=True)
class FileEditPlan:
    path: Path
    original: str
    modified: str
    diff: str
    change_count: int
```

建议方法：

```python
def plan_edit(self, call: ToolCall) -> FileEditPlan | ToolResult: ...
def apply_plan(self, call: ToolCall, plan: FileEditPlan) -> ToolResult: ...
```

`plan_edit()` 执行所有当前校验，但不写文件：

| 校验 | 保持 |
|---|---|
| `path` 非空 | 是 |
| `old_string` 非空 | 是 |
| `new_string` 是 str | 是 |
| spill root 拒绝 | 是 |
| file exists / not dir | 是 |
| 1 GiB cap | 是 |
| UTF-8 decode | 是 |
| no-op 拒绝 | 是 |
| occurrence count | 是 |

确认通过后，`execute()` 可以复用 `plan_edit()` 再写入。为避免 TOCTOU，写入前需要重新读取并确认 original 未变化：

```python
current = path.read_text(encoding="utf-8")
if current != plan.original:
    return _error(call, "file changed after permission preview; please retry")
```

## write_file Preview

`WriteFileTool` 需要在确认前展示 create/overwrite diff：

| 场景 | preview |
|---|---|
| 新文件 | diff from empty to content |
| 覆盖已有文件 | diff from old content to new content |
| 父目录不存在 | 标记将创建 parent dirs |
| content 超限 | 直接 error，不问 |
| 非 UTF-8 不是问题 | content 已是 str |

建议：

```python
@dataclass(slots=True, frozen=True)
class WriteFilePlan:
    path: Path
    existed: bool
    old_content: str
    new_content: str
    diff: str
    size_bytes: int
```

同样写入前做 TOCTOU 校验：

| 场景 | 校验 |
|---|---|
| preview 时文件不存在 | 写入前仍不存在，否则返回 stale preview error |
| preview 时文件存在 | 写入前内容仍等于 old_content，否则返回 stale preview error |

## Diff 格式

使用 `difflib.unified_diff`，限制行数，避免 overlay 太高：

```python
def build_unified_diff(old: str, new: str, *, fromfile: str, tofile: str) -> str:
    ...
```

截断策略：

| 字段 | 建议 |
|---|---|
| max lines | 120 |
| max chars | 12_000 |
| truncate marker | `... N more diff lines elided ...` |

CLI overlay 不要把完整 diff 放进 scrollback。只渲染当前 prompt 的 preview。

## notebook_edit

首版可以只展示 cell 级 summary，不必生成完整 `.ipynb` diff：

| 字段 | 内容 |
|---|---|
| title | `Edit notebook` |
| subtitle | notebook path |
| body | cell index、cell type、operation |
| diff | 对 cell source 生成局部 diff |

如果当前 `NotebookEditTool` 结构复杂，可以先纳入 PR 7.5/7.6 验收，但危险工具仍必须 ask。

## 测试

新增 `tests/tools/test_file_permission_preview.py`：

```python
def test_file_edit_preview_builds_diff_without_writing(): ...
def test_file_edit_reject_path_leaves_file_unchanged(): ...
def test_file_edit_accept_path_writes_once(): ...
def test_file_edit_stale_file_after_preview_returns_error(): ...
def test_write_file_preview_new_file_without_creating_parent(): ...
def test_write_file_reject_does_not_create_file_or_parent(): ...
def test_write_file_accept_creates_file_after_permission(): ...
def test_write_file_stale_existing_file_returns_error(): ...
def test_large_diff_is_truncated_in_preview(): ...
```

新增 engine 集成测试：

```python
def test_rejected_file_edit_does_not_mutate_disk_through_engine(): ...
def test_rejected_write_file_does_not_create_file_through_engine(): ...
def test_accept_session_skips_second_file_edit_prompt(): ...
```

## 验收门

- preview 构建没有任何磁盘写入。
- 拒绝 `file_edit` / `write_file` 后文件系统完全不变。
- 批准后只执行一次写入，不出现 preview 执行和真实执行的双写。
- 文件在 preview 和写入之间变化时拒绝执行并要求重试。
- diff 可读、截断可控，不污染输出框。

