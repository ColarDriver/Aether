# PR 9.3 — `write_file` Permission Modal Fallback Fix

## 目标 / Goal

修复截图 `image copy 11.png` 中的退化：`Overwrite file` 权限弹框在某些边缘情况下展示原始 JSON `{"path": "...", "content": "\"\"\"..."}`，而 Claude Code 的对照（`image copy 9.png`）一直展示一份格式化的 unified diff。

让权限弹框具备 **"永远不展示原始 JSON"** 的保证 —— 即使工具的 preview 不完整，也至少要展示一条 human-readable body。

不依赖 PR 9.1。

## 当前问题 / Current Problem

### 1. `_build_diff` 在边缘情况返回空字符串

`aether/tools/builtins/write_file.py:225-241`：

```python
@staticmethod
def _build_diff(plan: WriteFilePlan) -> str:
    diff_lines = list(
        difflib.unified_diff(
            plan.old_content.splitlines(keepends=True),
            plan.new_content.splitlines(keepends=True),
            fromfile=str(plan.path) if plan.existed else "/dev/null",
            tofile=str(plan.path),
            n=2,
        )
    )
    ...
    return "".join(diff_lines)
```

`difflib.unified_diff` 在以下情况返回空迭代器：

- `plan.old_content == plan.new_content`（覆写同内容）
- 两侧都是空字符串
- 极少数纯空白调整且 `n=0` 时

返回 `""`。

### 2. Wire 上传一个 `diff=""` 的 preview

`aether/tools/builtins/write_file.py:91-101`：

```python
return ToolPermissionPreview(
    title="Overwrite file" if plan.existed else "Create file",
    subtitle=str(plan.path),
    path=str(plan.path),
    diff=self._build_diff(plan),       # 可能是 ""
    metadata={
        "existed": plan.existed,
        "size_bytes": plan.size_bytes,
        "parent_exists": plan.path.parent.exists(),
    },
)
```

`body` 没填，于是 TUI 端收到 `{diff: "", body: null, command: null}`。

### 3. TS 端的 falsy 判断退化到 JSON dump

`tui/src/overlays/PermissionModal.tsx:218-253`：

```typescript
if (preview?.diff) { return <DiffView ... /> }   // "" 是 falsy → 跳过
if (preview?.command) { return <... /> }          // null → 跳过
if (preview?.body) { return <... /> }             // null → 跳过
// fallback
const args = JSON.stringify(request.arguments ?? {}, null, 2)
return <Text>{truncate(args, 600)}</Text>
```

这就是截图里看到的 `{"path": "...", "content": "\"\"\"..."}`。

## 改动 / Changes

### 1. Python：`write_file.build_permission_preview` 三种 case 全覆盖

**`aether/tools/builtins/write_file.py:82-101`** 改为：

```python
def build_permission_preview(
    self,
    call: ToolCall,
    context: TurnContext,
) -> ToolPermissionPreview | ToolResult:
    plan = self.plan_write(call)
    if isinstance(plan, ToolResult):
        return plan
    # Case A: 同内容写入 —— 视为 no-op，直接返回 ToolResult，不弹权限框
    if plan.existed and plan.old_content == plan.new_content:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"no-op: {plan.path} already has identical content",
            is_error=False,
            metadata={
                "path": str(plan.path),
                "no_op": True,
                "size_bytes": plan.size_bytes,
                "existed": True,
            },
        )
    context.metadata.setdefault("_tool_permission_preview_plans", {})[call.id] = plan
    diff_text = self._build_diff(plan)
    # Case B: 有 diff —— 走 diff 分支
    # Case C: 无 diff（极少数情况，例如新建空文件）—— 用 body 兜底
    body = None
    if not diff_text:
        body = self._fallback_body(plan)
    return ToolPermissionPreview(
        title="Overwrite file" if plan.existed else "Create file",
        subtitle=str(plan.path),
        path=str(plan.path),
        diff=diff_text or None,            # 显式 None 而非空串
        body=body,
        metadata={
            "existed": plan.existed,
            "size_bytes": plan.size_bytes,
            "parent_exists": plan.path.parent.exists(),
            "old_bytes": len(plan.old_content.encode("utf-8")),
            "new_bytes": len(plan.new_content.encode("utf-8")),
        },
    )

@staticmethod
def _fallback_body(plan: WriteFilePlan) -> str:
    verb = "Overwrite" if plan.existed else "Create"
    line_count = plan.new_content.count("\n") + (0 if not plan.new_content else 1)
    return (
        f"{verb} {plan.path}\n"
        f"size: {plan.size_bytes} bytes ({line_count} lines)"
    )
```

### 2. Python：相同改动应用到 `file_edit`

**`aether/tools/builtins/file_edit.py`** —— 在 `build_permission_preview`（约 124 行附近）中，对 `_build_diff(plan)` 返回空字符串的情况同样补 body。`file_edit` 已经在 `plan_edit` 阶段拒绝 no-op（`old_string == new_string` 检查），所以 Case A 不需要。但 Case C 仍然需要兜底 body。

### 3. TS：把 falsy 判断改成 `trim()`-aware

**`tui/src/overlays/PermissionModal.tsx:218-253`** —— `PreviewBody` 函数收紧：

```typescript
function PreviewBody({
  preview,
  request,
}: {
  preview: PermissionPreview | null
  request: PermissionToolRequest
}): ReactElement | null {
  if (preview?.diff && preview.diff.trim().length > 0) {
    return (
      <Box marginTop={1} flexDirection="column">
        <DiffView diff={preview.diff} expanded={false} />
      </Box>
    )
  }
  if (preview?.command && preview.command.trim().length > 0) {
    return <ShellCommandPreview command={preview.command} />  // PR 9.4 引入
  }
  if (preview?.body && preview.body.trim().length > 0) {
    return (
      <Box marginTop={1} flexDirection="column">
        {preview.body.split('\n').slice(0, 8).map((line, idx) => (
          <Text key={idx} {...theme.colorProps('dim')}>{line || ' '}</Text>
        ))}
      </Box>
    )
  }
  // Fall back —— 这里现在不应该被触发；触发即为 bug，打 telemetry
  reportPermissionPreviewFallback({
    toolName: request.tool_name,
    hasPreview: preview !== null,
    previewKeys: preview ? Object.keys(preview).filter((k) => preview[k as keyof PermissionPreview] != null) : [],
  })
  const args = JSON.stringify(request.arguments ?? {}, null, 2)
  return (
    <Box marginTop={1}>
      <Text {...theme.colorProps('dim')}>{truncate(args, 600)}</Text>
    </Box>
  )
}
```

### 4. TS：新增轻量遥测 hook

**新文件 `tui/src/lib/permissionTelemetry.ts`**：

```typescript
import { gatewayClient } from '../gatewayClient.js'

export function reportPermissionPreviewFallback(payload: {
  toolName: string
  hasPreview: boolean
  previewKeys: string[]
}): void {
  // notifications.telemetry 是单向 fire-and-forget；失败不影响 UX
  try {
    gatewayClient.notify('telemetry.event', {
      kind: 'permission_preview_fallback',
      payload,
    })
  } catch {
    // 静默丢弃 —— telemetry 永远不能 break 权限流
  }
}
```

如果 gateway 没有 `telemetry.event` 通知方法，临时退化为 `console.error` —— 至少在开发模式下能看到。最终落地需要在 gateway 端加一个 ignore-list 的 notification 处理器（5 行）；这部分在 PR 9.3 中作为 minor 兜底，**不阻塞主 PR**。

### 5. Pydantic 校验保险

**`aether/gateway/protocol.py:296-307`** —— `PermissionPreview` 的字段保持 optional，但加一个 `model_validator`：

```python
@model_validator(mode='after')
def _at_least_one_body_channel(self) -> 'PermissionPreview':
    has_visual = bool(
        (self.diff and self.diff.strip())
        or (self.command and self.command.strip())
        or (self.body and self.body.strip())
    )
    if not has_visual:
        # 不抛错（已经在 wire 上，无法 raise） —— 写入默认 body 兜底
        self.body = f"{self.title}{(': ' + self.subtitle) if self.subtitle else ''}"
    return self
```

这样即使工具忘了填，wire 校验阶段也会塞一条 title-based body 防止 JSON fallback。

## 测试 / Tests

### Python

扩展 `aether/tests/tools/test_file_permission_preview.py`：

- `test_write_file_identical_content_short_circuits_to_no_op_result` —— `plan.old_content == plan.new_content` 时 `build_permission_preview` 返回 `ToolResult`（不是 preview），`metadata['no_op'] is True`
- `test_write_file_empty_new_content_emits_body_when_no_diff` —— 创建空文件，`preview.diff is None`, `preview.body` 含 `"Create"` 与 path
- `test_write_file_normal_overwrite_keeps_diff_branch` —— 旧+新不同，`preview.diff` 非空，`preview.body is None`
- `test_file_edit_empty_diff_emits_body_fallback` —— 构造一个 `file_edit` no-op（虽然 plan_edit 应该拒绝，但仍 verify body 兜底逻辑）

新建 `aether/tests/gateway/test_permission_preview_validator.py`：

- `test_preview_with_all_empty_channels_gets_synthetic_body` —— 构造一个 `diff=""`, `body=None`, `command=None` 的 PermissionPreview，`model_validator` 后 `body` 不为空

### TS

`tui/src/__tests__/permissionModal.test.tsx`：

- `does NOT fall back to JSON dump when diff is empty string but body is present` —— preview `{diff: '', body: 'Overwrite /x ...'}` 渲染 body 不渲染 JSON
- `does NOT fall back to JSON dump when diff is whitespace-only` —— `preview.diff = '   \n  '` 应被视为空
- `reports telemetry when truly falling back to JSON dump` —— mock telemetry，构造一个 `{preview: null}` 的请求，断言 `reportPermissionPreviewFallback` 被调用一次

## 验收 / Acceptance

- `uv run pytest aether/tests/tools aether/tests/gateway` 全绿
- `npm test --prefix tui` 全绿
- 手动场景：
  1. 让模型 `write_file` 写一个已存在且内容完全相同的文件 —— 不弹权限框，直接得 `no-op` result
  2. 让模型 `write_file` 创建一个全新空文件 —— 弹框显示 `Create /path` + `size: 0 bytes (0 lines)`
  3. 让模型 `write_file` 覆写一个文件成完全不同的内容 —— 弹框显示彩色 unified diff（旧行为不退化）
  4. 让模型 `write_file` 覆写为完全相同内容（如果跳过了 no-op 短路）—— 弹框显示 body fallback，**不再有 JSON dump**
- 跑一次 demo 流程：grep gateway log 中 `permission_preview_fallback` 应为 0

## 不在本 PR

- 调整 `PermissionPreview` 字段的 wire 形状（保持 additive；新增字段留给未来）
- 修改 `file_edit` 的 no-op 检测（已存在，但行为不变）
- 多文件 batch preview
