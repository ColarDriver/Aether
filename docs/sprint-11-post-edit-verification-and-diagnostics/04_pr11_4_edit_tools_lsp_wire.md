# PR 11.4 — Edit Tools 接入 LSP + Tracker

## 目标 / Goal

让 `file_edit` / `write_file` / `notebook_edit` 三个写工具在 *成功执行后* 自动：

1. 在 *写* 之前调 `DiagnosticTracker.before_file_edited(path)` 抓 baseline。
2. 在 *写* 之后通过 PR 11.2 的 `post_tool_use` hook 触发 `DiagnosticTracker.notify_file_changed(path, content)`（异步 fire-and-forget）。
3. 所有动作均 *附加* 在工具原行为之上，**绝不**因 LSP 不可用、超时、抛异常而拒绝原本会成功的写操作。

参考 `open-claude-code/src/tools/FileEditTool/FileEditTool.ts:425,493-514` 与 `src/tools/FileWriteTool/FileWriteTool.ts:247,307-326`。

## 当前问题 / Current Problem

### 1. `file_edit.py` 与 `write_file.py` 完全没有 LSP / diagnostic 调用

直接 grep：

```bash
$ grep -n "lsp\|diagnostic" aether/tools/builtins/file_edit.py
# (only a comment about "verify the change without re-reading the file")

$ grep -n "lsp\|diagnostic" aether/tools/builtins/write_file.py
# (no matches)
```

—— 两个最常用的写工具与诊断回路 *零集成*。

### 2. 工具直接持有 LSPManager 会污染单一职责

错误做法：在 `file_edit.py` 里 import `LSPManager` 并自己调 `change_file` / `save_file`。这会让工具承担 *两套不同的副作用*（文件 IO + LSP 状态），未来 unit test 必须 mock 两条路径，且 `LSPTool`（已存在）与 `file_edit` 的 LSP 使用方式会快速漂移。

**正确做法**：工具只负责
- 写之前调 `tracker.before_file_edited(path)`（同步、无副作用、无依赖泄漏）
- 在 `ToolResult.metadata` 写入 `edited_paths: list[str]`

然后 PR 11.2 的 `post_tool_use` hook 在 engine 侧根据 metadata 调度 LSP 通知 —— 工具本身不持有 LSPManager。

### 3. `notebook_edit.py` 同样需要

虽然 `.ipynb` 文件本身 LSP 不支持，但 OCC 也把 notebook 走相同管道（语言服务器看不懂 cell 就直接返回空诊断）；保持 *所有写工具一致行为* 比省一点代码更重要。

## 改动 / Changes

### 1. 工具 base 类 / 协议

`aether/tools/base.py` 已有 `ToolDescriptor` / `ToolCall` / `ToolResult`。`ToolResult.metadata` 是 `dict[str, Any]`，无需 schema 变更。新增约定（写入 README 但不需类型约束）：

> 写文件类工具应在 `ToolResult.metadata["edited_paths"]` 里写入它实际修改的文件**绝对路径列表**。Engine 会用这个键发现需要触发 LSP 诊断的目标。

### 2. `aether/tools/builtins/file_edit.py`

文件结构里找到执行编辑的入口（约 `async def call`）。改动：

```python
# 头部
from aether.runtime.diagnostics import DiagnosticTracker

# 在 call 内部，定位到读取文件 + 即将写入之前
tracker: DiagnosticTracker | None = context.metadata.get("_diagnostic_tracker")

# resolved_path 是即将写入的目标绝对路径
if tracker is not None and tracker.enabled:
    tracker.before_file_edited(resolved_path)

# (写入文件)
self._write(resolved_path, new_content)

# 返回结果时附带 metadata
return ToolResult(
    text=diff_text,
    metadata={
        **(existing_metadata or {}),
        "edited_paths": [str(resolved_path)],
    },
)
```

`context.metadata["_diagnostic_tracker"]` 由 engine 在每次 turn 开头注入（与 `_lsp_manager` 同款）：

```python
# agent.py 里 _prepare_turn_context 或类似位置（line ~1532 附近）
context.metadata["_lsp_manager"] = getattr(self, "_lsp_manager", None)
context.metadata["_diagnostic_tracker"] = getattr(self, "_diagnostic_tracker", None)
```

### 3. `aether/tools/builtins/write_file.py`

同样改造：

```python
tracker = context.metadata.get("_diagnostic_tracker")
if tracker is not None and tracker.enabled:
    tracker.before_file_edited(resolved_path)
self._write(resolved_path, content)
return ToolResult(text=..., metadata={..., "edited_paths": [str(resolved_path)]})
```

### 4. `aether/tools/builtins/notebook_edit.py`

```python
tracker = context.metadata.get("_diagnostic_tracker")
if tracker is not None and tracker.enabled:
    tracker.before_file_edited(notebook_path)
self._apply_cell_edit(...)
return ToolResult(..., metadata={..., "edited_paths": [str(notebook_path)]})
```

### 5. Engine `post_tool_use` hook 安装

`aether/agents/core/agent.py` 提供一个内置 hook（**不是** 用户写的 hook，是 engine 自己的），其唯一职责是在 tool 返回后调度 LSP didChange + didSave + 等诊断采集：

```python
# 新文件：aether/agents/core/internal_hooks.py
"""Built-in EngineHooks used by the engine itself.

These cannot be replaced by user-supplied hooks — they live on a
*separate* hook channel that always fires.  User hooks are still
invoked via the existing ``self._hooks`` field.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolResult
from aether.runtime.diagnostics import DiagnosticTracker


class DiagnosticDispatchHook:
    """Schedule LSP didChange/didSave for any file the tool edited."""

    def __init__(
        self,
        tracker: DiagnosticTracker | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._tracker = tracker
        self._loop = loop

    def post_tool_use(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolResult,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        if self._tracker is None or not self._tracker.enabled:
            return
        edited = (result.metadata or {}).get("edited_paths") or []
        if not edited:
            return
        for raw in edited:
            path = Path(raw)
            content = _read_or_none(path)
            if content is None:
                continue
            asyncio.run_coroutine_threadsafe(
                self._tracker.notify_file_changed(path, content),
                self._loop,
            )

    def post_tool_use_failure(self, **_: Any) -> None:
        return None


def _read_or_none(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None
```

在 `AgentEngine.__init__` 末尾：

```python
self._diagnostic_hook = DiagnosticDispatchHook(
    self._diagnostic_tracker,
    asyncio.get_event_loop(),
)
```

主循环里在调用 `self._safe_post_tool_use(...)`（PR 11.2 引入）之后 *再调一次* 内部 hook：

```python
self._safe_post_tool_use(...)  # user hooks
if self._diagnostic_hook is not None:
    try:
        self._diagnostic_hook.post_tool_use(
            session_id=session_id,
            iteration=iteration,
            tool_call_id=call.id,
            tool_name=call.name,
            tool_args=call.args or {},
            result=result,
            elapsed_ms=elapsed_ms,
            context_metadata=context.metadata,
        )
    except Exception:
        self.services.logger.exception("internal_hook.diagnostic_dispatch")
```

—— 内置 hook 独立于用户 hook：用户写自己的 `EngineHooks` 子类不会替换掉诊断分发逻辑。

### 6. `LSPTool` 行为不变

`tools/builtins/lsp.py` 不动。模型显式调 `lsp` 工具仍然能查到诊断（包括 *尚未通过本 PR 自动注入* 的诊断），与 OCC 的 `mcp__ide__getDiagnostics` 并行存在。

## 测试 / Tests

新建 `aether/tests/tools/builtins/test_file_edit_diagnostic_wire.py`：

- `test_file_edit_calls_before_file_edited` —— mock tracker；执行一次 `file_edit` → `tracker.before_file_edited(p)` 被调用且 p 是绝对路径。
- `test_file_edit_emits_edited_paths_metadata` —— 成功执行 → `result.metadata["edited_paths"] == [str(absolute_p)]`。
- `test_file_edit_works_without_tracker` —— `context.metadata["_diagnostic_tracker"] is None` → 工具行为完全不变；不抛错。
- `test_file_edit_does_not_fire_lsp_directly` —— 工具内部不持有 LSPManager 引用（white-box：assert grep "lsp_manager" 在 file_edit.py 内 0 命中）。
- `test_write_file_carries_edited_paths` —— 同上对 `write_file`。
- `test_notebook_edit_carries_edited_paths` —— 同上对 `notebook_edit`。

新建 `aether/tests/agents/test_internal_diagnostic_hook.py`：

- `test_dispatch_hook_calls_notify_file_changed` —— mock tracker；engine 跑一个 `file_edit` 成功 → tracker.`notify_file_changed(p, content)` 被调度。
- `test_dispatch_hook_skips_when_no_edited_paths` —— result.metadata 空 → tracker 不被触碰。
- `test_dispatch_hook_skips_when_tracker_disabled` —— `DiagnosticTracker(None)` → 即使工具产 metadata，也无副作用。
- `test_dispatch_hook_failure_does_not_propagate` —— mock `notify_file_changed` 永久 hang → 主循环继续推进；ScheduledFuture 被 GC 时不引发 unhandled exception 警告（用 `pytest.warns` 反验证）。

新建 `aether/tests/integration/test_edit_to_diagnostic_pipeline.py`（mock LSP 端到端，不依赖真 pyright）：

- `test_edit_then_baseline_then_new_diagnostic_visible` —— mock LSPManager 返回的诊断在 edit 前是 `[A]`，edit 后变 `[A, B]` → 紧接着 `tracker.get_new_diagnostics([p])` 返回仅含 `B`。
- `test_two_consecutive_edits_baseline_resets` —— 第一次 edit 引入 `B` → 第二次 edit 引入 `C` → 第二次 `get_new_diagnostics` 仅返回 `C`（baseline 已重置）。

## 验收 / Acceptance

- `uv run pytest aether/tests/tools/builtins/test_file_edit_diagnostic_wire.py aether/tests/agents/test_internal_diagnostic_hook.py aether/tests/integration/test_edit_to_diagnostic_pipeline.py` 全绿。
- 既有 `aether/tests/tools/builtins/test_file_edit.py` / `test_write_file.py` / `test_notebook_edit.py` 零回归。
- `uv run pyright` 零新增告警。
- 真实 pyright 集成手测：在 Aether 自己仓库里跑 `uv run aether`，prompt "rename `_format_skill_listing` to `_render_skill_listing` in skill_listing.py" → 改完后 `_diagnostic_tracker.get_new_diagnostics()` 应返回该文件因仍有旧名调用而产生的 NameError 诊断（即使诊断 attachment 还没注入到 prompt，可在 log 中 verify）。
- 性能：mock LSP 立即响应时一次 edit → tracker round-trip ≤ 20ms（含 file read）。

## 不在本 PR / Deferred to other PRs

- **PR 11.5** 才把这些诊断注入给模型；本 PR 只完成"采集"，模型仍看不到。
- **多文件批量 edit**：现阶段 `file_edit` 一次一文件；如未来引入 `MultiEdit` 工具，复用本 PR 的 `edited_paths` metadata 数组即可。
- **写入后 syntax check 兜底**：当工程没装 LSP 时如何给模型至少看到 Python `SyntaxError`？OCC 用 Bun + TS 自带 parser；Aether 的简化方案是让 `verifier` 子 agent（PR 11.6）默认带 `python -m compileall` 命令。本 PR 不做这层兜底。
