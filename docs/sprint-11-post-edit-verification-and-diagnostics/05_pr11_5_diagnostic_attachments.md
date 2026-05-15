# PR 11.5 — Diagnostic Attachments 注入下一轮

## 目标 / Goal

把 PR 11.3 / 11.4 后台采集到的"新诊断"，在下一轮 LLM 调用 *之前* 渲染成 `<diagnostics>` 块、以 user-role 消息注入到 messages 列表里。这是让模型 *真正看见* 错误的临门一脚。

参考 `open-claude-code/src/utils/attachments.ts:956` 的 `getDiagnosticAttachments(toolUseContext)` 与 `src/utils/messages.ts:3728-3738` 的 `wrapMessagesInSystemReminder` 包装。

落地后，"写完代码看不见错"的链路就闭环了：
- PR 11.1 让模型 *愿意* 读 `<diagnostics>`。
- PR 11.3 / 11.4 让 tracker 攒着 diff。
- 本 PR 把 diff 写进下一轮 user turn。

## 当前问题 / Current Problem

### 1. 没有"per-turn attachment"机制

Aether 在 PRE_LLM 边界目前能注入两类东西：

- **system 头**：在 `_prepare_session_and_system_prompt` 时一次写死；turn 之间不变（PR 11.1 加的 verification directive 走这条）。
- **user-role 单条 nudge**：Sprint 10.1 引入的 `skill_nudge`（`agent.py:1574-1599`）—— 但它只在固定 interval 触发，与 *本轮发生过什么 tool call* 无关。

没有"根据本轮 tool 结果决定要不要补一条 system-reminder"的 attachment 框架。

### 2. 单 nudge 路径不够通用

`skill_nudge` 直接在 `agent.py` 里手写了一段：

```python
messages.append({
    "role": "user",
    "content": listing,
    "metadata": {"source": "skill_nudge"},
})
```

—— 如果再加 `<diagnostics>` 走类似手法，`agent.py` 会变成"各种 nudge 拼装中心"。本 PR 顺手抽出 `AttachmentDispatcher` 让 nudge 类可插拔。

### 3. Subagent 透传

PR 11.3 已经让 `DefaultSubagentBuilder` 传 `diagnostic_tracker`。本 PR 必须确保 child engine 也跑同一套 attachment 逻辑——否则父能看见 child 不能看见，两边对同一仓库的认知就漂移了。

## 改动 / Changes

### 1. 新模块 `aether/runtime/diagnostics/attachments.py`

```python
"""Render DiagnosticTracker output into LLM-visible attachments.

Parity with open-claude-code ``src/utils/attachments.ts:getDiagnosticAttachments``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import DiagnosticFile


_HEADER = (
    "The following diagnostics appeared after your most recent edits. "
    "These were introduced by your changes (or newly visible because of "
    "them). Fix them before reporting the task complete."
)


def render_diagnostics_block(files: list[DiagnosticFile]) -> str:
    """Render a `<diagnostics>` system-reminder block.

    Returns ``""`` if *files* is empty.
    """
    if not files:
        return ""
    lines = ["<diagnostics>", _HEADER, ""]
    for f in files:
        lines.append(f"## {f.path}")
        for d in f.diagnostics:
            code = f" [{d.code}]" if d.code else ""
            lines.append(
                f"  {d.severity.upper():<7} {d.line}:{d.column}  "
                f"{d.source}{code}: {d.message}"
            )
        lines.append("")
    lines.append("</diagnostics>")
    return "\n".join(lines)


def collect_pending_diagnostics(
    tracker: DiagnosticTracker | None,
    *,
    paths: Iterable[str] | None = None,
) -> list[DiagnosticFile]:
    """Drain any new diagnostics queued by the tracker.

    *paths* (str) is the list of files touched in the *most recent* turn,
    drawn from ``ToolResult.metadata['edited_paths']``.  If the model
    didn't edit anything this turn, returns ``[]`` even if older edits
    have unresolved diagnostics — those have already been delivered
    once (see ``DiagnosticTracker._delivered``).
    """
    if tracker is None or not tracker.enabled:
        return []
    if paths is None:
        return tracker.get_new_diagnostics()
    from pathlib import Path
    return tracker.get_new_diagnostics([Path(p) for p in paths])


__all__ = ["collect_pending_diagnostics", "render_diagnostics_block"]
```

### 2. `AttachmentDispatcher`（抽 skill_nudge + diagnostics 的共同抽象）

新文件 `aether/runtime/attachments/dispatcher.py`：

```python
"""Per-turn attachment producers, invoked before each LLM call.

Each producer returns a ``UserMessage`` dict (or ``None``).  Order is
deterministic: producers are invoked in their registration order and
their outputs are appended to ``messages`` in that order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

UserMessage = dict[str, Any]
AttachmentProducer = Callable[[dict[str, Any]], UserMessage | None]


class AttachmentDispatcher:
    def __init__(self) -> None:
        self._producers: list[tuple[str, AttachmentProducer]] = []

    def register(self, name: str, producer: AttachmentProducer) -> None:
        self._producers.append((name, producer))

    def dispatch(self, turn_state: dict[str, Any]) -> list[UserMessage]:
        out: list[UserMessage] = []
        for name, producer in self._producers:
            try:
                msg = producer(turn_state)
            except Exception:
                # Attachment producers must never block the LLM turn.
                msg = None
            if msg is not None:
                msg.setdefault("metadata", {}).setdefault("source", name)
                out.append(msg)
        return out


__all__ = ["AttachmentDispatcher", "AttachmentProducer", "UserMessage"]
```

### 3. Engine 集成

`aether/agents/core/agent.py` 的 `__init__`：

```python
self._attachments = AttachmentDispatcher()

# Skill nudge (moved out of inline code)
if self._skill_catalog is not None:
    self._attachments.register("skill_nudge", self._produce_skill_nudge)

# Diagnostics (this PR)
if self._diagnostic_tracker is not None:
    self._attachments.register("diagnostics", self._produce_diagnostic_attachment)
```

新方法：

```python
def _produce_diagnostic_attachment(self, state: dict[str, Any]) -> UserMessage | None:
    last_edited: list[str] = state.get("last_edited_paths") or []
    if not last_edited:
        # If nothing was edited this turn, still emit any orphaned
        # diagnostics from prior turns the model hasn't seen yet — but
        # only *once*: the tracker's _delivered set ensures this.
        files = collect_pending_diagnostics(self._diagnostic_tracker)
    else:
        files = collect_pending_diagnostics(
            self._diagnostic_tracker, paths=last_edited
        )
    block = render_diagnostics_block(files)
    if not block:
        return None
    return {"role": "user", "content": block}


def _produce_skill_nudge(self, state: dict[str, Any]) -> UserMessage | None:
    # (refactored from existing inline code at line 1574-1599)
    ...
```

在 PRE_LLM 边界（紧挨 `_maybe_set_skill_review` 之后、`provider.create_message` 之前）改为：

```python
turn_state = {
    "iteration": iteration,
    "last_edited_paths": context.metadata.get("_last_edited_paths") or [],
    "context_metadata": context.metadata,
}
for msg in self._attachments.dispatch(turn_state):
    messages.append(msg)
context.metadata["_last_edited_paths"] = []  # consume
```

### 4. `_last_edited_paths` 累积

`agent.py` 主循环里 *PR 11.4* 之后已经能从 `result.metadata["edited_paths"]` 拿到本次 tool call 编辑的文件。每次 tool 完成 hook 之后追加：

```python
edited = (result.metadata or {}).get("edited_paths") or []
if edited:
    context.metadata.setdefault("_last_edited_paths", []).extend(edited)
```

下次 PRE_LLM 时 dispatcher 取走并清空。

### 5. Subagent 透传

`DefaultSubagentBuilder` 无需改动 —— `diagnostic_tracker` 透传已在 PR 11.3 落地；child engine 自己构造 `AttachmentDispatcher` 并注册同样的 producer。

### 6. TUI 行为

`<diagnostics>` 块是 user-role message，但来自 engine 自己而不是用户输入。当前 TUI 已经会过滤掉 *`<system-reminder>` / `<diagnostics>` 等纯上下文标签*（通过 `tui/src/lib/phantomTool.ts` 的 `stripToolBlocks`？需要再确认）。如果还没过滤，本 PR 在 `tui/src/lib/phantomTool.ts` 里补 regex：

```ts
const PHANTOM_BLOCK_RE = /<(?:diagnostics|system-reminder|tool_use_contract|verification_directive|faithful_reporting|task-notification)>[\s\S]*?<\/\1>/g
```

—— 用户在 TUI 上看不到 `<diagnostics>` 块；它只活在上下文。

## 测试 / Tests

新建 `aether/tests/runtime/diagnostics/test_attachments.py`：

- `test_render_empty_returns_empty_string` —— `render_diagnostics_block([])` → `""`。
- `test_render_includes_path_and_diagnostics` —— 一个 file 含两条 diagnostic → 输出含路径、行列号、severity、message。
- `test_render_sorts_within_file_by_line_then_column` —— 同一文件多个 diagnostic → 按 (line, column) 排序后输出。
- `test_collect_with_paths_filters_to_those_files` —— tracker mock 返回 path A 与 B 都有新诊断 → `paths=[A]` → 输出只含 A。
- `test_collect_with_no_paths_drains_all` —— `paths=None` → 两者都被返回。

新建 `aether/tests/runtime/attachments/test_dispatcher.py`：

- `test_dispatch_preserves_registration_order` —— 注册 producer A、B → dispatch 返回 [A_msg, B_msg]。
- `test_failing_producer_does_not_block_others` —— A 抛 → 输出仅含 B 的 msg；A 的错被吞。
- `test_producer_returning_none_skipped` —— A 返 None → 输出不含 A 占位。
- `test_dispatch_sets_metadata_source` —— msg.metadata 含 `source` 字段为注册名。

新建 `aether/tests/agents/test_diagnostic_attachment_pipeline.py`（scripted provider 端到端，mock LSP）：

- `test_first_edit_then_diagnostic_appears_in_next_turn_messages` —— turn 0 模型发起一次 `file_edit` 引入 NameError → turn 1 LLM 请求的 messages 末尾含一条 user-role 消息且 content 含 `<diagnostics>` 与 `NameError`。
- `test_diagnostic_attachment_only_delivered_once` —— 模型在 turn 1 没修 → turn 2 LLM 请求的 messages 中不再追加同样诊断（已 delivered）。
- `test_resolving_diagnostic_makes_it_disappear` —— turn 1 模型 fix 了 NameError → turn 2 既无新诊断也无旧诊断（diff 后为空）。
- `test_no_attachment_when_tracker_disabled` —— `DiagnosticTracker(None)` → 即便 tools 报 `edited_paths`，messages 中不含 `<diagnostics>`。

## 验收 / Acceptance

- `uv run pytest aether/tests/runtime/diagnostics/ aether/tests/runtime/attachments/ aether/tests/agents/test_diagnostic_attachment_pipeline.py` 全绿。
- `uv run pyright` 零新增告警。
- `npm run test --prefix tui` 已有 297 个 test 仍全绿；新增 phantom 标签过滤如果触动 `phantomTool.ts` 需配套 vitest 用例。
- **关键手测**：在 Aether 仓库里 prompt "把 `aether/runtime/tools/skill_catalog.py` 里的 `_format_entry` 改名为 `_format_listing_entry`" → agent 改完后 *下一轮* 必须主动说 "我看到 `skill_listing.py` 还在调旧名字" 并修复，而不是先报告 done 等用户骂。
- **回归手测**：把 `EngineConfig(skill_listing_enabled=False, verification_directive_enabled=False)` —— 完全旧行为，无 attachment 注入；模型行为应与 sprint 10 之前一致。

## 不在本 PR / Deferred to other PRs

- **跨 turn 诊断衰减**：现在 `tracker._delivered` 是一个 set，永不清理；如果模型 50 turn 都不修，这条诊断只显示一次。如果用户希望 "每 N turn 重提醒"，独立小 PR 在 `DiagnosticTracker` 上加 `_delivered_at` 时间戳与 nudge interval。
- **Token budget**：本 PR 里 `<diagnostics>` 块不做截断；如果一次 edit 引入 200 条诊断，prompt 会膨胀。OCC 也只做了简单的 "first N files" 截断（`attachments.ts:2876` 附近）；Aether 留待 Sprint 12+。
- **TUI 端 "Diagnostics" 面板**：仍然不做；诊断对用户保持透明。
