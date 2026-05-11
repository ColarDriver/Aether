# PR 4.1 — 基础设施：合法-空判定 + 结构化错误工具 + 计数器消费

> **角色**：Sprint 4 的地基。把 Sprint 4 后续 PR 反复要用的"判 / 算 / 写"三件事抽出来：
> （1）一个判定器告诉调用者"这个空响应是合法的还是 bug"；（2）一个错误格式化工具
> 把模型友好的错误信息标准化（PR 4.4 的雏形）；（3）现有 `EMPTY_RESPONSE` 的 surface
> 路径整理清楚，让 PR 4.2 的 7 步降级 + Codex finalise pre-hook 有干净的入口可挂。
> 
> 借鉴 [`open-claude-code/src/utils/queryHelpers.ts:isResultSuccessful`](../../tmp/claude-code-references)
> 与 [`open-claude-code/src/utils/toolErrors.ts:formatZodValidationError`](../../tmp/claude-code-references)。

## 一、目标

1. **新增 `runtime/response_classification.py`**：实现 `is_legitimate_empty(response, *,
   stop_reason, finish_reason) -> ResponseClassification`，给 Sprint 4 后续 PR 一个统一入口
   判断"空响应是不是 bug"。
2. **新增 `runtime/tool_error_format.py`**：实现 `format_invalid_tool_args_error(tool_name, exc, raw_args)`
   把 `JSONDecodeError` 的裸消息升级成"分类 + 位置 + 上下文"的可读文本。**先不接 schema validator**
   （Aether 当前没有等价 Zod 的 schema 体系），仅做 JSON 语法错的友好版；schema 错误格式化等 PR 4.4 拓展。
3. **重构 `EMPTY_RESPONSE` 的 surface 路径**：把当前 `agent.py` 里散落在 4 个分支的"递增计数 + break"
   归一到一个 `_finalize_empty_response()` helper，**为 PR 4.2 的 7 步降级 + Codex finalise pre-hook 留挂载点**。
4. **`metadata` schema 标准化**：在 `session_runtime.py` 里追加 Sprint 4 计数器 / flag 的 const，
   让 PR 4.2/4.3 直接 import 不需要再推。
5. **保持纯重构语义**：本 PR **不**改任何外部行为；既有测试全 green，且新增的 helper 自身有
   单测覆盖。

## 二、现状分析

### 2.1 Aether 的 EMPTY_RESPONSE 入口现在长什么样

`backend/harness/aether/agents/core/agent.py:307-921` 关键片段：

```python
# 入口预设：
exit_reason = ExitReason.EMPTY_RESPONSE   # agent.py:311

# 在 LLM 响应处理后：
if final_response:
    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
    if phantom_outcome == "exhausted":
        ...
    else:
        exit_reason = ExitReason.LENGTH_RECOVERED if prefix else ExitReason.TEXT_RESPONSE
else:
    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = (
        int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)) + 1
    )
    exit_reason = ExitReason.EMPTY_RESPONSE
state_machine.transition(LoopState.FINALIZE)
break
```

**问题**：
- 计数器 `TURN_KEY_EMPTY_RESPONSE_RETRIES` 只递增、**没有任何代码读它**
  （`agent.py:898-900` 注释明确说"Sprint 4 will introduce..."）。
- 默认 exit_reason 在第 311 行预设为 EMPTY_RESPONSE，第 919 行又显式设一次 —— 两次设值容易出 bug。
- "空响应"判断是简单的 `if final_response`，没有把 `stop_reason` / `finish_reason` /
  `thinking_*` 字段纳入考虑——对 thinking-only 模型必误判。

### 2.2 claude-code 的对应

[`src/utils/queryHelpers.ts:56-94`](../../tmp/claude-code-references)：

```typescript
export function isResultSuccessful(
  message: Message | undefined,
  stopReason: string | null = null,
): message is Message {
  if (!message) return false
  if (message.type === 'assistant') {
    const lastContent = last(message.message.content)
    return (
      lastContent?.type === 'text' ||
      lastContent?.type === 'thinking' ||
      lastContent?.type === 'redacted_thinking'
    )
  }
  if (message.type === 'user') {
    const content = message.message.content
    if (
      Array.isArray(content) &&
      content.length > 0 &&
      content.every(block => 'type' in block && block.type === 'tool_result')
    ) {
      return true
    }
  }
  // Carve-out: API completed (message_delta set stop_reason) but yielded
  // no assistant content — last(messages) is still this turn's prompt.
  return stopReason === 'end_turn'
}
```

四种 success：assistant text / assistant thinking / user-tool_result / 显式 end_turn。
我们要把这套判定迁移到 Aether 的 `NormalizedResponse` 上。

[`src/utils/toolErrors.ts:66-132`](../../tmp/claude-code-references) `formatZodValidationError`：

把 ZodError 的 `issues` 数组分类成三组：
- `invalid_type` & `received undefined` → "缺参数"
- `unrecognized_keys` → "多参数"
- `invalid_type` & 其它 → "类型错"（含 expected / received）

claude-code 的输出示例（注意 claude-code 用 `Read` 工具且参数叫 `file_path`）：
```
Read failed due to the following 2 issues:
The required parameter `file_path` is missing
The parameter `offset` type is expected as `number` but provided as `string`
```

> Codex 反馈 #7 提醒：Aether 的 `read_file` 工具用 `path` 不是 `file_path`
> （`backend/harness/aether/tools/builtins/read_file.py:62-85`），等价输出形如：
> ```
> read_file failed due to the following 2 issues:
> The required parameter `path` is missing
> The parameter `offset` type is expected as `integer` but provided as `string`
> ```
> PR 4.4 文档里的所有测试 case 一律按 Aether 的真实 schema (`path`)，不改 read_file
> schema 兼容 claude-code（read_file 是已上线工具，重命名是 breaking change）。

我们要把这个**分类 + 友好措辞**的设计借鉴过来，但**先聚焦 JSON 语法错**（Aether 当前主路径），
schema validation 错误的同款格式化等 PR 4.4 接 schema validator 时一并做。

### 2.3 其它已有的 metadata 标记位

`backend/harness/aether/runtime/session_runtime.py:56-103` 已经有：
- `TURN_KEY_EMPTY_RESPONSE_RETRIES`（仅累加未消费）
- `TURN_KEY_PROVIDER_ERROR_RETRIES`
- `TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES`
- `TURN_KEY_INVALID_JSON_RETRIES`
- `TURN_KEY_PHANTOM_TOOL_RETRIES`
- `TURN_KEY_PHANTOM_TOOL_SYNTHESIZED`

**Sprint 4 要新增**（PR 4.1 在 session_runtime.py 加常量；PR 4.2/4.3 才真正写）：
- `TURN_KEY_THINKING_PREFILL_RETRIES`
- `TURN_KEY_POST_TOOL_EMPTY_RETRIED`（bool flag，不是计数）
- `TURN_KEY_CODEX_ACK_RETRIES`
- `TURN_KEY_STREAMED_ASSISTANT_TEXT`（累计字符串）
- `TURN_KEY_TRUNCATED_RESPONSE_PREFIX`（PR 1.2 已有但散在 metadata；本次正式标常量）
- `TURN_KEY_EMPTY_RECOVERY_LAST_STEP`（observability 用，告诉调用者"上次走的哪步"）

> **Codex 反馈 #5（已采纳）**：housekeeping fallback 需要"上一轮 → 这一轮"的状态。
> 当前 `agent.py:1139` 是 `metadata = dict(request.metadata)`——TurnContext.metadata 每 turn
> 从 `request.metadata` 浅拷贝重置。把 housekeeping 状态放在 turn key 里跨 turn 读不到。
> 因此本 PR 把这两项放到 **`SessionRuntimeState`**（每 session 持久，跨 turn 自动可见，
> 与既有的 `memory_nudge_counter` / `skill_nudge_counter` 同等地位），而**不是** turn key。
> 详见 § 3.4 SessionRuntimeState 增量。

## 三、设计

### 3.1 `runtime/response_classification.py`（新文件）

```python
"""Classify an LLM response as ``legitimate empty`` vs ``empty needing recovery``.

Sprint 4 / PR 4.1 — replaces the bare ``if final_response`` check that
sits in the run loop today.  The classifier is the single source of
truth for "should this response trigger the 9-step empty-response
degradation pipeline (PR 4.2)?".

Design borrowed from ``open-claude-code/src/utils/queryHelpers.ts:isResultSuccessful``
(see 00_overview.md § 1.2).  Adapted to Aether's NormalizedResponse and
multi-provider stop_reason / finish_reason vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from aether.runtime.contracts import NormalizedResponse


class EmptyKind(str, Enum):
    """Why this response was classified as empty."""

    NOT_EMPTY = "not_empty"
    """Response has user-visible text content; no recovery needed."""

    LEGITIMATE_END_TURN = "legitimate_end_turn"
    """Empty content + explicit stop_reason="end_turn" / finish_reason="stop".
    The model deliberately said "I have nothing to say".  Treat as success.

    Inspired by claude-code's stopReason === 'end_turn' carve-out."""

    THINKING_ONLY = "thinking_only"
    """Response has only thinking / reasoning blocks, no visible content.
    For Anthropic extended-thinking and OpenAI o1-style models this can
    be a legitimate single-turn response (the thinking IS the answer).
    For Aether's multi-provider story we treat it as **not recoverable**
    by retry but **also not a bug** — surface as success but mark for
    UI hint."""

    BUG_EMPTY = "bug_empty"
    """Empty response with no clear "I'm done" signal.  This is what the
    9-step degradation pipeline (PR 4.2) is for.  Examples:

    - Reasoning model returned ``content="" `` + ``stop_reason="length"``
    - Network glitch killed the stream mid-flight
    - Model emitted only ``<think>...</think>`` then stopped
    - Provider returned 200 OK with empty body
    """


@dataclass(frozen=True)
class ResponseClassification:
    """Verdict for a single LLM response.

    ``kind`` decides the next branch in the run-loop:

    - ``NOT_EMPTY`` / ``LEGITIMATE_END_TURN`` / ``THINKING_ONLY`` → no recovery.
    - ``BUG_EMPTY`` → enter the 9-step degradation (PR 4.2).

    ``has_thinking`` and ``has_streamed_partial`` are convenience flags
    consumed by individual recovery steps so the classifier doesn't
    re-walk the same fields.
    """

    kind: EmptyKind
    has_thinking: bool = False
    has_streamed_partial: bool = False
    visible_text_chars: int = 0
    raw_stop_reason: Optional[str] = None
    raw_finish_reason: Optional[str] = None

    @property
    def is_recoverable(self) -> bool:
        """``True`` iff the 9-step empty-response degradation should run."""
        return self.kind == EmptyKind.BUG_EMPTY

    @property
    def is_success(self) -> bool:
        """``True`` iff the run-loop can proceed to FINALIZE without recovery.

        Both ``NOT_EMPTY`` and ``LEGITIMATE_END_TURN`` count.  ``THINKING_ONLY``
        also counts because for reasoning-only models the thinking IS the
        answer — but the caller may want to surface a UI hint.
        """
        return self.kind in (
            EmptyKind.NOT_EMPTY,
            EmptyKind.LEGITIMATE_END_TURN,
            EmptyKind.THINKING_ONLY,
        )


# stop_reason / finish_reason values across providers that mean
# "the model is done deliberately".  We treat these as legitimate
# even when content is empty.
_DELIBERATE_STOPS: frozenset[str] = frozenset({
    "end_turn",       # Anthropic
    "stop",           # OpenAI
    "stop_sequence",  # Anthropic legacy
    "completed",      # Codex / OpenAI Responses API
    "finish",         # Some Chinese providers
})


def is_legitimate_empty(
    response: NormalizedResponse,
    *,
    streamed_assistant_text: str = "",
) -> ResponseClassification:
    """Classify ``response`` and return a verdict the run-loop can branch on.

    Args:
        response: NormalizedResponse from the provider.
        streamed_assistant_text: Text accumulated by the streaming callback
            BEFORE the response object materialised.  Empty if non-streaming
            or if no chunks arrived.  Pre-classifier strips ``<think>...</think>``
            blocks before measuring.

    Returns:
        ResponseClassification with ``kind`` set to one of NOT_EMPTY /
        LEGITIMATE_END_TURN / THINKING_ONLY / BUG_EMPTY.
    """
    visible = (response.content or "").strip()
    visible_chars = len(visible)
    has_thinking = _detect_thinking_in_response(response)
    has_partial = bool(_strip_think_tags(streamed_assistant_text).strip())

    finish = (response.finish_reason or "").lower() if response.finish_reason else None
    stop = _extract_stop_reason(response)

    if visible:
        return ResponseClassification(
            kind=EmptyKind.NOT_EMPTY,
            has_thinking=has_thinking,
            has_streamed_partial=has_partial,
            visible_text_chars=visible_chars,
            raw_stop_reason=stop,
            raw_finish_reason=finish,
        )

    # Empty content path — decide between LEGITIMATE / THINKING / BUG.

    deliberate = (stop or "") in _DELIBERATE_STOPS or (finish or "") in _DELIBERATE_STOPS

    if has_thinking and not has_partial:
        # Pure thinking-only response.  This is legitimate for Anthropic
        # extended-thinking and OpenAI o1; surface as THINKING_ONLY so the
        # caller can branch (e.g. attach a UI hint, but don't kick recovery).
        return ResponseClassification(
            kind=EmptyKind.THINKING_ONLY,
            has_thinking=True,
            has_streamed_partial=False,
            visible_text_chars=0,
            raw_stop_reason=stop,
            raw_finish_reason=finish,
        )

    if deliberate and not has_partial:
        return ResponseClassification(
            kind=EmptyKind.LEGITIMATE_END_TURN,
            has_thinking=has_thinking,
            has_streamed_partial=False,
            visible_text_chars=0,
            raw_stop_reason=stop,
            raw_finish_reason=finish,
        )

    # Anything else (length stop, unknown stop, partial-but-content-empty)
    # is a bug worth recovering from.
    return ResponseClassification(
        kind=EmptyKind.BUG_EMPTY,
        has_thinking=has_thinking,
        has_streamed_partial=has_partial,
        visible_text_chars=0,
        raw_stop_reason=stop,
        raw_finish_reason=finish,
    )


# ----- internal helpers -----

import re

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text or "") if text else ""


def _detect_thinking_in_response(response: NormalizedResponse) -> bool:
    """Detect thinking content in the normalised response.

    NOTE (Codex 反馈 #1): ``NormalizedResponse`` does NOT have a top-level
    ``reasoning`` field as of Sprint 4 — see
    ``backend/harness/aether/runtime/contracts.py:147``.  All providers that
    expose reasoning stash it under ``response.metadata`` (Codex writes
    ``metadata["reasoning_content"]``; Anthropic extended-thinking writes
    ``metadata["reasoning_details"]``).  This helper is the single place
    that normalises that read so the rest of Sprint 4 doesn't have to
    care which provider produced the response.

    Looks for:
    - ``response.metadata["reasoning_content"]`` non-empty
    - ``response.metadata["reasoning_details"]`` non-empty
    - ``<think>`` / ``<reasoning>`` tags in ``response.content``
    """
    md: dict[str, Any] = getattr(response, "metadata", None) or {}
    if md.get("reasoning_content") or md.get("reasoning_details"):
        return True
    content = response.content or ""
    if "<think>" in content.lower() or "<reasoning>" in content.lower():
        return True
    return False


def _extract_stop_reason(response: NormalizedResponse) -> Optional[str]:
    """Extract Anthropic-style stop_reason from the normalised response.

    NormalizedResponse uses ``finish_reason`` as the primary field; some
    providers also stash ``stop_reason`` in ``metadata``.  We check both.
    """
    md = getattr(response, "metadata", None) or {}
    raw = md.get("stop_reason")
    if raw:
        return str(raw).lower()
    return None
```

### 3.2 `runtime/tool_error_format.py`（新文件）

```python
"""Format invalid-tool-arguments errors into model-friendly text.

Sprint 4 / PR 4.1 — JSON syntax errors only.  Schema validation errors
(missing required fields, type mismatches, unknown fields) get the
same treatment in PR 4.4 once we wire in pydantic-style validators.

Design borrowed from ``open-claude-code/src/utils/toolErrors.ts:formatZodValidationError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FormattedToolError:
    """Outcome of :func:`format_invalid_tool_args_error`.

    Attributes:
        text: Model-friendly text suitable for injection as the
            ``content`` field of a ``role=tool`` corrective message.
        category: ``"json_syntax"`` for now; PR 4.4 will add ``"missing"``,
            ``"unexpected"``, ``"type_mismatch"`` for schema errors.
        line: 1-based line number of the syntax error if available.
        column: 1-based column number of the syntax error if available.
    """

    text: str
    category: str
    line: Optional[int] = None
    column: Optional[int] = None


_MAX_HINT_CHARS = 120


def format_invalid_tool_args_error(
    tool_name: str,
    exc: Exception,
    raw_args: str,
) -> FormattedToolError:
    """Build a friendly error message for an invalid-JSON tool-arg payload.

    The output text is consumed by ``_validate_tool_call_arguments`` (see
    PR 4.4 for the upgrade) when it injects a ``role=tool`` corrective
    message.  Format goal: tell the model **which tool, which line/column,
    and what the parser saw** so it can self-correct on the next turn.

    Example output for a missing comma::

        Invalid JSON arguments for tool `read_file` at line 1 column 23:
        ``Expecting ',' delimiter``.
        The parser stopped after: ...lid_path": "/foo/...
        Hint: ensure every key/value pair is comma-separated and the
        whole payload is wrapped in `{}`.

    Args:
        tool_name: name of the tool the model attempted to call.
        exc: the parse exception (typically ``json.JSONDecodeError``;
            other exceptions are accepted and rendered with ``str(exc)``).
        raw_args: the raw arguments string that failed to parse.

    Returns:
        FormattedToolError with ``text`` ready to drop into a ``role=tool``
        corrective message.
    """
    line, column, parser_msg = _extract_jsondecodeerror_metadata(exc)

    pieces: list[str] = []

    if line is not None and column is not None:
        pieces.append(
            f"Invalid JSON arguments for tool `{tool_name}` at line {line} column {column}: "
            f"`{parser_msg}`."
        )
    else:
        pieces.append(
            f"Invalid JSON arguments for tool `{tool_name}`: `{parser_msg}`."
        )

    snippet = _build_snippet(raw_args, line, column)
    if snippet:
        pieces.append(f"The parser stopped after: {snippet}")

    pieces.append(
        "Hint: emit a single JSON object as the `arguments` field. "
        "Every key/value pair must be comma-separated and the whole payload "
        "must be wrapped in `{}`. For tools with no required parameters, use `{}`."
    )

    return FormattedToolError(
        text="\n".join(pieces),
        category="json_syntax",
        line=line,
        column=column,
    )


# ----- internal helpers -----


def _extract_jsondecodeerror_metadata(
    exc: Exception,
) -> tuple[Optional[int], Optional[int], str]:
    """Extract (line, column, message) from a JSONDecodeError, gracefully
    handling other exception types."""
    if isinstance(exc, json.JSONDecodeError):
        return (
            exc.lineno if exc.lineno > 0 else None,
            exc.colno if exc.colno > 0 else None,
            exc.msg or str(exc),
        )
    return (None, None, str(exc) or exc.__class__.__name__)


def _build_snippet(
    raw_args: str,
    line: Optional[int],
    column: Optional[int],
) -> Optional[str]:
    """Build a short "...the parser stopped after..." snippet around the
    error position, capped at ``_MAX_HINT_CHARS`` for readability."""
    if not raw_args:
        return None
    if line is not None and column is not None:
        # Convert 1-based (line, column) to absolute character offset.
        try:
            offset = _line_col_to_offset(raw_args, line, column)
        except Exception:
            offset = None
    else:
        offset = None
    if offset is None:
        # Fallback: just use the start of the string.
        return _truncate(raw_args, _MAX_HINT_CHARS)
    start = max(0, offset - _MAX_HINT_CHARS // 2)
    end = min(len(raw_args), offset + _MAX_HINT_CHARS // 2)
    snippet = raw_args[start:end]
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(raw_args):
        snippet = f"{snippet}..."
    return snippet


def _line_col_to_offset(text: str, line: int, column: int) -> int:
    """Convert 1-based (line, column) to absolute character offset.
    Raises if line is out of range."""
    lines = text.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        raise ValueError("line out of range")
    offset = sum(len(l) for l in lines[: line - 1])
    return offset + max(0, column - 1)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}...{text[-half:]}"
```

### 3.3 `agent.py` 重构 — `_finalize_empty_response()` helper

把当前散落在 `agent.py:890-921` 的"final_response 判定 + 计数 + exit_reason 设置"
归一到一个新方法：

```python
def _finalize_empty_response(
    self,
    *,
    response: "NormalizedResponse",
    response_to_store: "NormalizedResponse",
    messages: list[dict],
    context: TurnContext,
    state_machine: LoopStateMachine,
    request: EngineRequest,
    phantom_outcome: str,
) -> tuple[Optional[str], ExitReason, Optional[str]]:
    """Classify the assistant response and decide the terminal exit.

    Returns: ``(final_response, exit_reason, error_text)``.

    Sprint 4 / PR 4.1 — extracted from the main run-loop body so that
    PR 4.2's 9-step degradation can hook in **before** this helper
    declares EMPTY_RESPONSE.  This PR makes the helper a 1-1 lift of
    the existing logic; PR 4.2 inserts the recovery branch at the top
    of the helper.
    """
    self._append_assistant_text_message(messages, response_to_store)
    final_response = response_to_store.content or ""

    # Streaming fallback emit (unchanged from previous behaviour).
    stream_callback_wrapped = self._maybe_get_stream_callback(request, context)
    if (
        stream_callback_wrapped
        and final_response
        and not context.metadata.get("streamed_output")
    ):
        stream_callback_wrapped(final_response)
        context.metadata["stream_fallback_emitted"] = True

    classification = is_legitimate_empty(
        response_to_store,
        streamed_assistant_text=str(
            context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""
        ),
    )
    context.metadata["empty_recovery"] = context.metadata.get("empty_recovery") or {}
    context.metadata["empty_recovery"]["classification"] = classification.kind.value

    if final_response:
        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
        if phantom_outcome == "exhausted":
            ...  # existing logic, unchanged
        else:
            prefix = context.metadata.get(TURN_KEY_TRUNCATED_RESPONSE_PREFIX)
            exit_reason = ExitReason.LENGTH_RECOVERED if prefix else ExitReason.TEXT_RESPONSE
        return final_response, exit_reason, None

    # No visible content.  Today (PR 4.1): unconditionally finalise as
    # EMPTY_RESPONSE, after bumping the per-turn counter.  PR 4.2 will
    # insert the 9-step degradation HERE.
    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = (
        int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)) + 1
    )

    if classification.is_success:
        # New behaviour PR 4.1 introduces: a deliberately-empty end_turn
        # is now a TEXT_RESPONSE with empty final_response, NOT an
        # EMPTY_RESPONSE error.  This unblocks reasoning-only models
        # without waiting for PR 4.2's full pipeline.
        context.metadata["empty_recovery"]["last_step"] = (
            "legitimate_empty_passthrough"
        )
        return "", ExitReason.TEXT_RESPONSE, None

    context.metadata["empty_recovery"]["last_step"] = "no_recovery_yet"
    return "", ExitReason.EMPTY_RESPONSE, None
```

调用点替换（`agent.py:890-921`）：

```python
# Before (current):
self._append_assistant_text_message(messages, response_to_store)
final_response = response_to_store.content or ""
if stream_callback_wrapped and final_response and not context.metadata.get("streamed_output"):
    stream_callback_wrapped(final_response)
    context.metadata["stream_fallback_emitted"] = True
if final_response:
    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
    if phantom_outcome == "exhausted":
        ...
    else:
        exit_reason = ExitReason.LENGTH_RECOVERED if prefix else ExitReason.TEXT_RESPONSE
else:
    context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = ...
    exit_reason = ExitReason.EMPTY_RESPONSE
state_machine.transition(LoopState.FINALIZE)
break

# After (PR 4.1):
final_response, exit_reason, error_text = self._finalize_empty_response(
    response=response,
    response_to_store=response_to_store,
    messages=messages,
    context=context,
    state_machine=state_machine,
    request=request,
    phantom_outcome=phantom_outcome,
)
state_machine.transition(LoopState.FINALIZE)
break
```

### 3.4 `runtime/session_runtime.py` 增量 — per-turn 常量

新增 6 个 const（其中部分仅做"占位声明"，PR 4.2/4.3 才真正写入）。
**注意**：原计划的 `TURN_KEY_LAST_CONTENT_WITH_TOOLS` /
`TURN_KEY_LAST_CONTENT_TOOLS_ALL_HOUSEKEEPING` **不再**作为 turn key
出现——见 § 3.4b。

```python
# Sprint 4 / PR 4.1: empty-response 9-step degradation runtime keys.
# PR 4.1 only declares the constants; PR 4.2 wires the actual writers.
TURN_KEY_THINKING_PREFILL_RETRIES: Final[str] = "thinking_prefill_retries"
TURN_KEY_POST_TOOL_EMPTY_RETRIED: Final[str] = "post_tool_empty_retried"
TURN_KEY_CODEX_ACK_RETRIES: Final[str] = "codex_intermediate_ack_retries"

# Sprint 4 / PR 4.1: streaming-side cumulative state used by the
# partial-stream-recovery branch (the partial-stream step of the
# empty-response degradation).  PR 4.2 wires writers; PR 4.1 declares
# the constants only.
TURN_KEY_STREAMED_ASSISTANT_TEXT: Final[str] = "streamed_assistant_text"

# NOTE: housekeeping-fallback inputs (last assistant content + the
# all-housekeeping flag) are NOT turn keys — they need to survive across
# turns so this turn's empty-response path can read what the PREVIOUS
# turn produced.  TurnContext.metadata is reset each turn (see
# agent.py:1139 ``metadata = dict(request.metadata)``), so we put them
# on ``SessionRuntimeState`` instead.  See § 3.4b for the field additions.

# Sprint 1 / PR 1.2 already populated this — Sprint 4 / PR 4.1 is just
# raising it to a named constant so PR 4.2's step 1 (truncated-prefix
# concatenation, formerly step 9 in the original 9-step plan) doesn't
# have to hard-code the string key.
TURN_KEY_TRUNCATED_RESPONSE_PREFIX: Final[str] = "truncated_response_prefix"

# Sprint 4 / PR 4.1: observability helper — the 7-step degradation
# always writes the name of the last step it ran here so logs / result
# metadata can read it without crawling all the per-step counters.
TURN_KEY_EMPTY_RECOVERY_LAST_STEP: Final[str] = "empty_recovery_last_step"
```

并把这些 key 加入 `TURN_RETRY_COUNTER_KEYS`（仅纯计数器）和 `__all__`：

```python
TURN_RETRY_COUNTER_KEYS: Final[frozenset[str]] = frozenset(
    {
        TURN_KEY_EMPTY_RESPONSE_RETRIES,
        TURN_KEY_PROVIDER_ERROR_RETRIES,
        TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES,
        TURN_KEY_INVALID_JSON_RETRIES,
        TURN_KEY_PHANTOM_TOOL_RETRIES,
        TURN_KEY_PHANTOM_TOOL_SYNTHESIZED,
        # Sprint 4 / PR 4.1:
        TURN_KEY_THINKING_PREFILL_RETRIES,
        TURN_KEY_CODEX_ACK_RETRIES,
    }
)
```

### 3.4b `runtime/session_runtime.py` 增量 — `SessionRuntimeState` 字段

> **来源**：Codex 反馈 #5 修订点。housekeeping fallback 在第 N+1 轮 fire 时需要看
> 第 N 轮的 assistant 输出快照；TurnContext.metadata 在第 N+1 轮入口已被
> `dict(request.metadata)` 重置，必然读不到。改放 `SessionRuntimeState`，
> 与 `memory_nudge_counter` / `skill_nudge_counter` 同级。

```python
@dataclass(slots=True)
class SessionRuntimeState:
    """Mutable state for one session, persisted across turns. (existing docstring)"""

    # ... existing fields ...
    memory_nudge_counter: int = 0
    skill_nudge_counter: int = 0

    # Sprint 4 / PR 4.1 (declared) + PR 4.2 (writer wired) — last
    # assistant turn's visible text and an "all tool_calls were
    # housekeeping" flag.  Read by PR 4.2's housekeeping-fallback step
    # of the empty-response degradation; written at end-of-turn by
    # ``_record_last_content_for_housekeeping_fallback`` in PR 4.2.
    #
    # Cross-turn lifetime is required: this turn ends in BUG_EMPTY,
    # next turn's step-2 looks at last_assistant_text_with_tools to
    # decide whether to substitute it as the final response.  Holding
    # them on TurnContext.metadata wouldn't survive
    # ``metadata = dict(request.metadata)`` at agent.py:1139.
    last_assistant_text_with_tools: str = ""
    last_assistant_tools_all_housekeeping: bool = False
```

PR 4.1 仅追加字段并保持默认值；不接 reader/writer（PR 4.2 接）。
`SessionRuntimeRegistry.get(session_id)` 已经在每轮入口被 agent 拿到（既有逻辑），
PR 4.2 的 step 2 直接 `state = self.services.session_runtime_registry.get(...)`
读这两个字段即可。

### 3.5 `EngineConfig` 增量（PR 4.1 仅声明，不在本 PR 启用）

`backend/harness/aether/config/schema.py`：

```python
# Sprint 4 / PR 4.1: declare 9-step degradation knobs.  PR 4.2 wires
# the consumers — until then these have no observable effect.  Default
# values match the Sprint 4 plan in 00_overview.md § 六.
empty_response_recovery_enabled: bool = True
empty_response_max_retries: int = 3
empty_response_partial_stream_recovery_enabled: bool = True
housekeeping_fallback_enabled: bool = True
housekeeping_tool_names: frozenset[str] = frozenset(
    {"memory", "update_todo", "skill_manage", "session_search"}
)
post_tool_empty_nudge_enabled: bool = True
thinking_prefill_enabled: bool = True
thinking_prefill_max_retries: int = 2
codex_intermediate_ack_enabled: bool = True
codex_intermediate_ack_max_retries: int = 2

# Sprint 4 / PR 4.4 (early declaration): tool-error structured format.
tool_error_structured_format_enabled: bool = True
```

> **注意**：`error_withholding_enabled` 留给 PR 4.3 自己加，本 PR 不引入。

### 3.6 `EngineResult.metadata` 输出契约扩展

`agent.py::_build_result` 增加 `empty_recovery` 子字典：

```python
# in _build_result(...)
empty_recovery_metadata = dict(context.metadata.get("empty_recovery") or {})
md["empty_recovery"] = {
    "classification": empty_recovery_metadata.get("classification"),
    "last_step": empty_recovery_metadata.get("last_step"),
    "retries": {
        "empty": int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)),
        # Below all 0 in PR 4.1; PR 4.2 will populate.
        "thinking_prefill": int(
            context.metadata.get(TURN_KEY_THINKING_PREFILL_RETRIES, 0)
        ),
        "codex_ack": int(context.metadata.get(TURN_KEY_CODEX_ACK_RETRIES, 0)),
    },
    "post_tool_empty_retried": bool(
        context.metadata.get(TURN_KEY_POST_TOOL_EMPTY_RETRIED, False)
    ),
}
```

### 3.7 `NormalizedResponse` 字段确认（不改，仅文档化）

`runtime/contracts.py:147` 的 `NormalizedResponse` 当前**只**有：
- `content: str | None`
- `tool_calls: list[ToolCall]`
- `finish_reason: str = "stop"`
- `metadata: dict[str, Any]` —— 这里塞 provider 原始 `stop_reason` / `reasoning_content` /
  `reasoning_details` 等

> **Codex 反馈 #1（已采纳）**：原方案打算"新增 `reasoning: str | None` 字段"。否决。
> 当前只有 Codex provider 在写 reasoning（`models/provider/codex.py:486-487` 写的是
> `metadata["reasoning_content"]`），单一 provider 不值得拓宽公共契约；Sprint 5 的
> reasoning-block 抽取本来也计划走 metadata 路径（contracts.py:214-215 已为
> `metadata["reasoning"]` 预留位置）。**PR 4.1 / PR 4.2 / PR 4.3 / PR 4.4 一律
> 通过 `response.metadata.get("reasoning_content")` / `["reasoning_details"]`
> 读 reasoning，不假设 `response.reasoning` 字段存在。**

PR 4.1 **不改**字段，只在 docstring 里追加一段说明：

```python
class NormalizedResponse:
    """...

    Sprint 4 / PR 4.1 — the following metadata keys are read by
    :func:`aether.runtime.response_classification.is_legitimate_empty`
    and the empty-response degradation pipeline (PR 4.2):

    - ``stop_reason`` (str): Anthropic-style explicit stop reason.
    - ``reasoning_content`` (str): plain-text reasoning summary
      (Codex / OpenAI Responses API; some Chinese providers).
    - ``reasoning_details`` (list[dict]): Anthropic verbose reasoning blocks.

    Providers SHOULD populate these so downstream classifiers don't
    misjudge thinking-only responses.

    NOTE: ``NormalizedResponse`` does NOT have a top-level ``reasoning``
    field as of Sprint 4.  Read reasoning via the metadata keys above.
    Sprint 5's reasoning-block extraction may revisit this contract
    (see metadata['reasoning'] reserved shape above).
    """
```

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/runtime/response_classification.py` | **新文件** | `EmptyKind` enum + `ResponseClassification` dataclass + `is_legitimate_empty` + helpers | ~180 |
| `backend/harness/aether/runtime/tool_error_format.py` | **新文件** | `FormattedToolError` dataclass + `format_invalid_tool_args_error` + helpers | ~150 |
| `backend/harness/aether/runtime/session_runtime.py` | 修改 | 加 6 个 `TURN_KEY_*` 常量；扩 `TURN_RETRY_COUNTER_KEYS` 与 `__all__`；`SessionRuntimeState` 加 2 个跨 turn 字段（`last_assistant_text_with_tools` / `last_assistant_tools_all_housekeeping`） | ~50 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | 抽 `_finalize_empty_response()` helper；调用点替换；`_build_result` 加 `empty_recovery` 子字典 | ~80 净增 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 11 个 `EngineConfig` 字段（仅声明默认值，PR 4.2/4.3/4.4 才消费） | ~50 |
| `backend/harness/aether/runtime/contracts.py` | 修改 | `NormalizedResponse` docstring 追加 metadata 约定 | ~15 |
| `backend/harness/aether/tests/runtime/test_response_classification.py` | **新文件** | 见 § 五 测试组 A-D | ~250 |
| `backend/harness/aether/tests/runtime/test_tool_error_format.py` | **新文件** | 见 § 五 测试组 E | ~150 |
| `backend/harness/aether/tests/agents/core/test_finalize_empty_response.py` | **新文件** | 见 § 五 测试组 F | ~200 |

## 五、测试用例（详细）

### 5.1 `test_response_classification.py`

**测试组 A：NOT_EMPTY**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | response.content="Hello world" | kind=NOT_EMPTY；is_success=True；visible_text_chars=11 |
| **T-A2** | response.content="  hello  " | strip 后非空 → NOT_EMPTY |
| **T-A3** | content="Hi" + reasoning="thinking..." | kind=NOT_EMPTY；has_thinking=True |
| **T-A4** | content="Hello" + finish_reason="length" | 仍 NOT_EMPTY（有可见文本就不需要 recovery） |

**测试组 B：LEGITIMATE_END_TURN**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | content="" + finish_reason="stop" | kind=LEGITIMATE_END_TURN；is_success=True |
| **T-B2** | content="" + metadata["stop_reason"]="end_turn" | 同上 |
| **T-B3** | content="   " (全空白) + finish_reason="stop" | strip 后为空，stop deliberate → LEGITIMATE |
| **T-B4** | content="" + finish_reason="completed" | 同 |
| **T-B5** | content="" + finish_reason="STOP"（大写） | case-insensitive 命中 |
| **T-B6** | content=""+stop_reason="stop_sequence" | 仍 deliberate → LEGITIMATE |

**测试组 C：THINKING_ONLY**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | content="" + reasoning="The user asked..." + finish_reason="length" | kind=THINKING_ONLY；is_success=True；has_thinking=True |
| **T-C2** | content="" + metadata["reasoning_content"]="..." + finish_reason="" | THINKING_ONLY |
| **T-C3** | content="<think>foo</think>" + finish_reason="stop" | content 全是 thinking 标签 → has_thinking=True；strip 后空 → THINKING_ONLY |
| **T-C4** | content="" + reasoning="..." + has_streamed_partial=True | 优先 partial → BUG_EMPTY 不是 THINKING_ONLY（partial 路径优先于 thinking-only） |
| **T-C5** | content="" + metadata["reasoning_details"]=[{...}] + finish_reason="" | THINKING_ONLY |

**测试组 D：BUG_EMPTY**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | content="" + finish_reason="length" | kind=BUG_EMPTY；is_recoverable=True |
| **T-D2** | content="" + finish_reason=None | BUG_EMPTY（无 deliberate 信号） |
| **T-D3** | content="" + finish_reason="" | BUG_EMPTY |
| **T-D4** | content="" + streamed_assistant_text="some partial text" | BUG_EMPTY；has_streamed_partial=True |
| **T-D5** | content="" + streamed_assistant_text="<think>only think</think>" | strip think 后空 → has_streamed_partial=False；no thinking field → BUG_EMPTY（无 deliberate） |
| **T-D6** | content="" + finish_reason="content_filter"（OpenAI 过滤） | BUG_EMPTY（content_filter 不在 deliberate 集合） |

### 5.2 `test_tool_error_format.py`

**测试组 E：JSON 错误格式化**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | `JSONDecodeError`（缺 comma），raw_args='{"a":1 "b":2}' | text 含 "line 1 column 8"；含 "Hint:"；category="json_syntax"；line=1，column=8 |
| **T-E2** | `JSONDecodeError`（缺右大括号），raw_args='{"a":1' | text 含位置信息；snippet 反映被截断 |
| **T-E3** | 任意 `Exception`（非 JSONDecodeError） | text 含 `str(exc)`；line=None，column=None |
| **T-E4** | raw_args="" | snippet=None 不输出；其它字段正常 |
| **T-E5** | raw_args 巨长（10KB），错误位置在中间 | snippet 限定在 _MAX_HINT_CHARS=120 字符内；前后用 `...` |
| **T-E6** | raw_args 巨长，错误在末尾 | snippet 不再追加 trailing `...` |
| **T-E7** | tool_name 含特殊字符 `"my-tool::v2"` | text 用反引号包住完整名 |
| **T-E8** | `JSONDecodeError` lineno=0 colno=0（异常情况） | line=column=None；不抛 |

### 5.3 `test_finalize_empty_response.py`

**测试组 F：helper 行为**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | response.content="hello" | final_response="hello"；exit_reason=TEXT_RESPONSE；EMPTY_RESPONSE_RETRIES 重置为 0 |
| **T-F2** | response.content="" + finish_reason="stop" | final_response=""；exit_reason=**TEXT_RESPONSE**（PR 4.1 新行为）；metadata["empty_recovery"]["last_step"]="legitimate_empty_passthrough" |
| **T-F3** | response.content="" + finish_reason="length" | final_response=""；exit_reason=EMPTY_RESPONSE；EMPTY_RESPONSE_RETRIES 递增；last_step="no_recovery_yet" |
| **T-F4** | response.content="hello" + truncated_prefix="prev..." | exit_reason=LENGTH_RECOVERED |
| **T-F5** | response.content="hello" + phantom_outcome="exhausted" | 走 phantom_outcome 分支（保持原 behavior） |
| **T-F6** | response.content="" + reasoning=non-empty + finish_reason="length" | THINKING_ONLY → exit_reason=TEXT_RESPONSE（is_success=True 路径）|
| **T-F7** | classification 写入 metadata["empty_recovery"]["classification"] | 任何 final_response 路径都写 |
| **T-F8** | EngineResult.metadata["empty_recovery"]["retries"] 包含 3 个 key | empty=N，thinking_prefill=0，codex_ack=0 |

**测试组 G：回归（既有行为不变）**

| ID | 场景 | 验证 |
|---|---|---|
| **T-G1** | 整个 Sprint 0/1/2 测试套件 | 全 green |
| **T-G2** | `test_run_loop_empty_response_*` 既有用例 | 行为兼容（除 T-F2 / T-F6 新增的"deliberate empty 不再算 EMPTY_RESPONSE"） |

> **重要**：T-F2 / T-F6 是 PR 4.1 引入的**外部行为变化**（先于 PR 4.2 上线）。
> 既有测试如果依赖"空响应必然返回 EMPTY_RESPONSE"需要更新。本 PR 主体工作之一。

### 5.4 集成 smoke

| ID | 场景 | 验证 |
|---|---|---|
| **T-I1** | mock provider 返回 content="" + finish_reason="stop" | engine.run_turn 返回 EngineResult，exit_reason=TEXT_RESPONSE，final_response="" |
| **T-I2** | mock provider 返回 content="" + finish_reason="length" | exit_reason=EMPTY_RESPONSE（同今天） |

## 六、验收门

- [ ] 所有新测试 green（约 30 case）
- [ ] 既有测试不回归（除 T-G2 注明的兼容性更新）
- [ ] `result.metadata["empty_recovery"]["classification"]` 在每次 turn 末尾都有值
- [ ] `result.metadata["empty_recovery"]["retries"]` 包含 3 个 key
- [ ] `runtime/response_classification.py` 与 `runtime/tool_error_format.py` 不依赖任何
      `agent.py` / `models/` / `services/` 内部符号（仅依赖 `runtime/contracts.py`）

## 七、回滚开关

- 因 PR 4.1 有"deliberate-empty 不再算 EMPTY_RESPONSE"的小行为变化，引入紧急回退路径：

```python
# config/schema.py
legitimate_empty_passthrough_enabled: bool = True
```

设为 `False` 时退回原"空响应一律 EMPTY_RESPONSE"行为。
- `tool_error_structured_format_enabled=False`：PR 4.4 才会消费它，本 PR 仅声明。
- 完全 revert PR：删除两个新文件 + revert agent.py / session_runtime.py / config 改动。

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. `runtime/response_classification.py` + 测试组 A-D | 3h | 25 case |
| 2. `runtime/tool_error_format.py` + 测试组 E | 2h | 8 case |
| 3. `runtime/session_runtime.py` 加 6 个 const + 2 个 `SessionRuntimeState` 字段 | 30min | 编译通过；__all__ 完整 |
| 4. `config/schema.py` 加 11 字段 | 30min | dataclass valid |
| 5. `agent.py::_finalize_empty_response` 抽取 + 接入 | 2h | run-loop 调用点替换 |
| 6. `agent.py::_build_result` 加 `empty_recovery` 子字典 | 1h | EngineResult.metadata 输出 |
| 7. `test_finalize_empty_response.py` 测试组 F-G | 2h | 9 case |
| 8. 既有测试集回归 + 修少量兼容性测试 | 1h | full suite green |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| `is_legitimate_empty` 把 deliberate-empty 误判成 BUG，让 PR 4.2 后续走 7 步降级 | 低 | 测试组 A-D 共 22 case 严格覆盖；T-D5 专测"thinking + partial 优先于 deliberate" |
| 抽 helper 时把 phantom_outcome 分支搞错 | 中 | T-F4/T-F5 专测原 phantom 行为不变；保留 phantom_outcome 完整 dispatch 表 |
| `legitimate_empty_passthrough` 行为变化让线上脚本依赖 EMPTY_RESPONSE 失败 | 中 | 加 `legitimate_empty_passthrough_enabled` flag（默认 True）；CHANGELOG 注明 |
| `tool_error_format.py` 在 raw_args 含特殊字符时 snippet 异常 | 低 | T-E5/T-E6/T-E7 覆盖；`_truncate` 用纯字符索引不解析内容 |
| metadata schema 扩展破坏下游消费者 | 低 | `empty_recovery` 是新增子字典，对未升级的消费者透明 |

## 十、与后续 PR 的衔接

- **PR 4.2** 在 `_finalize_empty_response()` 的 `if not final_response and not classification.is_success`
  分支内部插入 7 步降级；同分支头部还插入 Codex finalise pre-hook（处理非空 ack）。helper 已留出两个"插入点"。
- **PR 4.3** 在 `_invoke_provider_with_recovery` 中使用 `is_legitimate_empty` 判断"这次失败是不是该 hold"。
- **PR 4.4** 升级 `_validate_tool_call_arguments` 的 `f"Error: Invalid JSON arguments. {err_msg}"` 为
  `format_invalid_tool_args_error(...).text`；同时引入 schema validator 后扩展 `tool_error_format.py`
  支持 missing / unexpected / type_mismatch 三类。
