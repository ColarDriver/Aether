"""Hook proxy that fans out async-subagent lifecycle events to TaskStore.

Wraps the child engine's existing :class:`EngineHooks` so user-installed
hooks (e.g. the gateway's event sink) keep firing.  This proxy adds:

- Heartbeat on every ``pre_llm_call`` / ``on_session_start`` so
  :func:`recover_task_store` can tell live tasks from orphans.
- Assistant text + tool messages appended to ``messages.jsonl`` so
  ``TaskOutput`` (PR 10.6) can show progress to the parent.
- Tool-name + truncated content appended to ``output.log`` so a
  ``tail -f`` of the live task gives a human-readable view.
- ``tool_use_count`` progress increments on each tool call.

The proxy never raises — store failures are logged and swallowed so
they cannot break the child's loop.
"""

from __future__ import annotations

import logging
from typing import Any

from aether.runtime.core.contracts import NormalizedResponse, ToolResult
from aether.runtime.core.hooks import EngineHooks, HookOutcome
from aether.runtime.tasks import TaskStore

logger = logging.getLogger(__name__)


# Truncation budgets for what lands in messages.jsonl / output.log.
# The full text already lives in the child's NormalizedResponse and
# tool result, which the parent sees via SubagentResult.engine_result;
# the store copy is for streaming observation only.
_MESSAGE_CONTENT_MAX = 2_000
_OUTPUT_PREVIEW_MAX = 500


class TaskStoreFanoutHooks(EngineHooks):
    """``EngineHooks`` proxy that mirrors lifecycle events into a TaskStore."""

    __slots__ = ("_inner", "_store", "_task_id")

    def __init__(self, *, inner: EngineHooks, store: TaskStore, task_id: str) -> None:
        # EngineHooks is a slots dataclass with no fields; init the
        # base so dataclass machinery doesn't complain on subclassing.
        super().__init__()
        self._inner = inner
        self._store = store
        self._task_id = task_id

    # ------------------------------------------------------------ helpers

    def _safe_store(self, op: str, fn) -> None:
        """Run a TaskStore op, swallowing failures with a warning."""
        try:
            fn()
        except Exception:  # noqa: BLE001 - hook contract: never raise
            logger.warning(
                "task store fanout %s failed for task %s",
                op,
                self._task_id,
                exc_info=True,
            )

    # ------------------------------------------------------------ lifecycle

    def on_session_start(self, *, session_id: str, context_metadata: dict[str, Any]) -> None:
        self._safe_store("heartbeat", lambda: self._store.record_heartbeat(self._task_id))
        return self._inner.on_session_start(
            session_id=session_id, context_metadata=context_metadata
        )

    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        self._safe_store("heartbeat", lambda: self._store.record_heartbeat(self._task_id))
        return self._inner.pre_llm_call(
            session_id=session_id,
            iteration=iteration,
            messages=messages,
            context_metadata=context_metadata,
        )

    def pre_api_request(
        self,
        *,
        session_id: str,
        iteration: int,
        model: str,
        provider: str,
        api_mode: str,
        api_call_count: int,
        message_count: int,
        tool_count: int,
        approx_input_tokens: int,
        request_char_count: int,
        max_tokens: int | None,
        context_metadata: dict[str, Any],
    ) -> None:
        return self._inner.pre_api_request(
            session_id=session_id,
            iteration=iteration,
            model=model,
            provider=provider,
            api_mode=api_mode,
            api_call_count=api_call_count,
            message_count=message_count,
            tool_count=tool_count,
            approx_input_tokens=approx_input_tokens,
            request_char_count=request_char_count,
            max_tokens=max_tokens,
            context_metadata=context_metadata,
        )

    def post_api_request(
        self,
        *,
        session_id: str,
        iteration: int,
        model: str,
        provider: str,
        api_mode: str,
        api_call_count: int,
        elapsed_ms: float,
        response_finish_reason: str | None,
        error: Exception | None,
        context_metadata: dict[str, Any],
    ) -> None:
        return self._inner.post_api_request(
            session_id=session_id,
            iteration=iteration,
            model=model,
            provider=provider,
            api_mode=api_mode,
            api_call_count=api_call_count,
            elapsed_ms=elapsed_ms,
            response_finish_reason=response_finish_reason,
            error=error,
            context_metadata=context_metadata,
        )

    def post_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        response_text: str,
        context_metadata: dict[str, Any],
    ) -> None:
        # Capture the assistant turn for messages.jsonl + bump
        # token counters from the canonical usage block (populated
        # by the engine's ``_accumulate_usage`` step).
        text = response_text or ""
        self._safe_store(
            "append_message",
            lambda: self._store.append_message(
                self._task_id,
                {
                    "role": "assistant",
                    "content": text[:_MESSAGE_CONTENT_MAX],
                    "iteration": iteration,
                },
            ),
        )
        # Iteration count is the canonical "how far has this task gotten"
        # signal; record it unconditionally.  Token counters are only
        # bumped when the engine has populated a ``usage`` block on the
        # context — scripted-provider tests don't, so guard the lookup.
        in_tokens = 0
        out_tokens = 0
        usage = context_metadata.get("usage")
        if isinstance(usage, dict):
            try:
                in_tokens = int(usage.get("input_tokens_delta") or 0)
                out_tokens = int(usage.get("output_tokens_delta") or 0)
            except (TypeError, ValueError):
                in_tokens, out_tokens = 0, 0
        self._safe_store(
            "record_progress",
            lambda: self._store.record_progress(
                self._task_id,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                iterations=iteration,
            ),
        )
        return self._inner.post_llm_call(
            session_id=session_id,
            iteration=iteration,
            response_text=response_text,
            context_metadata=context_metadata,
        )

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
        content = result.content or ""
        self._safe_store(
            "append_tool_message",
            lambda: self._store.append_message(
                self._task_id,
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content[:_MESSAGE_CONTENT_MAX],
                    "is_error": bool(result.is_error),
                    "elapsed_ms": elapsed_ms,
                },
            ),
        )
        preview = content[:_OUTPUT_PREVIEW_MAX]
        self._safe_store(
            "append_output",
            lambda: self._store.append_output(
                self._task_id, f"\n[{tool_name}] {preview}\n"
            ),
        )
        self._safe_store(
            "tool_use_count",
            lambda: self._store.record_progress(
                self._task_id, tool_use_count_delta=1
            ),
        )
        return self._inner.post_tool_use(
            session_id=session_id,
            iteration=iteration,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
            elapsed_ms=elapsed_ms,
            context_metadata=context_metadata,
        )

    def post_tool_use_failure(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        error: Exception,
        elapsed_ms: float,
        context_metadata: dict[str, Any],
    ) -> None:
        self._safe_store(
            "append_failure_message",
            lambda: self._store.append_message(
                self._task_id,
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "is_error": True,
                    "error": str(error),
                    "elapsed_ms": elapsed_ms,
                },
            ),
        )
        self._safe_store(
            "tool_use_count",
            lambda: self._store.record_progress(
                self._task_id, tool_use_count_delta=1
            ),
        )
        return self._inner.post_tool_use_failure(
            session_id=session_id,
            iteration=iteration,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            error=error,
            elapsed_ms=elapsed_ms,
            context_metadata=context_metadata,
        )

    def on_session_end(
        self,
        *,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        # Final heartbeat — recovery only kills RUNNING records, but
        # this still gives observability tools a fresh ts.
        self._safe_store("heartbeat", lambda: self._store.record_heartbeat(self._task_id))
        return self._inner.on_session_end(
            session_id=session_id,
            completed=completed,
            interrupted=interrupted,
            context_metadata=context_metadata,
        )

    def on_task_cleanup(
        self,
        *,
        task_id: str,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        return self._inner.on_task_cleanup(
            task_id=task_id,
            session_id=session_id,
            completed=completed,
            interrupted=interrupted,
            context_metadata=context_metadata,
        )


__all__ = ["TaskStoreFanoutHooks"]
