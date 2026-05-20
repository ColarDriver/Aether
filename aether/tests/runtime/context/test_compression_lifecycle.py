from __future__ import annotations

from typing import Any

from aether.runtime.context import (
    CompressionLifecycleService,
    CompressionRequest,
    ContextEngineResult,
)
from aether.runtime.core.contracts import TurnContext
from aether.runtime.core.hooks import EngineHooks


class _Engine:
    name = "fake"

    def __init__(
        self,
        *,
        should_compress: bool = True,
        result: ContextEngineResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.should_compress = should_compress
        self.result = result
        self.error = error
        self.should_calls = 0
        self.compact_calls = 0

    def should_compress_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> bool:
        del messages, context
        self.should_calls += 1
        return self.should_compress

    def compact_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> ContextEngineResult:
        del context
        self.compact_calls += 1
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return ContextEngineResult(messages=messages, reason=trigger_reason)

    def apply_provider_projection(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        return messages


class _Hooks(EngineHooks):
    def __init__(self, *, fail_before: bool = False) -> None:
        self.fail_before = fail_before
        self.events: list[tuple[str, dict[str, Any]]] = []

    def before_context_compression(
        self,
        *,
        session_id: str,
        trigger_reason: str,
        source_message_count: int,
        context_metadata: dict[str, Any],
    ) -> None:
        self.events.append(
            (
                "before",
                {
                    "session_id": session_id,
                    "trigger_reason": trigger_reason,
                    "source_message_count": source_message_count,
                    "has_messages": "messages" in context_metadata,
                },
            )
        )
        if self.fail_before:
            raise RuntimeError("hook failed")

    def after_context_compression(
        self,
        *,
        session_id: str,
        trigger_reason: str,
        source_message_count: int,
        result_message_count: int,
        context_metadata: dict[str, Any],
    ) -> None:
        self.events.append(
            (
                "after",
                {
                    "session_id": session_id,
                    "trigger_reason": trigger_reason,
                    "source_message_count": source_message_count,
                    "result_message_count": result_message_count,
                    "generation": context_metadata.get("compression_lineage", {}).get("generation"),
                },
            )
        )

    def context_compression_failed(
        self,
        *,
        session_id: str,
        trigger_reason: str,
        source_message_count: int,
        error: str,
        context_metadata: dict[str, Any],
    ) -> None:
        self.events.append(
            (
                "failed",
                {
                    "session_id": session_id,
                    "trigger_reason": trigger_reason,
                    "source_message_count": source_message_count,
                    "error": error,
                    "has_messages": "messages" in context_metadata,
                },
            )
        )


def _context() -> TurnContext:
    return TurnContext(session_id="s", iteration=1, metadata={})


def test_skips_when_engine_says_compression_not_needed() -> None:
    engine = _Engine(should_compress=False)
    service = CompressionLifecycleService(context_engine=engine)
    messages = [{"role": "user", "content": "hello"}]
    context = _context()

    result = service.compress(
        CompressionRequest(messages=messages, context=context, trigger_reason="preflight")
    )

    assert result.status == "skipped"
    assert result.messages is messages
    assert engine.should_calls == 1
    assert engine.compact_calls == 0
    assert context.metadata["context_engine"]["compression"] == {
        "status": "skipped",
        "trigger_reason": "preflight",
        "source_message_count": 1,
        "result_message_count": 1,
        "reason": "not_needed",
    }


def test_force_compression_bypasses_preflight_decision() -> None:
    compressed = [{"role": "user", "content": "summary"}]
    engine = _Engine(
        should_compress=False,
        result=ContextEngineResult(
            messages=compressed,
            changed=True,
            reason="manual",
            metadata={"tokens_before": 100, "tokens_after": 20},
        ),
    )
    service = CompressionLifecycleService(context_engine=engine)

    result = service.compress(
        CompressionRequest(
            messages=[{"role": "user", "content": "hello"}],
            context=_context(),
            trigger_reason="manual",
            force=True,
        )
    )

    assert result.status == "compressed"
    assert result.messages == compressed
    assert engine.should_calls == 0
    assert engine.compact_calls == 1
    assert result.metadata["source_message_count"] == 1
    assert result.metadata["result_message_count"] == 1
    assert result.metadata["result_tokens"] == 20


def test_failed_compression_preserves_input_messages() -> None:
    engine = _Engine(error=RuntimeError("boom"))
    service = CompressionLifecycleService(context_engine=engine)
    messages = [{"role": "user", "content": "hello"}]
    context = _context()

    result = service.compress(
        CompressionRequest(messages=messages, context=context, trigger_reason="preflight")
    )

    assert result.status == "failed"
    assert result.messages is messages
    assert result.error == "RuntimeError"
    assert context.metadata["context_engine"]["compression"]["error"] == "RuntimeError"


def test_invalid_message_output_becomes_failure() -> None:
    engine = _Engine(
        result=ContextEngineResult(
            messages=[{"content": "missing role"}],
            changed=True,
        )
    )
    service = CompressionLifecycleService(context_engine=engine)
    messages = [{"role": "user", "content": "hello"}]

    result = service.compress(
        CompressionRequest(messages=messages, context=_context(), trigger_reason="preflight")
    )

    assert result.status == "failed"
    assert result.messages is messages
    assert result.error == "missing_message_role"


def test_metadata_is_json_safe_and_counts_successful_compressions() -> None:
    engine = _Engine(
        result=ContextEngineResult(
            messages=[{"role": "user", "content": "summary"}],
            changed=True,
            metadata={
                "tokens_before": 100,
                "tokens_after": 20,
                "live_object": object(),
                "tiers_run": ["tier5"],
            },
        )
    )
    service = CompressionLifecycleService(context_engine=engine)
    context = _context()

    result = service.compress(
        CompressionRequest(
            messages=[{"role": "user", "content": "hello"}],
            context=context,
            trigger_reason="preflight",
        )
    )

    assert result.status == "compressed"
    metadata = context.metadata["context_engine"]
    assert metadata["name"] == "fake"
    assert metadata["compression_count"] == 1
    assert metadata["last_trigger_reason"] == "preflight"
    assert metadata["compression"]["engine"]["live_object"] == "object"
    assert metadata["compression"]["engine"]["tiers_run"] == ["tier5"]


def test_successful_compression_records_lineage_and_hooks() -> None:
    engine = _Engine(
        result=ContextEngineResult(
            messages=[{"role": "user", "content": "summary"}],
            changed=True,
        )
    )
    hooks = _Hooks()
    service = CompressionLifecycleService(context_engine=engine, hooks=hooks)
    context = _context()

    result = service.compress(
        CompressionRequest(
            messages=[{"role": "user", "content": "hello"}],
            context=context,
            trigger_reason="preflight",
        )
    )

    assert result.status == "compressed"
    lineage = context.metadata["compression_lineage"]
    assert lineage["generation"] == 1
    assert lineage["trigger_reason"] == "preflight"
    assert lineage["source_message_count"] == 1
    assert lineage["result_message_count"] == 1
    assert context.metadata["diagnostics"]["compression_generation"] == 1
    assert [event for event, _payload in hooks.events] == ["before", "after"]
    assert hooks.events[1][1]["generation"] == 1


def test_failed_compression_fires_failed_hook_without_raw_messages() -> None:
    hooks = _Hooks()
    service = CompressionLifecycleService(
        context_engine=_Engine(error=RuntimeError("boom")),
        hooks=hooks,
    )

    result = service.compress(
        CompressionRequest(
            messages=[{"role": "user", "content": "hello"}],
            context=_context(),
            trigger_reason="preflight",
        )
    )

    assert result.status == "failed"
    assert hooks.events == [
        (
            "before",
            {
                "session_id": "s",
                "trigger_reason": "preflight",
                "source_message_count": 1,
                "has_messages": False,
            },
        ),
        (
            "failed",
            {
                "session_id": "s",
                "trigger_reason": "preflight",
                "source_message_count": 1,
                "error": "RuntimeError",
                "has_messages": False,
            },
        ),
    ]


def test_hook_failure_does_not_abort_compression() -> None:
    hooks = _Hooks(fail_before=True)
    service = CompressionLifecycleService(
        context_engine=_Engine(
            result=ContextEngineResult(
                messages=[{"role": "user", "content": "summary"}],
                changed=True,
            )
        ),
        hooks=hooks,
    )
    context = _context()

    result = service.compress(
        CompressionRequest(
            messages=[{"role": "user", "content": "hello"}],
            context=context,
            trigger_reason="preflight",
        )
    )

    assert result.status == "compressed"
    assert context.metadata["context_engine"]["hook_errors"] == [
        {"hook": "before_context_compression", "error": "RuntimeError"}
    ]
