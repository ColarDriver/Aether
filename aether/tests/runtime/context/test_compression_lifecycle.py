from __future__ import annotations

from typing import Any

from aether.runtime.context import (
    CompressionLifecycleService,
    CompressionRequest,
    ContextEngineResult,
)
from aether.runtime.core.contracts import TurnContext


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
