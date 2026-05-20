from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether.runtime.context import DefaultContextEngine
from aether.runtime.core.contracts import TurnContext


@dataclass(slots=True)
class _RawCompaction:
    compressed_messages: list[dict[str, Any]]
    tokens_before: int = 10
    tokens_after: int = 5
    tiers_run: list[str] | None = None
    exhausted: bool = False


class _Adapter:
    def __init__(self, raw: _RawCompaction | None = None) -> None:
        self.raw = raw
        self.compaction_calls = 0
        self.projection_calls = 0

    def maybe_compact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> _RawCompaction | None:
        del messages, context, trigger_reason
        self.compaction_calls += 1
        return self.raw

    def apply_collapse_view(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        del context
        self.projection_calls += 1
        return [*messages, {"role": "user", "content": "projected"}]


def test_default_engine_returns_unchanged_messages_when_adapter_skips() -> None:
    adapter = _Adapter()
    engine = DefaultContextEngine(adapter=adapter)
    messages = [{"role": "user", "content": "hello"}]
    context = TurnContext(session_id="s", iteration=1, metadata={})

    result = engine.compact_preflight(
        messages,
        context=context,
        trigger_reason="preflight",
    )

    assert adapter.compaction_calls == 1
    assert result.messages is messages
    assert result.changed is False
    assert result.metadata["status"] == "skipped"


def test_default_engine_wraps_changed_compaction_result() -> None:
    compacted = [{"role": "user", "content": "summary"}]
    adapter = _Adapter(
        _RawCompaction(
            compressed_messages=compacted,
            tiers_run=["tier2_snip"],
        )
    )
    engine = DefaultContextEngine(adapter=adapter)

    result = engine.compact_preflight(
        [{"role": "user", "content": "hello"}],
        context=TurnContext(session_id="s", iteration=1, metadata={}),
        trigger_reason="preflight",
    )

    assert result.messages == compacted
    assert result.changed is True
    assert result.reason == "preflight"
    assert result.metadata == {
        "status": "compressed",
        "trigger_reason": "preflight",
        "tokens_before": 10,
        "tokens_after": 5,
        "tiers_run": ["tier2_snip"],
        "exhausted": False,
    }
    assert result.raw_result is adapter.raw


def test_provider_projection_returns_new_list_without_mutating_canonical() -> None:
    adapter = _Adapter()
    engine = DefaultContextEngine(adapter=adapter)
    messages = [{"role": "user", "content": "canonical"}]

    projected = engine.apply_provider_projection(
        messages,
        context=TurnContext(session_id="s", iteration=1, metadata={}),
    )

    assert adapter.projection_calls == 1
    assert projected is not messages
    assert messages == [{"role": "user", "content": "canonical"}]
    assert projected[-1]["content"] == "projected"


def test_context_engine_metadata_is_json_safe() -> None:
    adapter = _Adapter(
        _RawCompaction(
            compressed_messages=[{"role": "user", "content": "summary"}],
            tiers_run=["tier5_autocompact"],
            exhausted=True,
        )
    )
    engine = DefaultContextEngine(adapter=adapter)

    result = engine.compact_preflight(
        [{"role": "user", "content": "hello"}],
        context=TurnContext(session_id="s", iteration=1, metadata={}),
        trigger_reason="manual",
    )

    assert result.metadata == {
        "status": "compressed",
        "trigger_reason": "manual",
        "tokens_before": 10,
        "tokens_after": 5,
        "tiers_run": ["tier5_autocompact"],
        "exhausted": True,
    }
