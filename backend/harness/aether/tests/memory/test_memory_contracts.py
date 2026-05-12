from __future__ import annotations

import pytest

from aether.config.schema import EngineConfig
from aether.memory import (
    MemoryBlock,
    MemoryBundle,
    MemoryKind,
    MemoryMode,
    MemoryQuery,
    MemoryScope,
    NullMemoryProvider,
    normalize_memory_mode,
    scopes_for_mode,
)
from aether.memory.contracts import MemoryProvider
from aether.agents.memory import MemoryProvider as LegacyMemoryProvider


def test_engine_config_memory_defaults_are_task_project_oriented() -> None:
    config = EngineConfig()

    assert config.memory_enabled is True
    assert config.memory_mode == "project"
    assert config.memory_user_profile_enabled is False
    assert config.memory_auto_write_enabled is False
    assert config.memory_token_budget_pct == 0.08
    assert config.memory_token_budget_max == 2500
    assert config.memory_block_token_max == 500
    assert config.memory_retrieval_timeout_ms == 200


def test_normalize_memory_mode_defaults_invalid_values_to_project() -> None:
    assert normalize_memory_mode("task") is MemoryMode.TASK
    assert normalize_memory_mode("PERSONAL_ASSISTANT") is MemoryMode.PERSONAL_ASSISTANT
    assert normalize_memory_mode("unknown") is MemoryMode.PROJECT


def test_scopes_for_project_mode_excludes_user_scope() -> None:
    assert scopes_for_mode(MemoryMode.PROJECT, user_profile_enabled=True) == (
        MemoryScope.SESSION,
        MemoryScope.TASK,
        MemoryScope.PROJECT,
    )


def test_scopes_for_personal_mode_requires_user_profile_flag() -> None:
    assert MemoryScope.USER not in scopes_for_mode(
        MemoryMode.PERSONAL_ASSISTANT,
        user_profile_enabled=False,
    )
    assert MemoryScope.USER in scopes_for_mode(
        MemoryMode.PERSONAL_ASSISTANT,
        user_profile_enabled=True,
    )


def test_memory_block_requires_provenance_and_token_estimate() -> None:
    with pytest.raises(ValueError, match="source"):
        MemoryBlock(
            id="m1",
            scope=MemoryScope.PROJECT,
            kind=MemoryKind.DECISION,
            text="Use outbound-only injection.",
            source="",
            token_estimate=8,
        )

    with pytest.raises(ValueError, match="token_estimate"):
        MemoryBlock(
            id="m1",
            scope=MemoryScope.PROJECT,
            kind=MemoryKind.DECISION,
            text="Use outbound-only injection.",
            source="memory.md",
            token_estimate=0,
        )


def test_memory_bundle_from_blocks_sums_token_estimates() -> None:
    block = MemoryBlock(
        id="m1",
        scope=MemoryScope.PROJECT,
        kind=MemoryKind.DECISION,
        text="Use outbound-only injection.",
        source="memory.md",
        token_estimate=8,
    )

    bundle = MemoryBundle.from_blocks([block])

    assert bundle.blocks == (block,)
    assert bundle.token_estimate == 8
    assert bundle.skipped_reason is None


def test_null_memory_provider_is_protocol_compatible_and_skips() -> None:
    provider = NullMemoryProvider()
    query = MemoryQuery(
        session_id="s1",
        task_id=None,
        user_message="What should I do next?",
        recent_messages=[],
    )

    assert isinstance(provider, MemoryProvider)
    assert LegacyMemoryProvider is MemoryProvider
    assert provider.retrieve(query).skipped_reason == "disabled"
    provider.observe_turn(session_id="s1", task_id=None, messages=[], metadata={})
    provider.before_compaction(session_id="s1", task_id=None, messages=[], metadata={})
