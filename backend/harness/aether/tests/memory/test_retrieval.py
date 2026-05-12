from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from aether.memory.budget import estimate_text_tokens, pack_memory_blocks
from aether.memory.contracts import (
    MemoryBlock,
    MemoryBundle,
    MemoryKind,
    MemoryMode,
    MemoryQuery,
    MemoryScope,
)
from aether.memory.project_store import ProjectMemoryStore
from aether.memory.retrieval import (
    BLOCK_TOKEN_MAX,
    MAX_PROJECT_BLOCKS,
    SCORE_INJECTION_THRESHOLD,
    QueryFeatures,
    RetrievalMemoryProvider,
    extract_query_features,
    recall_project_candidates,
    score_block,
)


def _make_block(
    *,
    id: str = "b1",
    scope: MemoryScope = MemoryScope.PROJECT,
    kind: MemoryKind = MemoryKind.PROJECT_FACT,
    text: str = "some memory content",
    source: str = "test",
    relevance: float = 0.0,
    confidence: str = "medium",
    tags: tuple[str, ...] = (),
    updated_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryBlock:
    return MemoryBlock(
        id=id,
        scope=scope,
        kind=kind,
        text=text,
        source=source,
        token_estimate=estimate_text_tokens(text),
        relevance=relevance,
        confidence=confidence,
        tags=tags,
        updated_at=updated_at,
        metadata=metadata or {},
    )


def _make_query(
    *,
    user_message: str = "continue",
    mode: MemoryMode = MemoryMode.PROJECT,
    token_budget: int = 2000,
    active_files: tuple[str, ...] = (),
) -> MemoryQuery:
    return MemoryQuery(
        session_id="test-session",
        task_id="test-task",
        user_message=user_message,
        recent_messages=[],
        mode=mode,
        token_budget=token_budget,
        active_files=active_files,
    )


def _seed_project_store(store: ProjectMemoryStore, entries: list[dict[str, Any]]) -> None:
    store.initialize()
    for entry in entries:
        store.write_entry(**entry)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_task_memory_wins_over_project_under_budget_pressure(tmp_path: Path) -> None:
    store = ProjectMemoryStore(tmp_path, store_root=tmp_path / ".aether" / "memory")
    _seed_project_store(store, [
        {"topic": "decisions", "text": "Use markdown store for project memory.", "tags": ("memory",)},
    ])

    provider = RetrievalMemoryProvider(project_store=store)
    provider.task_provider.observe_turn(
        session_id="test-session",
        task_id="test-task",
        messages=[{"role": "user", "content": "Implement memory retrieval. You must keep it deterministic."}],
        metadata={},
    )

    bundle = provider.retrieve(_make_query(
        user_message="Continue with memory implementation.",
        token_budget=80,
    ))

    assert bundle.blocks
    first = bundle.blocks[0]
    assert first.scope in (MemoryScope.TASK, MemoryScope.SESSION)


def test_project_path_match_boosts_relevance() -> None:
    block_matching = _make_block(
        id="match",
        text="Memory injection is outbound-only.",
        metadata={"paths": ["backend/harness/aether/agents/core/agent.py"]},
    )
    block_no_match = _make_block(
        id="nomatch",
        text="Memory injection is outbound-only.",
        metadata={"paths": ["frontend/components/button.tsx"]},
    )

    features = extract_query_features(_make_query(
        user_message="Fix memory injection.",
        active_files=("backend/harness/aether/agents/core/agent.py",),
    ))

    score_match = score_block(block_matching, features)
    score_no_match = score_block(block_no_match, features)

    assert score_match > score_no_match


def test_low_score_candidates_are_not_injected() -> None:
    block = _make_block(
        text="Completely unrelated topic about cooking recipes.",
        scope=MemoryScope.USER,
    )
    features = QueryFeatures(
        keywords={"retrieval", "ranking", "memory"},
        mode=MemoryMode.PROJECT,
    )

    score = score_block(block, features)
    assert score < SCORE_INJECTION_THRESHOLD


def test_user_memory_excluded_in_project_mode(tmp_path: Path) -> None:
    store = ProjectMemoryStore(tmp_path, store_root=tmp_path / ".aether" / "memory")
    store.initialize()

    provider = RetrievalMemoryProvider(project_store=store)
    provider.task_provider.observe_turn(
        session_id="s1",
        task_id="t1",
        messages=[{"role": "user", "content": "Check my preferences."}],
        metadata={},
    )

    bundle = provider.retrieve(_make_query(
        user_message="Check my preferences.",
        mode=MemoryMode.PROJECT,
    ))

    for block in bundle.blocks:
        assert block.scope != MemoryScope.USER


def test_budget_packer_reserves_estimation_headroom() -> None:
    blocks = [
        _make_block(id=f"b{i}", text=f"Memory fact number {i} with enough detail.", relevance=10.0 - i)
        for i in range(5)
    ]

    budget = 100
    result = pack_memory_blocks(blocks, token_budget=budget, block_token_max=500, reserve_pct=0.10)

    assert result.blocks
    total_tokens = sum(b.token_estimate for b in result.blocks)
    assert total_tokens <= int(budget * 0.90)


def test_large_block_is_truncated_before_injection() -> None:
    large_text = "Important memory fact. " * 300
    blocks = [_make_block(text=large_text, relevance=10.0)]

    result = pack_memory_blocks(blocks, token_budget=200, block_token_max=BLOCK_TOKEN_MAX)

    assert result.blocks
    assert result.blocks[0].token_estimate <= BLOCK_TOKEN_MAX
    assert "[memory truncated]" in result.blocks[0].text


def test_retrieval_timeout_returns_skipped_bundle(tmp_path: Path) -> None:
    store = ProjectMemoryStore(tmp_path, store_root=tmp_path / ".aether" / "memory")
    store.initialize()

    def _slow_load_index() -> dict[str, Any]:
        raise OSError("simulated timeout / IO error")

    provider = RetrievalMemoryProvider(project_store=store)

    with mock.patch.object(store, "load_index", side_effect=_slow_load_index), \
         mock.patch.object(store, "rebuild_index", side_effect=OSError("also fails")):
        bundle = provider.retrieve(_make_query(
            user_message="What decisions have we made about memory?",
            mode=MemoryMode.PROJECT,
        ))

    assert bundle.skipped_reason in ("no_candidates", "no_relevant_blocks") or bundle.blocks == ()


def test_corrupted_project_index_rebuilds_or_skips_safely(tmp_path: Path) -> None:
    store = ProjectMemoryStore(tmp_path, store_root=tmp_path / ".aether" / "memory")
    _seed_project_store(store, [
        {"topic": "decisions", "text": "Use deterministic retrieval.", "tags": ("memory", "retrieval")},
    ])

    store.index_path.write_text("NOT VALID JSON", encoding="utf-8")

    provider = RetrievalMemoryProvider(project_store=store)
    bundle = provider.retrieve(_make_query(
        user_message="What decisions about retrieval?",
        mode=MemoryMode.PROJECT,
    ))

    assert bundle.blocks is not None
    has_project = any(b.scope == MemoryScope.PROJECT for b in bundle.blocks)
    assert has_project or bundle.skipped_reason is not None
