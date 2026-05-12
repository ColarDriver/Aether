from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aether.config.schema import EngineConfig
from aether.memory.project_store import ProjectMemoryStore
from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.tools.builtins.memory import (
    MemoryForgetTool,
    MemoryListTool,
    MemoryReadTool,
    MemoryUpdateTool,
    MemoryWriteTool,
)


def _ctx(
    tmp_path: Path,
    *,
    memory_mode: str = "project",
    auto_write: bool = True,
) -> tuple[TurnContext, ProjectMemoryStore]:
    store = ProjectMemoryStore(tmp_path, store_root=tmp_path / ".aether" / "memory")
    store.initialize()

    config = EngineConfig(
        memory_mode=memory_mode,
        memory_auto_write_enabled=auto_write,
        use_builtin_tools=False,
    )
    ctx = TurnContext(session_id="test-session", iteration=1)
    ctx.metadata["_project_memory_store"] = store
    ctx.metadata["_engine_config"] = config
    return ctx, store


def _call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=args)


def test_memory_read_returns_short_relevant_summary(tmp_path: Path) -> None:
    ctx, store = _ctx(tmp_path, auto_write=True)
    store.write_entry(topic="decisions", text="Use markdown store for all project memory persistence.")

    tool = MemoryReadTool()
    result = tool.execute(_call("memory_read", query="markdown"), ctx)

    assert not result.is_error
    assert "markdown" in result.content.lower()
    assert len(result.content) < 500


def test_memory_write_project_scope_requires_reason(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path)
    tool = MemoryWriteTool()

    result = tool.execute(
        _call("memory_write", topic="decisions", text="Some fact."),
        ctx,
    )

    assert result.is_error
    assert "reason_required" in result.content


def test_memory_write_user_scope_denied_in_project_mode(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path, memory_mode="project")
    tool = MemoryWriteTool()

    result = tool.execute(
        _call(
            "memory_write",
            topic="decisions",
            text="User preference.",
            scope="user",
            reason="User asked to save.",
        ),
        ctx,
    )

    assert result.is_error
    assert "user_scope_denied" in result.content


def test_memory_write_rejects_secret_like_text(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path)
    tool = MemoryWriteTool()

    result = tool.execute(
        _call(
            "memory_write",
            topic="pitfalls",
            text="API key is sk-abc123def456ghi789jkl012",
            reason="User asked to save.",
        ),
        ctx,
    )

    assert result.is_error
    assert "secret_detected" in result.content


def test_memory_update_preserves_entry_id_and_source(tmp_path: Path) -> None:
    ctx, store = _ctx(tmp_path)

    store.write_entry(
        topic="decisions",
        text="Original text.",
        entry_id="keep-this-id",
        source_session="original-session",
    )

    tool = MemoryUpdateTool()
    result = tool.execute(
        _call(
            "memory_update",
            entry_id="keep-this-id",
            topic="decisions",
            text="Updated text.",
            reason="Correcting the decision.",
        ),
        ctx,
    )

    assert not result.is_error
    assert "keep-this-id" in result.content

    entries = store.read_entries(topic="decisions")
    matching = [e for e in entries if e.id == "keep-this-id"]
    assert len(matching) == 1
    assert matching[0].text == "Updated text."
    assert matching[0].created_at


def test_memory_forget_writes_tombstone(tmp_path: Path) -> None:
    ctx, store = _ctx(tmp_path)

    store.write_entry(
        topic="decisions",
        text="This will be forgotten.",
        entry_id="forget-me",
    )

    tool = MemoryForgetTool()
    result = tool.execute(
        _call(
            "memory_forget",
            entry_id="forget-me",
            topic="decisions",
            reason="No longer relevant.",
        ),
        ctx,
    )

    assert not result.is_error
    assert "tombstone" in result.content.lower()

    entries = store.read_entries(topic="decisions")
    matching = [e for e in entries if e.id == "forget-me"]
    assert len(matching) == 1
    assert "[deleted]" in matching[0].text
    assert "deleted" in matching[0].tags


def test_non_interactive_memory_write_denied_by_default(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path, auto_write=False)
    tool = MemoryWriteTool()

    result = tool.execute(
        _call(
            "memory_write",
            topic="decisions",
            text="Should be denied.",
            reason="Trying to auto-write.",
        ),
        ctx,
    )

    assert result.is_error
    assert "auto_write_disabled" in result.content


def test_memory_tool_results_do_not_dump_full_topic_file(tmp_path: Path) -> None:
    ctx, store = _ctx(tmp_path)

    for i in range(5):
        store.write_entry(
            topic="architecture",
            text=f"Architecture decision number {i} with enough detail to be meaningful.",
        )

    read_tool = MemoryReadTool()
    read_result = read_tool.execute(
        _call("memory_read", query="architecture"), ctx,
    )

    list_tool = MemoryListTool()
    list_result = list_tool.execute(
        _call("memory_list", topic="architecture"), ctx,
    )

    assert len(read_result.content) < 500
    assert len(list_result.content) < 500
