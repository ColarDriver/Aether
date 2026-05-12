from __future__ import annotations

import copy
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.memory import MemoryBundle, MemoryQuery, MemoryScope, RetrievalMemoryProvider, TaskMemoryProvider
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import EngineRequest, EngineStatus, NormalizedResponse, TurnContext
from aether.tools.base import ToolDescriptor


class RecordingProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"
    context_window = 32_000

    def __init__(self) -> None:
        self.model = "test-model"
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback=None,
        stream_silent_callback=None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append(copy.deepcopy(messages))
        return NormalizedResponse(content="We will keep the task memory local.")


class FailingObserveMemoryProvider:
    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        return MemoryBundle.skipped("no_candidates")

    def observe_turn(self, *, session_id: str, task_id: str | None, messages: list[dict], metadata: dict) -> None:
        raise RuntimeError("observe failed")

    def before_compaction(self, *, session_id: str, task_id: str | None, messages: list[dict], metadata: dict) -> None:
        return None


def test_task_snapshot_created_per_session() -> None:
    provider = TaskMemoryProvider()

    provider.observe_turn(
        session_id="s1",
        task_id="t1",
        messages=[{"role": "user", "content": "Implement task memory. You must keep it local."}],
        metadata={},
    )

    snapshot = provider.snapshot_for("s1", "t1")
    assert snapshot is not None
    assert snapshot.goal == "Implement task memory. You must keep it local."
    assert snapshot.constraints == ["You must keep it local."]
    assert snapshot.token_estimate > 0


def test_task_memory_does_not_leak_between_sessions() -> None:
    provider = TaskMemoryProvider()
    provider.observe_turn(
        session_id="s1",
        task_id="t1",
        messages=[{"role": "user", "content": "Remember alpha. You must keep alpha."}],
        metadata={},
    )
    provider.observe_turn(
        session_id="s2",
        task_id="t1",
        messages=[{"role": "user", "content": "Remember beta. You must keep beta."}],
        metadata={},
    )

    s1 = provider.retrieve(
        MemoryQuery(session_id="s1", task_id="t1", user_message="continue", recent_messages=[], token_budget=1000)
    )
    s2 = provider.retrieve(
        MemoryQuery(session_id="s2", task_id="t1", user_message="continue", recent_messages=[], token_budget=1000)
    )

    assert "alpha" in "\n".join(block.text for block in s1.blocks)
    assert "beta" not in "\n".join(block.text for block in s1.blocks)
    assert "beta" in "\n".join(block.text for block in s2.blocks)
    assert "alpha" not in "\n".join(block.text for block in s2.blocks)


def test_task_memory_extracts_explicit_constraints_and_decisions() -> None:
    provider = TaskMemoryProvider()

    provider.observe_turn(
        session_id="s1",
        task_id="t1",
        messages=[
            {"role": "user", "content": "Build memory. You must not write user profile memory."},
            {"role": "assistant", "content": "We will use a local task snapshot."},
            {"role": "tool", "name": "read_file", "content": "Found memory contracts in aether/memory."},
        ],
        metadata={},
    )

    snapshot = provider.snapshot_for("s1", "t1")
    assert snapshot is not None
    assert snapshot.constraints == ["You must not write user profile memory."]
    assert snapshot.decisions == ["We will use a local task snapshot."]
    assert snapshot.recent_findings == ["read_file: Found memory contracts in aether/memory."]


def test_task_memory_prioritizes_goal_under_small_budget() -> None:
    provider = TaskMemoryProvider()
    provider.observe_turn(
        session_id="s1",
        task_id="t1",
        messages=[
            {
                "role": "user",
                "content": "Implement task memory. You must keep messages canonical. You must keep memory local.",
            },
            {"role": "assistant", "content": "We will use task snapshots."},
        ],
        metadata={},
    )

    bundle = provider.retrieve(
        MemoryQuery(session_id="s1", task_id="t1", user_message="continue", recent_messages=[], token_budget=24)
    )

    assert bundle.blocks
    assert bundle.blocks[0].kind.value == "task_state"
    assert "Current goal" in bundle.blocks[0].text


def test_before_compaction_updates_snapshot_without_mutating_messages() -> None:
    provider = TaskMemoryProvider()
    messages = [{"role": "user", "content": "Continue. You must preserve canonical messages."}]
    before = copy.deepcopy(messages)

    provider.before_compaction(
        session_id="s1",
        task_id="t1",
        messages=messages,
        metadata={"memory": {"injected_count": 1}},
    )

    assert messages == before
    snapshot = provider.snapshot_for("s1", "t1")
    assert snapshot is not None
    assert snapshot.constraints == ["You must preserve canonical messages."]


def test_retrieved_project_memory_is_not_promoted_to_task_snapshot() -> None:
    provider = TaskMemoryProvider()

    provider.observe_turn(
        session_id="s1",
        task_id="t1",
        messages=[{"role": "assistant", "content": "ok"}],
        metadata={"memory": {"scopes": ["project"], "injected_count": 1}},
    )

    assert provider.snapshot_for("s1", "t1") is None


def test_default_agent_task_memory_persists_across_turns_by_session() -> None:
    model = RecordingProvider()
    engine = AgentEngine(model, config=EngineConfig(use_builtin_tools=False))

    first = engine.run_turn(
        EngineRequest(
            session_id="task-memory-session",
            user_message="Implement task memory. You must keep it session scoped.",
        )
    )
    second = engine.run_turn(
        EngineRequest(session_id="task-memory-session", user_message="Continue.")
    )

    assert first.status is EngineStatus.COMPLETED
    assert second.status is EngineStatus.COMPLETED
    assert len(model.calls) == 2
    second_user = model.calls[1][-1]
    assert "<memory_context>" in second_user["content"]
    assert "Current goal" in second_user["content"]
    assert "session scoped" in second_user["content"]
    assert second.metadata["memory"]["scopes"] == [MemoryScope.SESSION.value]


def test_default_agent_task_memory_lives_in_session_runtime() -> None:
    model = RecordingProvider()
    engine = AgentEngine(model, config=EngineConfig(use_builtin_tools=False))

    result = engine.run_turn(
        EngineRequest(
            session_id="runtime-backed-memory",
            user_message="Implement task memory. You must keep lifecycle session scoped.",
        )
    )

    assert result.status is EngineStatus.COMPLETED
    state = engine._session_runtime.get("runtime-backed-memory")
    assert state.task_memory_snapshots

    engine._session_runtime.discard("runtime-backed-memory")
    provider = engine.services.memory_provider
    assert isinstance(provider, RetrievalMemoryProvider)
    assert provider.task_provider.snapshot_for("runtime-backed-memory") is None


def test_task_memory_observe_failure_does_not_fail_turn() -> None:
    model = RecordingProvider()
    engine = AgentEngine(
        model,
        memory_provider=FailingObserveMemoryProvider(),
        config=EngineConfig(use_builtin_tools=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="observe-fails", user_message="Continue.")
    )

    assert result.status is EngineStatus.COMPLETED
    assert result.metadata["memory"]["error"] == "RuntimeError"
