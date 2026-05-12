from __future__ import annotations

import copy
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.memory import (
    MemoryBlock,
    MemoryBundle,
    MemoryKind,
    MemoryQuery,
    MemoryScope,
)
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks, HookOutcome
from aether.tools.base import ToolDescriptor


class RecordingProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"
    context_window = 32_000

    def __init__(self) -> None:
        self.model = "test-model"
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback=None,
        stream_silent_callback=None,
    ) -> NormalizedResponse:
        del tools, config, stream_callback, stream_silent_callback
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "memory": dict(context.metadata.get("memory") or {}),
            }
        )
        return NormalizedResponse(content="ok")


class StaticMemoryProvider:
    def __init__(self, blocks: list[MemoryBlock]) -> None:
        self.blocks = blocks
        self.queries: list[MemoryQuery] = []

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        self.queries.append(query)
        return MemoryBundle.from_blocks(self.blocks, latency_ms=1.0)

    def observe_turn(self, *, session_id: str, task_id: str | None, messages: list[dict], metadata: dict) -> None:
        return None

    def before_compaction(self, *, session_id: str, task_id: str | None, messages: list[dict], metadata: dict) -> None:
        return None


class FailingMemoryProvider(StaticMemoryProvider):
    def __init__(self) -> None:
        super().__init__([])

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        self.queries.append(query)
        raise RuntimeError("memory backend unavailable")


class UserContextHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        return HookOutcome(inject_user_context="hook provided context")


class ShortCircuitHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        return HookOutcome(short_circuit_response=NormalizedResponse(content="from hook"))


def _project_block() -> MemoryBlock:
    return MemoryBlock(
        id="project-decision-1",
        scope=MemoryScope.PROJECT,
        kind=MemoryKind.DECISION,
        text="Memory injection is outbound-only.",
        source=".aether/memory/topics/decisions.md#project-decision-1",
        token_estimate=12,
        relevance=10.0,
    )


def _user_block() -> MemoryBlock:
    return MemoryBlock(
        id="user-pref-1",
        scope=MemoryScope.USER,
        kind=MemoryKind.USER_PREFERENCE,
        text="User prefers personal long-term memory.",
        source="user/profile.md#user-pref-1",
        token_estimate=12,
        relevance=10.0,
    )


def test_memory_injection_only_affects_provider_payload() -> None:
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-inject", user_message="What do we know?")
    )

    assert result.status is EngineStatus.COMPLETED
    assert len(memory.queries) == 1
    assert memory.queries[0].user_message == "What do we know?"
    provider_user = provider.calls[0]["messages"][-1]
    assert provider_user["role"] == "user"
    assert "<memory_context>" in provider_user["content"]
    assert "Memory injection is outbound-only." in provider_user["content"]
    assert "Retrieved memory may be stale" in provider_user["content"]
    assert result.messages[0]["content"] == "What do we know?"
    assert "<memory_context>" not in result.messages[0]["content"]
    assert result.metadata["memory"]["candidate_count"] == 1
    assert result.metadata["memory"]["injected_count"] == 1
    assert result.metadata["memory"]["scopes"] == ["project"]


def test_memory_context_merges_with_existing_hook_user_context() -> None:
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        hooks=UserContextHook(),
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-hook-merge", user_message="Continue")
    )

    assert result.status is EngineStatus.COMPLETED
    provider_user = provider.calls[0]["messages"][-1]
    assert provider_user["content"].count("<hook_context>") == 1
    assert "hook provided context" in provider_user["content"]
    assert "<memory_context>" in provider_user["content"]


def test_user_memory_is_not_injected_in_project_mode() -> None:
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_user_block(), _project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False, memory_mode="project"),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-project-mode", user_message="Continue")
    )

    provider_user = provider.calls[0]["messages"][-1]
    assert "Memory injection is outbound-only." in provider_user["content"]
    assert "User prefers personal long-term memory." not in provider_user["content"]
    assert result.metadata["memory"]["candidate_count"] == 2
    assert result.metadata["memory"]["injected_count"] == 1
    assert result.metadata["memory"]["scopes"] == ["project"]


def test_memory_provider_exception_is_soft_failure() -> None:
    provider = RecordingProvider()
    memory = FailingMemoryProvider()
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-failure", user_message="Continue")
    )

    assert result.status is EngineStatus.COMPLETED
    provider_user = provider.calls[0]["messages"][-1]
    assert "<memory_context>" not in provider_user["content"]
    assert result.metadata["memory"]["skipped_reason"] == "provider_error"
    assert result.metadata["memory"]["error"] == "RuntimeError"


def test_memory_disabled_skips_provider_retrieval() -> None:
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False, memory_enabled=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-disabled", user_message="Continue")
    )

    assert result.status is EngineStatus.COMPLETED
    assert memory.queries == []
    assert result.metadata["memory"]["enabled"] is False
    assert result.metadata["memory"]["skipped_reason"] == "disabled"


def test_short_circuit_hook_skips_memory_retrieval() -> None:
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        hooks=ShortCircuitHook(),
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-short-circuit", user_message="Continue")
    )

    assert result.status is EngineStatus.COMPLETED
    assert result.final_response == "from hook"
    assert provider.calls == []
    assert memory.queries == []
    assert result.metadata["memory"]["skipped_reason"] == "short_circuit"
