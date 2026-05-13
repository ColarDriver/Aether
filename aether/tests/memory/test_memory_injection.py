from __future__ import annotations

import copy
from collections import deque
from typing import Any, Iterable

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
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks, HookOutcome
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.strategies import ClassifiedRecoveryStrategy
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


# ---------------------------------------------------------------------
# PR 8.2 acceptance — budget, metadata privacy, compaction boundary.
# ---------------------------------------------------------------------


def test_no_memory_injection_when_budget_too_small() -> None:
    """When the effective memory budget falls under the floor, skip injection.

    Setting ``memory_token_budget_max`` below ``DEFAULT_MIN_MEMORY_TOKENS``
    (300) forces ``resolve_memory_token_budget`` to report
    ``budget_too_small``; the engine must not retrieve memory and must
    record the reason in metadata.
    """
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(
            use_builtin_tools=False,
            memory_token_budget_max=100,  # < DEFAULT_MIN_MEMORY_TOKENS (300)
        ),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-tiny-budget", user_message="Continue")
    )

    assert result.status is EngineStatus.COMPLETED
    # Memory provider not consulted when budget is too small.
    assert memory.queries == []
    provider_user = provider.calls[0]["messages"][-1]
    assert "<memory_context>" not in provider_user["content"]
    assert result.metadata["memory"]["skipped_reason"] == "budget_too_small"
    assert result.metadata["memory"]["injected_count"] == 0
    assert result.metadata["memory"]["injected_tokens"] == 0


def test_memory_metadata_does_not_include_block_text_by_default() -> None:
    """Metadata records counts/scopes/source-side facts, not block contents.

    Logs and turn telemetry must be safe to ship without leaking the
    text inside retrieved memory blocks.  The default metadata shape
    deliberately excludes ``text`` / ``content`` / ``source`` of any
    individual block.
    """
    provider = RecordingProvider()
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(use_builtin_tools=False),
    )

    result = engine.run_turn(
        EngineRequest(session_id="memory-metadata", user_message="What do we know?")
    )

    assert result.status is EngineStatus.COMPLETED
    meta = result.metadata["memory"]

    # Stable shape — assert the schema explicitly so future changes hit
    # this test before they hit telemetry.
    expected_keys = {
        "enabled",
        "mode",
        "retrieval_ms",
        "candidate_count",
        "injected_count",
        "injected_tokens",
        "scopes",
        "skipped_reason",
        "write_count",
        "error",
    }
    assert set(meta.keys()) == expected_keys

    # Block text must not appear anywhere in metadata, even indirectly.
    block = _project_block()
    serialised = repr(meta)
    assert block.text not in serialised
    assert block.source not in serialised
    assert "Memory injection is outbound-only." not in serialised


def test_preflight_compaction_input_excludes_retrieved_memory() -> None:
    """Preflight compaction must operate on canonical messages.

    Memory injection happens AFTER preflight in the run loop, so the
    summariser fork must never see a ``<memory_context>`` block.  We
    assert that property directly on the fork's recorded payload.
    """
    provider = _MemoryCompactionProvider(
        main_script=[NormalizedResponse(content="ok after preflight")],
        fork_script=[
            NormalizedResponse(content="## summary\n\n* preflight " + ("x " * 200))
        ],
    )
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(
            use_builtin_tools=False,
            tool_use_contract_enabled=False,
            compression_enabled=True,
            autocompact_enabled=True,
            # Force preflight to fire every turn regardless of token count.
            compression_pre_llm_pct=0.0,
            compression_autocompact_pct=0.0,
            compression_protect_first_n=1,
            compression_protect_last_n=1,
            compression_target_summary_tokens=100,
        ),
    )

    seed = [
        {"role": "user", "content": f"seed-{i} " + ("x" * 800)}
        for i in range(6)
    ]
    result = engine.run_turn(
        EngineRequest(
            session_id="memory-preflight",
            messages=seed,
            user_message="answer",
        )
    )

    assert result.status is EngineStatus.COMPLETED
    # Preflight fork ran before any memory injection.
    assert provider.fork_calls >= 1
    fork_messages = provider.last_fork_messages or []
    fork_payload = "\n".join(
        m.get("content", "") for m in fork_messages if isinstance(m.get("content"), str)
    )
    assert "<memory_context>" not in fork_payload, (
        "preflight compaction saw injected memory; canonical → outbound boundary leaked"
    )


def test_recovery_compaction_strips_memory_before_summariser() -> None:
    """Context-overflow recovery → summariser must not see injected memory.

    Setup:

    * ``context_window=10_000`` and ``compression_pre_llm_pct=0.5`` so
      preflight does NOT fire on a small seed (we want only the
      recovery fork in this test).
    * Memory budget formula gives a healthy positive budget so memory
      IS injected on the first main call.
    * First main call raises ``context_overflow``; engine recovery
      runs the summariser fork and retries.

    Assertion: the recovery fork's input and the retry's main payload
    are both free of ``<memory_context>`` — PR 8.2 boundary.
    """
    provider = _MemoryCompactionProvider(
        main_script=[
            _context_overflow_error(),
            NormalizedResponse(content="ok after compress"),
        ],
        fork_script=[
            NormalizedResponse(content="## summary\n\n* recovery"),
        ],
        context_window=10_000,
    )
    memory = StaticMemoryProvider([_project_block()])
    engine = AgentEngine(
        provider,
        memory_provider=memory,
        config=EngineConfig(
            max_iterations=4,
            use_builtin_tools=False,
            tool_use_contract_enabled=False,
            compression_enabled=True,
            autocompact_enabled=True,
            # High enough that preflight stays inert on a small prompt.
            compression_pre_llm_pct=0.5,
            compression_autocompact_pct=0.0,
            compression_protect_first_n=1,
            compression_protect_last_n=1,
            compression_target_summary_tokens=100,
            compression_max_failures=3,
            classified_recovery_enabled=True,
        ),
        recovery_strategy=ClassifiedRecoveryStrategy(
            max_attempts=3,
            base_wait_seconds=0.0,
            rate_limit_fallback_threshold_seconds=0.0,
        ),
    )

    # Seed sized so preflight stays inert AND memory budget remains
    # positive AND the recovery summariser can strictly reduce tokens:
    # ~4 × 500 chars ≈ ~500 tokens, threshold 5_000, remaining ~4_500.
    seed = [
        {"role": "user", "content": f"seed-{i} " + ("x" * 500)}
        for i in range(4)
    ]
    result = engine.run_turn(
        EngineRequest(
            session_id="memory-recovery",
            messages=seed,
            user_message="please answer",
        )
    )

    assert result.status is EngineStatus.COMPLETED, result.exit_reason
    assert result.final_response == "ok after compress"

    # Sanity: memory WAS injected on the first attempt (otherwise the
    # boundary assertions below would be vacuous).
    assert len(memory.queries) >= 1
    first_main_payload = "\n".join(
        m.get("content", "")
        for m in provider.main_message_log[0]
        if isinstance(m.get("content"), str)
    )
    assert "<memory_context>" in first_main_payload, (
        "test premise broken: memory should have been injected on the "
        "first attempt so the recovery strip has something to strip"
    )

    # Recovery fork ran exactly once; preflight stayed inert.
    assert provider.fork_calls == 1, provider.fork_calls
    assert provider.main_calls == 2

    # PR 8.2 boundary: recovery fork did NOT see memory.
    recovery_fork_payload = "\n".join(
        m.get("content", "")
        for m in provider.fork_message_log[0]
        if isinstance(m.get("content"), str)
    )
    assert "<memory_context>" not in recovery_fork_payload, (
        "recovery compaction summariser saw <memory_context>; "
        "PR 8.2 boundary requires it to be stripped before compaction"
    )

    # And the retry's main payload is also memory-free — the doc says
    # "重试前应重新计算 memory budget; 不能复用旧注入文本".  We satisfy
    # this by stripping for the retry; a fresh injection happens only
    # on a subsequent outer-loop iteration.
    retry_main_payload = "\n".join(
        m.get("content", "")
        for m in provider.main_message_log[1]
        if isinstance(m.get("content"), str)
    )
    assert "<memory_context>" not in retry_main_payload


# ---------------------------------------------------------------------
# Test doubles for compaction-aware paths.
# ---------------------------------------------------------------------


def _context_overflow_error() -> ProviderInvocationError:
    return ProviderInvocationError(
        status_code=400,
        body_summary="prompt is too long: context length exceeded",
    )


class _MemoryCompactionProvider(ModelProvider):
    """Provider that branches on compaction sub-calls vs main loop calls.

    Adapted from ``aether/tests/engine/test_compaction_integration.py``
    with extra per-call message logging so PR 8.2 boundary assertions
    can inspect *each* fork / main payload (not just the last one).
    """

    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(
        self,
        *,
        main_script: Iterable[Any],
        fork_script: Iterable[Any] | None = None,
        model: str = "test-model",
        context_window: int = 100,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self._main: deque[Any] = deque(main_script)
        self._fork: deque[Any] = deque(fork_script or [])
        self.main_calls = 0
        self.fork_calls = 0
        self.last_main_messages: list[dict] | None = None
        self.last_fork_messages: list[dict] | None = None
        self.main_message_log: list[list[dict]] = []
        self.fork_message_log: list[list[dict]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del tools, config, stream_callback, stream_silent_callback
        snapshot = copy.deepcopy(messages)
        if context.metadata.get("_compaction_in_progress"):
            self.fork_calls += 1
            self.last_fork_messages = snapshot
            self.fork_message_log.append(snapshot)
            if not self._fork:
                raise RuntimeError(f"{type(self).__name__}: fork script exhausted")
            outcome = self._fork.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        self.main_calls += 1
        self.last_main_messages = snapshot
        self.main_message_log.append(snapshot)
        if not self._main:
            raise RuntimeError(f"{type(self).__name__}: main script exhausted")
        outcome = self._main.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
