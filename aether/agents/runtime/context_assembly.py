"""Provider-bound context assembly for AgentEngine PRE_LLM."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from aether.runtime.context.default_engine import DefaultContextEngine
from aether.runtime.context.engine import ContextEngine, ContextEngineResult
from aether.runtime.core.contracts import EngineRequest, TurnContext
from aether.runtime.core.hooks import EngineHooks, HookOutcome
from aether.runtime.core.services import EngineServices


@dataclass(slots=True)
class ContextAssemblyInput:
    request: EngineRequest
    messages: list[dict[str, Any]]
    context: TurnContext
    iteration: int


@dataclass(slots=True)
class ContextAssemblyResult:
    canonical_messages: list[dict[str, Any]]
    prepared_messages: list[dict[str, Any]]
    hook_outcome: HookOutcome
    preflight_compaction: ContextEngineResult | None = None


class ContextAssemblyAdapter(Protocol):
    def maybe_compact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> Any | None: ...

    def register_skill_nudge(self, context: TurnContext) -> None: ...

    def maybe_inject_skill_nudge(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...

    def drain_pending_messages(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...

    def maybe_inject_diagnostic_attachment(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...

    def maybe_inject_verifier_reminder(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...

    def maybe_inject_plan_mode_attachment(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
        *,
        session_id: str | None,
    ) -> list[dict[str, Any]]: ...

    def collect_pre_llm_hook_outcome(self, name: str, **kwargs: Any) -> HookOutcome: ...

    def consume_messages_override(self, context: TurnContext) -> list[dict[str, Any]] | None: ...

    def merge_memory_context_into_hook_outcome(
        self,
        messages: list[dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> HookOutcome: ...

    def apply_hook_outcome_to_messages(
        self,
        messages: list[dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...

    def apply_collapse_view(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...


class LegacyContextAssemblyAdapter:
    """Bridge the new pipeline to the existing AgentEngine helpers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def maybe_compact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> Any | None:
        return self._engine._maybe_compact_messages(  # noqa: SLF001
            messages,
            context=context,
            trigger_reason=trigger_reason,
        )

    def register_skill_nudge(self, context: TurnContext) -> None:
        self._engine._register_skill_nudge(context)  # noqa: SLF001

    def maybe_inject_skill_nudge(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._engine._maybe_inject_skill_nudge(messages, context)  # noqa: SLF001

    def drain_pending_messages(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._engine._drain_pending_messages(messages, context)  # noqa: SLF001

    def maybe_inject_diagnostic_attachment(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._engine._maybe_inject_diagnostic_attachment(messages, context)  # noqa: SLF001

    def maybe_inject_verifier_reminder(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._engine._maybe_inject_verifier_reminder(messages, context)  # noqa: SLF001

    def maybe_inject_plan_mode_attachment(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
        *,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        return self._engine._maybe_inject_plan_mode_attachment(  # noqa: SLF001
            messages,
            context,
            session_id=session_id,
        )

    def collect_pre_llm_hook_outcome(self, name: str, **kwargs: Any) -> HookOutcome:
        return self._engine._collect_pre_llm_hook_outcome(name, **kwargs)  # noqa: SLF001

    def consume_messages_override(self, context: TurnContext) -> list[dict[str, Any]] | None:
        loop_messages = context.metadata.pop("_messages_override", None)
        return loop_messages if isinstance(loop_messages, list) else None

    def merge_memory_context_into_hook_outcome(
        self,
        messages: list[dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> HookOutcome:
        return self._engine._merge_memory_context_into_hook_outcome(  # noqa: SLF001
            messages,
            outcome,
            context=context,
        )

    def apply_hook_outcome_to_messages(
        self,
        messages: list[dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._engine._apply_hook_outcome_to_messages(  # noqa: SLF001
            messages,
            outcome,
            context=context,
        )

    def apply_collapse_view(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        return self._engine._apply_collapse_view(messages, context)  # noqa: SLF001


class ContextAssemblyPipeline:
    """Build canonical and provider-bound messages in a stable order."""

    def __init__(
        self,
        *,
        services: EngineServices,
        hooks: EngineHooks,
        adapter: ContextAssemblyAdapter,
        context_engine: ContextEngine | None = None,
    ) -> None:
        self._services = services
        self._hooks = hooks
        self._adapter = adapter
        self._context_engine = context_engine or DefaultContextEngine(adapter=adapter)

    def assemble(self, assembly: ContextAssemblyInput) -> ContextAssemblyResult:
        messages = assembly.messages
        context = assembly.context
        preflight_compaction = None
        context_engine_meta = dict(context.metadata.get("context_engine") or {})
        context_engine_meta["name"] = self._context_engine.name
        context.metadata["context_engine"] = context_engine_meta

        if not context.metadata.get("_preflight_compaction_done"):
            context.metadata["_preflight_compaction_done"] = True
            if self._context_engine.should_compress_preflight(messages, context=context):
                preflight_compaction = self._context_engine.compact_preflight(
                    messages,
                    context=context,
                    trigger_reason="preflight",
                )
                messages = preflight_compaction.messages

        self._adapter.register_skill_nudge(context)
        messages = self._adapter.maybe_inject_skill_nudge(messages, context)
        messages = self._adapter.drain_pending_messages(messages, context)
        messages = self._adapter.maybe_inject_diagnostic_attachment(messages, context)
        messages = self._adapter.maybe_inject_verifier_reminder(messages, context)
        messages = self._adapter.maybe_inject_plan_mode_attachment(
            messages,
            context,
            session_id=assembly.request.session_id,
        )

        hook_outcome = self._adapter.collect_pre_llm_hook_outcome(
            "pre_llm_call",
            session_id=assembly.request.session_id,
            iteration=assembly.iteration,
            messages=copy.deepcopy(messages),
            context_metadata=context.metadata,
        )

        override = self._adapter.consume_messages_override(context)
        if override is not None:
            messages = override

        hook_outcome = self._adapter.merge_memory_context_into_hook_outcome(
            messages,
            hook_outcome,
            context=context,
        )
        outbound_messages = self._adapter.apply_hook_outcome_to_messages(
            messages,
            hook_outcome,
            context=context,
        )
        prepared_messages = self._services.middleware_pipeline.run_before_llm(
            outbound_messages,
            context,
        )
        prepared_messages = self._context_engine.apply_provider_projection(
            prepared_messages,
            context=context,
        )
        return ContextAssemblyResult(
            canonical_messages=messages,
            prepared_messages=prepared_messages,
            hook_outcome=hook_outcome,
            preflight_compaction=preflight_compaction,
        )
