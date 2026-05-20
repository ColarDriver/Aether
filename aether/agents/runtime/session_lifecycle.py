"""Session and turn lifecycle controller for AgentEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aether.runtime.core.contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    LoopState,
    TurnContext,
)
from aether.runtime.core.iteration_budget import IterationBudget
from aether.runtime.core.services import EngineServices
from aether.runtime.core.state_machine import EngineStateMachine


@dataclass(slots=True)
class TurnPreparationResult:
    state_machine: EngineStateMachine
    messages: list[dict[str, Any]]
    context: TurnContext
    active_system_prompt: str | None


@dataclass(slots=True)
class TurnFinalizationInput:
    request: EngineRequest
    messages: list[dict[str, Any]]
    context: TurnContext
    final_response: str | None
    error_text: str | None
    exit_reason: ExitReason
    iterations: int
    budget: IterationBudget | None
    state_machine: EngineStateMachine
    active_system_prompt: str | None


class SessionLifecycleAdapter(Protocol):
    def seed_session_cwd(self, request: EngineRequest) -> None: ...

    def prepare_turn_entry(
        self,
        request: EngineRequest,
    ) -> tuple[EngineStateMachine, list[dict[str, Any]], TurnContext]: ...

    def prepare_session_and_system_prompt(
        self,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> tuple[list[dict[str, Any]], str | None]: ...

    def handle_max_iterations(
        self,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> str | None: ...

    def observe_memory_turn(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> None: ...

    def build_result(
        self,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        iterations: int,
        final_response: str | None,
        error_text: str | None,
        exit_reason: ExitReason,
        *,
        context: TurnContext,
        active_system_prompt: str | None,
    ) -> EngineResult: ...

    def save_trajectory_if_enabled(
        self,
        *,
        result: EngineResult,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> None: ...

    def cleanup_task_resources_if_needed(
        self,
        *,
        result: EngineResult,
        context: TurnContext,
    ) -> None: ...

    def cleanup_task_resources(
        self,
        *,
        context: TurnContext,
        completed: bool,
        interrupted: bool,
    ) -> dict[str, Any]: ...

    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool: ...

    def clear_interrupt(self, session_id: str | None = None) -> None: ...

    def safe_call_hook(self, name: str, **kwargs: Any) -> Any: ...


class LegacySessionLifecycleAdapter:
    """Bridge lifecycle controller to AgentEngine's existing helpers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def seed_session_cwd(self, request: EngineRequest) -> None:
        self._engine._seed_session_cwd(request)  # noqa: SLF001

    def prepare_turn_entry(
        self,
        request: EngineRequest,
    ) -> tuple[EngineStateMachine, list[dict[str, Any]], TurnContext]:
        return self._engine._prepare_turn_entry(request)  # noqa: SLF001

    def prepare_session_and_system_prompt(
        self,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> tuple[list[dict[str, Any]], str | None]:
        return self._engine._prepare_session_and_system_prompt(  # noqa: SLF001
            request,
            messages,
            context,
        )

    def handle_max_iterations(
        self,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> str | None:
        return self._engine._handle_max_iterations(request, messages, context)  # noqa: SLF001

    def observe_memory_turn(
        self,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> None:
        self._engine._observe_memory_turn(messages, context)  # noqa: SLF001

    def build_result(
        self,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        iterations: int,
        final_response: str | None,
        error_text: str | None,
        exit_reason: ExitReason,
        *,
        context: TurnContext,
        active_system_prompt: str | None,
    ) -> EngineResult:
        return self._engine._build_result(  # noqa: SLF001
            request,
            messages,
            iterations,
            final_response,
            error_text,
            exit_reason,
            context=context,
            active_system_prompt=active_system_prompt,
        )

    def save_trajectory_if_enabled(self, **kwargs: Any) -> None:
        self._engine._save_trajectory_if_enabled(**kwargs)  # noqa: SLF001

    def cleanup_task_resources_if_needed(self, **kwargs: Any) -> None:
        self._engine._cleanup_task_resources_if_needed(**kwargs)  # noqa: SLF001

    def cleanup_task_resources(self, **kwargs: Any) -> dict[str, Any]:
        return self._engine._cleanup_task_resources(**kwargs)  # noqa: SLF001

    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool:
        return self._engine._is_interrupted(session_id, context)  # noqa: SLF001

    def clear_interrupt(self, session_id: str | None = None) -> None:
        self._engine.clear_interrupt(session_id=session_id)

    def safe_call_hook(self, name: str, **kwargs: Any) -> Any:
        return self._engine._safe_call_hook(name, **kwargs)  # noqa: SLF001


class SessionLifecycleController:
    """Coordinate per-turn setup and finalization."""

    def __init__(
        self,
        *,
        services: EngineServices,
        adapter: SessionLifecycleAdapter,
    ) -> None:
        self._services = services
        self._adapter = adapter

    def prepare_turn(self, request: EngineRequest) -> TurnPreparationResult:
        self._adapter.seed_session_cwd(request)
        state_machine, messages, context = self._adapter.prepare_turn_entry(request)
        messages, active_system_prompt = self._adapter.prepare_session_and_system_prompt(
            request,
            messages,
            context,
        )
        return TurnPreparationResult(
            state_machine=state_machine,
            messages=messages,
            context=context,
            active_system_prompt=active_system_prompt,
        )

    def finalize_turn(self, finalization: TurnFinalizationInput) -> EngineResult:
        final_response = finalization.final_response
        if (
            finalization.exit_reason == ExitReason.MAX_ITERATIONS
            and not final_response
            and finalization.budget is not None
        ):
            summary_text = self._adapter.handle_max_iterations(
                finalization.request,
                finalization.messages,
                finalization.context,
            )
            if summary_text:
                final_response = summary_text

        if finalization.state_machine.state == LoopState.FINALIZE:
            finalization.state_machine.transition(LoopState.DONE)

        pending_steer = self._services.steer_inbox.drain(finalization.request.session_id)
        if pending_steer:
            finalization.context.metadata["pending_steer"] = pending_steer

        self._adapter.observe_memory_turn(finalization.messages, finalization.context)
        result = self._adapter.build_result(
            finalization.request,
            finalization.messages,
            finalization.iterations,
            final_response,
            finalization.error_text,
            finalization.exit_reason,
            context=finalization.context,
            active_system_prompt=finalization.active_system_prompt,
        )
        self._adapter.save_trajectory_if_enabled(
            result=result,
            messages=finalization.messages,
            context=finalization.context,
        )
        self._adapter.cleanup_task_resources_if_needed(
            result=result,
            context=finalization.context,
        )
        self._services.interrupt_controller.clear(finalization.request.session_id)
        self._adapter.safe_call_hook(
            "on_session_end",
            session_id=finalization.request.session_id,
            completed=result.status in {EngineStatus.COMPLETED, EngineStatus.MAX_ITERATIONS},
            interrupted=result.status == EngineStatus.INTERRUPTED,
            context_metadata=finalization.context.metadata,
        )
        return result

    def cleanup_after_turn(self, context: TurnContext | None) -> None:
        if context is None:
            return
        if not context.metadata.get("_task_cleanup_done"):
            self._adapter.cleanup_task_resources(
                context=context,
                completed=False,
                interrupted=self._adapter.is_interrupted(context.session_id, context),
            )
        self._adapter.clear_interrupt(session_id=context.session_id)
