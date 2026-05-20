"""Recovery coordination for provider invocation attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from aether.agents.runtime.provider_invocation import (
    ProviderInvocationRequest,
    ProviderInvocationResult,
)
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    TurnContext,
)
from aether.runtime.core.exceptions import EngineInterrupted
from aether.runtime.core.services import EngineServices
from aether.runtime.recovery.error_classifier import FailoverReason
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.rate_guard import RateGuardCheck
from aether.runtime.recovery.strategies import (
    AttemptState,
    RecoveryDecision,
    wait_interruptible,
)
from aether.runtime.session.session_runtime import (
    TURN_KEY_PROVIDER_ERROR_RETRIES,
    TURN_KEY_STREAMED_ASSISTANT_TEXT,
)


ProviderInvoker = Callable[[ProviderInvocationRequest], ProviderInvocationResult]


@dataclass(slots=True)
class RecoveryAttemptInput:
    request: EngineRequest
    canonical_messages: list[dict[str, Any]]
    prepared_messages: list[dict[str, Any]]
    context: TurnContext
    stream_callback: Any = None
    stream_silent_callback: Any = None
    invoke_provider: ProviderInvoker | None = None


@dataclass(slots=True)
class RecoveryAttemptResult:
    response: NormalizedResponse | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False
    exit_reason: ExitReason | None = None
    error_text: str | None = None
    continue_loop: bool = False
    error: Exception | None = None


@dataclass(slots=True)
class RecoveryWithholdingState:
    pending_errors: list[ProviderInvocationError] = field(default_factory=list)
    cascade_log: list[str] = field(default_factory=list)
    suppressed_callback_notifications: int = 0
    compression_attempted_for: set[str] = field(default_factory=set)


class RecoveryControllerAdapter(Protocol):
    def check_rate_guard_before_provider_call(
        self,
        *,
        provider: ModelProvider,
        context: TurnContext,
    ) -> RateGuardCheck | None: ...

    def activate_rate_guard_fallback(
        self,
        *,
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> bool: ...

    def build_rate_guard_blocked_error(self, check: RateGuardCheck) -> ProviderInvocationError: ...

    def clear_rate_guard_after_success(
        self,
        *,
        provider: ModelProvider,
        context: TurnContext,
    ) -> None: ...

    def tool_descriptors_for_provider_call(self, context: TurnContext) -> list[Any]: ...

    def prepare_unicode_safe_payload(
        self,
        *,
        canonical_messages: list[dict[str, Any]],
        prepared_messages: list[dict[str, Any]],
        tools: list[Any],
        provider: ModelProvider,
        context: TurnContext,
    ) -> None: ...

    def record_interrupt_metadata(
        self,
        context: TurnContext,
        *,
        reason: str = "user-interrupt",
        partial_text: str = "",
        was_in_tool_call: bool = False,
    ) -> None: ...

    def is_interrupted(self, session_id: str, context: TurnContext) -> bool: ...

    def maybe_recover_unicode_error(
        self,
        error: Exception,
        *,
        canonical_messages: list[dict[str, Any]],
        prepared_messages: list[dict[str, Any]],
        tools: list[Any],
        provider: ModelProvider,
        context: TurnContext,
    ) -> bool: ...

    def maybe_write_rate_guard_lock(
        self,
        *,
        provider: ModelProvider,
        error: ProviderInvocationError,
        decision: RecoveryDecision,
        context: TurnContext,
    ) -> None: ...

    def maybe_apply_image_shrink_retry(
        self,
        *,
        decision: RecoveryDecision,
        messages: list[dict[str, Any]],
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> bool: ...

    def maybe_apply_schema_sanitizer_retry(
        self,
        *,
        decision: RecoveryDecision,
        tools: list[Any],
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> bool: ...

    def maybe_upgrade_decision_for_repeat_withholding(
        self,
        decision: RecoveryDecision,
        *,
        state: RecoveryWithholdingState,
        context: TurnContext,
    ) -> RecoveryDecision: ...

    def apply_recovery_decision_cascade(
        self,
        *,
        decision: RecoveryDecision,
        messages: list[dict[str, Any]],
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> bool: ...

    def apply_recovery_decision_singleshot(
        self,
        *,
        decision: RecoveryDecision,
        messages: list[dict[str, Any]],
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> bool: ...

    def observe_recovery_cascade(
        self,
        context: TurnContext,
        state: RecoveryWithholdingState,
        *,
        terminal: str,
    ) -> None: ...

    def maybe_dump_failed_request(
        self,
        *,
        error: Exception,
        reason: str,
        request: EngineRequest,
        prepared_messages: list[dict[str, Any]],
        tools: list[Any],
        call_config: ModelCallConfig,
        provider: ModelProvider,
        context: TurnContext,
    ) -> None: ...

    def signal_for_context(self, context: TurnContext) -> Any: ...


class LegacyRecoveryControllerAdapter:
    """Delegate recovery side effects to AgentEngine's existing helpers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def check_rate_guard_before_provider_call(self, **kwargs: Any) -> RateGuardCheck | None:
        return self._engine._check_rate_guard_before_provider_call(**kwargs)  # noqa: SLF001

    def activate_rate_guard_fallback(self, **kwargs: Any) -> bool:
        return self._engine._activate_rate_guard_fallback(**kwargs)  # noqa: SLF001

    def build_rate_guard_blocked_error(self, check: RateGuardCheck) -> ProviderInvocationError:
        return self._engine._build_rate_guard_blocked_error(check)  # noqa: SLF001

    def clear_rate_guard_after_success(self, **kwargs: Any) -> None:
        self._engine._clear_rate_guard_after_success(**kwargs)  # noqa: SLF001

    def tool_descriptors_for_provider_call(self, context: TurnContext) -> list[Any]:
        return self._engine._tool_descriptors_for_provider_call(context)  # noqa: SLF001

    def prepare_unicode_safe_payload(self, **kwargs: Any) -> None:
        self._engine._prepare_unicode_safe_payload(**kwargs)  # noqa: SLF001

    def record_interrupt_metadata(self, context: TurnContext, **kwargs: Any) -> None:
        self._engine._record_interrupt_metadata(context, **kwargs)  # noqa: SLF001

    def is_interrupted(self, session_id: str, context: TurnContext) -> bool:
        return self._engine._is_interrupted(session_id, context)  # noqa: SLF001

    def maybe_recover_unicode_error(self, error: Exception, **kwargs: Any) -> bool:
        return self._engine._maybe_recover_unicode_error(error, **kwargs)  # noqa: SLF001

    def maybe_write_rate_guard_lock(self, **kwargs: Any) -> None:
        self._engine._maybe_write_rate_guard_lock(**kwargs)  # noqa: SLF001

    def maybe_apply_image_shrink_retry(self, **kwargs: Any) -> bool:
        return self._engine._maybe_apply_image_shrink_retry(**kwargs)  # noqa: SLF001

    def maybe_apply_schema_sanitizer_retry(self, **kwargs: Any) -> bool:
        return self._engine._maybe_apply_schema_sanitizer_retry(**kwargs)  # noqa: SLF001

    def maybe_upgrade_decision_for_repeat_withholding(
        self,
        decision: RecoveryDecision,
        **kwargs: Any,
    ) -> RecoveryDecision:
        return self._engine._maybe_upgrade_decision_for_repeat_withholding(  # noqa: SLF001
            decision,
            **kwargs,
        )

    def apply_recovery_decision_cascade(self, **kwargs: Any) -> bool:
        return self._engine._apply_recovery_decision_cascade(**kwargs)  # noqa: SLF001

    def apply_recovery_decision_singleshot(self, **kwargs: Any) -> bool:
        return self._engine._apply_recovery_decision_singleshot(**kwargs)  # noqa: SLF001

    def observe_recovery_cascade(
        self,
        context: TurnContext,
        state: RecoveryWithholdingState,
        *,
        terminal: str,
    ) -> None:
        self._engine._observe_recovery_cascade(context, state, terminal=terminal)  # noqa: SLF001

    def maybe_dump_failed_request(self, **kwargs: Any) -> None:
        self._engine._maybe_dump_failed_request(**kwargs)  # noqa: SLF001

    def signal_for_context(self, context: TurnContext) -> Any:
        return self._engine._signal_for_context(context)  # noqa: SLF001


class RecoveryController:
    """Coordinate provider retries and recovery decisions."""

    def __init__(
        self,
        *,
        services: EngineServices,
        config: EngineConfig,
        adapter: RecoveryControllerAdapter,
    ) -> None:
        self._services = services
        self._config = config
        self._adapter = adapter

    def invoke_with_recovery(self, attempt: RecoveryAttemptInput) -> RecoveryAttemptResult:
        if attempt.invoke_provider is None:
            raise ValueError("RecoveryAttemptInput.invoke_provider is required")

        attempt_state = AttemptState()
        last_error: Exception | None = None
        decisions_log = attempt.context.metadata.setdefault("recovery_decisions", [])
        if self._services.fallback_chain is not None:
            attempt.context.metadata["active_provider_name"] = (
                self._services.fallback_chain.current_slot_name
            )
        withholding_state = RecoveryWithholdingState()
        max_provider_attempts = max(
            1,
            int(getattr(self._config, "max_provider_recovery_attempts", 8)),
        )

        provider: ModelProvider = self._services.provider
        tools: list[Any] = []
        call_config = attempt.request.model_config

        while attempt_state.attempt < max_provider_attempts:
            attempt.context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = ""
            try:
                call_config = self._call_config_for_attempt(attempt)
                provider = self._services.provider
                rate_guard_check = self._adapter.check_rate_guard_before_provider_call(
                    provider=provider,
                    context=attempt.context,
                )
                if rate_guard_check is not None and rate_guard_check.blocked:
                    if self._adapter.activate_rate_guard_fallback(
                        context=attempt.context,
                        state=withholding_state,
                    ):
                        attempt_state = AttemptState()
                        continue
                    error = self._adapter.build_rate_guard_blocked_error(rate_guard_check)
                    attempt.context.metadata["recovery_terminal_exit_reason"] = (
                        ExitReason.RATE_LIMITED.value
                    )
                    return RecoveryAttemptResult(
                        messages=attempt.prepared_messages,
                        error=error,
                    )

                tools = self._adapter.tool_descriptors_for_provider_call(attempt.context)
                self._adapter.prepare_unicode_safe_payload(
                    canonical_messages=attempt.canonical_messages,
                    prepared_messages=attempt.prepared_messages,
                    tools=tools,
                    provider=provider,
                    context=attempt.context,
                )
                invocation_result = attempt.invoke_provider(
                    ProviderInvocationRequest(
                        request=attempt.request,
                        canonical_messages=attempt.canonical_messages,
                        prepared_messages=attempt.prepared_messages,
                        tools=tools,
                        call_config=call_config,
                        context=attempt.context,
                        stream_callback=attempt.stream_callback,
                        stream_silent_callback=attempt.stream_silent_callback,
                        on_valid_response=lambda current_provider, current_context: (
                            self._adapter.clear_rate_guard_after_success(
                                provider=current_provider,
                                context=current_context,
                            )
                        ),
                    )
                )
                if invocation_result.interrupted:
                    return RecoveryAttemptResult(
                        messages=attempt.prepared_messages,
                        interrupted=True,
                    )
                if invocation_result.error is not None:
                    raise invocation_result.error
                response = invocation_result.response
                assert response is not None
                if withholding_state.pending_errors or withholding_state.cascade_log:
                    self._adapter.observe_recovery_cascade(
                        attempt.context,
                        withholding_state,
                        terminal="success",
                    )
                return RecoveryAttemptResult(
                    response=response,
                    messages=attempt.prepared_messages,
                )
            except EngineInterrupted as exc:
                self._adapter.record_interrupt_metadata(
                    attempt.context,
                    reason=exc.reason,
                    partial_text=exc.partial_text,
                    was_in_tool_call=exc.was_in_tool_call,
                )
                return RecoveryAttemptResult(
                    messages=attempt.prepared_messages,
                    interrupted=True,
                )
            except ProviderInvocationError as exc:
                interrupted = self._handle_provider_error_interrupt(attempt, exc)
                if interrupted:
                    return interrupted

                attempt.context.metadata[TURN_KEY_PROVIDER_ERROR_RETRIES] = (
                    int(attempt.context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)) + 1
                )
                attempt_state.attempt += 1
                attempt_state.errors.append(exc)
                last_error = exc
                withholding_state.pending_errors.append(exc)

                raw_error = exc.raw if isinstance(exc.raw, UnicodeEncodeError) else exc
                if self._adapter.maybe_recover_unicode_error(
                    raw_error,
                    canonical_messages=attempt.canonical_messages,
                    prepared_messages=attempt.prepared_messages,
                    tools=tools,
                    provider=provider,
                    context=attempt.context,
                ):
                    continue

                decision = self._services.recovery_strategy.decide(
                    error=exc,
                    attempt_state=attempt_state,
                    context=attempt.context,
                )
                if getattr(self._config, "error_withholding_enabled", True):
                    decision = self._adapter.maybe_upgrade_decision_for_repeat_withholding(
                        decision,
                        state=withholding_state,
                        context=attempt.context,
                    )

                self._adapter.maybe_write_rate_guard_lock(
                    provider=provider,
                    error=exc,
                    decision=decision,
                    context=attempt.context,
                )

                image_shrink_retry, decision = self._maybe_apply_image_shrink_retry(
                    decision=decision,
                    messages=attempt.prepared_messages,
                    context=attempt.context,
                    state=withholding_state,
                )
                if image_shrink_retry:
                    continue

                decisions_log.append(self._decision_log_entry(attempt_state, decision, exc))

                if decision.strip_thinking:
                    attempt.context.metadata["recovery_strip_thinking_requested"] = True

                if self._adapter.maybe_apply_schema_sanitizer_retry(
                    decision=decision,
                    tools=tools,
                    context=attempt.context,
                    state=withholding_state,
                ):
                    continue

                if getattr(self._config, "error_withholding_enabled", True):
                    applied = self._adapter.apply_recovery_decision_cascade(
                        decision=decision,
                        messages=attempt.prepared_messages,
                        context=attempt.context,
                        state=withholding_state,
                    )
                else:
                    applied = self._adapter.apply_recovery_decision_singleshot(
                        decision=decision,
                        messages=attempt.prepared_messages,
                        context=attempt.context,
                        state=withholding_state,
                    )
                if applied:
                    if attempt.context.metadata.pop("_recovery_reset_attempt_state", False):
                        attempt_state = AttemptState()
                    continue

                terminal = self._maybe_surface_terminal_after_recovery(
                    attempt=attempt,
                    decision=decision,
                    error=exc,
                    tools=tools,
                    call_config=call_config,
                    provider=provider,
                    state=withholding_state,
                )
                if terminal is not None:
                    return terminal

                if not decision.retry:
                    if (
                        decision.classified_reason == FailoverReason.rate_limit.value
                        and not attempt.context.metadata.get("recovery_terminal_exit_reason")
                    ):
                        attempt.context.metadata["recovery_terminal_exit_reason"] = (
                            ExitReason.RATE_LIMITED.value
                        )
                    self._adapter.observe_recovery_cascade(
                        attempt.context,
                        withholding_state,
                        terminal="surface",
                    )
                    self._adapter.maybe_dump_failed_request(
                        error=exc,
                        reason="non_retryable_provider_error",
                        request=attempt.request,
                        prepared_messages=attempt.prepared_messages,
                        tools=tools,
                        call_config=call_config,
                        provider=provider,
                        context=attempt.context,
                    )
                    return RecoveryAttemptResult(
                        messages=attempt.prepared_messages,
                        error=exc,
                    )

                if decision.wait_seconds > 0:
                    completed = wait_interruptible(
                        decision.wait_seconds,
                        interrupt_controller=self._services.interrupt_controller,
                        session_id=attempt.request.session_id,
                        interrupt_signal=self._adapter.signal_for_context(attempt.context),
                    )
                    attempt_state.total_wait_seconds += decision.wait_seconds
                    if not completed:
                        return RecoveryAttemptResult(
                            messages=attempt.prepared_messages,
                            interrupted=True,
                        )
                continue
            except Exception as exc:
                interrupted = self._handle_non_provider_interrupt(attempt)
                if interrupted:
                    return interrupted
                attempt.context.metadata[TURN_KEY_PROVIDER_ERROR_RETRIES] = (
                    int(attempt.context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)) + 1
                )
                if self._adapter.maybe_recover_unicode_error(
                    exc,
                    canonical_messages=attempt.canonical_messages,
                    prepared_messages=attempt.prepared_messages,
                    tools=tools,
                    provider=provider,
                    context=attempt.context,
                ):
                    continue
                last_error = exc
                self._adapter.maybe_dump_failed_request(
                    error=last_error,
                    reason="non_retryable_client_error",
                    request=attempt.request,
                    prepared_messages=attempt.prepared_messages,
                    tools=tools,
                    call_config=call_config,
                    provider=provider,
                    context=attempt.context,
                )
                return RecoveryAttemptResult(
                    messages=attempt.prepared_messages,
                    error=last_error,
                )

        self._adapter.observe_recovery_cascade(
            attempt.context,
            withholding_state,
            terminal="exhausted",
        )
        if last_error is not None:
            self._adapter.maybe_dump_failed_request(
                error=last_error,
                reason="max_retries_exhausted",
                request=attempt.request,
                prepared_messages=attempt.prepared_messages,
                tools=tools,
                call_config=call_config,
                provider=provider,
                context=attempt.context,
            )
        return RecoveryAttemptResult(messages=attempt.prepared_messages, error=last_error)

    @staticmethod
    def _call_config_for_attempt(attempt: RecoveryAttemptInput) -> ModelCallConfig:
        call_config = attempt.request.model_config
        ephemeral = attempt.context.metadata.pop("_ephemeral_max_output_tokens", None)
        if isinstance(ephemeral, int) and ephemeral > 0:
            call_config = ModelCallConfig(
                temperature=attempt.request.model_config.temperature,
                max_tokens=ephemeral,
                extra=dict(attempt.request.model_config.extra),
            )
        return call_config

    def _handle_provider_error_interrupt(
        self,
        attempt: RecoveryAttemptInput,
        error: ProviderInvocationError,
    ) -> RecoveryAttemptResult | None:
        del error
        if not self._adapter.is_interrupted(attempt.request.session_id, attempt.context):
            return None
        reason = attempt.context.interrupt_signal.reason() if attempt.context.interrupt_signal else None
        self._adapter.record_interrupt_metadata(
            attempt.context,
            reason=reason or "user-interrupt",
            partial_text=str(attempt.context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "")),
            was_in_tool_call=False,
        )
        return RecoveryAttemptResult(messages=attempt.prepared_messages, interrupted=True)

    def _handle_non_provider_interrupt(
        self,
        attempt: RecoveryAttemptInput,
    ) -> RecoveryAttemptResult | None:
        if not self._adapter.is_interrupted(attempt.request.session_id, attempt.context):
            return None
        reason = attempt.context.interrupt_signal.reason() if attempt.context.interrupt_signal else None
        self._adapter.record_interrupt_metadata(
            attempt.context,
            reason=reason or "user-interrupt",
            partial_text=str(attempt.context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "")),
            was_in_tool_call=False,
        )
        return RecoveryAttemptResult(messages=attempt.prepared_messages, interrupted=True)

    def _maybe_apply_image_shrink_retry(
        self,
        *,
        decision: RecoveryDecision,
        messages: list[dict[str, Any]],
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> tuple[bool, RecoveryDecision]:
        if decision.classified_reason != FailoverReason.image_too_large.value:
            return False, decision
        image_shrink_retry = self._adapter.maybe_apply_image_shrink_retry(
            decision=decision,
            messages=messages,
            context=context,
            state=state,
        )
        if not image_shrink_retry:
            return (
                False,
                RecoveryDecision.give_up(
                    "image-too-large:shrink-unavailable",
                    classified_reason=FailoverReason.image_too_large.value,
                ),
            )
        return True, decision

    @staticmethod
    def _decision_log_entry(
        attempt_state: AttemptState,
        decision: RecoveryDecision,
        error: ProviderInvocationError,
    ) -> dict[str, Any]:
        return {
            "attempt": attempt_state.attempt,
            "retry": decision.retry,
            "wait_seconds": decision.wait_seconds,
            "reason": decision.reason,
            "status_code": error.status_code,
            "is_network_error": error.is_network_error,
            "classified_reason": decision.classified_reason,
            "activate_fallback": decision.activate_fallback,
            "compress_context": decision.compress_context,
            "strip_thinking": decision.strip_thinking,
        }

    def _maybe_surface_terminal_after_recovery(
        self,
        *,
        attempt: RecoveryAttemptInput,
        decision: RecoveryDecision,
        error: ProviderInvocationError,
        tools: list[Any],
        call_config: ModelCallConfig,
        provider: ModelProvider,
        state: RecoveryWithholdingState,
    ) -> RecoveryAttemptResult | None:
        terminal_after_recovery = attempt.context.metadata.get(
            "recovery_terminal_exit_reason"
        )
        if not terminal_after_recovery or not (
            decision.compress_context or decision.activate_fallback
        ):
            return None
        self._adapter.observe_recovery_cascade(
            attempt.context,
            state,
            terminal="surface",
        )
        self._adapter.maybe_dump_failed_request(
            error=error,
            reason="recovery_terminal",
            request=attempt.request,
            prepared_messages=attempt.prepared_messages,
            tools=tools,
            call_config=call_config,
            provider=provider,
            context=attempt.context,
        )
        return RecoveryAttemptResult(messages=attempt.prepared_messages, error=error)
