from __future__ import annotations

import logging
import unittest
from typing import Any

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.agents.runtime.provider_invocation import ProviderInvocationRequest, ProviderInvocationResult
from aether.agents.runtime.recovery_controller import (
    RecoveryAttemptInput,
    RecoveryController,
    RecoveryWithholdingState,
)
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, TurnContext
from aether.runtime.core.services import EngineServices
from aether.runtime.recovery.error_classifier import FailoverReason
from aether.runtime.recovery.provider_errors import ProviderInvocationError, ResponseInvalidError
from aether.runtime.recovery.strategies import (
    AttemptState,
    GenericBackoffStrategy,
    RecoveryDecision,
    RecoveryStrategy,
)
from aether.tools.registry import ToolRegistry


class _Adapter:
    def __init__(self) -> None:
        self.compaction_calls = 0
        self.cascade_terminals: list[str] = []
        self.dumps: list[str] = []
        self.cleared_success = 0

    def check_rate_guard_before_provider_call(self, **kwargs: Any) -> None:
        del kwargs
        return None

    def activate_rate_guard_fallback(self, **kwargs: Any) -> bool:
        del kwargs
        return False

    def build_rate_guard_blocked_error(self, check: Any) -> ProviderInvocationError:
        del check
        return ProviderInvocationError(status_code=429, body_summary="blocked")

    def clear_rate_guard_after_success(self, **kwargs: Any) -> None:
        del kwargs
        self.cleared_success += 1

    def tool_descriptors_for_provider_call(self, context: TurnContext) -> list[Any]:
        del context
        return []

    def prepare_unicode_safe_payload(self, **kwargs: Any) -> None:
        del kwargs

    def record_interrupt_metadata(self, context: TurnContext, **kwargs: Any) -> None:
        context.metadata["interrupt"] = dict(kwargs)

    def is_interrupted(self, session_id: str, context: TurnContext) -> bool:
        del session_id, context
        return False

    def maybe_recover_unicode_error(self, error: Exception, **kwargs: Any) -> bool:
        del error, kwargs
        return False

    def maybe_write_rate_guard_lock(self, **kwargs: Any) -> None:
        del kwargs

    def maybe_apply_image_shrink_retry(self, **kwargs: Any) -> bool:
        del kwargs
        return False

    def maybe_apply_schema_sanitizer_retry(self, **kwargs: Any) -> bool:
        del kwargs
        return False

    def maybe_upgrade_decision_for_repeat_withholding(
        self,
        decision: RecoveryDecision,
        **kwargs: Any,
    ) -> RecoveryDecision:
        del kwargs
        return decision

    def apply_recovery_decision_cascade(
        self,
        *,
        decision: RecoveryDecision,
        messages: list[dict[str, Any]],
        context: TurnContext,
        state: RecoveryWithholdingState,
    ) -> bool:
        del context, state
        if not decision.compress_context:
            return False
        self.compaction_calls += 1
        messages.append({"role": "user", "content": "compacted"})
        return True

    def apply_recovery_decision_singleshot(self, **kwargs: Any) -> bool:
        return self.apply_recovery_decision_cascade(**kwargs)

    def observe_recovery_cascade(
        self,
        context: TurnContext,
        state: RecoveryWithholdingState,
        *,
        terminal: str,
    ) -> None:
        del context, state
        self.cascade_terminals.append(terminal)

    def maybe_dump_failed_request(self, **kwargs: Any) -> None:
        self.dumps.append(str(kwargs.get("reason")))

    def signal_for_context(self, context: TurnContext) -> None:
        del context
        return None


class _CompressOnceStrategy(RecoveryStrategy):
    def decide(
        self,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
        context: TurnContext,
    ) -> RecoveryDecision:
        del error, attempt_state, context
        return RecoveryDecision.retry_after(
            0.0,
            reason="context-overflow:compress",
            compress_context=True,
            classified_reason=FailoverReason.context_overflow.value,
        )


def _services(strategy: RecoveryStrategy) -> EngineServices:
    return EngineServices(
        provider=ScriptedProvider([NormalizedResponse(content="unused")]),
        tool_registry=ToolRegistry(),
        middleware_pipeline=MiddlewarePipeline(),
        interrupt_controller=InterruptController(),
        logger=logging.getLogger(__name__),
        recovery_strategy=strategy,
    )


def _attempt(
    context: TurnContext,
    invoker: Any,
) -> RecoveryAttemptInput:
    messages = [{"role": "user", "content": "hi"}]
    return RecoveryAttemptInput(
        request=EngineRequest(session_id=context.session_id),
        canonical_messages=messages,
        prepared_messages=list(messages),
        context=context,
        invoke_provider=invoker,
    )


class RecoveryControllerTests(unittest.TestCase):
    def test_transient_provider_error_retries_to_success(self) -> None:
        adapter = _Adapter()
        controller = RecoveryController(
            services=_services(GenericBackoffStrategy(max_attempts=2, base_wait_seconds=0.0)),
            config=EngineConfig(),
            adapter=adapter,
        )
        calls = 0

        def invoker(invocation: ProviderInvocationRequest) -> ProviderInvocationResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ProviderInvocationResult(
                    error=ProviderInvocationError(
                        body_summary="temporary",
                        is_network_error=True,
                    )
                )
            if invocation.on_valid_response is not None:
                invocation.on_valid_response(invocation.context.metadata["provider"], invocation.context)
            return ProviderInvocationResult(response=NormalizedResponse(content="ok"))

        context = TurnContext(
            session_id="recovery",
            iteration=1,
            metadata={"provider": _services(GenericBackoffStrategy()).provider},
        )

        result = controller.invoke_with_recovery(_attempt(context, invoker))

        assert result.response is not None
        self.assertEqual(result.response.content, "ok")
        self.assertEqual(calls, 2)
        self.assertEqual(context.metadata["provider_error_retries"], 1)
        self.assertEqual(adapter.cleared_success, 1)

    def test_response_invalid_uses_existing_retry_budget(self) -> None:
        adapter = _Adapter()
        controller = RecoveryController(
            services=_services(GenericBackoffStrategy(max_attempts=3, base_wait_seconds=0.0)),
            config=EngineConfig(max_provider_recovery_attempts=1),
            adapter=adapter,
        )

        def invoker(invocation: ProviderInvocationRequest) -> ProviderInvocationResult:
            del invocation
            return ProviderInvocationResult(
                error=ResponseInvalidError(validation_errors=["bad shape"])
            )

        context = TurnContext(session_id="invalid", iteration=1, metadata={})

        result = controller.invoke_with_recovery(_attempt(context, invoker))

        self.assertIsInstance(result.error, ResponseInvalidError)
        self.assertEqual(adapter.cascade_terminals, ["exhausted"])
        self.assertEqual(adapter.dumps, ["max_retries_exhausted"])

    def test_context_overflow_delegates_to_compaction_adapter_then_retries(self) -> None:
        adapter = _Adapter()
        controller = RecoveryController(
            services=_services(_CompressOnceStrategy()),
            config=EngineConfig(),
            adapter=adapter,
        )
        calls = 0

        def invoker(invocation: ProviderInvocationRequest) -> ProviderInvocationResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ProviderInvocationResult(
                    error=ProviderInvocationError(
                        status_code=400,
                        body_summary="context overflow",
                    )
                )
            return ProviderInvocationResult(response=NormalizedResponse(content="ok"))

        context = TurnContext(session_id="compact", iteration=1, metadata={})

        result = controller.invoke_with_recovery(_attempt(context, invoker))

        assert result.response is not None
        self.assertEqual(result.response.content, "ok")
        self.assertEqual(calls, 2)
        self.assertEqual(adapter.compaction_calls, 1)


if __name__ == "__main__":
    unittest.main()
