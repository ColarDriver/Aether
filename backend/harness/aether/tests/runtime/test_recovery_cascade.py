from __future__ import annotations

import unittest
from typing import List

from aether import AgentEngine
from aether.agents.core.agent import _WithholdingState
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, TurnContext
from aether.runtime.fallback_chain import FallbackChain, ProviderSlot
from aether.runtime.recovery import FailoverReason, RecoveryDecision
from aether.tools.base import ToolDescriptor


class _NoopProvider(ModelProvider):
    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback=None,
        stream_silent_callback=None,
    ) -> NormalizedResponse:
        return NormalizedResponse(content="ok")


def _chain() -> FallbackChain:
    return FallbackChain(
        [
            ProviderSlot("primary", factory=_NoopProvider),
            ProviderSlot("fallback", factory=_NoopProvider),
        ]
    )


class RecoveryCascadeTests(unittest.TestCase):
    def test_repeat_withholdable_error_after_compression_forces_fallback(self) -> None:
        chain = _chain()
        engine = AgentEngine(
            chain.current_provider,
            fallback_chain=chain,
            config=EngineConfig(use_builtin_tools=False, fallback_chain_enabled=True),
        )
        state = _WithholdingState()
        state.compression_attempted_for.add(FailoverReason.payload_too_large.value)
        decision = RecoveryDecision.retry_after(
            0.0,
            reason="payload-too-large:compress",
            compress_context=True,
            classified_reason=FailoverReason.payload_too_large.value,
        )

        upgraded = engine._maybe_upgrade_decision_for_repeat_withholding(
            decision,
            state=state,
            context=TurnContext(session_id="s", iteration=0, metadata={}),
        )

        self.assertTrue(upgraded.activate_fallback)
        self.assertTrue(upgraded.compress_context)
        self.assertIn(
            "force_fallback_upgrade(reason=payload_too_large,after_compression=true)",
            state.cascade_log,
        )

    def test_repeat_withholdable_error_does_not_force_fallback_without_next_slot(self) -> None:
        chain = _chain()
        self.assertTrue(chain.activate_next())
        engine = AgentEngine(
            chain.current_provider,
            fallback_chain=chain,
            config=EngineConfig(use_builtin_tools=False, fallback_chain_enabled=True),
        )
        state = _WithholdingState()
        state.compression_attempted_for.add(FailoverReason.payload_too_large.value)
        decision = RecoveryDecision.retry_after(
            0.0,
            reason="payload-too-large:compress",
            compress_context=True,
            classified_reason=FailoverReason.payload_too_large.value,
        )

        upgraded = engine._maybe_upgrade_decision_for_repeat_withholding(
            decision,
            state=state,
            context=TurnContext(session_id="s", iteration=0, metadata={}),
        )

        self.assertFalse(upgraded.activate_fallback)
        self.assertEqual(state.cascade_log, [])

    def test_observe_recovery_cascade_writes_stable_metadata_shape(self) -> None:
        engine = AgentEngine(_NoopProvider(), config=EngineConfig(use_builtin_tools=False))
        context = TurnContext(session_id="s", iteration=0, metadata={})
        state = _WithholdingState()
        state.cascade_log.extend(["compress_context(freed=10,reason=payload_too_large)"])
        state.suppressed_callback_notifications = 2

        engine._observe_recovery_cascade(context, state, terminal="success")

        self.assertEqual(
            context.metadata["recovery"],
            {
                "cascade_log": ["compress_context(freed=10,reason=payload_too_large)"],
                "pending_errors_count": 0,
                "terminal": "success",
                "suppressed_callback_notifications": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
