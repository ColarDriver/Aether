"""Sprint 1 / PR 1.1 — response-shape validation contract.

Pins down two layers:

1. ``ModelProvider.validate_response`` default returns ``(True, [])`` so
   existing providers (Scripted, Codex, Claude, ...) opt out for free.
   ``OpenAICompatibleModel.validate_response`` flags HTTP-200 responses
   that contain an ``error`` envelope or empty ``choices`` list as
   invalid.
2. ``AgentEngine`` raises ``ResponseInvalidError`` after a successful
   ``provider.generate`` if validation fails, routes it through the
   Sprint 0 recovery layer (so the engine retries the call), and surfaces
   ``ExitReason.RESPONSE_INVALID`` distinctly from ``PROVIDER_ERROR``
   when retries are exhausted.
"""

from __future__ import annotations

import unittest
from typing import Iterable, List

from aether import AgentEngine
from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.models.provider.openai_compatible import OpenAICompatibleModel
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    TurnContext,
)
from aether.runtime.provider_errors import ResponseInvalidError
from aether.runtime.recovery import GenericBackoffStrategy, NoRetryStrategy
from aether.tools.base import ToolDescriptor


# ---------------------------------------------------------------------------
# Provider-level validate_response
# ---------------------------------------------------------------------------


class ProviderValidateResponseTests(unittest.TestCase):
    def test_default_implementation_always_passes(self) -> None:
        # ScriptedProvider doesn't override, so it inherits the trivial
        # always-valid default.  The engine then never tries to retry.
        provider = ScriptedProvider([NormalizedResponse(content="ok")])
        ok, reasons = provider.validate_response(NormalizedResponse(content=""))
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_openai_provider_flags_embedded_error_envelope(self) -> None:
        # Some gateways (notably OpenRouter) reply HTTP 200 with the body
        # ``{"error": {"message": "...", "type": "..."}}``.  The parser
        # turns that into an empty NormalizedResponse but keeps the raw
        # dict in metadata; validate_response must recognise it.
        provider = _make_openai_provider()
        bad = NormalizedResponse(
            content="",
            tool_calls=[],
            finish_reason="stop",
            metadata={
                "raw": {
                    "error": {
                        "message": "upstream credit exhausted",
                        "type": "billing_error",
                    }
                }
            },
        )
        ok, reasons = provider.validate_response(bad)
        self.assertFalse(ok)
        self.assertTrue(any("upstream credit exhausted" in r for r in reasons))

    def test_openai_provider_flags_empty_choices_only_when_response_also_empty(self) -> None:
        # An empty choices list AND empty content/tool_calls is malformed.
        provider = _make_openai_provider()
        bad = NormalizedResponse(
            content="",
            tool_calls=[],
            finish_reason="stop",
            metadata={"raw": {"choices": []}},
        )
        ok, reasons = provider.validate_response(bad)
        self.assertFalse(ok)
        self.assertTrue(any("choices is empty" in r for r in reasons))

    def test_openai_provider_passes_legitimate_response(self) -> None:
        provider = _make_openai_provider()
        good = NormalizedResponse(
            content="hi",
            tool_calls=[],
            finish_reason="stop",
            metadata={
                "raw": {
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
                }
            },
        )
        ok, reasons = provider.validate_response(good)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])


# ---------------------------------------------------------------------------
# Engine-level wiring
# ---------------------------------------------------------------------------


class _ValidatingProvider(ModelProvider):
    """Programmable provider that always succeeds at the HTTP level but
    lets each test control whether ``validate_response`` reports invalid.

    The ``validation_script`` is a list of (valid, reasons) tuples consumed
    one-per-call so we can express "first two responses invalid, third
    valid" in tests.
    """

    def __init__(
        self,
        responses: Iterable[NormalizedResponse],
        validation_script: list[tuple[bool, list[str]]],
    ) -> None:
        self._responses = list(responses)
        self._validation_script = list(validation_script)
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        if not self._responses:
            raise RuntimeError("ValidatingProvider response script exhausted")
        self.calls += 1
        return self._responses.pop(0)

    def validate_response(self, response: NormalizedResponse) -> tuple[bool, list[str]]:
        if not self._validation_script:
            return True, []
        ok, reasons = self._validation_script.pop(0)
        return ok, list(reasons)


class EngineValidationIntegrationTests(unittest.TestCase):
    def test_invalid_then_valid_recovers_via_recovery_layer(self) -> None:
        # First response is "structurally" valid but flagged invalid by
        # the provider's validate_response.  Sprint 0's
        # GenericBackoffStrategy should retry once, the second response
        # passes validation, the turn completes successfully.
        provider = _ValidatingProvider(
            responses=[
                NormalizedResponse(content="bad"),
                NormalizedResponse(content="good"),
            ],
            validation_script=[(False, ["raw.error.test"]), (True, [])],
        )
        engine = AgentEngine(
            provider,
            recovery_strategy=GenericBackoffStrategy(
                max_attempts=3, base_wait_seconds=0.0
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="vr", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "good")
        self.assertEqual(provider.calls, 2)

        # The retry must show up in the recovery decisions trail with
        # is_network_error=True (Sprint 1 stop-gap classification).
        decisions = result.metadata["turn"]["recovery_decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0]["retry"])
        self.assertTrue(decisions[0]["is_network_error"])

    def test_persistently_invalid_surfaces_response_invalid_exit(self) -> None:
        # All responses fail validation.  After the budget is exhausted
        # the engine must surface ExitReason.RESPONSE_INVALID — distinct
        # from PROVIDER_ERROR — so observers can tell the failure mode
        # apart from a transport-level fault.
        provider = _ValidatingProvider(
            responses=[
                NormalizedResponse(content="x"),
                NormalizedResponse(content="x"),
                NormalizedResponse(content="x"),
            ],
            validation_script=[
                (False, ["raw.error.always"]),
                (False, ["raw.error.always"]),
                (False, ["raw.error.always"]),
            ],
        )
        engine = AgentEngine(
            provider,
            recovery_strategy=GenericBackoffStrategy(
                max_attempts=3, base_wait_seconds=0.0
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="vr2", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.RESPONSE_INVALID)
        self.assertEqual(provider.calls, 3)

    def test_no_retry_strategy_fails_on_first_invalid(self) -> None:
        # Operators that opt out of retries (NoRetryStrategy) should still
        # see RESPONSE_INVALID, not PROVIDER_ERROR.
        provider = _ValidatingProvider(
            responses=[NormalizedResponse(content="x")],
            validation_script=[(False, ["raw.error.test"])],
        )
        engine = AgentEngine(provider, recovery_strategy=NoRetryStrategy())

        result = engine.run_turn(EngineRequest(session_id="vr3", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.RESPONSE_INVALID)
        self.assertEqual(provider.calls, 1)


# ---------------------------------------------------------------------------
# ResponseInvalidError surface
# ---------------------------------------------------------------------------


class ResponseInvalidErrorTests(unittest.TestCase):
    def test_default_body_summary_built_from_validation_errors(self) -> None:
        exc = ResponseInvalidError(validation_errors=["a", "b"])
        self.assertTrue(exc.is_network_error)
        self.assertIn("invalid response", str(exc))
        self.assertIn("a", str(exc))
        self.assertIn("b", str(exc))

    def test_explicit_body_summary_preserved(self) -> None:
        exc = ResponseInvalidError(
            validation_errors=["x"], body_summary="custom summary"
        )
        self.assertIn("custom summary", str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_provider() -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        model="m1",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        request_timeout_sec=5,
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
