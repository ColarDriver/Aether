from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from aether.agents.runtime.response_finalization import (
    ResponseFinalizationController,
    ResponseFinalizationInput,
)
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    TurnContext,
)


@dataclass(slots=True)
class _Finalized:
    final_response: str | object | None
    exit_reason: ExitReason | None
    error_text: str | None = None


class _Adapter:
    def __init__(
        self,
        *,
        phantom_outcome: str = "none",
        finalized: _Finalized | None = None,
        continue_finalization: bool = False,
    ) -> None:
        self.phantom_outcome = phantom_outcome
        self.finalized = finalized or _Finalized("ok", ExitReason.TEXT_RESPONSE)
        self.continue_finalization = continue_finalization
        self.finalize_calls: list[dict[str, Any]] = []

    def maybe_recover_phantom_tool_intent(
        self,
        *,
        response_to_store: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> str:
        del context
        if self.phantom_outcome == "synthesized":
            response_to_store.tool_calls = [
                ToolCall(id="phantom_1", name="read_file", arguments={"path": "README.md"})
            ]
            response_to_store.content = ""
        if self.phantom_outcome == "retry":
            messages.append({"role": "user", "content": "use tools properly"})
        return self.phantom_outcome

    def finalize_empty_response(self, **kwargs: Any) -> _Finalized:
        self.finalize_calls.append(dict(kwargs))
        return self.finalized

    def is_continue_loop_finalization(self, finalized: Any) -> bool:
        del finalized
        return self.continue_finalization


def _input(
    response: NormalizedResponse,
    *,
    messages: list[dict[str, Any]] | None = None,
    context: TurnContext | None = None,
) -> ResponseFinalizationInput:
    return ResponseFinalizationInput(
        response=response,
        messages=messages if messages is not None else [],
        context=context or TurnContext(session_id="finalize", iteration=1),
        request=EngineRequest(session_id="finalize", user_message="hi"),
    )


class ResponseFinalizationControllerTests(unittest.TestCase):
    def test_plain_text_final_response(self) -> None:
        adapter = _Adapter(finalized=_Finalized("hello", ExitReason.TEXT_RESPONSE))
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(_input(NormalizedResponse(content="hello")))

        self.assertEqual(result.action, "finalize")
        self.assertEqual(result.final_response, "hello")
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(adapter.finalize_calls[0]["phantom_outcome"], "none")

    def test_prefix_is_combined_once_before_finalization(self) -> None:
        context = TurnContext(
            session_id="finalize-prefix",
            iteration=1,
            metadata={"truncated_response_prefix_parts": ["first", "second"]},
        )
        adapter = _Adapter(finalized=_Finalized("first second third", ExitReason.LENGTH_RECOVERED))
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(
            _input(NormalizedResponse(content="third"), context=context)
        )

        self.assertEqual(result.action, "finalize")
        stored = adapter.finalize_calls[0]["response_to_store"]
        self.assertEqual(stored.content, "first second third")
        self.assertNotIn("truncated_response_prefix_parts", context.metadata)

    def test_empty_response_finalization(self) -> None:
        adapter = _Adapter(finalized=_Finalized("", ExitReason.EMPTY_RESPONSE))
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(_input(NormalizedResponse(content="")))

        self.assertEqual(result.action, "finalize")
        self.assertEqual(result.final_response, "")
        self.assertEqual(result.exit_reason, ExitReason.EMPTY_RESPONSE)

    def test_empty_response_recovery_retry(self) -> None:
        adapter = _Adapter(
            finalized=_Finalized(object(), None),
            continue_finalization=True,
        )
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(_input(NormalizedResponse(content="")))

        self.assertEqual(result.action, "continue")

    def test_phantom_retry_returns_continue(self) -> None:
        messages: list[dict[str, Any]] = []
        adapter = _Adapter(phantom_outcome="retry")
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(
            _input(NormalizedResponse(content="<function=read_file>"), messages=messages)
        )

        self.assertEqual(result.action, "continue")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(adapter.finalize_calls, [])

    def test_phantom_exhausted_reaches_finalization(self) -> None:
        adapter = _Adapter(
            phantom_outcome="exhausted",
            finalized=_Finalized("broken prose", ExitReason.PHANTOM_TOOL_INTENT),
        )
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(_input(NormalizedResponse(content="broken prose")))

        self.assertEqual(result.action, "finalize")
        self.assertEqual(result.exit_reason, ExitReason.PHANTOM_TOOL_INTENT)
        self.assertEqual(adapter.finalize_calls[0]["phantom_outcome"], "exhausted")

    def test_synthesized_tool_call_returns_dispatch_request(self) -> None:
        adapter = _Adapter(phantom_outcome="synthesized")
        controller = ResponseFinalizationController(adapter=adapter)

        result = controller.finalize(_input(NormalizedResponse(content="<function=read_file>")))

        self.assertEqual(result.action, "dispatch_synthesized")
        assert result.synthesized_response is not None
        self.assertEqual(result.synthesized_response.content, "")
        self.assertEqual(result.synthesized_response.tool_calls[0].name, "read_file")
        self.assertEqual(adapter.finalize_calls, [])


if __name__ == "__main__":
    unittest.main()
