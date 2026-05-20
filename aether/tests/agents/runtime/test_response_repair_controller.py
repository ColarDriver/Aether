from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from aether.agents.runtime.response_repair import (
    ResponseRepairController,
    ResponseRepairInput,
)
from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    TurnContext,
)


@dataclass(slots=True)
class _LengthOutcome:
    action: str
    messages: list[dict[str, Any]]
    final_response: str | None = None
    exit_reason: ExitReason = ExitReason.TEXT_RESPONSE


@dataclass(slots=True)
class _ValidationOutcome:
    action: str
    invalid_json_args: list[tuple[str, str]] = field(default_factory=list)
    injection_messages: list[dict[str, Any]] = field(default_factory=list)


class _Adapter:
    def __init__(
        self,
        *,
        length_outcome: _LengthOutcome | None = None,
        validation_outcome: _ValidationOutcome | None = None,
    ) -> None:
        self.length_outcome = length_outcome
        self.validation_outcome = validation_outcome or _ValidationOutcome("ok")
        self.length_tool_calls = 0
        self.length_text_calls = 0
        self.pending_steer_calls = 0

    def handle_length_with_tool_calls(self, **kwargs: Any) -> _LengthOutcome:
        self.length_tool_calls += 1
        return self.length_outcome or _LengthOutcome("continue", kwargs["messages"])

    def handle_length_finish_reason(self, **kwargs: Any) -> _LengthOutcome:
        self.length_text_calls += 1
        return self.length_outcome or _LengthOutcome("continue", kwargs["messages"])

    def validate_tool_call_arguments(self, **kwargs: Any) -> _ValidationOutcome:
        del kwargs
        return self.validation_outcome

    def get_messages_up_to_last_assistant(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(messages[:-1])

    def extract_visible_text(self, content: str) -> str:
        return content.strip()

    def apply_pending_steer_to_tool_results(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        del messages, kwargs
        self.pending_steer_calls += 1


def _repair_input(
    response: NormalizedResponse,
    *,
    messages: list[dict[str, Any]] | None = None,
    context: TurnContext | None = None,
) -> ResponseRepairInput:
    return ResponseRepairInput(
        response=response,
        messages=messages if messages is not None else [],
        request=EngineRequest(session_id="repair", user_message="hi"),
        context=context or TurnContext(session_id="repair", iteration=1),
    )


class ResponseRepairControllerTests(unittest.TestCase):
    def test_length_prose_continuation(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        adapter = _Adapter(length_outcome=_LengthOutcome("continue", messages + [{"role": "user", "content": "continue"}]))
        controller = ResponseRepairController(config=EngineConfig(), adapter=adapter)

        result = controller.repair(
            _repair_input(
                NormalizedResponse(content="partial", finish_reason="length"),
                messages=messages,
            )
        )

        self.assertEqual(result.action, "continue")
        self.assertEqual(adapter.length_text_calls, 1)
        self.assertEqual(result.messages[-1]["content"], "continue")

    def test_length_tool_call_uses_truncated_tool_path(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        adapter = _Adapter(length_outcome=_LengthOutcome("continue", messages))
        controller = ResponseRepairController(config=EngineConfig(), adapter=adapter)

        result = controller.repair(
            _repair_input(
                NormalizedResponse(
                    tool_calls=[ToolCall(id="c1", name="shell", arguments={"cmd": "pwd"})],
                    finish_reason="length",
                ),
                messages=messages,
            )
        )

        self.assertEqual(result.action, "continue")
        self.assertEqual(adapter.length_tool_calls, 1)
        self.assertEqual(adapter.length_text_calls, 0)

    def test_truncated_tool_call_exhausted_finalizes_without_poisoning_history(self) -> None:
        context = TurnContext(session_id="repair-truncated", iteration=1)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "new"},
        ]
        adapter = _Adapter(validation_outcome=_ValidationOutcome("truncated"))
        controller = ResponseRepairController(config=EngineConfig(), adapter=adapter)

        result = controller.repair(
            _repair_input(
                NormalizedResponse(
                    content="visible prefix",
                    tool_calls=[ToolCall(id="c1", name="shell", arguments={})],
                ),
                messages=messages,
                context=context,
            )
        )

        self.assertEqual(result.action, "finalize")
        self.assertEqual(result.exit_reason, ExitReason.TOOL_CALL_TRUNCATED)
        self.assertEqual(result.final_response, "visible prefix")
        self.assertEqual(result.messages, messages[:-1])
        self.assertTrue(context.metadata["partial"])
        self.assertEqual(context.metadata["length_exit_reason"], "tool_call_truncated")
        self.assertEqual(context.metadata["truncated_response_prefix_parts"], ["visible prefix"])

    def test_invalid_json_error_injection_continues(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        injection = [
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "invalid json"},
        ]
        adapter = _Adapter(
            validation_outcome=_ValidationOutcome(
                "inject_error",
                injection_messages=injection,
            )
        )
        controller = ResponseRepairController(config=EngineConfig(), adapter=adapter)

        result = controller.repair(
            _repair_input(
                NormalizedResponse(tool_calls=[ToolCall(id="c1", name="shell", arguments={})]),
                messages=messages,
            )
        )

        self.assertEqual(result.action, "continue")
        self.assertEqual(result.messages, [{"role": "user", "content": "hi"}, *injection])
        self.assertEqual(adapter.pending_steer_calls, 1)

    def test_valid_tool_call_proceeds_to_dispatch(self) -> None:
        adapter = _Adapter(validation_outcome=_ValidationOutcome("ok"))
        controller = ResponseRepairController(config=EngineConfig(), adapter=adapter)

        result = controller.repair(
            _repair_input(
                NormalizedResponse(tool_calls=[ToolCall(id="c1", name="shell", arguments={})])
            )
        )

        self.assertEqual(result.action, "proceed")

    def test_disabled_detection_skips_validation(self) -> None:
        adapter = _Adapter(validation_outcome=_ValidationOutcome("truncated"))
        controller = ResponseRepairController(
            config=EngineConfig(truncated_tool_call_detection_enabled=False),
            adapter=adapter,
        )

        result = controller.repair(
            _repair_input(
                NormalizedResponse(tool_calls=[ToolCall(id="c1", name="shell", arguments={})])
            )
        )

        self.assertEqual(result.action, "proceed")


if __name__ == "__main__":
    unittest.main()
