from __future__ import annotations

from pathlib import Path

from aether.gateway.handlers.run_event_adapter import (
    permission_request_to_gateway_payload,
    safe_metadata,
    service_event_to_gateway_payload,
)
from aether.services.runs import (
    AssistantDelta,
    IterationFinished,
    IterationStarted,
    LoopStateChanged,
    PermissionRequested,
    ReasoningDelta,
    RunCancelled,
    RunFailed,
    RunFinished,
    RunStarted,
    RunStatusChanged,
    SilentProgress,
    TokenUsageUpdated,
    ToolFinished,
    ToolStarted,
)


def test_service_events_map_to_existing_gateway_event_shapes() -> None:
    assert service_event_to_gateway_payload(RunStarted("ses", "run")) is None
    assert service_event_to_gateway_payload(AssistantDelta("ses", "run", "hi", 0)) == {
        "type": "text.delta",
        "session_id": "ses",
        "run_id": "run",
        "text": "hi",
        "sequence": 0,
    }
    assert service_event_to_gateway_payload(ReasoningDelta("ses", "run", "why", 1)) == {
        "type": "reasoning.delta",
        "session_id": "ses",
        "run_id": "run",
        "text": "why",
        "sequence": 1,
    }
    assert service_event_to_gateway_payload(SilentProgress("ses", "run", 7, 2)) == {
        "type": "stream.progress",
        "session_id": "ses",
        "run_id": "run",
        "chars": 7,
        "sequence": 2,
    }
    assert service_event_to_gateway_payload(RunStatusChanged("ses", "run", "tool_use", "grep")) == {
        "type": "status",
        "session_id": "ses",
        "run_id": "run",
        "kind": "tool_use",
        "detail": "grep",
    }
    assert service_event_to_gateway_payload(LoopStateChanged("ses", "run", "tool")) == {
        "type": "loop.state",
        "session_id": "ses",
        "run_id": "run",
        "state": "tool",
    }
    assert service_event_to_gateway_payload(IterationStarted("ses", "run", 3)) == {
        "type": "iteration.start",
        "session_id": "ses",
        "run_id": "run",
        "iteration": 3,
    }
    assert service_event_to_gateway_payload(IterationFinished("ses", "run", 3)) == {
        "type": "iteration.end",
        "session_id": "ses",
        "run_id": "run",
        "iteration": 3,
    }


def test_tool_usage_terminal_and_permission_payloads_match_gateway_wire_names() -> None:
    assert service_event_to_gateway_payload(
        ToolStarted("ses", "run", "tc", "grep", {"pattern": "x"}, 0)
    ) == {
        "type": "tool.call",
        "session_id": "ses",
        "run_id": "run",
        "tool_call_id": "tc",
        "tool_name": "grep",
        "arguments": {"pattern": "x"},
        "iteration": 0,
    }
    assert service_event_to_gateway_payload(
        ToolFinished(
            "ses",
            "run",
            "tc",
            "grep",
            "matches",
            is_error=True,
            iteration=0,
            metadata={"count": 1, "path": Path("not-json")},
        )
    ) == {
        "type": "tool.result",
        "session_id": "ses",
        "run_id": "run",
        "tool_call_id": "tc",
        "tool_name": "grep",
        "content": "matches",
        "is_error": True,
        "iteration": 0,
        "metadata": {"count": 1},
    }
    assert service_event_to_gateway_payload(TokenUsageUpdated("ses", "run", 1, 2, 3, 4)) == {
        "type": "usage",
        "session_id": "ses",
        "run_id": "run",
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_tokens": 3,
        "cache_write_tokens": 4,
    }
    assert service_event_to_gateway_payload(RunFinished("ses", "run", "done", "done")) == {
        "type": "done",
        "session_id": "ses",
        "run_id": "run",
        "final_text": "done",
        "exit_reason": "done",
    }
    assert service_event_to_gateway_payload(RunCancelled("ses", "run", "user", "part")) == {
        "type": "cancelled",
        "session_id": "ses",
        "run_id": "run",
        "reason": "user",
        "partial_text": "part",
    }
    assert service_event_to_gateway_payload(RunFailed("ses", "run", "boom")) == {
        "type": "error",
        "session_id": "ses",
        "run_id": "run",
        "message": "boom",
    }
    permission = PermissionRequested(
        "ses",
        "run",
        "plan",
        tool_call_id="tc",
        plan_text="plan",
        plan_path="/tmp/plan.md",
        deadline_ms=123,
    )
    assert service_event_to_gateway_payload(permission) is None
    assert permission_request_to_gateway_payload(permission) == {
        "kind": "plan",
        "session_id": "ses",
        "run_id": "run",
        "tool_call_id": "tc",
        "plan_text": "plan",
        "plan_path": "/tmp/plan.md",
        "questions": [],
        "deadline_ms": 123,
    }


def test_gateway_run_event_metadata_sanitizer_matches_agent_rules() -> None:
    assert safe_metadata({"ok": True, "bad": Path("not-json")}) == {"ok": True}
