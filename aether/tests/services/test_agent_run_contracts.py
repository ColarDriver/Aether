from __future__ import annotations

from pathlib import Path

from aether.services.runs import (
    AgentRunCancelRequest,
    AgentRunOptions,
    AgentRunRequest,
    AgentRunResult,
    AgentRunSnapshot,
    AgentRunStatus,
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
    event_to_public_dict,
)


def test_run_contracts_are_plain_service_data() -> None:
    request = AgentRunRequest(
        session_id="ses",
        user_message="hello",
        run_id="run",
        options=AgentRunOptions(max_iterations=3, temperature=0.2),
    )
    result = AgentRunResult(
        session_id="ses",
        run_id="run",
        final_text="done",
        exit_reason="done",
        usage={"input_tokens": 1},
    )
    snapshot = AgentRunSnapshot(
        session_id="ses",
        run_id="run",
        status=AgentRunStatus.COMPLETED,
        result=result,
    )
    cancel = AgentRunCancelRequest(session_id="ses", reason="user")

    assert request.options.max_iterations == 3
    assert result.usage["input_tokens"] == 1
    assert snapshot.status == AgentRunStatus.COMPLETED
    assert cancel.reason == "user"


def test_every_service_event_serializes_to_public_safe_data() -> None:
    events = [
        RunStarted("ses", "run"),
        AssistantDelta("ses", "run", "hi", 0),
        ReasoningDelta("ses", "run", "thinking", 1),
        SilentProgress("ses", "run", 5, 2),
        RunStatusChanged("ses", "run", "thinking"),
        LoopStateChanged("ses", "run", "llm"),
        IterationStarted("ses", "run", 0),
        IterationFinished("ses", "run", 0),
        ToolStarted("ses", "run", "tc", "read_file", {"path": "x"}, 0),
        ToolFinished(
            "ses",
            "run",
            "tc",
            "read_file",
            "content",
            metadata={"ok": True, "path": Path("not-json")},
        ),
        PermissionRequested("ses", "run", "plan", plan_text="plan", deadline_ms=1000),
        TokenUsageUpdated("ses", "run", input_tokens=1, output_tokens=2),
        RunFinished("ses", "run", "done", "done"),
        RunFailed("ses", "run", "boom"),
        RunCancelled("ses", "run", "user", "partial"),
    ]

    for event in events:
        payload = event_to_public_dict(event)
        assert payload["event"] == type(event).__name__
        assert payload["session_id"] == "ses"
        assert payload["run_id"] == "run"


def test_run_service_events_do_not_import_gateway_protocol() -> None:
    import aether.services.runs.events as events_module

    assert "aether.gateway" not in Path(events_module.__file__).read_text(encoding="utf-8")
