from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from aether.runtime.tools.tool_permissions import ToolPermissionRequest
from aether.services.runs import (
    AgentRunCancelRequest,
    AgentRunRequest,
    AgentRunResult,
    AgentRunService,
    AssistantDelta,
    RunEventSink,
    RunFinished,
    RunStarted,
    TokenUsageUpdated,
    ToolFinished,
    ToolStarted,
)
from aether.services.workspace import WorkspaceService
from aether.web.app import create_app


class _FakeRunService:
    def __init__(self) -> None:
        self.requests: list[AgentRunRequest] = []
        self.cancel_requests: list[AgentRunCancelRequest] = []

    def start(self, request: AgentRunRequest, sink: RunEventSink | None = None) -> AgentRunResult:
        self.requests.append(request)
        if sink is not None:
            sink.emit(RunStarted(session_id=request.session_id, run_id=request.run_id or "run-1"))
            sink.emit(AssistantDelta(session_id=request.session_id, run_id=request.run_id or "run-1", text="hel", sequence=0))
            sink.emit(AssistantDelta(session_id=request.session_id, run_id=request.run_id or "run-1", text="lo", sequence=1))
            sink.emit(TokenUsageUpdated(session_id=request.session_id, run_id=request.run_id or "run-1", input_tokens=3, output_tokens=2))
            sink.emit(ToolStarted(session_id=request.session_id, run_id=request.run_id or "run-1", tool_call_id="tc1", tool_name="read_file", arguments={"path": "x.py"}))
            sink.emit(ToolFinished(session_id=request.session_id, run_id=request.run_id or "run-1", tool_call_id="tc1", tool_name="read_file", content="ok"))
            sink.emit(RunFinished(session_id=request.session_id, run_id=request.run_id or "run-1", final_text="hello"))
        return AgentRunResult(session_id=request.session_id, run_id=request.run_id or "run-1", final_text="hello")

    def cancel(self, request: AgentRunCancelRequest) -> bool:
        self.cancel_requests.append(request)
        return True

    def status(self, run_id_or_session_id: str) -> None:
        del run_id_or_session_id
        return None


class _PromptRunService(_FakeRunService):
    def start(self, request: AgentRunRequest, sink: RunEventSink | None = None) -> AgentRunResult:
        self.requests.append(request)
        decision = request.tool_permission_prompter.request_tool_permission(
            ToolPermissionRequest(
                session_id=request.session_id,
                tool_call_id="tc-perm",
                tool_name="shell",
                arguments={"command": "echo ok"},
                category="shell",
                risk="medium",
            )
        )
        assert decision.type.value == "allow_once"
        approval = request.approval_prompter.confirm_plan("## Plan\n\n- Do it")
        assert approval["confirmed"] is True
        if sink is not None:
            sink.emit(RunFinished(session_id=request.session_id, run_id=request.run_id or "run-prompt", final_text="approved"))
        return AgentRunResult(session_id=request.session_id, run_id=request.run_id or "run-prompt", final_text="approved")


def test_run_websocket_streams_service_events() -> None:
    service = _FakeRunService()
    app = create_app(auth_enabled=False, run_service=cast(AgentRunService, service))
    client = TestClient(app)

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(
            {
                "type": "run.start",
                "id": "run-1",
                "payload": {
                    "session_id": "ses",
                    "user_message": "hello",
                    "attachments": [
                        {
                            "type": "image",
                            "name": "plot.png",
                            "mimeType": "image/png",
                            "data": "data:image/png;base64,abc",
                        }
                    ],
                },
            }
        )
        assert ws.receive_json()["type"] == "run.accepted"
        types = [ws.receive_json()["type"] for _ in range(8)]

    assert types == [
        "run.started",
        "assistant.delta",
        "assistant.delta",
        "token.usage",
        "tool.started",
        "tool.finished",
        "run.finished",
        "run.result",
    ]
    assert service.requests[0].session_id == "ses"
    assert service.requests[0].attachments == [
        {
            "type": "image",
            "name": "plot.png",
            "mimeType": "image/png",
            "data": "data:image/png;base64,abc",
        }
    ]


def test_run_websocket_enriches_workspace_reference_attachments(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Notes\n\nUse workspace context.", encoding="utf-8")
    service = _FakeRunService()
    app = create_app(
        auth_enabled=False,
        run_service=cast(AgentRunService, service),
        workspace_service=WorkspaceService(root=tmp_path),
    )
    client = TestClient(app)

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(
            {
                "type": "run.start",
                "id": "run-workspace",
                "payload": {
                    "session_id": "ses",
                    "user_message": "summarize",
                    "attachments": [
                        {
                            "type": "text",
                            "name": "notes.md",
                            "path": "docs/notes.md",
                            "note": "workspace reference",
                        }
                    ],
                },
            }
        )
        assert ws.receive_json()["type"] == "run.accepted"
        assert ws.receive_json()["type"] == "run.started"

    attachment = service.requests[0].attachments[0]
    assert attachment["path"] == "docs/notes.md"
    assert attachment["_llm_language"] == "markdown"
    assert attachment["_llm_content"] == "# Notes\n\nUse workspace context."


def test_run_websocket_cancel_and_ping() -> None:
    service = _FakeRunService()
    app = create_app(auth_enabled=False, run_service=cast(AgentRunService, service))
    client = TestClient(app)

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json(
            {
                "type": "run.cancel",
                "payload": {"session_id": "ses", "run_id": "run-1", "reason": "test"},
            }
        )
        frame = ws.receive_json()

    assert frame["type"] == "run.cancel.accepted"
    assert frame["payload"]["cancelled"] is True
    assert service.cancel_requests[0].session_id == "ses"
    assert service.cancel_requests[0].run_id == "run-1"


def test_run_websocket_permission_and_approval_prompt_round_trip() -> None:
    service = _PromptRunService()
    app = create_app(auth_enabled=False, run_service=cast(AgentRunService, service))
    client = TestClient(app)

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(
            {
                "type": "run.start",
                "id": "run-prompt",
                "payload": {"session_id": "ses", "user_message": "needs prompts"},
            }
        )
        assert ws.receive_json()["type"] == "run.accepted"
        permission = ws.receive_json()
        assert permission["type"] == "permission.requested"
        assert permission["payload"]["session_id"] == "ses"
        assert permission["payload"]["request"]["session_id"] == "ses"
        prompt_id = permission["payload"]["prompt_id"]
        ws.send_json(
            {
                "type": "permission.respond",
                "payload": {"prompt_id": prompt_id, "decision": {"type": "allow_once"}},
            }
        )
        resolved = ws.receive_json()
        assert resolved["type"] == "prompt.resolved"
        assert resolved["payload"]["result"]["decision"] == {"type": "allow_once"}
        approval = ws.receive_json()
        assert approval["type"] == "approval.requested"
        approval_prompt_id = approval["payload"]["prompt_id"]
        ws.send_json(
            {
                "type": "approval.respond",
                "payload": {"prompt_id": approval_prompt_id, "confirmed": True},
            }
        )
        resolved = ws.receive_json()
        assert resolved["type"] == "prompt.resolved"
        assert resolved["payload"]["result"]["confirmed"] is True
        assert ws.receive_json()["type"] == "run.finished"
        assert ws.receive_json()["type"] == "run.result"


def test_run_websocket_replays_pending_prompt_after_reconnect() -> None:
    service = _PromptRunService()
    app = create_app(auth_enabled=False, run_service=cast(AgentRunService, service))
    client = TestClient(app)

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(
            {
                "type": "run.start",
                "id": "run-reconnect",
                "payload": {"session_id": "ses", "user_message": "needs reconnect"},
            }
        )
        assert ws.receive_json()["type"] == "run.accepted"
        permission = ws.receive_json()
        assert permission["type"] == "permission.requested"
        prompt_id = permission["payload"]["prompt_id"]

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        replayed = ws.receive_json()
        assert replayed["type"] == "permission.requested"
        assert replayed["payload"]["prompt_id"] == prompt_id
        assert replayed["payload"]["session_id"] == "ses"
        ws.send_json(
            {
                "type": "permission.respond",
                "payload": {"prompt_id": prompt_id, "decision": {"type": "allow_once"}},
            }
        )
        assert ws.receive_json()["type"] == "prompt.resolved"
        approval = ws.receive_json()
        assert approval["type"] == "approval.requested"
        approval_prompt_id = approval["payload"]["prompt_id"]
        ws.send_json(
            {
                "type": "approval.respond",
                "payload": {"prompt_id": approval_prompt_id, "confirmed": True},
            }
        )
        assert ws.receive_json()["type"] == "prompt.resolved"
        assert ws.receive_json()["type"] == "run.finished"
        assert ws.receive_json()["type"] == "run.result"


def test_run_websocket_rejects_invalid_message() -> None:
    app = create_app(auth_enabled=False)
    client = TestClient(app)

    with client.websocket_connect("/api/runs/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text("not-json")
        frame = ws.receive_json()

    assert frame["type"] == "error"
    assert frame["payload"]["code"] == "invalid_json"
