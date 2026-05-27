from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aether.cli.sessions import session_file
from aether.runtime.context import CompressionResult
from aether.runtime.session.plan_artifact import read_plan, write_plan
from aether.runtime.session.session_state import get_cwd
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.services.config import PrefsService
from aether.services.context import ContextService
from aether.services.docs import DocsService
from aether.services.environment import EnvironmentService
from aether.services.providers import ModelSelectionService, ProviderService
from aether.services.runs import AgentRunService
from aether.services.sessions import SessionService
from aether.services.skills import SkillService
from aether.services.tasks import TaskService
from aether.services.workspace import WorkspaceService
from aether.web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    sessions = SessionService(session_dir=tmp_path / "sessions")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text("print(\"hi\")\n", encoding="utf-8")
    (workspace_root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "README.md").write_text("# Test Docs\n\nHello docs.\n", encoding="utf-8")
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "python"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Python\n\nUse when editing Python code.\n",
        encoding="utf-8",
    )
    skill_service = SkillService(config=SimpleNamespace(skill_search_paths=(str(skills_root),)))
    task_store = TaskStore(root=tmp_path / "tasks")
    app = create_app(
        auth_enabled=False,
        session_service=sessions,
        provider_service=ProviderService(environ={}),
        model_selection_service=ModelSelectionService(environ={}),
        skill_service=skill_service,
        task_service=TaskService(store=task_store),
        environment_service=EnvironmentService(env_path=tmp_path / ".env", environ={}),
        docs_service=DocsService(docs_root=docs_root),
        workspace_service=WorkspaceService(root=workspace_root),
        run_service=AgentRunService(session_service=sessions),
    )
    return TestClient(app)


def test_session_routes_create_list_current_resume_messages_and_delete(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "web_ses"},
    )
    assert created.status_code == 200
    assert created.json()["session_id"] == "web_ses"

    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["session_id"] == "web_ses"

    current = client.get("/api/sessions/current")
    assert current.status_code == 200
    assert current.json()["session"]["session_id"] == "web_ses"

    resumed = client.post("/api/sessions/web_ses/resume")
    assert resumed.status_code == 200
    assert resumed.json()["session_id"] == "web_ses"
    assert resumed.json()["messages"] == []

    detail = client.get("/api/sessions/web_ses")
    assert detail.status_code == 200
    assert detail.json()["session_id"] == "web_ses"
    assert detail.json()["info"]["session_id"] == "web_ses"

    updated = client.patch(
        "/api/sessions/web_ses",
        json={"model": "gpt-5.4-mini", "system_prompt": "Be concise", "update_system_prompt": True},
    )
    assert updated.status_code == 200
    assert updated.json()["model"] == "gpt-5.4-mini"
    assert updated.json()["system_prompt"] == "Be concise"

    search = client.get("/api/sessions/search?q=web")
    assert search.status_code == 200
    assert search.json()["sessions"][0]["session_id"] == "web_ses"

    exported = client.get("/api/sessions/web_ses/export")
    assert exported.status_code == 200
    assert exported.json()["session_id"] == "web_ses"
    assert exported.json()["data"]["model"] == "gpt-5.4-mini"

    imported = client.post(
        "/api/sessions/import",
        json={
            "data": exported.json()["data"],
            "new_session_id": "imported_web_ses",
            "make_current": False,
        },
    )
    assert imported.status_code == 200
    assert imported.json()["source_session_id"] == "web_ses"
    assert imported.json()["info"]["session_id"] == "imported_web_ses"
    assert imported.json()["overwritten"] is False

    conflict = client.post(
        "/api/sessions/import",
        json={"data": exported.json()["data"], "new_session_id": "imported_web_ses"},
    )
    assert conflict.status_code == 409

    renamed = client.post(
        "/api/sessions/imported_web_ses/rename",
        json={"new_session_id": "renamed_web_ses"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["session_id"] == "renamed_web_ses"

    messages = client.get("/api/sessions/web_ses/messages")
    assert messages.status_code == 200
    assert messages.json() == {"session_id": "web_ses", "messages": []}

    write_plan("web_ses", "# stale plan\n")
    store = client.app.state.aether_services.tasks._store
    assert isinstance(store, TaskStore)
    store.create(
        TaskRecord(
            task_id="delete-task",
            parent_session_id="web_ses",
            subagent_type="worker",
            prompt="delete me",
            status=TaskStatus.COMPLETED,
            started_at=1.0,
            finished_at=2.0,
        )
    )

    sessions_dir = client.app.state.aether_services.sessions._session_dir
    assert session_file("web_ses", base=sessions_dir).is_file()

    deleted = client.delete("/api/sessions/web_ses")
    assert deleted.status_code == 204
    assert not session_file("web_ses", base=sessions_dir).exists()
    assert read_plan("web_ses") is None
    assert store.read("delete-task") is None

    relisted = client.get("/api/sessions")
    assert relisted.status_code == 200
    assert [item["session_id"] for item in relisted.json()["sessions"]] == ["renamed_web_ses"]

    deleted_import = client.delete("/api/sessions/renamed_web_ses")
    assert deleted_import.status_code == 204
    final_list = client.get("/api/sessions")
    assert [item["session_id"] for item in final_list.json()["sessions"]] == []

    current_after_delete = client.get("/api/sessions/current")
    assert current_after_delete.status_code == 200
    assert current_after_delete.json()["session"] is None

    missing = client.get("/api/sessions/web_ses/messages")
    assert missing.status_code == 404


def test_session_fork_route_copies_transcript_prefix(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "source_ses"},
    )
    assert created.status_code == 200
    services = client.app.state.aether_services
    services.sessions.persist_run_result(
        "source_ses",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ],
    )

    forked = client.post(
        "/api/sessions/source_ses/fork",
        json={"message_index": 1, "new_session_id": "forked_ses"},
    )

    assert forked.status_code == 200
    body = forked.json()
    assert body["source_session_id"] == "source_ses"
    assert body["info"]["session_id"] == "forked_ses"
    assert body["messages_copied"] == 2
    assert [message["text"] for message in body["messages"]] == ["first", "answer"]
    current = client.get("/api/sessions/current")
    assert current.json()["session"]["session_id"] == "forked_ses"


def test_session_action_fork_route_accepts_stable_turn_target(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "source_target_ses"},
    )
    assert created.status_code == 200
    services = client.app.state.aether_services
    services.sessions.persist_run_result(
        "source_target_ses",
        messages=[
            {"role": "user", "id": "turn-1", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "id": "turn-2", "content": "second"},
            {"role": "assistant", "content": "second answer"},
        ],
    )

    forked = client.post(
        "/api/sessions/source_target_ses/actions/fork",
        json={"target_user_message_id": "turn-2", "expected_content": "second", "new_session_id": "forked_target_ses"},
    )

    assert forked.status_code == 200
    body = forked.json()
    assert body["info"]["session_id"] == "forked_target_ses"
    assert body["forked_from_index"] == 2
    assert [message["text"] for message in body["messages"]] == ["first", "answer", "second"]


def test_session_rewind_route_truncates_transcript(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "rewind_ses"},
    )
    assert created.status_code == 200
    services = client.app.state.aether_services
    services.sessions.persist_run_result(
        "rewind_ses",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ],
    )

    rewound = client.post(
        "/api/sessions/rewind_ses/rewind",
        json={"message_index": 0},
    )

    assert rewound.status_code == 200
    body = rewound.json()
    assert body["session_id"] == "rewind_ses"
    assert body["rewound_to_index"] == 0
    assert body["messages_kept"] == 1
    assert body["messages_removed"] == 2
    assert body["info"]["message_count"] == 1
    assert [message["text"] for message in body["messages"]] == ["first"]
    current = client.get("/api/sessions/current")
    assert current.json()["session"]["session_id"] == "rewind_ses"


def test_session_turn_checkpoint_diff_route_returns_file_diff(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "diff_web"},
    )
    assert created.status_code == 200
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    services = client.app.state.aether_services
    services.sessions.persist_run_result(
        "diff_web",
        messages=[
            {"role": "user", "id": "turn-1", "content": "edit app.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "file_edit", "arguments": {"path": "app.py"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "file_edit",
                "content": "edited",
                "metadata": {
                    "path": "app.py",
                    "diff": diff,
                    "workspace_checkpoint": {"checkpoint_id": "cp-1", "root": "/repo"},
                },
            },
            {"role": "assistant", "content": "done"},
        ],
    )

    result = client.get(
        "/api/sessions/diff_web/turn-checkpoints/diff?targetUserMessageId=turn-1&path=app.py",
    )

    assert result.status_code == 200
    body = result.json()
    assert body["state"] == "ok"
    assert body["diff"] == diff
    assert body["target"]["target_user_message_id"] == "turn-1"
    assert body["checkpoint_id"] == "cp-1"


def test_config_prefs_and_diagnostics_routes(client: TestClient) -> None:
    config = client.get("/api/config")
    assert config.status_code == 200
    assert "values" in config.json()

    paths = client.get("/api/config/paths")
    assert paths.status_code == 200
    assert paths.json()["aether_home"]

    prefs = client.get("/api/prefs")
    assert prefs.status_code == 200
    assert prefs.json()["version"] >= 1

    saved = client.put("/api/prefs", json={"key": "ui.theme", "value": "dark"})
    assert saved.status_code == 200
    assert saved.json() == {"ok": True, "key": "ui.theme", "value": "dark"}

    pref = client.get("/api/prefs/ui.theme")
    assert pref.status_code == 200
    assert pref.json() == {"key": "ui.theme", "value": "dark"}

    deleted = client.request("DELETE", "/api/prefs", json={"key": "ui.theme"})
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True, "key": "ui.theme", "deleted": True}

    invalid = client.put("/api/prefs", json={"key": "version", "value": 2})
    assert invalid.status_code == 400

    diagnostics = client.get("/api/diagnostics")
    assert diagnostics.status_code == 200
    assert "enabled" in diagnostics.json()


def test_analytics_route_reports_sessions(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "analytics_web"},
    )
    assert created.status_code == 200

    report = client.get("/api/analytics?days=30&limit=5")
    assert report.status_code == 200
    body = report.json()
    assert body["days"] == 30
    assert body["summary"]["session_count"] == 1
    assert body["models"][0]["provider"] == "openai"


def test_docs_routes_list_and_read_project_markdown(client: TestClient) -> None:
    listed = client.get("/api/docs")
    assert listed.status_code == 200
    assert listed.json()["default_path"] == "README.md"
    assert listed.json()["documents"][0]["title"] == "Test Docs"

    content = client.get("/api/docs/README.md")
    assert content.status_code == 200
    assert content.json()["content"].startswith("# Test Docs")

    missing = client.get("/api/docs/missing.md")
    assert missing.status_code == 404


def test_commands_route_exposes_slash_catalog(client: TestClient) -> None:
    result = client.get("/api/commands")

    assert result.status_code == 200
    by_name = {item["name"]: item for item in result.json()["commands"]}
    assert by_name["/plan"]["category"] == "session"
    assert by_name["/help"]["description"]


def test_provider_preflight_route_reports_current_run_readiness(client: TestClient) -> None:
    result = client.get("/api/providers/preflight?provider=openai&model=gpt-5.4")

    assert result.status_code == 200
    body = result.json()
    assert body["provider_name"] == "openai"
    assert body["model"] == "gpt-5.4"
    assert body["status"] == "error"
    assert body["ready"] is False
    assert body["chat_completions_url"] == "https://api.openai.com/v1/chat/completions"
    assert "OPENAI_API_KEY" in body["issues"][0]


def test_context_routes_report_status_and_compress_session(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "ctx_web"},
    )
    assert created.status_code == 200

    status = client.get("/api/context/ctx_web/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["session_id"] == "ctx_web"
    assert status_body["context_engine"] == "default"
    assert status_body["compression_count"] == 0
    assert status_body["message_count"] == 0

    skipped = client.post("/api/context/ctx_web/compress", json={"focus": "auth"})
    assert skipped.status_code == 200
    skipped_body = skipped.json()
    assert skipped_body["status"] == "skipped"
    assert skipped_body["last_compression"]["reason"] == "not_enough_context"

    services = client.app.state.aether_services
    services.sessions.persist_run_result(
        "ctx_web",
        messages=[{"role": "user", "content": f"message {idx}"} for idx in range(6)],
    )

    class FakeCompressionService:
        def __init__(self) -> None:
            self.focus = None

        def compress(self, request):
            self.focus = request.focus
            return CompressionResult(
                messages=[{"role": "user", "content": "summary"}],
                status="compressed",
                metadata={
                    "status": "compressed",
                    "source_message_count": 6,
                    "result_message_count": 1,
                },
            )

    fake = FakeCompressionService()
    services.context = ContextService(
        session_service=services.sessions,
        compression_service_factory=lambda _record: fake,
    )

    compressed = client.post("/api/context/ctx_web/compress", json={"focus": "auth", "force": True})
    assert compressed.status_code == 200
    compressed_body = compressed.json()
    assert compressed_body["status"] == "compressed"
    assert compressed_body["compression_count"] == 1
    assert compressed_body["message_count"] == 1
    assert compressed_body["last_compression"]["source_message_count"] == 6
    assert fake.focus == "auth"
    assert services.sessions.transcript("ctx_web")[0].text == "summary"

    missing = client.get("/api/context/missing/status")
    assert missing.status_code == 404


def test_mcp_status_route_reports_runtime_integration_state(client: TestClient) -> None:
    result = client.get("/api/mcp/status")

    assert result.status_code == 200
    body = result.json()
    assert body["status"] in {"available", "not_configured"}
    assert isinstance(body["servers"], list)
    assert isinstance(body["imported_tools"], list)


def test_mcp_resources_route_reports_resource_boundary(client: TestClient) -> None:
    result = client.get("/api/mcp/resources")

    assert result.status_code == 200
    body = result.json()
    assert body["status"] in {"not_configured", "not_available", "available"}
    assert isinstance(body["resources"], list)
    assert "message" in body


def test_mcp_resource_read_route_reports_resource_boundary(client: TestClient) -> None:
    result = client.get("/api/mcp/resources/read?server=filesystem&uri=file%3A%2F%2F%2FREADME.md")

    assert result.status_code == 200
    body = result.json()
    assert body["status"] in {"not_configured", "server_not_found", "not_available", "available"}
    assert body["server"] == "filesystem"
    assert body["uri"] == "file:///README.md"
    assert isinstance(body["contents"], list)
    assert "message" in body


def test_mcp_config_routes_manage_local_servers(client: TestClient) -> None:
    listed = client.get("/api/mcp/config")
    assert listed.status_code == 200
    assert listed.json()["exists"] is False
    assert listed.json()["servers"] == []

    saved = client.put(
        "/api/mcp/servers",
        json={
            "name": "local fs",
            "command": "node",
            "args": ["server.js"],
            "env": {"TOKEN": "${MCP_TOKEN}"},
            "timeout": 5,
            "connect_timeout": 2,
        },
    )
    assert saved.status_code == 200
    saved_body = saved.json()
    assert saved_body["ok"] is True
    assert saved_body["server"]["name"] == "local_fs"
    assert saved_body["server"]["env_keys"] == ["TOKEN"]

    relisted = client.get("/api/mcp/config")
    assert relisted.status_code == 200
    servers = relisted.json()["servers"]
    assert [(server["name"], server["command"], server["args"]) for server in servers] == [
        ("local_fs", "node", ["server.js"])
    ]

    refreshed = client.post("/api/mcp/refresh")
    assert refreshed.status_code == 200
    assert "status" in refreshed.json()

    deleted = client.delete("/api/mcp/servers/local_fs")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True
    assert client.get("/api/mcp/config").json()["servers"] == []


def test_web_search_routes_report_status_and_test_configuration(client: TestClient) -> None:
    status = client.get("/api/web-search/status")
    assert status.status_code == 200
    body = status.json()
    assert body["provider"] in {"brave", "tavily", "bocha"}
    assert body["status"] in {"ready", "missing_credential", "invalid_provider"}
    assert "WEB_SEARCH_API_KEY" == body["credential_name"]

    tested = client.post("/api/web-search/test", json={"query": "docs", "max_results": 1})
    assert tested.status_code == 200
    test_body = tested.json()
    assert test_body["query"] == "docs"
    assert "provider" in test_body


def test_task_routes_list_session_tasks_and_output_tail(client: TestClient) -> None:
    store = client.app.state.aether_services.tasks._store
    assert isinstance(store, TaskStore)
    store.create(
        TaskRecord(
            task_id="task-running",
            parent_session_id="task_web",
            subagent_type="explorer",
            prompt="Inspect renderer",
            status=TaskStatus.RUNNING,
            started_at=20.0,
            model="gpt-5.4",
            background=True,
            tool_use_count=2,
            input_tokens=100,
            output_tokens=25,
            iterations=1,
            agent_type_def_snapshot={"name": "explorer", "description": "Inspect code"},
        )
    )
    store.write_result("task-running", {"status": "running", "summary": "partial result"})
    store.append_output("task-running", "first line\nsecond line\n")
    store.append_message("task-running", {"role": "assistant", "content": "checking files", "iteration": 1})
    store.append_message(
        "task-running",
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call-1",
            "content": "file contents",
            "elapsed_ms": 12.5,
        },
    )
    store.enqueue_pending_message("task-running", "please inspect src/auth.ts")
    store.create(
        TaskRecord(
            task_id="task-child",
            parent_session_id="task_web",
            subagent_type="verifier",
            prompt="Verify renderer",
            status=TaskStatus.COMPLETED,
            started_at=21.0,
            finished_at=23.0,
            parent_task_id="task-running",
            child_depth=2,
            summary="verified",
        )
    )
    store.append_message("task-child", {"role": "assistant", "content": "child checked it", "iteration": 1})
    store.enqueue_pending_message("task-child", "parent follow-up for child")
    assert store.drain_pending_messages("task-child") == ["parent follow-up for child"]
    store.create(
        TaskRecord(
            task_id="task-done",
            parent_session_id="other_session",
            subagent_type="worker",
            prompt="Patch files",
            status=TaskStatus.COMPLETED,
            started_at=10.0,
            finished_at=12.0,
            summary="patched",
        )
    )

    session_tasks = client.get("/api/sessions/task_web/tasks")
    assert session_tasks.status_code == 200
    body = session_tasks.json()
    assert body["total_count"] == 2
    assert body["active_count"] == 1
    by_task_id = {task["task_id"]: task for task in body["tasks"]}
    assert by_task_id["task-running"]["metadata"]["agent_type"] == "explorer"
    assert by_task_id["task-running"]["output_tail"] is None
    assert by_task_id["task-child"]["parent_task_id"] == "task-running"

    detail = client.get("/api/tasks/task-running")
    assert detail.status_code == 200
    assert detail.json()["output_tail"] == "first line\nsecond line\n"

    result = client.get("/api/tasks/task-running/result")
    assert result.status_code == 200
    result_body = result.json()
    assert result_body["task_id"] == "task-running"
    assert result_body["result_path"].endswith("result.json")
    assert result_body["result"] == {"status": "running", "summary": "partial result"}

    messages = client.get("/api/tasks/task-running/messages")
    assert messages.status_code == 200
    message_body = messages.json()
    assert message_body["task_id"] == "task-running"
    assert message_body["messages"][0]["role"] == "assistant"
    assert message_body["messages"][0]["content"] == "checking files"
    assert message_body["messages"][1]["name"] == "read_file"
    assert message_body["messages"][1]["elapsed_ms"] == 12.5
    assert message_body["pending_messages"][0]["message"] == "please inspect src/auth.ts"
    assert message_body["delivered_messages"] == []

    sent_message = client.post(
        "/api/tasks/task-running/messages",
        json={"message": "browser follow-up", "summary": "follow up"},
    )
    assert sent_message.status_code == 200
    sent_body = sent_message.json()
    assert sent_body["queued"] is True
    assert sent_body["task_id"] == "task-running"
    queued_messages = client.get("/api/tasks/task-running/messages")
    assert queued_messages.status_code == 200
    assert [item["message"] for item in queued_messages.json()["pending_messages"]] == [
        "please inspect src/auth.ts",
        "browser follow-up",
    ]

    assert store.drain_pending_messages("task-running") == [
        "please inspect src/auth.ts",
        "browser follow-up",
    ]
    delivered_messages = client.get("/api/tasks/task-running/messages")
    assert delivered_messages.status_code == 200
    delivered_body = delivered_messages.json()
    assert delivered_body["pending_messages"] == []
    assert delivered_body["delivered_messages"][0]["message"] == "please inspect src/auth.ts"
    assert delivered_body["delivered_messages"][1]["message"] == "browser follow-up"
    assert isinstance(delivered_body["delivered_messages"][0]["delivered_at"], float)

    terminal_message = client.post(
        "/api/tasks/task-child/messages",
        json={"message": "too late"},
    )
    assert terminal_message.status_code == 409

    invalid_messages = client.get("/api/tasks/task-running/messages?limit=0")
    assert invalid_messages.status_code == 400

    child_messages = client.get("/api/tasks/task-running/children/messages")
    assert child_messages.status_code == 200
    child_body = child_messages.json()
    assert child_body["task_id"] == "task-running"
    assert child_body["streams"][0]["task"]["task_id"] == "task-child"
    assert child_body["streams"][0]["messages"][0]["content"] == "child checked it"
    assert child_body["streams"][0]["delivered_messages"][0]["message"] == "parent follow-up for child"

    invalid_child_messages = client.get("/api/tasks/task-running/children/messages?per_task_limit=0")
    assert invalid_child_messages.status_code == 400

    global_active = client.get("/api/tasks?active_only=true")
    assert global_active.status_code == 200
    assert [task["task_id"] for task in global_active.json()["tasks"]] == ["task-running"]

    client.app.state.aether_services.runs.stop_task = lambda task_id: task_id == "task-running"
    stopped = client.post("/api/tasks/task-running/stop")
    assert stopped.status_code == 200
    stopped_body = stopped.json()
    assert stopped_body["task_id"] == "task-running"
    assert stopped_body["delivered"] is True
    assert stopped_body["status"] == "running"

    terminal_stop = client.post("/api/tasks/task-child/stop")
    assert terminal_stop.status_code == 200
    assert terminal_stop.json()["delivered"] is False

    missing = client.get("/api/tasks/missing")
    assert missing.status_code == 404

    missing_result = client.get("/api/tasks/task-child/result")
    assert missing_result.status_code == 404

    invalid = client.get("/api/tasks?limit=0")
    assert invalid.status_code == 400


def test_plan_routes_read_and_update_session_mode(client: TestClient) -> None:
    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "plan_web"},
    )
    assert created.status_code == 200

    current = client.get("/api/plan/plan_web")
    assert current.status_code == 200
    assert current.json()["mode"] == "agent"
    assert current.json()["has_plan"] is False
    assert current.json()["plan_path"].endswith("plan_web.md")

    updated = client.put("/api/plan/plan_web/mode", json={"mode": "plan"})
    assert updated.status_code == 200
    assert updated.json()["mode"] == "plan"
    assert updated.json()["info"]["mode"] == "plan"

    write_plan("plan_web", "# Plan\n")
    with_artifact = client.get("/api/plan/plan_web")
    assert with_artifact.status_code == 200
    assert with_artifact.json()["has_plan"] is True
    assert with_artifact.json()["plan_content"] == "# Plan\n"

    cleared = client.post("/api/plan/plan_web/clear")
    assert cleared.status_code == 200
    assert cleared.json()["mode"] == "agent"
    assert cleared.json()["has_plan"] is False
    assert cleared.json()["plan_content"] is None
    assert cleared.json()["info"]["mode"] == "agent"

    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["mode"] == "agent"

    invalid = client.put("/api/plan/plan_web/mode", json={"mode": "invalid"})
    assert invalid.status_code == 400


def test_workspace_routes_list_read_and_search_files(client: TestClient) -> None:
    tree = client.get("/api/workspace/tree")
    assert tree.status_code == 200
    assert tree.json()["entries"][0]["path"] == "app.py"

    file = client.get("/api/workspace/file?path=app.py")
    assert file.status_code == 200
    assert file.json()["content"] == 'print("hi")\n'
    assert file.json()["language"] == "python"
    assert file.json()["mime_type"] == "text/x-python"

    image = client.get("/api/workspace/file?path=logo.png")
    assert image.status_code == 200
    assert image.json()["binary"] is True
    assert image.json()["mime_type"] == "image/png"

    raw_image = client.get("/api/workspace/raw?path=logo.png")
    assert raw_image.status_code == 200
    assert raw_image.content == b"\x89PNG\r\n\x1a\n"
    assert raw_image.headers["content-type"].startswith("image/png")

    saved = client.put("/api/workspace/file", json={"path": "app.py", "content": 'print("bye")\n'})
    assert saved.status_code == 200
    assert saved.json()["content"] == 'print("bye")\n'
    assert saved.json()["language"] == "python"
    assert client.get("/api/workspace/file?path=app.py").json()["content"] == 'print("bye")\n'

    search = client.get("/api/workspace/search?q=app")
    assert search.status_code == 200
    assert search.json()["entries"][0]["path"] == "app.py"

    created_file = client.post("/api/workspace/file", json={"path": "notes/todo.txt", "content": "x"})
    assert created_file.status_code == 404

    created_dir = client.post("/api/workspace/directory", json={"path": "notes"})
    assert created_dir.status_code == 200
    assert created_dir.json()["path"] == "notes"

    created_file = client.post("/api/workspace/file", json={"path": "notes/todo.txt", "content": "x"})
    assert created_file.status_code == 200
    assert created_file.json()["path"] == "notes/todo.txt"
    assert created_file.json()["content"] == "x"

    renamed = client.patch("/api/workspace/path", json={"path": "notes/todo.txt", "new_path": "notes/done.txt"})
    assert renamed.status_code == 200
    assert renamed.json()["path"] == "notes/done.txt"

    delete_nonempty = client.delete("/api/workspace/path?path=notes")
    assert delete_nonempty.status_code == 409

    deleted = client.delete("/api/workspace/path?path=notes&recursive=true")
    assert deleted.status_code == 204
    assert client.get("/api/workspace/file?path=notes/done.txt").status_code == 404

    escaped = client.get("/api/workspace/file?path=../secret.py")
    assert escaped.status_code == 400

    escaped_save = client.put("/api/workspace/file", json={"path": "../secret.py", "content": "x"})
    assert escaped_save.status_code == 400

    image_save = client.put("/api/workspace/file", json={"path": "logo.png", "content": "not an image"})
    assert image_save.status_code == 400

    escaped_raw = client.get("/api/workspace/raw?path=../secret.py")
    assert escaped_raw.status_code == 400


def test_workspace_root_routes_switch_persist_and_update_session_cwd(client: TestClient, tmp_path) -> None:
    original_root = client.app.state.aether_services.workspace.root
    alternate = tmp_path / "alternate-workspace"
    alternate.mkdir()
    (alternate / "alt.py").write_text("print('alt')\n", encoding="utf-8")

    current = client.get("/api/workspace/root")
    assert current.status_code == 200
    assert current.json()["root"] == str(original_root)
    assert current.json()["recent_roots"][0] == str(original_root)

    created = client.post(
        "/api/sessions",
        json={"provider": "openai", "model": "gpt-5.4", "session_id": "root_ses"},
    )
    assert created.status_code == 200

    switched = client.put(
        "/api/workspace/root",
        json={"path": str(alternate), "session_id": "root_ses"},
    )
    assert switched.status_code == 200
    body = switched.json()
    assert body["root"] == str(alternate.resolve())
    assert body["recent_roots"][:2] == [str(alternate.resolve()), str(original_root)]
    assert client.app.state.aether_services.prefs.get("workspace.active_root") == str(alternate.resolve())
    assert get_cwd("root_ses") == str(alternate.resolve())

    tree = client.get("/api/workspace/tree")
    assert tree.status_code == 200
    assert [entry["path"] for entry in tree.json()["entries"]] == ["alt.py"]

    invalid = client.put("/api/workspace/root", json={"path": str(alternate / "missing")})
    assert invalid.status_code == 404
    assert client.app.state.aether_services.workspace.root == alternate.resolve()


def test_create_app_uses_remembered_workspace_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path / "home"))
    remembered = tmp_path / "remembered"
    remembered.mkdir()
    prefs = PrefsService()
    prefs.set("workspace.active_root", str(remembered))
    prefs.set("workspace.recent_roots", [str(remembered)])

    app = create_app(auth_enabled=False, prefs_service=prefs)

    assert app.state.aether_services.workspace.root == remembered.resolve()


def test_workspace_git_routes_expose_status_diff_restore_and_checkpoints(client: TestClient) -> None:
    root = client.app.state.aether_services.workspace.root
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Aether Test")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "initial")
    (root / "app.py").write_text("print('changed')\n", encoding="utf-8")

    status = client.get("/api/workspace/git/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["available"] is True
    assert status_body["clean"] is False
    assert status_body["files"][0]["path"] == "app.py"

    diff = client.get("/api/workspace/git/diff?path=app.py")
    assert diff.status_code == 200
    assert "+print('changed')" in diff.json()["diff"]

    checkpoint = client.post("/api/workspace/checkpoints", json={"label": "web checkpoint"})
    assert checkpoint.status_code == 200
    assert checkpoint.json()["label"] == "web checkpoint"
    checkpoint_id = checkpoint.json()["checkpoint_id"]

    restored = client.post("/api/workspace/git/restore", json={"path": "app.py"})
    assert restored.status_code == 200
    assert (root / "app.py").read_text(encoding="utf-8") == 'print("hi")\n'

    checkpoints = client.get("/api/workspace/checkpoints")
    assert checkpoints.status_code == 200
    assert checkpoints.json()["checkpoints"][0]["checkpoint_id"] == checkpoint_id

    restored_checkpoint = client.post(f"/api/workspace/checkpoints/{checkpoint_id}/restore")
    assert restored_checkpoint.status_code == 200
    assert (root / "app.py").read_text(encoding="utf-8") == "print('changed')\n"

    (root / "app.py").write_text("print('agent changed')\n", encoding="utf-8")
    restored_paths = client.post(
        f"/api/workspace/checkpoints/{checkpoint_id}/restore-paths",
        json={"paths": ["app.py"]},
    )
    assert restored_paths.status_code == 200
    assert (root / "app.py").read_text(encoding="utf-8") == "print('changed')\n"

    changes = client.get("/api/workspace/changes")
    assert changes.status_code == 200
    assert changes.json()["changes"][0]["path"] == "app.py"
    current_hash = changes.json()["changes"][0]["current_hash"]

    accepted = client.post("/api/workspace/changes/accept", json={"paths": ["app.py"]})
    assert accepted.status_code == 200
    assert accepted.json()["action"] == "accepted"
    assert client.get("/api/workspace/changes").json()["changes"][0]["accepted"] is True

    (root / "app.py").write_text("print('conflict')\n", encoding="utf-8")
    conflict = client.post(
        "/api/workspace/changes/reject",
        json={"paths": ["app.py"], "expected_hashes": {"app.py": current_hash}},
    )
    assert conflict.status_code == 409

    rejected = client.post("/api/workspace/changes/reject", json={"paths": ["app.py"]})
    assert rejected.status_code == 200
    assert rejected.json()["action"] == "rejected"
    assert (root / "app.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_environment_routes_list_set_reveal_and_delete(client: TestClient) -> None:
    listed = client.get("/api/env")
    assert listed.status_code == 200
    assert listed.json()["env_path"].endswith(".env")
    assert any(item["key"] == "OPENAI_API_KEY" for item in listed.json()["variables"])

    saved = client.put("/api/env", json={"key": "OPENAI_API_KEY", "value": "sk-test-secret"})
    assert saved.status_code == 200
    assert saved.json()["key"] == "OPENAI_API_KEY"

    reveal = client.post("/api/env/reveal", json={"key": "OPENAI_API_KEY"})
    assert reveal.status_code == 200
    assert reveal.json()["value"] == "sk-test-secret"

    audit = client.get("/api/env/reveal-audit")
    assert audit.status_code == 200
    assert audit.json()["events"][0]["key"] == "OPENAI_API_KEY"

    deleted = client.request("DELETE", "/api/env", json={"key": "OPENAI_API_KEY"})
    assert deleted.status_code == 200

    missing = client.post("/api/env/reveal", json={"key": "OPENAI_API_KEY"})
    assert missing.status_code == 404


def _git(root, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def test_provider_model_and_auxiliary_routes(client: TestClient) -> None:
    providers = client.get("/api/providers")
    assert providers.status_code == 200
    assert {item["name"] for item in providers.json()["providers"]} >= {"openai", "claude", "codex"}

    current = client.get("/api/providers/current?provider=openai&model=gpt-5.4")
    assert current.status_code == 200
    assert current.json()["provider_name"] == "openai"
    assert current.json()["model"] == "gpt-5.4"

    models = client.get("/api/providers/openai/models")
    assert models.status_code == 200
    assert any(item["id"] == "gpt-5" for item in models.json()["models"])
    assert models.json()["discovery"]["kind"] == "static"

    selected = client.post(
        "/api/model/select",
        json={"provider": "openai", "model": "gpt-5.4", "persist_last_model": False},
    )
    assert selected.status_code == 200
    assert selected.json()["provider"] == "openai"
    assert selected.json()["ready"] is False
    assert selected.json()["missing_credentials"]

    auxiliary = client.get("/api/model/auxiliary")
    assert auxiliary.status_code == 200
    assert any(slot["slot"] == "subagent" for slot in auxiliary.json()["slots"])


def test_tools_and_skills_routes(client: TestClient) -> None:
    tools = client.get("/api/tools")
    assert tools.status_code == 200
    assert any(item["name"] == "read_file" for item in tools.json()["tools"])

    groups = client.get("/api/tools/groups")
    assert groups.status_code == 200
    assert any(group["name"] == "filesystem" for group in groups.json()["groups"])

    skills = client.get("/api/skills")
    assert skills.status_code == 200
    assert skills.json()["skills"][0]["name"] == "python"

    detail = client.get("/api/skills/python")
    assert detail.status_code == 200
    assert detail.json()["name"] == "python"

    missing = client.get("/api/skills/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_logs_routes_list_and_filter_runtime_logs(client: TestClient, tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "gateway_crash.log").write_text(
        "INFO gateway ready\nERROR gateway failed\nWARNING tools slow\n",
        encoding="utf-8",
    )

    files = client.get("/api/logs/files")
    assert files.status_code == 200
    assert any(item["key"] == "gateway" for item in files.json()["files"])

    logs = client.get("/api/logs?file=gateway&level=ERROR&lines=10")
    assert logs.status_code == 200
    assert logs.json()["exists"] is True
    assert logs.json()["lines"] == ["ERROR gateway failed"]

    invalid = client.get("/api/logs?file=../secret")
    assert invalid.status_code == 400


def test_run_status_and_cancel_routes(client: TestClient) -> None:
    missing = client.get("/api/runs/nope")
    assert missing.status_code == 404

    cancel = client.post("/api/runs/nope/cancel", json={"reason": "test"})
    assert cancel.status_code == 200
    assert cancel.json() == {"cancelled": False}
