from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aether.cli.sessions import session_file
from aether.runtime.session.plan_artifact import read_plan, write_plan
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
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
    assert [item["session_id"] for item in relisted.json()["sessions"]] == []

    current_after_delete = client.get("/api/sessions/current")
    assert current_after_delete.status_code == 200
    assert current_after_delete.json()["session"] is None

    missing = client.get("/api/sessions/web_ses/messages")
    assert missing.status_code == 404


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

    assert store.drain_pending_messages("task-running") == ["please inspect src/auth.ts"]
    delivered_messages = client.get("/api/tasks/task-running/messages")
    assert delivered_messages.status_code == 200
    delivered_body = delivered_messages.json()
    assert delivered_body["pending_messages"] == []
    assert delivered_body["delivered_messages"][0]["message"] == "please inspect src/auth.ts"
    assert isinstance(delivered_body["delivered_messages"][0]["delivered_at"], float)

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

    search = client.get("/api/workspace/search?q=app")
    assert search.status_code == 200
    assert search.json()["entries"][0]["path"] == "app.py"

    escaped = client.get("/api/workspace/file?path=../secret.py")
    assert escaped.status_code == 400


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
