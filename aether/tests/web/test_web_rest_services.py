from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aether.services.environment import EnvironmentService
from aether.services.providers import ModelSelectionService, ProviderService
from aether.services.runs import AgentRunService
from aether.services.sessions import SessionService
from aether.services.skills import SkillService
from aether.web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    sessions = SessionService(session_dir=tmp_path / "sessions")
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "python"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Python\n\nUse when editing Python code.\n",
        encoding="utf-8",
    )
    skill_service = SkillService(config=SimpleNamespace(skill_search_paths=(str(skills_root),)))
    app = create_app(
        auth_enabled=False,
        session_service=sessions,
        provider_service=ProviderService(environ={}),
        model_selection_service=ModelSelectionService(environ={}),
        skill_service=skill_service,
        environment_service=EnvironmentService(env_path=tmp_path / ".env", environ={}),
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

    search = client.get("/api/sessions/search?q=web")
    assert search.status_code == 200
    assert search.json()["sessions"][0]["session_id"] == "web_ses"

    messages = client.get("/api/sessions/web_ses/messages")
    assert messages.status_code == 200
    assert messages.json() == {"session_id": "web_ses", "messages": []}

    deleted = client.delete("/api/sessions/web_ses")
    assert deleted.status_code == 204

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

    diagnostics = client.get("/api/diagnostics")
    assert diagnostics.status_code == 200
    assert "enabled" in diagnostics.json()


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
