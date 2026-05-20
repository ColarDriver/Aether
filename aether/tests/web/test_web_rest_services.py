from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

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


def test_run_status_and_cancel_routes(client: TestClient) -> None:
    missing = client.get("/api/runs/nope")
    assert missing.status_code == 404

    cancel = client.post("/api/runs/nope/cancel", json={"reason": "test"})
    assert cancel.status_code == 200
    assert cancel.json() == {"cancelled": False}
