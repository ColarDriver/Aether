from __future__ import annotations

import ast
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import pytest

from aether.services.runs import AssistantDelta, event_to_public_dict
from aether.services.sessions import SessionCreateRequest, SessionService
from aether.services.providers import ProviderService
from aether.services.tools import ToolService


CLI_ROOT = Path(__file__).resolve().parents[2] / "cli"


def test_cli_modules_do_not_import_gateway_handlers() -> None:
    offenders: list[str] = []
    for path in sorted(CLI_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            for module in imported:
                if module.startswith("aether.gateway.handlers"):
                    offenders.append(f"{path.name} -> {module}")

    assert offenders == []


def test_cli_can_consume_services_without_gateway_handlers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    sessions = SessionService(session_dir=tmp_path / "sessions")
    created = sessions.create(
        SessionCreateRequest(
            session_id="cli-ses",
            provider="openai",
            model="gpt-5",
        )
    )

    listed = sessions.list()
    providers = ProviderService(environ={}).list_providers()
    tools = ToolService().list_tools()

    assert created.session_id == "cli-ses"
    assert [item.session_id for item in listed.sessions] == ["cli-ses"]
    assert {provider.name for provider in providers} >= {"openai", "claude"}
    assert any(tool.name == "read_file" for tool in tools.tools)


def test_service_contracts_are_json_compatible_without_gateway_protocol() -> None:
    payloads = [
        event_to_public_dict(AssistantDelta("ses", "run", "hello", 0)),
        asdict(SessionCreateRequest(provider="openai", model="gpt-5")),
    ]

    for payload in payloads:
        assert not (is_dataclass(payload) and type(payload).__module__.startswith("aether.gateway"))
        json.dumps(payload)
