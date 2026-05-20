from __future__ import annotations

import pytest

from aether.services.config import PrefsService
from aether.services.providers import ModelSelectionService, ProviderSelectionRequest


def test_model_selection_uses_prefs_and_persists_choice(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    prefs = PrefsService()
    prefs.set_last_model("openai", "gpt-from-prefs")
    service = ModelSelectionService(
        prefs=prefs,
        environ={"OPENAI_API_KEY": "sk-secret"},
    )

    selected = service.select(ProviderSelectionRequest(provider="openai", persist_last_model=True))

    assert selected.provider == "openai"
    assert selected.model == "gpt-from-prefs"
    assert selected.ready is True
    assert selected.missing_credentials == []
    assert prefs.get_last_model("openai") == "gpt-from-prefs"


def test_model_selection_reports_missing_credentials(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    service = ModelSelectionService(environ={})

    selected = service.select(ProviderSelectionRequest(provider="openai", model="gpt-5.4"))

    assert selected.ready is False
    assert selected.missing_credentials == ["OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"]
