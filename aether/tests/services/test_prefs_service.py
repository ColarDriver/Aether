from __future__ import annotations

import pytest

from aether.services.common import ServiceValidationError
from aether.services.config import PrefsService


def test_prefs_service_round_trips_scoped_keys(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    service = PrefsService()

    service.set("last_model_by_provider.claude", "claude-sonnet-4-6")
    service.set("custom.flag", {"enabled": True})

    assert service.get("last_model_by_provider.claude") == "claude-sonnet-4-6"
    assert service.get("last_model_by_provider") == {"claude": "claude-sonnet-4-6"}
    assert service.get("custom.flag") == {"enabled": True}
    assert service.all()["custom.flag"] == {"enabled": True}

    assert service.delete("last_model_by_provider.claude") is True
    assert service.get("last_model_by_provider.claude") is None
    assert service.delete("custom.flag") is True
    assert service.get("custom.flag") is None


def test_last_model_helpers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    service = PrefsService()

    service.set_last_model("openai", "gpt-5")

    assert service.get_last_model("openai") == "gpt-5"


def test_prefs_service_rejects_invalid_mutations(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    service = PrefsService()

    with pytest.raises(ServiceValidationError):
        service.set("version", 2)
    with pytest.raises(ServiceValidationError):
        service.set("last_model_by_provider", "bad")
    with pytest.raises(ServiceValidationError):
        service.get("")
