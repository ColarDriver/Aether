from __future__ import annotations

import pytest

from aether.config.schema import EngineConfig
from aether.services.config import ConfigService


def test_config_service_reports_paths_and_defaults(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    service = ConfigService()

    paths = service.paths()
    defaults = service.defaults()
    statuses = service.environment_paths()

    assert paths.aether_home == str(tmp_path)
    assert paths.sessions_dir == str(tmp_path / "sessions")
    assert paths.prefs_file == str(tmp_path / "prefs.json")
    assert defaults.values["max_iterations"] == EngineConfig().max_iterations
    assert {status.name for status in statuses} == {"AETHER_HOME", "AETHER_PREFS", "AETHER_SESSIONS"}


def test_config_service_effective_values_are_public_safe() -> None:
    service = ConfigService(config=EngineConfig(web_search_api_key="secret"))

    values = service.effective().values

    assert values["web_search_api_key"] is True
    assert "secret" not in repr(values)
