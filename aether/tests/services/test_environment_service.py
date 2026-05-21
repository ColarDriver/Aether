from __future__ import annotations

import pytest

from aether.services.common import ServiceNotFoundError, ServiceValidationError
from aether.services.environment import EnvironmentService


def test_environment_catalog_redacts_file_and_process_values(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-test-secret\nAETHER_MODEL=gpt-5.4\n", encoding="utf-8")
    service = EnvironmentService(env_path=env_path, environ={"ANTHROPIC_API_KEY": "anthropic-secret"})

    catalog = service.catalog()

    values = {item.key: item for item in catalog.variables}
    assert catalog.env_path == str(env_path)
    assert values["OPENAI_API_KEY"].source == "file"
    assert values["OPENAI_API_KEY"].redacted_value == "sk-t...cret"
    assert values["ANTHROPIC_API_KEY"].source == "process"
    assert values["AETHER_MODEL"].is_secret is False


def test_environment_set_delete_and_reveal_preserve_comments(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\nOPENAI_API_KEY=old\n", encoding="utf-8")
    environ: dict[str, str] = {}
    service = EnvironmentService(env_path=env_path, environ=environ)

    service.set("OPENAI_API_KEY", "new value")
    assert environ["OPENAI_API_KEY"] == "new value"
    service.set("WEB_SEARCH_PROVIDER", "brave")

    assert service.reveal("OPENAI_API_KEY").value == "new value"
    text = env_path.read_text(encoding="utf-8")
    assert "# comment" in text
    assert 'OPENAI_API_KEY="new value"' in text
    assert "WEB_SEARCH_PROVIDER=brave" in text

    deleted = service.delete("WEB_SEARCH_PROVIDER")
    assert deleted.ok is True
    assert "WEB_SEARCH_PROVIDER" not in environ
    assert "WEB_SEARCH_PROVIDER" not in env_path.read_text(encoding="utf-8")


def test_environment_rejects_invalid_and_missing_keys(tmp_path) -> None:
    service = EnvironmentService(env_path=tmp_path / ".env", environ={})

    with pytest.raises(ServiceValidationError):
        service.set("../SECRET", "x")
    with pytest.raises(ServiceNotFoundError):
        service.delete("OPENAI_API_KEY")
    with pytest.raises(ServiceNotFoundError):
        service.reveal("OPENAI_API_KEY")
