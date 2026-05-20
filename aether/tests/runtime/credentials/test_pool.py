from __future__ import annotations

import json

from aether.runtime.credentials.pool import CredentialPool


def _pool(strategy: str = "fill_first") -> CredentialPool:
    return CredentialPool.from_mapping(
        {
            "strategy": strategy,
            "providers": {
                "openai": [
                    {"name": "primary", "api_key_env": "OPENAI_API_KEY"},
                    {"name": "backup", "api_key_env": "OPENAI_API_KEY_2"},
                ]
            },
        },
        environ={
            "OPENAI_API_KEY": "sk-primary-secret",
            "OPENAI_API_KEY_2": "sk-backup-secret",
        },
    )


def test_pool_loads_configured_credentials() -> None:
    pool = _pool()

    selected = pool.select("openai")

    assert selected is not None
    assert selected.credential.name == "primary"
    assert selected.credential.credential.value == "sk-primary-secret"


def test_fill_first_always_selects_first_healthy() -> None:
    pool = _pool("fill_first")

    assert pool.select("openai").credential.name == "primary"  # type: ignore[union-attr]
    assert pool.select("openai").credential.name == "primary"  # type: ignore[union-attr]


def test_round_robin_cycles_healthy_credentials() -> None:
    pool = _pool("round_robin")

    assert pool.select("openai").credential.name == "primary"  # type: ignore[union-attr]
    assert pool.select("openai").credential.name == "backup"  # type: ignore[union-attr]
    assert pool.select("openai").credential.name == "primary"  # type: ignore[union-attr]


def test_unhealthy_credential_is_skipped() -> None:
    pool = _pool("fill_first")

    assert pool.mark_unhealthy("openai", "primary", reason="rate-limit") is True

    selected = pool.select("openai")

    assert selected is not None
    assert selected.credential.name == "backup"


def test_all_unhealthy_returns_none() -> None:
    pool = _pool()
    pool.mark_unhealthy("openai", "primary")
    pool.mark_unhealthy("openai", "backup")

    assert pool.select("openai") is None


def test_public_metadata_redacts_raw_secret() -> None:
    pool = _pool()

    metadata = pool.public_metadata()

    assert metadata["enabled"] is True
    rendered = json.dumps(metadata, sort_keys=True)
    assert "sk-primary-secret" not in rendered
    assert "sk-backup-secret" not in rendered
    assert "OPENAI_API_KEY" in rendered


def test_missing_pool_file_is_empty(tmp_path) -> None:
    pool = CredentialPool.from_file(tmp_path / "missing.json", environ={})

    assert pool.select("openai") is None
    assert pool.public_metadata()["enabled"] is False
