from __future__ import annotations

from aether.runtime.credentials import (
    CredentialLookup,
    EnvCredentialSource,
    default_credential_lookup,
)


def test_env_source_returns_credential_with_metadata() -> None:
    source = EnvCredentialSource(environ={"OPENAI_API_KEY": "sk-test-secret-1234"})

    value = source.get("OPENAI_API_KEY")

    assert value is not None
    assert value.value == "sk-test-secret-1234"
    assert value.source == "env"
    assert value.key_name == "OPENAI_API_KEY"


def test_env_source_missing_returns_none() -> None:
    source = EnvCredentialSource(environ={})

    assert source.get("OPENAI_API_KEY") is None


def test_lookup_tries_names_in_order() -> None:
    lookup = CredentialLookup(
        (EnvCredentialSource(environ={"OPENAI_API_KEY_2": "sk-backup-secret"}),)
    )

    value = lookup.get_first(("OPENAI_API_KEY", "OPENAI_API_KEY_2"))

    assert value is not None
    assert value.key_name == "OPENAI_API_KEY_2"
    assert value.value == "sk-backup-secret"


def test_default_lookup_accepts_explicit_environ() -> None:
    lookup = default_credential_lookup(environ={"ANTHROPIC_API_KEY": "sk-ant-secret"})

    value = lookup.get_first(("ANTHROPIC_API_KEY",))
    assert value is not None
    assert value.value == "sk-ant-secret"


def test_credential_repr_and_public_metadata_are_redacted() -> None:
    value = EnvCredentialSource(environ={"OPENAI_API_KEY": "sk-test-secret-1234"}).get(
        "OPENAI_API_KEY"
    )
    assert value is not None

    assert "sk-test-secret-1234" not in repr(value)
    metadata = value.public_metadata()
    assert metadata == {
        "source": "env",
        "name": "OPENAI_API_KEY",
        "configured": True,
        "redacted": "sk-t...1234",
    }
    assert "sk-test-secret-1234" not in str(metadata)
