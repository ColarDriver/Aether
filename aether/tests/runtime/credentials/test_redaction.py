from __future__ import annotations

from aether.runtime.credentials import (
    contains_secret_like_text,
    redact_mapping,
    redact_secret,
    redact_text,
)


def test_redact_secret_preserves_suffix_for_long_values() -> None:
    assert redact_secret("sk-test-secret-1234") == "sk-t...1234"


def test_redact_secret_handles_empty_and_short_values() -> None:
    assert redact_secret("") == ""
    assert redact_secret("short") == "***"


def test_redact_mapping_redacts_nested_auth_headers() -> None:
    payload = {
        "api_key": "sk-test-secret-1234",
        "nested": {
            "headers": {
                "Authorization": "Bearer sk-another-secret-5678",
            }
        },
        "model": "gpt-5.4",
        "base_url": "https://api.example.test/v1",
    }

    redacted = redact_mapping(payload)

    assert redacted["api_key"] == "sk-t...1234"
    assert redacted["nested"]["headers"]["Authorization"] == "Bear...5678"
    assert redacted["model"] == "gpt-5.4"
    assert redacted["base_url"] == "https://api.example.test/v1"
    assert "sk-test-secret-1234" not in str(redacted)
    assert "sk-another-secret-5678" not in str(redacted)


def test_redact_text_redacts_bearer_tokens_and_prefix_keys() -> None:
    text = "Authorization: Bearer sk-test-secret-1234 body tvly-abcdef123456"

    redacted = redact_text(text)

    assert "sk-test-secret-1234" not in redacted
    assert "tvly-abcdef123456" not in redacted
    assert "Authorization: Bearer" in redacted


def test_contains_secret_like_text() -> None:
    assert contains_secret_like_text("Bearer sk-test-secret-1234") is True
    assert contains_secret_like_text("model gpt-5.4") is False


def test_redact_mapping_preserves_non_secret_tuples_and_lists() -> None:
    redacted = redact_mapping(
        {
            "api_key_env_names": ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
            "items": ["gpt-5.4", "claude-sonnet-4-6"],
        }
    )

    assert redacted["api_key_env_names"] == ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert redacted["items"] == ["gpt-5.4", "claude-sonnet-4-6"]
