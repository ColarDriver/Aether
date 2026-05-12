"""Unicode payload sanitizers for provider-call recovery."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable


def strip_surrogates(text: str) -> str:
    """Return ``text`` with lone UTF-16 surrogate code points removed."""

    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def strip_non_ascii(text: str) -> str:
    """Return ``text`` with characters outside the ASCII range removed."""

    return "".join(ch for ch in text if ord(ch) < 128)


def sanitize_structure_surrogates(value: Any) -> bool:
    """Remove surrogate code points from a mutable structure in place."""

    _, changed = _sanitize_value(value, strip_surrogates, seen=set())
    return changed


def sanitize_structure_non_ascii(value: Any) -> bool:
    """Remove non-ASCII characters from a mutable structure in place."""

    _, changed = _sanitize_value(value, strip_non_ascii, seen=set())
    return changed


def sanitize_provider_credentials_non_ascii(provider: Any) -> bool:
    """Strip non-ASCII chars from common provider credential/header fields."""

    changed = False
    for attr in (
        "api_key",
        "_api_key",
        "_access_token",
        "_oauth_access_token",
        "auth_token",
    ):
        if hasattr(provider, attr):
            current = getattr(provider, attr)
            if isinstance(current, str):
                cleaned = strip_non_ascii(current)
                if cleaned != current:
                    setattr(provider, attr, cleaned)
                    changed = True

    for attr in ("default_headers", "headers", "_headers", "_client_kwargs"):
        if hasattr(provider, attr):
            current = getattr(provider, attr)
            if sanitize_structure_non_ascii(current):
                changed = True

    for client_attr in ("client", "_client"):
        client = getattr(provider, client_attr, None)
        if client is None:
            continue
        for attr in ("api_key", "auth_token"):
            if hasattr(client, attr):
                current = getattr(client, attr)
                if isinstance(current, str):
                    cleaned = strip_non_ascii(current)
                    if cleaned != current:
                        setattr(client, attr, cleaned)
                        changed = True
    return changed


def _sanitize_value(
    value: Any,
    cleaner: Callable[[str], str],
    *,
    seen: set[int],
) -> tuple[Any, bool]:
    if isinstance(value, str):
        cleaned = cleaner(value)
        return cleaned, cleaned != value

    if isinstance(value, (bytes, bytearray, memoryview)) or value is None:
        return value, False

    value_id = id(value)
    if value_id in seen:
        return value, False
    seen.add(value_id)

    if isinstance(value, list):
        changed = False
        for idx, item in enumerate(value):
            cleaned, item_changed = _sanitize_value(item, cleaner, seen=seen)
            if item_changed:
                value[idx] = cleaned
                changed = True
        return value, changed

    if isinstance(value, dict):
        changed = False
        rebuilt: dict[Any, Any] = {}
        for key, item in list(value.items()):
            cleaned_key, key_changed = _sanitize_value(key, cleaner, seen=seen)
            cleaned_item, item_changed = _sanitize_value(item, cleaner, seen=seen)
            rebuilt[cleaned_key] = cleaned_item
            changed = changed or key_changed or item_changed
        if changed:
            value.clear()
            value.update(rebuilt)
        return value, changed

    if isinstance(value, tuple):
        changed = False
        cleaned_items = []
        for item in value:
            cleaned, item_changed = _sanitize_value(item, cleaner, seen=seen)
            cleaned_items.append(cleaned)
            changed = changed or item_changed
        return tuple(cleaned_items) if changed else value, changed

    if isinstance(value, set):
        changed = False
        cleaned_items = []
        for item in value:
            cleaned, item_changed = _sanitize_value(item, cleaner, seen=seen)
            cleaned_items.append(cleaned)
            changed = changed or item_changed
        if changed:
            value.clear()
            value.update(cleaned_items)
        return value, changed

    if is_dataclass(value) and not isinstance(value, type):
        changed = False
        for field in fields(value):
            if not hasattr(value, field.name):
                continue
            current = getattr(value, field.name)
            cleaned, field_changed = _sanitize_value(current, cleaner, seen=seen)
            if field_changed:
                try:
                    setattr(value, field.name, cleaned)
                    changed = True
                except (AttributeError, TypeError):
                    pass
        return value, changed

    return value, False
