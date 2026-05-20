"""JSON serialization helpers for web adapters."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if _is_json_scalar(value):
        return value
    return str(value)


def json_safe_dict(value: Any) -> dict[str, Any]:
    payload = to_jsonable(value)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in payload.items():
        try:
            json.dumps(item)
        except (TypeError, ValueError):
            continue
        out[str(key)] = item
    return out


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


__all__ = ["json_safe_dict", "to_jsonable"]
