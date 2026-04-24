"""Engine configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class ModelCallConfig:
    """Provider call configuration for a single turn."""

    temperature: float | None = None
    max_tokens: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineConfig:
    """Runtime configuration for the loop engine."""

    max_iterations: int = 8
    fail_on_tool_error: bool = False
    raise_on_middleware_error: bool = False
    fail_on_unknown_tool: bool = True
    enable_todo_hydration: bool = False
    memory_nudge_interval: int = 0
    skill_nudge_interval: int = 0
