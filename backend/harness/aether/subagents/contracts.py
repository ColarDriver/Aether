"""Contracts for subagent delegation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import EngineRequest, EngineResult


class SubagentStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(slots=True)
class SubagentTask:
    task_id: str
    goal: str
    request: EngineRequest
    provider: ModelProvider | None = None
    max_iterations: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubagentResult:
    task_id: str
    status: SubagentStatus
    summary: str | None
    engine_result: EngineResult | None
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
