"""Builder abstraction for subagents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aether.subagents.contracts import SubagentTask

if TYPE_CHECKING:
    from aether.agents.core.agent import AgentEngine


class SubagentBuilder(ABC):
    """Build child agent instances for delegated tasks."""

    @abstractmethod
    def build_child(
        self, parent: "AgentEngine", task: SubagentTask, child_depth: int
    ) -> "AgentEngine":
        raise NotImplementedError
