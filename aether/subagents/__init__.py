"""Subagent execution module."""

from .contracts import SubagentResult, SubagentStatus, SubagentTask
from .manager import SubagentManager

__all__ = [
    "SubagentManager",
    "SubagentTask",
    "SubagentResult",
    "SubagentStatus",
]
