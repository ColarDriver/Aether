from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True, frozen=True)
class AgentTypeDefinition:
    """Frozen description of a subagent persona."""

    agent_type: str
    description: str
    system_prompt: str = ""
    tools: Optional[tuple[str, ...]] = None
    disallowed_tools: tuple[str, ...] = ()
    model: Optional[str] = None
    skills: tuple[str, ...] = ()
    max_turns: Optional[int] = None
    isolation: Optional[str] = None
    background: bool = False
    source: str = "builtin"
    source_path: Optional[Path] = None

    def to_snapshot(self) -> dict[str, object]:
        return {
            "agent_type": self.agent_type,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools) if self.tools is not None else None,
            "disallowed_tools": list(self.disallowed_tools),
            "model": self.model,
            "skills": list(self.skills),
            "max_turns": self.max_turns,
            "isolation": self.isolation,
            "background": self.background,
            "source": self.source,
            "source_path": str(self.source_path) if self.source_path else None,
        }
