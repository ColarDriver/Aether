"""Agent type registry — discovery over builtins and markdown files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from aether.agents.types.builtin import BUILTIN_AGENT_TYPES
from aether.agents.types.definition import AgentTypeDefinition
from aether.agents.types.markdown_loader import load_markdown_agent_type

logger = logging.getLogger(__name__)


class AgentTypeRegistry:
    """Resolve agent type names to ``AgentTypeDefinition`` instances."""

    def __init__(self, *, search_paths: Iterable[Path]) -> None:
        self.search_paths: list[Path] = [Path(p) for p in search_paths]
        self._types: dict[str, AgentTypeDefinition] = {}
        self._loaded = False

    def discover(self, *, force: bool = False) -> None:
        if self._loaded and not force:
            return
        self._types.clear()
        for builtin in BUILTIN_AGENT_TYPES:
            self._types[builtin.agent_type] = builtin
        for root in self.search_paths:
            if not root.exists() or not root.is_dir():
                continue
            source = self._infer_source(root)
            for md in sorted(root.glob("*.md")):
                definition = load_markdown_agent_type(md, source=source)
                if definition is not None:
                    self._types[definition.agent_type] = definition
                else:
                    logger.warning("skipping malformed agent type at %s", md)
        self._loaded = True

    def get(self, name: str) -> AgentTypeDefinition | None:
        self.discover()
        if not name:
            return None
        if name in self._types:
            return self._types[name]
        lowered = name.lower()
        for key, value in self._types.items():
            if key.lower() == lowered:
                return value
        return None

    def list_all(self) -> list[AgentTypeDefinition]:
        self.discover()
        return list(self._types.values())

    @staticmethod
    def _infer_source(root: Path) -> str:
        parts = root.parts
        if ".aether" in parts:
            return "user"
        if ".claude" in parts:
            return "project"
        return "external"
