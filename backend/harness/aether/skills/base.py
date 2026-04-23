"""Skill provider abstraction placeholder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class SkillProvider(ABC):
    """Hook point for reusable skill bundles."""

    @abstractmethod
    def list_skills(self) -> List[str]:
        raise NotImplementedError
