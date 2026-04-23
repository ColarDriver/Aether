"""Memory provider abstraction placeholder."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryProvider(ABC):
    """Hook point for future memory integrations."""

    @abstractmethod
    def build_context(self, session_id: str) -> str:
        raise NotImplementedError
