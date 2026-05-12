"""No-op memory provider used when memory is disabled or unconfigured."""

from __future__ import annotations

from .contracts import MemoryBundle, MemoryProvider, MemoryQuery


class NullMemoryProvider:
    """Memory provider that never retrieves or writes memory."""

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        return MemoryBundle.skipped("disabled")

    def observe_turn(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict],
        metadata: dict,
    ) -> None:
        return None

    def before_compaction(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict],
        metadata: dict,
    ) -> None:
        return None


__all__ = ["NullMemoryProvider", "MemoryProvider"]
