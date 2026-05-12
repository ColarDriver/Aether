"""Thread-safe bridge from the sync engine to the AetherApp permission UI."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING

from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionRequest,
)

if TYPE_CHECKING:
    from aether.cli.app import AetherApp


class AetherToolPermissionPrompter:
    """Engine-facing prompter backed by ``AetherApp``'s bottom overlay."""

    def __init__(self, app: "AetherApp", loop: asyncio.AbstractEventLoop) -> None:
        self._app = app
        self._loop = loop

    def is_interactive(self) -> bool:
        return True

    def request_tool_permission(
        self,
        request: ToolPermissionRequest,
        *,
        timeout: float | None = None,
    ) -> ToolPermissionDecision:
        future: Future[ToolPermissionDecision] = Future()
        self._loop.call_soon_threadsafe(
            self._app.enqueue_permission_request,
            request,
            future,
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            self._loop.call_soon_threadsafe(
                self._app.cancel_permission_request,
                future,
            )
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                feedback="permission prompt timed out",
                source="timeout",
            )


__all__ = ["AetherToolPermissionPrompter"]
