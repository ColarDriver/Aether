"""Interrupt signal primitive for event-driven cancellation."""

from __future__ import annotations

import threading
from typing import Callable

InterruptListener = Callable[[str | None], None]


class InterruptSignal:
    """Thread-safe cancellation primitive with optional parent inheritance."""

    def __init__(self, parent: "InterruptSignal | None" = None) -> None:
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._reason: str | None = None
        self._listeners: list[InterruptListener] = []
        self._parent = parent
        self._parent_listener: InterruptListener | None = None
        if parent is not None:
            def _inherit(reason: str | None) -> None:
                self.abort(reason)
            self._parent_listener = _inherit
            parent.add_listener(_inherit)
            if parent.is_aborted():
                self.abort(parent.reason())

    def abort(self, reason: str | None = None) -> bool:
        listeners: list[InterruptListener] = []
        with self._lock:
            if self._event.is_set():
                if self._reason is None and reason is not None:
                    self._reason = reason
                return False
            self._reason = reason
            self._event.set()
            listeners = list(self._listeners)
        for listener in listeners:
            listener(reason)
        return True

    def is_aborted(self) -> bool:
        return self._event.is_set()

    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def add_listener(self, listener: InterruptListener) -> None:
        reason: str | None = None
        should_fire = False
        with self._lock:
            if self._event.is_set():
                should_fire = True
                reason = self._reason
            else:
                self._listeners.append(listener)
        if should_fire:
            listener(reason)

    def remove_listener(self, listener: InterruptListener) -> None:
        with self._lock:
            self._listeners = [existing for existing in self._listeners if existing is not listener]

    def close(self) -> None:
        if self._parent is not None and self._parent_listener is not None:
            self._parent.remove_listener(self._parent_listener)
            self._parent_listener = None


__all__ = ["InterruptSignal", "InterruptListener"]
