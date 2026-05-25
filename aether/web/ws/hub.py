"""Shared WebSocket run hub for reconnectable browser sessions."""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable

FrameSender = Callable[[dict[str, Any]], None]


class WebRunSocketHub:
    """Fan out live run frames to currently connected browser sockets."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._senders: list[FrameSender] = []

    def attach(self, sender: FrameSender) -> None:
        with self._lock:
            if not any(existing is sender for existing in self._senders):
                self._senders.append(sender)

    def detach(self, sender: FrameSender) -> None:
        with self._lock:
            self._senders = [existing for existing in self._senders if existing is not sender]

    def broadcast(self, frame: dict[str, Any]) -> None:
        with self._lock:
            senders = list(self._senders)
        for sender in senders:
            try:
                sender(frame)
            except RuntimeError:
                self.detach(sender)


__all__ = ["FrameSender", "WebRunSocketHub"]
