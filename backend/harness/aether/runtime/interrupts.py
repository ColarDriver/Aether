"""Thread-safe session-scoped interrupt controller."""

from __future__ import annotations

from threading import RLock
from typing import Dict

from aether.runtime.interrupt_signal import InterruptSignal


class InterruptController:
    """Session-scoped interrupt signals with legacy flag-style API."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._signals: Dict[str, InterruptSignal] = {}

    def signal_for(self, session_id: str) -> InterruptSignal:
        with self._lock:
            signal = self._signals.get(session_id)
            if signal is None:
                signal = InterruptSignal()
                self._signals[session_id] = signal
            return signal

    def request(self, session_id: str, reason: str | None = None) -> None:
        self.signal_for(session_id).abort(reason)

    def clear(self, session_id: str) -> None:
        with self._lock:
            signal = self._signals.pop(session_id, None)
        if signal is not None:
            signal.close()

    def is_interrupted(self, session_id: str) -> bool:
        with self._lock:
            signal = self._signals.get(session_id)
        return False if signal is None else signal.is_aborted()

    def reason(self, session_id: str) -> str | None:
        with self._lock:
            signal = self._signals.get(session_id)
        return None if signal is None else signal.reason()


__all__ = ["InterruptController"]
