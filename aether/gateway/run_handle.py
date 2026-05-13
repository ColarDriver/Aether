"""In-flight ``agent.run`` tracking for the gateway."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from aether.runtime.control.interrupt_signal import InterruptSignal


@dataclass(slots=True)
class RunHandle:
    """Cancellation state for one active agent run."""

    session_id: str
    run_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    interrupt_signal: InterruptSignal = field(default_factory=InterruptSignal)

    def cancel(self, reason: str = "rpc-cancel") -> None:
        self.cancel_event.set()
        self.interrupt_signal.abort(reason)


class RunRegistry:
    """Thread-safe session_id -> run handle map."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunHandle] = {}

    def register(self, handle: RunHandle) -> bool:
        with self._lock:
            if handle.session_id in self._runs:
                return False
            self._runs[handle.session_id] = handle
            return True

    def get(self, session_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(session_id)

    def cancel(self, session_id: str, reason: str = "rpc-cancel") -> bool:
        handle = self.get(session_id)
        if handle is None:
            return False
        handle.cancel(reason)
        return True

    def unregister(self, session_id: str, handle: RunHandle) -> None:
        with self._lock:
            if self._runs.get(session_id) is handle:
                self._runs.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


running_runs = RunRegistry()


__all__ = ["RunHandle", "RunRegistry", "running_runs"]
