"""In-process run handle registry."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

from aether.runtime.control.interrupt_signal import InterruptSignal


@dataclass(slots=True)
class RunHandle:
    session_id: str
    run_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    interrupt_signal: InterruptSignal = field(default_factory=InterruptSignal)

    def cancel(self, reason: str = "rpc-cancel") -> None:
        self.cancel_event.set()
        self.interrupt_signal.abort(reason)


class RunRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_session: dict[str, RunHandle] = {}
        self._by_run: dict[str, RunHandle] = {}

    def register(self, handle: RunHandle) -> bool:
        with self._lock:
            if handle.session_id in self._by_session or handle.run_id in self._by_run:
                return False
            self._by_session[handle.session_id] = handle
            self._by_run[handle.run_id] = handle
            return True

    def get(self, session_id: str) -> RunHandle | None:
        return self.get_by_session(session_id)

    def get_by_session(self, session_id: str) -> RunHandle | None:
        with self._lock:
            return self._by_session.get(session_id)

    def get_by_run(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._by_run.get(run_id)

    def cancel(
        self,
        session_id: str | None = None,
        *,
        run_id: str | None = None,
        reason: str = "rpc-cancel",
    ) -> bool:
        handle = self.get_by_run(run_id) if run_id else None
        if handle is None and session_id:
            handle = self.get_by_session(session_id)
        if handle is None:
            return False
        handle.cancel(reason)
        return True

    def unregister(self, session_id: str, handle: RunHandle) -> None:
        with self._lock:
            if self._by_session.get(session_id) is handle:
                self._by_session.pop(session_id, None)
                self._by_run.pop(handle.run_id, None)

    def clear(self) -> None:
        with self._lock:
            self._by_session.clear()
            self._by_run.clear()


__all__ = ["RunHandle", "RunRegistry"]
