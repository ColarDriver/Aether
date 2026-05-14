"""Transport abstraction for the Aether gateway.

A :class:`Transport` is anything that can accept a JSON-serialisable
dict and forward it to the peer as one newline-terminated frame.  The
current implementation ships exactly one transport —
:class:`StdioTransport` — which writes to ``sys.stdout`` (resolved
lazily so tests can swap the stream). Additional transports can be
added later without changing handler code.

The "current" transport for a given request is tracked in a
:class:`contextvars.ContextVar`. This lets handlers running on worker
threads write back to the right peer when multiple peers coexist. In
the current stdio-only setup the contextvar transparently falls back to
a module-level ``StdioTransport`` singleton.

Framing rules:

* One frame is one UTF-8 JSON object terminated by ``\\n``.
* JSON is serialised with ``ensure_ascii=False`` and compact separators.
  All in-string control characters (``\\n``, ``\\r``, ``\\t`` …) are
  escaped by :func:`json.dumps` itself, so a frame is always exactly
  one terminal line.
* stdout is the transport sink.  stderr is reserved for crash and
  signal diagnostics and is not routed through this module.
"""

from __future__ import annotations

import contextvars
import errno
import json
import logging
import sys
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


_PEER_GONE_ERRNOS = frozenset(
    {
        errno.EPIPE,
        errno.ECONNRESET,
        errno.EBADF,
        errno.ESHUTDOWN,
        getattr(errno, "WSAECONNRESET", -1),
        getattr(errno, "WSAESHUTDOWN", -1),
    }
    - {-1}
)


@runtime_checkable
class Transport(Protocol):
    """Anything that can write a JSON frame and be closed."""

    def write(self, frame: dict[str, Any]) -> None:
        """Serialise ``frame`` as one JSON line and send to the peer."""

    def close(self) -> None:
        """Mark the transport closed.  Idempotent; subsequent writes are no-ops."""


class StdioTransport:
    """Newline-delimited JSON frames over a text stream (default ``sys.stdout``).

    The stream is resolved lazily via ``stream_getter`` so tests can
    monkey-patch ``sys.stdout`` or inject a ``StringIO``.  Writes are
    serialised under a lock so concurrent ``write`` calls from worker
    threads do not interleave.

    Peer-gone errors (``EPIPE``, ``ECONNRESET``, ``EBADF``, ``ESHUTDOWN``)
    are silently swallowed and flip the transport to closed.  Any other
    ``OSError`` re-raises so a real I/O fault still surfaces.
    """

    __slots__ = ("_stream_getter", "_lock", "_closed")

    def __init__(self, stream_getter: Optional[Callable[[], Any]] = None) -> None:
        self._stream_getter: Callable[[], Any] = stream_getter or (lambda: sys.stdout)
        self._lock = threading.Lock()
        self._closed = False

    def write(self, frame: dict[str, Any]) -> None:
        if self._closed:
            return

        line = json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n"

        try:
            with self._lock:
                stream = self._stream_getter()
                stream.write(line)
                stream.flush()
        except (BrokenPipeError, ConnectionResetError):
            self._mark_closed(OSError("peer disconnected"))
        except OSError as exc:
            if exc.errno in _PEER_GONE_ERRNOS:
                self._mark_closed(exc)
                return
            raise

    def close(self) -> None:
        self._mark_closed(OSError("peer disconnected"))

    @property
    def closed(self) -> bool:
        return self._closed

    def _mark_closed(self, reason: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        _notify_peer_gone(reason or OSError("peer disconnected"))


_current: contextvars.ContextVar[Optional[Transport]] = contextvars.ContextVar(
    "aether_gateway_transport",
    default=None,
)

_fallback_lock = threading.Lock()
_fallback: Optional[Transport] = None
_peer_gone_lock = threading.Lock()
_peer_gone_callbacks: list[Callable[[BaseException], None]] = []


def _get_fallback() -> Transport:
    """Lazily build the module-level stdio fallback transport."""
    global _fallback
    with _fallback_lock:
        if _fallback is None:
            _fallback = StdioTransport()
        return _fallback


def bind_transport(t: Transport) -> contextvars.Token:
    """Bind ``t`` as the current transport for this context.

    Returns a token that must be passed to :func:`reset_transport` to
    restore the prior binding.  The pairing is per-context: copying the
    context (e.g. ``contextvars.copy_context()`` before
    ``ThreadPoolExecutor.submit``) propagates the binding to the worker.
    """
    return _current.set(t)


def reset_transport(token: contextvars.Token) -> None:
    """Undo a prior :func:`bind_transport`.  Idempotent on a stale token is OK."""
    _current.reset(token)


def current_transport() -> Transport:
    """Return the transport bound to this context, or the stdio fallback."""
    t = _current.get()
    return t if t is not None else _get_fallback()


def reset_transport_for_tests() -> None:
    """Drop the cached fallback transport so tests can reseed it.

    Test-only; production code never calls this.  Public (no leading
    underscore) so tests don't have to alias-rename it on import.
    """
    global _fallback
    with _fallback_lock:
        _fallback = None


def add_peer_gone_callback(callback: Callable[[BaseException], None]) -> None:
    """Register a process-local callback for transport disconnects."""
    with _peer_gone_lock:
        if callback not in _peer_gone_callbacks:
            _peer_gone_callbacks.append(callback)


def remove_peer_gone_callback(callback: Callable[[BaseException], None]) -> None:
    """Remove a callback previously registered by ``add_peer_gone_callback``."""
    with _peer_gone_lock:
        _peer_gone_callbacks[:] = [
            existing for existing in _peer_gone_callbacks if existing is not callback
        ]


def _notify_peer_gone(reason: BaseException) -> None:
    with _peer_gone_lock:
        callbacks = list(_peer_gone_callbacks)
    for callback in callbacks:
        try:
            callback(reason)
        except Exception:
            logger.exception("transport peer-gone callback failed")
