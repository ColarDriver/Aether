"""Transport abstraction for the Aether gateway.

A :class:`Transport` is anything that can accept a JSON-serialisable
dict and forward it to the peer as one newline-terminated frame.  The
PR 1 implementation ships exactly one transport — :class:`StdioTransport`
— which writes to ``sys.stdout`` (resolved lazily so tests can swap the
stream).  Later PRs will add ``WebSocketTransport`` and ``TeeTransport``
for the web UI path without changing handler code.

The "current" transport for a given request is tracked in a
:class:`contextvars.ContextVar`.  This lets handlers running on worker
threads (PR 2's long-handler pool) write back to the right peer when
multiple peers (future web sessions) coexist.  In the PR 1 stdio-only
world the contextvar transparently falls back to a module-level
``StdioTransport`` singleton.

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
            self._closed = True
        except OSError as exc:
            if exc.errno in _PEER_GONE_ERRNOS:
                self._closed = True
                return
            raise

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


_current: contextvars.ContextVar[Optional[Transport]] = contextvars.ContextVar(
    "aether_gateway_transport",
    default=None,
)

_fallback_lock = threading.Lock()
_fallback: Optional[Transport] = None


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
