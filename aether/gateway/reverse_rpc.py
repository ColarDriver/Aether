"""Server-initiated JSON-RPC request bridge.

Used by gateway prompters to ask the TS client for approval or tool
permission decisions while the synchronous engine thread blocks for the
answer.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from typing import Any

from aether.gateway.transport import (
    add_peer_gone_callback,
    current_transport,
)


class ReverseRpcError(RuntimeError):
    """Base class for reverse-RPC bridge failures."""


class UnknownPendingResponse(ReverseRpcError):
    """Raised when the client answers an id that is not pending."""


_counter = itertools.count(1)
_pending_lock = threading.Lock()
_pending: dict[str, Future[dict[str, Any]]] = {}


def call(
    method: str,
    params: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Send a server-initiated request and block for its response."""

    request_id = _next_request_id(method)
    future: Future[dict[str, Any]] = Future()
    with _pending_lock:
        _pending[request_id] = future

    frame = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    transport = current_transport()
    try:
        transport.write(frame)
        if bool(getattr(transport, "closed", False)):
            raise OSError("peer disconnected")
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        _drop_pending(request_id)
        raise TimeoutError(f"reverse RPC timed out: {method}") from exc
    except Exception as exc:
        _drop_pending(request_id)
        if not future.done():
            future.set_exception(exc)
        raise
    finally:
        _drop_pending(request_id)


def complete(request_id: str, result: dict[str, Any]) -> None:
    """Resolve one pending reverse-RPC request."""

    with _pending_lock:
        future = _pending.pop(request_id, None)
    if future is None:
        raise UnknownPendingResponse(f"unknown pending reverse RPC id: {request_id}")
    future.set_result(result)


def fail(request_id: str, error: BaseException) -> None:
    """Reject one pending reverse-RPC request."""

    with _pending_lock:
        future = _pending.pop(request_id, None)
    if future is None:
        raise UnknownPendingResponse(f"unknown pending reverse RPC id: {request_id}")
    future.set_exception(error)


def reject_all(error: BaseException | None = None) -> None:
    """Reject every pending request, typically after peer disconnect."""

    reason = error or OSError("peer disconnected")
    with _pending_lock:
        pending = list(_pending.values())
        _pending.clear()
    for future in pending:
        if not future.done():
            future.set_exception(reason)


def pending_ids_for_tests() -> list[str]:
    with _pending_lock:
        return sorted(_pending)


def reset_for_tests() -> None:
    reject_all(RuntimeError("reverse rpc reset"))


def _next_request_id(method: str) -> str:
    if method.startswith("approval."):
        prefix = "srv_app"
    elif method.startswith("permission."):
        prefix = "srv_perm"
    else:
        prefix = "srv"
    return f"{prefix}_{next(_counter)}"


def _drop_pending(request_id: str) -> None:
    with _pending_lock:
        _pending.pop(request_id, None)


def _reject_on_peer_gone(reason: BaseException) -> None:
    reject_all(reason if str(reason) else OSError("peer disconnected"))


add_peer_gone_callback(_reject_on_peer_gone)


__all__ = [
    "ReverseRpcError",
    "UnknownPendingResponse",
    "call",
    "complete",
    "fail",
    "pending_ids_for_tests",
    "reject_all",
    "reset_for_tests",
]
