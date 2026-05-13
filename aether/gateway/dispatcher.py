"""Method registry and dispatch loop for the Aether gateway.

The gateway exposes RPC methods via a small decorator-based registry:

    @method("session.list", long=True)
    def _list_sessions(params: dict | None) -> dict:
        ...

* Short handlers run on the dispatcher thread.  They MUST return
  quickly (a few ms) because they block the loop from reading the
  next frame.
* Long handlers run on a shared :class:`~concurrent.futures.ThreadPoolExecutor`.
  The dispatcher submits the call and returns ``None`` so the loop can
  keep accepting frames; the worker writes the response itself when
  the handler returns.

Handler signature is ``(params: dict | None) -> dict | None``:

* Return a dict → wrapped in :class:`RpcResponse.result`.
* Return ``None`` → handler will respond later via ``respond(id, ...)``
  (used by server-initiated request flows in PR 5; the dispatcher
  itself never emits a response on its own when the handler returned
  None).
* Raise :class:`GatewayError` → mapped to ``-32000`` application error.
* Raise any other ``Exception`` → mapped to ``-32603`` internal error
  plus a crash log entry via the panic hook chain.

Per-request context binding: every long-handler submission first calls
``contextvars.copy_context()`` so the worker thread inherits the
current transport binding.  In the PR 1+2 stdio-only world this is
redundant (one global transport), but it costs nothing and unlocks
the per-peer transport routing PR 5 + the eventual web UI will need.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextvars
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel

from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INTERNAL,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_PARSE,
    GatewayError,
    JSONRPC_VERSION,
    RpcNotification,
    RpcRequest,
    RpcResponse,
    make_error_response,
    make_success_response,
)
from aether.gateway.transport import Transport, current_transport

logger = logging.getLogger(__name__)


# ── Registry ─────────────────────────────────────────────────────────


Handler = Callable[[Optional[dict[str, Any]]], Optional[dict[str, Any]]]

_METHODS: dict[str, Handler] = {}
_LONG_METHODS: set[str] = set()
_current_request_id: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "aether_gateway_request_id",
    default=None,
)


def method(name: str, *, long: bool = False) -> Callable[[Handler], Handler]:
    """Register ``fn`` as an RPC handler under ``name``.

    ``long=True`` routes the handler to the worker pool so the
    dispatcher loop stays responsive.  Re-registering an existing
    name overwrites — this is intentional so tests can monkey-patch
    handlers freely; production paths should use unique names.
    """

    def _wrap(fn: Handler) -> Handler:
        _METHODS[name] = fn
        if long:
            _LONG_METHODS.add(name)
        else:
            _LONG_METHODS.discard(name)
        return fn

    return _wrap


def unregister_method(name: str) -> None:
    """Remove a handler.  Used by tests; production has no remove flow yet."""
    _METHODS.pop(name, None)
    _LONG_METHODS.discard(name)


def list_registered_methods() -> list[str]:
    """Return the sorted list of currently-registered method names."""
    return sorted(_METHODS.keys())


def current_request_id() -> Any | None:
    """Return the JSON-RPC id for the handler currently executing."""
    return _current_request_id.get()


# ── Worker pool ──────────────────────────────────────────────────────


_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _resolve_max_workers() -> int:
    raw = os.environ.get("AETHER_GATEWAY_MAX_WORKERS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return min(8, (os.cpu_count() or 4) + 2)


def _get_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_resolve_max_workers(),
                thread_name_prefix="aether-gw-worker",
            )
            atexit.register(_shutdown_pool)
        return _pool


def _shutdown_pool() -> None:
    """Stop the pool without waiting for in-flight tasks.

    Aligned with Hermes' ``tui_gateway`` lifecycle: at process exit we
    cannot block on a wedged handler, so we cancel pending work and
    let the OS reap workers.  The shutdown grace in ``entry.py``
    gives the pool a brief window to drain naturally.
    """
    global _pool
    with _pool_lock:
        pool = _pool
        _pool = None
    if pool is None:
        return
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # pragma: no cover - shutdown is best-effort
        pass


def reset_dispatcher_for_tests() -> None:
    """Drop the registry + pool so each test starts clean.  Test-only.

    Public so tests can call it without reaching into private symbols;
    production code never calls this.
    """
    _METHODS.clear()
    _LONG_METHODS.clear()
    _shutdown_pool()


# ── Parsing ──────────────────────────────────────────────────────────


def parse_frame(line: str) -> tuple[RpcRequest | RpcNotification | None, RpcResponse | None]:
    """Parse one inbound JSON line.

    Returns ``(envelope, error_response)``.  Exactly one is non-None:

    * Successful parse → ``(envelope, None)``.
    * Parse / validation failure → ``(None, error_response)``.

    Batch frames (top-level JSON arrays) are not supported in this
    protocol and are rejected as -32600 invalid request.  The id on
    the error frame is ``None`` because we cannot read it from
    something that did not validate.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, make_error_response(
            None,
            ERROR_PARSE,
            "frame is not valid JSON",
            data={"reason": str(exc)},
        )

    if isinstance(obj, list):
        return None, make_error_response(
            None,
            ERROR_INVALID_REQUEST,
            "batch requests are not supported",
        )
    if not isinstance(obj, dict):
        return None, make_error_response(
            None,
            ERROR_INVALID_REQUEST,
            "envelope must be a JSON object",
        )

    if ("result" in obj or "error" in obj) and "method" not in obj:
        response_request, response_error = _parse_reverse_response(obj)
        if response_request is not None or response_error is not None:
            return response_request, response_error

    has_id = "id" in obj

    # Pydantic validates jsonrpc/method/params + extra=forbid.
    try:
        if has_id:
            return RpcRequest.model_validate(obj), None
        return RpcNotification.model_validate(obj), None
    except Exception as exc:
        # ``id`` may or may not be present; surface it on the error
        # envelope when we can read it so the peer can correlate.
        request_id = obj.get("id") if isinstance(obj.get("id"), (str, int)) else None
        return None, make_error_response(
            request_id,
            ERROR_INVALID_REQUEST,
            "envelope validation failed",
            data={"reason": _short_validation_message(exc)},
        )


def _parse_reverse_response(obj: dict[str, Any]) -> tuple[RpcRequest | None, RpcResponse | None]:
    """Map client responses to server-initiated requests back into handlers."""

    request_id = obj.get("id")
    if not isinstance(request_id, str):
        return None, make_error_response(
            request_id if isinstance(request_id, int) else None,
            ERROR_INVALID_REQUEST,
            "reverse response id must be a string",
        )
    if request_id.startswith("srv_app_"):
        method_name = "approval.response"
    elif request_id.startswith("srv_perm_"):
        method_name = "permission.response"
    else:
        return None, make_error_response(
            request_id,
            ERROR_INVALID_REQUEST,
            "unknown server-initiated response id prefix",
        )

    if obj.get("jsonrpc") != JSONRPC_VERSION:
        return None, make_error_response(
            request_id,
            ERROR_INVALID_REQUEST,
            "jsonrpc version must be 2.0",
        )

    has_result = "result" in obj
    has_error = "error" in obj
    if has_result == has_error:
        return None, make_error_response(
            request_id,
            ERROR_INVALID_REQUEST,
            "reverse response must have exactly one of result/error",
        )

    payload = obj.get("result") if has_result else {"error": obj.get("error")}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    try:
        return RpcRequest(id=request_id, method=method_name, params=payload), None
    except Exception as exc:
        return None, make_error_response(
            request_id,
            ERROR_INVALID_REQUEST,
            "reverse response validation failed",
            data={"reason": _short_validation_message(exc)},
        )


def _short_validation_message(exc: Exception) -> str:
    """Pydantic ValidationError messages can be huge; keep just the first line."""
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return msg[:200]


# ── Dispatch ─────────────────────────────────────────────────────────


def dispatch_request(
    envelope: RpcRequest | RpcNotification,
    transport: Transport | None = None,
) -> RpcResponse | None:
    """Route ``envelope`` to its handler and return the response (if any).

    Returns ``None`` for notifications, for handlers tagged as
    ``long=True`` (their response is written asynchronously by the
    worker), and for handlers that explicitly returned ``None`` to
    signal "I will respond later".
    """
    method_name = envelope.method
    handler = _METHODS.get(method_name)

    is_request = isinstance(envelope, RpcRequest)

    if handler is None:
        if not is_request:
            # Notifications addressed to unknown methods are dropped silently.
            return None
        return make_error_response(
            envelope.id,
            ERROR_METHOD_NOT_FOUND,
            f"unknown method: {method_name}",
        )

    if method_name in _LONG_METHODS:
        ctx = contextvars.copy_context()
        _get_pool().submit(_run_long, ctx, handler, envelope, transport)
        return None

    return _run_short(handler, envelope)


def _run_short(handler: Handler, envelope: RpcRequest | RpcNotification) -> RpcResponse | None:
    """Execute a short handler synchronously on the dispatcher thread."""
    is_request = isinstance(envelope, RpcRequest)
    token = _current_request_id.set(envelope.id if is_request else None)
    try:
        result = handler(envelope.params)
    except GatewayError as exc:
        return make_error_response(
            envelope.id if is_request else None,
            exc.code,
            exc.message,
            data=exc.data,
        )
    except Exception as exc:  # noqa: BLE001 - any escaped exception is internal
        _log_handler_exception(envelope.method, exc)
        if not is_request:
            return None
        return make_error_response(
            envelope.id,
            ERROR_INTERNAL,
            "handler raised an unhandled exception",
            data={"exception": type(exc).__name__},
        )
    finally:
        _current_request_id.reset(token)

    if not is_request:
        return None
    if result is None:
        # Handler will respond later via respond() / notify().
        return None
    return make_success_response(envelope.id, result)


def _run_long(
    ctx: contextvars.Context,
    handler: Handler,
    envelope: RpcRequest | RpcNotification,
    transport: Transport | None,
) -> None:
    """Worker entrypoint for long handlers.

    Runs the handler under the captured request context so the
    transport binding propagates from dispatcher → worker; writes the
    response (success or error) on the supplied transport if any, or
    the current transport otherwise.
    """
    response = ctx.run(_run_short, handler, envelope)
    if response is None:
        return
    sink = transport if transport is not None else current_transport()
    write_envelope(sink, response)


# ── Outbound writes ──────────────────────────────────────────────────


def notify(method: str, params: dict[str, Any] | None = None) -> None:
    """Emit a JSON-RPC notification on the current-context transport.

    Used by handlers (e.g. for streaming events) and by the gateway
    itself for lifecycle frames like ``gateway.ready``.  The current
    transport is resolved via :func:`aether.gateway.transport.current_transport`,
    which falls back to the module-level stdio sink when nothing is
    explicitly bound.
    """
    frame = RpcNotification(method=method, params=params)
    write_envelope(current_transport(), frame)


def respond(request_id: Any, result: Any) -> None:
    """Emit a deferred success response for a request whose handler returned None.

    Reserved for the PR 5 approval / permission bridge — the dispatcher
    itself never calls this.  Documented here so handlers know the
    contract: returning ``None`` from a short handler means "I will
    call ``respond()`` later".
    """
    response = make_success_response(request_id, result)
    write_envelope(current_transport(), response)


def respond_error(
    request_id: Any,
    code: int,
    message: str,
    *,
    data: dict[str, Any] | None = None,
) -> None:
    """Emit a deferred error response — same contract as :func:`respond`."""
    response = make_error_response(request_id, code, message, data=data)
    write_envelope(current_transport(), response)


def write_envelope(transport: Transport, envelope: BaseModel) -> None:
    """Serialise an envelope and write it as one frame.

    JSON-RPC 2.0 requires the ``id`` member to be present on every
    response — explicitly ``null`` when the server could not read it
    from a malformed request (parse / invalid-request errors).  We
    serialise with ``exclude_none=True`` for compactness then
    reintroduce a null ``id`` for responses where it was elided.

    Public so the entry-point loop can reuse the same serialisation
    contract without reaching into private helpers.
    """
    payload = envelope.model_dump(mode="json", exclude_none=True)
    if isinstance(envelope, RpcResponse) and "id" not in payload:
        payload["id"] = None
    transport.write(payload)


# ── Logging / observability ─────────────────────────────────────────


def _log_handler_exception(method_name: str, exc: BaseException) -> None:
    """Surface unexpected handler failures through the existing crash plumbing.

    The full traceback goes to the panic / thread hooks installed by
    ``entry.py``.  Here we just emit a structured log line so anyone
    grepping the gateway log can see *which* method failed without
    having to open the crash file.
    """
    logger.exception(
        "gateway handler raised: method=%s exception=%s",
        method_name,
        type(exc).__name__,
    )


# ── Built-in methods ─────────────────────────────────────────────────


def _ping(params: dict[str, Any] | None) -> dict[str, Any]:
    """Liveness check.  Echoes back ``params['echo']`` if provided."""
    response: dict[str, Any] = {"pong": True, "timestamp": time.time()}
    if params and "echo" in params:
        response["echo"] = params["echo"]
    return response


def register_builtins() -> None:
    """Register methods that ship in the gateway itself.

    Called explicitly by :func:`aether.gateway.entry.main` at boot
    and from test ``setUp`` after :func:`reset_dispatcher_for_tests`
    clears the registry.  Idempotent.
    """
    method("gateway.ping", long=False)(_ping)


__all__ = [
    "Handler",
    "current_request_id",
    "dispatch_request",
    "list_registered_methods",
    "method",
    "notify",
    "parse_frame",
    "register_builtins",
    "reset_dispatcher_for_tests",
    "respond",
    "respond_error",
    "unregister_method",
    "write_envelope",
]
