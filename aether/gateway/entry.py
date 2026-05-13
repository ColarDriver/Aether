"""Aether gateway process entrypoint.

Boots the gateway: installs crash hooks, signal handlers, transport,
and runs the dispatcher's request loop.

Frame-level lifecycle:

1. ``main()`` installs ``sys.excepthook`` / ``threading.excepthook`` /
   signal handlers, registers the dispatcher's builtin and handler
   methods, and binds a :class:`StdioTransport` to the current
   context.
2. The first outbound frame is a ``gateway.ready`` notification
   announcing the gateway's version, capability tags, and the full
   list of registered methods so a TS client can feature-detect
   without sending probe requests.
3. ``_request_loop`` reads stdin line by line, parses each line into
   a request / notification envelope via
   :func:`~aether.gateway.dispatcher.parse_frame`, and dispatches to
   the registered handler.  Short handlers respond inline; long
   handlers respond from the worker pool asynchronously.
4. Stdin EOF or SIGTERM / SIGINT trigger an orderly shutdown.  After
   the loop exits the gateway waits up to the configured grace
   (default 1s, overridable via ``AETHER_GATEWAY_SHUTDOWN_GRACE_S``)
   for background threads to drain before returning.

Crash diagnostics: any unhandled exception in the main or worker
threads is appended to ``$AETHER_HOME/logs/gateway_crash.log`` AND
emits a single ``[gateway-crash] ...`` line to stderr that the parent
TUI can surface in its activity bar without opening the log file.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from types import FrameType
from typing import Optional

from aether.gateway.dispatcher import (
    dispatch_request,
    list_registered_methods,
    notify,
    parse_frame,
    register_builtins,
    write_envelope,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.transport import (
    StdioTransport,
    Transport,
    bind_transport,
    reset_transport,
)

logger = logging.getLogger(__name__)


# Capability tag list emitted in the ``gateway.ready`` event.  Stays
# small on purpose — TS clients use it for feature detection so a
# meaningful name beats an exhaustive method dump.  Annotated as
# ``tuple[str, ...]`` (rather than the inferred literal tuple type)
# so list concatenation in ``_capabilities`` returns a plain
# ``list[str]`` without invariance complaints.
_BUILTIN_CAPABILITIES: tuple[str, ...] = ("ping",)
_HANDLER_CAPABILITIES: tuple[str, ...] = (
    "sessions",
    "prefs",
    "providers",
    "commands",
    "agent",
    "approvals",
    "permissions",
)

GATEWAY_VERSION = "0.5.0"  # bump per PR; PR 5 adds approval/permission bridge.


_DEFAULT_SHUTDOWN_GRACE_S = 1.0

_shutdown_event = threading.Event()


def _aether_home() -> Path:
    return Path(os.environ.get("AETHER_HOME") or (Path.home() / ".aether"))


def _crash_log_path() -> Path:
    return _aether_home() / "logs" / "gateway_crash.log"


def _append_crash_log(trace: str) -> None:
    """Append a timestamped traceback block to the crash log.

    Best-effort: if the disk is full / read-only / permissions are
    wrong, we swallow the secondary failure rather than mask the
    original exception.
    """
    path = _crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n=== unhandled exception · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            f.write(trace)
            if not trace.endswith("\n"):
                f.write("\n")
    except Exception:  # pragma: no cover - secondary failure path
        pass


def _stderr_panic_line(exc_type: type[BaseException], exc_value: BaseException) -> None:
    msg = str(exc_value).strip()
    first = msg.splitlines()[0] if msg else exc_type.__name__
    try:
        print(
            f"[gateway-crash] {exc_type.__name__}: {first}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # pragma: no cover - stderr also gone
        pass


def _panic_hook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb,
) -> None:
    """Replacement for ``sys.excepthook``.  Logs and chains to default."""
    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _append_crash_log(trace)
    _stderr_panic_line(exc_type, exc_value)
    # Chain so the interpreter still terminates as it would have.
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _thread_panic_hook(args) -> None:
    """Replacement for ``threading.excepthook`` — same diagnostics, thread tag."""
    trace = "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    name = getattr(args.thread, "name", "<unknown>")
    _append_crash_log(f"[thread={name}]\n{trace}")
    try:
        print(
            f"[gateway-crash] thread={name} "
            f"{args.exc_type.__name__}: {args.exc_value}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # pragma: no cover
        pass


_SIGNAL_NAMES: dict[int, str] = {
    signal.SIGTERM: "SIGTERM",
    signal.SIGINT: "SIGINT",
}
if hasattr(signal, "SIGPIPE"):
    _SIGNAL_NAMES[signal.SIGPIPE] = "SIGPIPE"


def _shutdown_grace_seconds() -> float:
    raw = (os.environ.get("AETHER_GATEWAY_SHUTDOWN_GRACE_S") or "").strip()
    if not raw:
        return _DEFAULT_SHUTDOWN_GRACE_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SHUTDOWN_GRACE_S
    return value if value > 0 else _DEFAULT_SHUTDOWN_GRACE_S


def _log_signal(signum: int, frame: Optional[FrameType]) -> None:
    """Log signal arrival and request shutdown for terminating signals.

    SIGPIPE is logged but does NOT request shutdown — the kernel has
    already delivered ``EPIPE`` to whichever ``write()`` triggered it,
    and :class:`~aether.gateway.transport.StdioTransport` handles that
    explicitly.  Without this handler the default SIGPIPE action would
    kill the process the instant a background thread tries to write to
    a TUI that has already gone away.
    """
    name = _SIGNAL_NAMES.get(signum, f"signal-{signum}")
    thread = threading.current_thread()
    logger.info("gateway received %s on thread %s", name, thread.name)

    if hasattr(signal, "SIGPIPE") and signum == signal.SIGPIPE:
        return

    _shutdown_event.set()
    # Raise SystemExit from the signal handler so a blocking stdin read
    # in the main thread unwinds promptly.  Python delivers the signal
    # between bytecodes; SystemExit propagates out of readline() and
    # through main() into __main__.
    raise SystemExit(0)


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _log_signal)
    signal.signal(signal.SIGINT, _log_signal)
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, _log_signal)


def _drain_background_threads(grace_seconds: float) -> None:
    """Wait up to ``grace_seconds`` for non-daemon threads to finish."""
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        # active_count includes the main thread itself.
        if threading.active_count() <= 1:
            return
        time.sleep(0.05)


def _capabilities() -> list[str]:
    """Capability tags advertised in the ``gateway.ready`` event.

    PR 2 advertised only ``ping``; PR 3 adds the four handler groups
    that landed in this PR.  PR 4+ will extend this list as more
    feature areas come online.
    """
    return list(_BUILTIN_CAPABILITIES) + list(_HANDLER_CAPABILITIES)


def _emit_ready() -> None:
    """Announce that the gateway has finished booting and is accepting frames."""
    notify(
        "gateway.ready",
        {
            "version": GATEWAY_VERSION,
            "capabilities": _capabilities(),
            "methods": list_registered_methods(),
        },
    )


def _request_loop(transport: Transport) -> None:
    """Read JSON frames from stdin, parse, dispatch, write responses.

    Each iteration handles one inbound line.  Parse errors and
    envelope-validation failures produce an error response with a
    null id (the peer cannot correlate it without one, but that is
    the cost of sending malformed JSON).  Successful parses are
    routed through :func:`dispatch_request`; the dispatcher returns
    ``None`` when the response will be written asynchronously (long
    handlers, deferred responders).
    """
    while not _shutdown_event.is_set():
        line = sys.stdin.readline()
        if line == "":  # EOF
            return
        stripped = line.strip()
        if not stripped:
            continue

        envelope, error_response = parse_frame(stripped)
        if error_response is not None:
            write_envelope(transport, error_response)
            continue
        assert envelope is not None  # parse_frame: exactly one is non-None

        response = dispatch_request(envelope, transport=transport)
        if response is not None:
            write_envelope(transport, response)


def main() -> int:
    """Boot the gateway, announce readiness, run the request loop."""
    sys.excepthook = _panic_hook
    threading.excepthook = _thread_panic_hook
    _install_signal_handlers()

    # Explicit registration: the dispatcher and handler modules used
    # to register themselves at import time.  Doing it here makes the
    # boot order and the set of advertised capabilities obvious from
    # ``main`` alone, and lets tests exercise the registry without
    # relying on import side effects.
    register_builtins()
    register_handler_methods()

    transport = StdioTransport()
    token = bind_transport(transport)
    try:
        _emit_ready()
        _request_loop(transport)
    except (KeyboardInterrupt, SystemExit):
        _shutdown_event.set()
    finally:
        reset_transport(token)

    _drain_background_threads(_shutdown_grace_seconds())
    return 0
