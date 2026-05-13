"""Aether gateway — Python-side RPC server for the TS Ink TUI.

This package hosts the JSON-RPC dispatcher, transport abstraction,
and method handlers that expose the agent engine, memory, sessions,
providers, and tools to a TypeScript client running as a parent
process.

The package is organised into three layers:

* :mod:`~aether.gateway.transport` — newline-delimited JSON frames
  over stdio (with ``Transport`` Protocol + contextvar-tracked
  current transport) so future :class:`WebSocketTransport` work can
  drop in without changing handler code.
* :mod:`~aether.gateway.protocol` — Pydantic schemas for the
  JSON-RPC envelope, error codes, and gateway lifecycle events.
* :mod:`~aether.gateway.dispatcher` — ``@method`` registry, request
  loop helpers, and the worker pool that runs ``long=True`` handlers.
* :mod:`~aether.gateway.handlers` — concrete RPC method
  implementations (sessions, prefs, providers, slash commands).

The gateway is exposed as the ``aether-gateway`` console script (see
``[project.scripts]`` in ``pyproject.toml``).  Production callers
spawn ``aether-gateway`` directly; tests bootstrap the entry function
via ``python -c``.
"""

from aether.gateway.transport import (
    StdioTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)

__all__ = [
    "StdioTransport",
    "Transport",
    "bind_transport",
    "current_transport",
    "reset_transport",
]
