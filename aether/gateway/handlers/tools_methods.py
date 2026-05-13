"""``tools.list`` RPC method.

Exposes the built-in tool catalog as ``[{name, description, parameters,
required}, ...]`` so the TS TUI can implement ``/tools`` without bundling
its own copy of the registry. The list is stable per gateway process —
it reflects whatever ``build_default_tool_registry`` returns when the
process boots.

The handler is intentionally lightweight: it does NOT instantiate engines
or sessions, and it does NOT depend on ``aether/cli/`` (which would pull
prompt_toolkit / rich at import time).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from aether.gateway.dispatcher import method


def tools_list(_params: dict[str, Any] | None) -> dict[str, Any]:
    # Lazy import: aether.tools.builtins eagerly instantiates LSP/browser
    # managers and other heavyweight state at import time. Pulling that into
    # the gateway module load path slows boot enough to race SIGINT handler
    # installation in tests; keeping the import inside the handler body
    # restores fast startup.
    from aether.tools.builtins import build_default_tool_registry

    registry = build_default_tool_registry()
    descriptors = registry.list_descriptors()
    catalog = [_descriptor_to_wire(asdict(descriptor)) for descriptor in descriptors]
    catalog.sort(key=lambda entry: entry["name"])
    return {"tools": catalog}


def _descriptor_to_wire(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(payload.get("name") or ""),
        "description": str(payload.get("description") or ""),
        "parameters": payload.get("parameters") or {},
        "required": list(payload.get("required") or []),
    }


def register() -> None:
    """Register ``tools.list`` on the dispatcher.  Idempotent."""
    method("tools.list", long=False)(tools_list)


__all__ = ["register", "tools_list"]
