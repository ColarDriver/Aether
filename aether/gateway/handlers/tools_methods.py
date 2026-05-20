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

from typing import Any

from aether.gateway.dispatcher import method
from aether.services.tools import ToolService, ToolSummary


def tools_list(_params: dict[str, Any] | None) -> dict[str, Any]:
    catalog = [_tool_to_wire(tool) for tool in ToolService().list_tools().tools]
    return {"tools": catalog}


def _tool_to_wire(tool: ToolSummary) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters or {}),
        "required": list(tool.required or []),
    }


def register() -> None:
    """Register ``tools.list`` on the dispatcher.  Idempotent."""
    method("tools.list", long=False)(tools_list)


__all__ = ["register", "tools_list"]
