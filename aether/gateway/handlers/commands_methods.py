"""``commands.catalog`` RPC method.

Exposes the slash-command registry from :mod:`aether.cli.commands` as
a flat list of ``{name, description, category}`` entries.  The
handler does NOT execute the registry's ``_cmd_*`` handlers — those
operate on ``ReplState`` (prompt_toolkit-bound state) which has no
meaning across a process boundary. The TS slash dispatcher pulls
this catalog and decides per-command whether to handle locally or
fan out to ``session.*`` / ``prefs.*`` / ``providers.*`` / ``agent.*``
RPC methods.

The import of :mod:`aether.cli.commands` is deferred to the handler
body so importing this module does not transitively pull
``prompt_toolkit`` / ``rich`` (the cli commands module pulls both at
import time).  Lazy import keeps gateway startup fast and isolates
any environment issue to the one method that needs them.
"""

from __future__ import annotations

from typing import Any

from aether.gateway.dispatcher import method
from aether.gateway.handlers.schemas import SlashCommandInfo


# Heuristic category map — used by the TS UI to group commands in
# ``/help``.  Unknown commands default to ``local``; the TS side
# treats unknown categories the same way, so this is purely
# informational.

_SESSION_COMMANDS = frozenset(
    {"/new", "/session", "/sessions", "/resume", "/system", "/model"}
)
_CONTROL_COMMANDS = frozenset({"/interrupt"})
_REMOTE_COMMANDS = frozenset({"/tools"})


def _categorise(name: str) -> str:
    if name in _SESSION_COMMANDS:
        return "session"
    if name in _CONTROL_COMMANDS:
        return "control"
    if name in _REMOTE_COMMANDS:
        return "remote"
    return "local"


def commands_catalog(_params: dict[str, Any] | None) -> dict[str, Any]:
    # Lazy import: aether.cli.commands pulls in rich/prompt_toolkit at
    # module import; we don't want those on every gateway boot.
    from aether.cli.commands import REGISTRY

    entries = sorted(REGISTRY.values(), key=lambda c: c.name)
    catalog = [
        SlashCommandInfo(
            name=cmd.name,
            description=cmd.description,
            category=_categorise(cmd.name),
        ).model_dump(mode="json", exclude_none=True)
        for cmd in entries
    ]
    return {"commands": catalog}


def register() -> None:
    """Register ``commands.catalog`` on the dispatcher.  Idempotent."""
    method("commands.catalog", long=False)(commands_catalog)


__all__ = [
    "commands_catalog",
    "register",
]
