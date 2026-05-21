"""``commands.catalog`` RPC method.

Exposes the slash-command metadata from :mod:`aether.cli.commands` as a flat
list of ``{name, description, category}`` entries. The TS slash dispatcher
pulls this catalog and decides per-command whether to handle locally or fan
out to ``session.*`` / ``prefs.*`` / ``providers.*`` / ``agent.*`` RPC methods.
"""

from __future__ import annotations

from typing import Any

from aether.cli.commands import catalog_entries
from aether.gateway.dispatcher import method
from aether.gateway.handlers.schemas import SlashCommandInfo


def commands_catalog(_params: dict[str, Any] | None) -> dict[str, Any]:
    catalog = [
        SlashCommandInfo(
            name=entry.name,
            description=entry.description,
            category=entry.category,
        ).model_dump(mode="json", exclude_none=True)
        for entry in catalog_entries()
    ]
    return {"commands": catalog}


def register() -> None:
    """Register ``commands.catalog`` on the dispatcher.  Idempotent."""
    _ = method("commands.catalog", long=False)(commands_catalog)


__all__ = [
    "commands_catalog",
    "register",
]
