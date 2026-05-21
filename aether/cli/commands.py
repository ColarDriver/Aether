"""Slash-command metadata shared with the TypeScript TUI.

The interactive command implementations live in ``tui/src/slash``.  Python
keeps only the catalog used by the gateway ``commands.catalog`` RPC and the
path-vs-command classifier pinned by tests.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class SlashCommandCatalogEntry:
    name: str
    description: str
    category: str


REGISTRY: dict[str, SlashCommand] = {
    "/help": SlashCommand("/help", "Show this help table"),
    "/exit": SlashCommand("/exit", "Exit the TUI"),
    "/new": SlashCommand("/new", "Start a new session"),
    "/clear": SlashCommand("/clear", "Clear conversation history"),
    "/refresh": SlashCommand("/refresh", "Refresh the visible terminal area"),
    "/session": SlashCommand("/session", "Show current session info"),
    "/sessions": SlashCommand("/sessions", "List previously saved sessions"),
    "/resume": SlashCommand("/resume", "Resume a previous session: /resume [<id|prefix>]"),
    "/system": SlashCommand("/system", "Show or set the system prompt: /system <text>"),
    "/tools": SlashCommand("/tools", "List registered tools"),
    "/model": SlashCommand("/model", "Show, switch, or pick the active model"),
    "/verbose": SlashCommand("/verbose", "Toggle per-turn verbose output"),
    "/stats": SlashCommand("/stats", "Show stats from the last turn"),
    "/interrupt": SlashCommand("/interrupt", "Interrupt the active turn"),
    "/plan": SlashCommand("/plan", "Enable plan mode or view the current session plan"),
}


_SESSION_COMMANDS = frozenset(
    {"/new", "/session", "/sessions", "/resume", "/system", "/model", "/plan"}
)
_CONTROL_COMMANDS = frozenset({"/interrupt"})
_REMOTE_COMMANDS = frozenset({"/tools"})

_COMMAND_NAME_RE = re.compile(r"^[a-zA-Z0-9:_-]+$")


def command_category(name: str) -> str:
    if name in _SESSION_COMMANDS:
        return "session"
    if name in _CONTROL_COMMANDS:
        return "control"
    if name in _REMOTE_COMMANDS:
        return "remote"
    return "local"


def catalog_entries() -> list[SlashCommandCatalogEntry]:
    return [
        SlashCommandCatalogEntry(
            name=command.name,
            description=command.description,
            category=command_category(command.name),
        )
        for command in sorted(REGISTRY.values(), key=lambda item: item.name)
    ]


def _looks_like_command_name(head: str) -> bool:
    return bool(_COMMAND_NAME_RE.fullmatch(head))


def is_slash(line: str) -> bool:
    """Return True when *line* should route to slash-command handling."""
    stripped = line.strip()
    if not stripped.startswith("/") or len(stripped) < 2:
        return False
    head = stripped[1:].split(maxsplit=1)[0]
    if not head or not _looks_like_command_name(head):
        return False
    try:
        if os.path.exists("/" + head):
            return False
    except OSError:
        pass
    return True


__all__ = [
    "REGISTRY",
    "SlashCommand",
    "SlashCommandCatalogEntry",
    "catalog_entries",
    "command_category",
    "is_slash",
]
