"""Interrupt marker helpers."""

from __future__ import annotations

INTERRUPT_MESSAGE = "[Request interrupted by user]"
INTERRUPT_MESSAGE_FOR_TOOL_USE = "[Request interrupted by user for tool use]"
FETCH_INTERRUPTED_MESSAGE = "[fetch interrupted by user before response]"


def select_interrupt_marker(*, was_in_tool_call: bool) -> str:
    return INTERRUPT_MESSAGE_FOR_TOOL_USE if was_in_tool_call else INTERRUPT_MESSAGE


__all__ = [
    "INTERRUPT_MESSAGE",
    "INTERRUPT_MESSAGE_FOR_TOOL_USE",
    "FETCH_INTERRUPTED_MESSAGE",
    "select_interrupt_marker",
]
