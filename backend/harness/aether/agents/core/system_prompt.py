"""System-prompt augmentation utilities used by the engine.

Currently this module owns one helper:
:func:`augment_system_with_tool_contract`, which prepends a small
``<tool_use_contract>`` block describing the registered tools and
forbidding prose-style tool emission.

The block is the single most effective lever against Kimi-class
models that *want* to invoke tools but write the call in
``\u0060\u0060\u0060bash`` fences, ``<function=NAME>``, ``<functions.shell:N>``
or ``<invoke>`` XML instead of populating the structured
``tool_calls`` field.  Because it is derived from the registry it
stays accurate as builtins are added or disabled, and because it
lives on the engine — not the CLI — every consumer (CLI, SDK,
subagents, tests) gets the same protection without duplicating
strings.
"""

from __future__ import annotations

from typing import Iterable

from aether.tools.base import ToolDescriptor


_CONTRACT_TEMPLATE = (
    "<tool_use_contract>\n"
    "You have these tools available: {names}.\n"
    "You MUST invoke them via the structured ``tool_calls`` field of your "
    "response. Do NOT write tool calls in markdown code blocks "
    "(```bash...```), ``<function=NAME>{{...}}``, ``<functions.shell:N>{{...}}``, "
    "``<invoke name=...>``, ``<tool_call>``, or any other prose form — "
    "such text will be discarded and the run loop will exit without "
    "executing anything.\n"
    "Common mappings: to run a shell command, call the ``shell`` tool; "
    "to read a file, call ``read_file``; to list a directory, call "
    "``list_dir``; to search file contents, call ``grep``; to find files "
    "by name, call ``glob``; to write a file, call ``write_file``.\n"
    "</tool_use_contract>"
)


def augment_system_with_tool_contract(
    system: str | None,
    descriptors: Iterable[ToolDescriptor],
) -> str | None:
    """Return *system* with a ``<tool_use_contract>`` block prepended.

    When *descriptors* is empty (no tools to advertise) the input is
    returned unchanged — the contract would be misleading without any
    tools to back it.
    """
    names = sorted({d.name for d in descriptors if d.name})
    if not names:
        return system

    contract = _CONTRACT_TEMPLATE.format(names=", ".join(f"``{n}``" for n in names))

    if system and system.strip():
        return f"{contract}\n\n{system}"
    return contract


__all__ = ["augment_system_with_tool_contract"]
