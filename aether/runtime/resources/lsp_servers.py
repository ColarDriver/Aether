"""LSP server resolution table.

Maps file extensions to languages and languages to launch commands.
Keeping the data in one place makes it easy to extend support without
touching the manager, client, or tool layers.

Operators that want to ship a custom server can override
:func:`resolve_server_for` indirectly through
``EngineConfig.lsp_server_overrides`` (the manager merges the override
table on top of these defaults).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional


__all__ = [
    "EXT_TO_LANG",
    "LANGUAGE_SERVERS",
    "language_for",
    "resolve_server_for",
    "supported_languages",
    "supported_extensions",
]


# A conservative seed list.  Add more as we ship more bundled support.
# Order *within* each value list is "preferred first" — the manager
# probes them in order and uses the first one that resolves on PATH.
LANGUAGE_SERVERS: Mapping[str, list[list[str]]] = {
    "python": [
        ["pylsp"],
        ["pyright-langserver", "--stdio"],
    ],
    "typescript": [["typescript-language-server", "--stdio"]],
    "javascript": [["typescript-language-server", "--stdio"]],
    "rust": [["rust-analyzer"]],
    "go": [["gopls"]],
    "ruby": [["solargraph", "stdio"]],
    "json": [["vscode-json-language-server", "--stdio"]],
    "html": [["vscode-html-language-server", "--stdio"]],
    "css": [["vscode-css-language-server", "--stdio"]],
}


EXT_TO_LANG: Mapping[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
}


def language_for(path: Path) -> Optional[str]:
    """Return the language id for ``path`` or ``None`` if unknown."""
    return EXT_TO_LANG.get(path.suffix.lower())


def resolve_server_for(
    path: Path,
    *,
    overrides: Optional[Mapping[str, list[list[str]]]] = None,
) -> Optional[list[list[str]]]:
    """Return the candidate launch commands for ``path``.

    Each candidate is a list of argv tokens.  The manager picks the
    first one whose binary resolves via :func:`shutil.which`; we don't
    do that resolution here so this module stays a pure data lookup
    and is trivially testable.
    """
    lang = language_for(path)
    if lang is None:
        return None
    if overrides and lang in overrides:
        return list(overrides[lang])
    return list(LANGUAGE_SERVERS.get(lang, []))


def supported_languages() -> Iterable[str]:
    return tuple(LANGUAGE_SERVERS.keys())


def supported_extensions() -> Iterable[str]:
    return tuple(EXT_TO_LANG.keys())
