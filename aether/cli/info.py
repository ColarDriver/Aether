"""Non-interactive CLI commands."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Final

from aether.cli.providers import PROVIDER_ALIASES, get_provider_defaults, list_providers

FALLBACK_VERSION: Final[str] = "1.0.0"


def _resolve_version() -> str:
    try:
        return package_version("aether-harness")
    except PackageNotFoundError:
        return FALLBACK_VERSION
    except Exception:
        return FALLBACK_VERSION


AETHER_VERSION: Final[str] = _resolve_version()


def cmd_providers(_args: argparse.Namespace) -> None:
    notes: dict[str, str] = {
        "claude": "Anthropic Claude - set ANTHROPIC_API_KEY",
        "openai": "OpenAI-compatible - set OPENAI_API_KEY [+ OPENAI_BASE_URL]",
        "codex": "OpenAI Codex - set CODEX_ACCESS_TOKEN",
    }
    rows: list[tuple[str, str, str]] = []
    for name in list_providers():
        defaults = get_provider_defaults(name)
        model = defaults.get("model")
        rows.append((name, str(model or "?"), notes.get(name, "")))

    name_w = max([len("name"), *(len(row[0]) for row in rows)])
    model_w = max([len("default model"), *(len(row[1]) for row in rows)])
    print("Aether providers")
    print(f"{'name'.ljust(name_w)}  {'default model'.ljust(model_w)}  notes")
    for name, model, note in rows:
        print(f"{name.ljust(name_w)}  {model.ljust(model_w)}  {note}")
    if PROVIDER_ALIASES:
        print()
        aliases = ", ".join(
            f"{alias} -> {target}" for alias, target in PROVIDER_ALIASES.items()
        )
        print("aliases: " + aliases)


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"Aether v{AETHER_VERSION}")
    print(f"python {sys.version.split()[0]}  -  platform {sys.platform}")
