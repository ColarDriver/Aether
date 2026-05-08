"""Aether env loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _workspace_env_path(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        env_path = candidate / '.env'
        if env_path.exists():
            return env_path
    return None


def load_aether_dotenv(project_root: str | os.PathLike | None = None) -> list[Path]:
    """Load .env files with local project env taking precedence.

    Search order (first hit wins):

    1. ``<project_root>/.env`` (or nearest parent ``.env``) when an explicit
       root is given.
    2. The Aether package root (``aether/.env``) and its parents — this is
       where the dev-time ``.env`` lives.
    3. The current working directory and its parents.

    Existing process env is kept unless overridden by file values.
    """

    loaded: list[Path] = []
    seen: set[Path] = set()

    if project_root is not None:
        roots: list[Path] = [Path(project_root)]
    else:
        # ``parents[1]`` is the aether package root (.../aether/), where the
        # canonical dev-time .env lives.  We also fall back to CWD so that a
        # user-supplied .env in their working tree still wins.
        pkg_root = Path(__file__).resolve().parents[1]
        roots = [pkg_root, Path.cwd()]

    for root in roots:
        env_path = _workspace_env_path(root)
        if env_path is None or env_path in seen:
            continue
        seen.add(env_path)
        load_dotenv(dotenv_path=env_path, override=True, encoding='utf-8')
        loaded.append(env_path)

    return loaded
