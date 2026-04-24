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

    Priority:
    1. <project_root>/.env (or nearest parent .env)
    2. Existing process env (kept unless overridden by file values)
    """

    loaded: list[Path] = []
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    env_path = _workspace_env_path(root)
    if env_path and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True, encoding='utf-8')
        loaded.append(env_path)
    return loaded
