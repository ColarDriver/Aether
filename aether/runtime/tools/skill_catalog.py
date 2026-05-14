"""Skill discovery and lookup.

Backs ``SkillTool`` (``tools/builtins/skill.py``).

A *skill* is a ``SKILL.md`` file under one of the configured search
paths.  It may begin with a YAML frontmatter block delimited by ``---``
lines that supplies metadata (``name``, ``description``, ``whenToUse``,
``version``).  When metadata is missing we fall back to:

* ``name`` — the directory containing ``SKILL.md`` joined by ``/``,
  ``-`` separated, relative to the search root.  E.g.
  ``software-development/test-driven-development/SKILL.md`` becomes
  ``software-development-test-driven-development``.  This matches
  Anthropic's "skill name" convention.
* ``description`` / ``when_to_use`` — empty.

The frontmatter parser is intentionally minimal (``key: value`` only,
no flow / block scalars, no nested mappings beyond the single line).
We deliberately avoid a runtime dependency on ``pyyaml`` — every
existing fixture uses simple key/value pairs, and the skill body is
plain markdown anyway.

If a path contains an unparseable file we log + skip rather than crash;
malformed skills must never take down the host process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


__all__ = ["Skill", "SkillCatalog", "build_default_skill_catalog"]


@dataclass(slots=True, frozen=True)
class Skill:
    """A discovered skill.  Hashable so callers can dedupe by identity."""

    name: str
    path: Path
    description: str = ""
    when_to_use: str = ""
    body: str = ""
    source: str = "local"
    version: Optional[str] = None


def build_default_skill_catalog(config: Any | None = None) -> SkillCatalog:
    """Build a catalog from configured search paths or sensible defaults."""
    search_paths: list[Path] = []
    configured_any = getattr(config, "skill_search_paths", ()) if config is not None else ()
    configured = tuple(configured_any) if configured_any else ()
    if configured:
        search_paths.extend(Path(p) for p in configured)
    else:
        cwd = Path.cwd()
        for candidate in (cwd / "skills", Path.home() / ".aether" / "skills"):
            search_paths.append(candidate)
    catalog = SkillCatalog(search_paths=search_paths)
    catalog.discover()
    return catalog


class SkillCatalog:
    """Discover and serve skills from configured directories.

    Discovery order: each path in ``search_paths`` is walked once, and
    later paths overwrite earlier ones on name collision.  This lets a
    user's ``~/.aether/skills`` override the bundled set.
    """

    def __init__(self, *, search_paths: Iterable[Path]):
        self.search_paths: list[Path] = [Path(p) for p in search_paths]
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    # ------------------------------------------------------------------ public

    def discover(self, *, force: bool = False) -> None:
        """Scan all search paths.  Idempotent unless ``force=True``."""
        if self._loaded and not force:
            return
        self._skills.clear()
        for root in self.search_paths:
            if not root.exists() or not root.is_dir():
                continue
            try:
                files = sorted(root.rglob("SKILL.md"))
            except OSError as exc:
                logger.warning("failed to scan skill root %s: %s", root, exc)
                continue
            for skill_md in files:
                skill = self._parse_skill_md(skill_md, root=root)
                if skill is not None:
                    self._skills[skill.name] = skill
        self._loaded = True

    def get(self, name: str) -> Optional[Skill]:
        """Look up a skill by name.

        Resolution order:
        1. Exact match
        2. Case-insensitive match
        3. Strip leading ``/`` and re-try (so models can write
           ``/test-driven-development``).
        """
        self.discover()
        if not name:
            return None
        if name in self._skills:
            return self._skills[name]
        normalized = name.lstrip("/").strip()
        if normalized in self._skills:
            return self._skills[normalized]
        lowered = normalized.lower()
        for k, v in self._skills.items():
            if k.lower() == lowered:
                return v
        return None

    def list_all(self) -> list[Skill]:
        self.discover()
        return list(self._skills.values())

    # ----------------------------------------------------------------- private

    def _parse_skill_md(self, path: Path, *, root: Path) -> Optional[Skill]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("skill %s unreadable: %s", path, exc)
            return None

        frontmatter, body = self._split_frontmatter(text)

        try:
            rel = path.parent.relative_to(root)
        except ValueError:
            rel = Path(path.parent.name)
        name_default = "-".join(rel.parts) if rel.parts else path.parent.name
        name = (frontmatter.get("name") or name_default).strip()
        if not name:
            logger.warning("skill at %s has empty name; skipping", path)
            return None

        return Skill(
            name=name,
            path=path,
            description=frontmatter.get("description", "").strip(),
            when_to_use=(frontmatter.get("whenToUse") or "").strip(),
            body=body,
            source="local",
            version=frontmatter.get("version") or None,
        )

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """Parse a YAML-like frontmatter block delimited by ``---`` lines.

        Returns ``({}, text)`` if no frontmatter is detected.

        Only top-level ``key: value`` lines are recognised — mappings
        (e.g. ``metadata:`` followed by an indented block) are skipped
        in their entirety, which is fine because we only consume four
        well-known fields.
        """
        if not text.startswith("---"):
            return {}, text
        lines = text.splitlines(keepends=False)
        if not lines or lines[0].strip() != "---":
            return {}, text
        # Find the closing fence.
        end = -1
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end = idx
                break
        if end == -1:
            return {}, text

        front: dict[str, str] = {}
        for line in lines[1:end]:
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith((" ", "\t")):
                # nested entry — skip; we only consume top-level keys.
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            front[key] = value

        body = "\n".join(lines[end + 1 :])
        # Preserve a trailing newline if the source had one — markdown
        # renderers occasionally rely on it.
        if text.endswith("\n") and not body.endswith("\n"):
            body += "\n"
        return front, body
