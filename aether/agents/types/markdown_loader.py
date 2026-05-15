"""Parse markdown agent-type definitions."""

from __future__ import annotations

from pathlib import Path

from aether.agents.types.definition import AgentTypeDefinition
from aether.runtime.tools.skill_catalog import SkillCatalog


def load_markdown_agent_type(
    path: Path,
    *,
    source: str = "project",
) -> AgentTypeDefinition | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    front, body = SkillCatalog._split_frontmatter(text)  # noqa: SLF001

    name = (front.get("name") or path.stem).strip()
    if not name:
        return None
    description = front.get("description", "").strip()
    if not description:
        return None

    return AgentTypeDefinition(
        agent_type=name,
        description=description,
        system_prompt=body.strip(),
        tools=_parse_list_field(front.get("tools")) or None,
        disallowed_tools=tuple(_parse_list_field(front.get("disallowed_tools")) or ()),
        model=(front.get("model") or "").strip() or None,
        skills=tuple(_parse_list_field(front.get("skills")) or ()),
        max_turns=_parse_int(front.get("max_turns")),
        isolation=(front.get("isolation") or "").strip() or None,
        background=_parse_bool(front.get("background")),
        source=source,
        source_path=path,
    )


def _parse_list_field(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not raw.strip():
        return None
    txt = raw.strip()
    inner = txt[1:-1] if txt.startswith("[") and txt.endswith("]") else txt
    items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
    return tuple(item for item in items if item)


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in ("true", "yes", "1", "on")
