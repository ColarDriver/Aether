"""Render skill metadata into a ``<system-reminder>`` block.

Only skill frontmatter metadata is exposed here: name, description, and
optional ``when_to_use``. Skill bodies remain lazy-loaded from disk by the
``skill`` tool itself.
"""

from __future__ import annotations

from aether.runtime.tools.skill_catalog import Skill, SkillCatalog

_HEADER = "The following skills are available for use with the `skill` tool:"
_FOOTER_HINT = (
    "Invoke a skill via `skill(skill=\"<name>\", args=\"<optional args>\")`. "
    "Skill bodies are loaded on demand."
)


def format_skill_listing(
    catalog: SkillCatalog | None,
    *,
    budget_chars: int = 4096,
) -> str:
    """Return a ``<system-reminder>`` block or ``""`` when no skills exist."""
    if catalog is None or budget_chars <= 0:
        return ""

    skills = sorted(catalog.list_all(), key=lambda s: s.name)
    if not skills:
        return ""

    prefix_lines = ["<system-reminder>", _HEADER, ""]
    suffix_lines = ["", _FOOTER_HINT, "</system-reminder>"]
    prefix_len = _joined_len(prefix_lines)
    suffix_len = _joined_len(suffix_lines)
    if prefix_len + suffix_len > budget_chars:
        return ""

    lines: list[str] = list(prefix_lines)
    used = prefix_len
    included = 0

    for skill in skills:
        entry = _format_entry(skill)
        omitted = len(skills) - (included + 1)
        omitted_line = _omitted_line(omitted) if omitted > 0 else None
        extra_len = len(entry) + 1
        if omitted_line is not None:
            extra_len += len(omitted_line) + 1
        if used + extra_len + suffix_len > budget_chars:
            break
        lines.append(entry)
        used += len(entry) + 1
        included += 1

    omitted = len(skills) - included
    if included == 0:
        return ""
    if omitted > 0:
        omitted_line = _omitted_line(omitted)
        if used + len(omitted_line) + 1 + suffix_len > budget_chars:
            return ""
        lines.append(omitted_line)

    lines.extend(suffix_lines)
    rendered = "\n".join(lines)
    return rendered if len(rendered) <= budget_chars else ""


def _format_entry(skill: Skill) -> str:
    desc = (skill.description or "").strip().replace("\n", " ")
    if skill.when_to_use:
        when = skill.when_to_use.strip().replace("\n", " ")
        return f"- {skill.name}: {desc} (use when: {when})" if desc else f"- {skill.name} (use when: {when})"
    return f"- {skill.name}: {desc}" if desc else f"- {skill.name}"


def _omitted_line(omitted: int) -> str:
    return (
        f"... ({omitted} more skills omitted; "
        f"use `skill` tool with the exact name to load any of them)"
    )


def _joined_len(lines: list[str]) -> int:
    return len("\n".join(lines))


__all__ = ["format_skill_listing"]
