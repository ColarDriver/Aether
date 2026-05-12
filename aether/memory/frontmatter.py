"""Minimal frontmatter parser/renderer for project memory files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class FrontmatterDocument:
    """One markdown document with YAML-like frontmatter."""

    metadata: dict[str, Any]
    body: str


def parse_frontmatter_documents(text: str) -> tuple[FrontmatterDocument, ...]:
    """Parse repeated ``--- metadata --- body`` markdown documents.

    This intentionally supports only the small subset we render ourselves:
    string scalars and ``key:`` followed by ``  - item`` lists.  Topic files
    are human-editable, so unknown/malformed lines are ignored instead of
    failing the whole store read.
    """

    lines = text.splitlines()
    docs: list[FrontmatterDocument] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "---":
            index += 1
            continue

        fm_start = index + 1
        index = fm_start
        while index < len(lines) and lines[index].strip() != "---":
            index += 1
        if index >= len(lines):
            break

        metadata = parse_frontmatter_block(lines[fm_start:index])
        body_start = index + 1
        index = body_start
        body_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != "---":
            body_lines.append(lines[index])
            index += 1
        body = "\n".join(body_lines).strip()
        docs.append(FrontmatterDocument(metadata=metadata, body=body))
    return tuple(docs)


def parse_frontmatter_block(lines: Iterable[str]) -> dict[str, Any]:
    """Parse the small frontmatter subset used by project memory entries."""

    metadata: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_list_key and line.startswith("  - "):
            value = _parse_scalar(line[4:].strip())
            metadata.setdefault(current_list_key, []).append(value)
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        if raw_value == "":
            metadata[key] = []
            current_list_key = key
            continue
        metadata[key] = _parse_scalar(raw_value)
    return metadata


def render_frontmatter_document(metadata: dict[str, Any], body: str) -> str:
    """Render metadata/body as one frontmatter document."""

    return f"---\n{render_frontmatter_block(metadata)}---\n\n{body.rstrip()}\n"


def render_frontmatter_block(metadata: dict[str, Any]) -> str:
    """Render metadata as deterministic YAML-like frontmatter."""

    lines: list[str] = []
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {_format_scalar(item)}")
            continue
        lines.append(f"{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _parse_scalar(value: str) -> str | bool | int | float | None:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _format_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    text = str(value)
    if not text:
        return '""'
    if any(char in text for char in "\n\r"):
        text = " ".join(text.split())
    if text.startswith(("{", "[", "-", "!", "@", "#", "&", "*")):
        return _quote(text)
    if ": " in text or text.strip() != text:
        return _quote(text)
    return text


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


__all__ = [
    "FrontmatterDocument",
    "parse_frontmatter_block",
    "parse_frontmatter_documents",
    "render_frontmatter_block",
    "render_frontmatter_document",
]
