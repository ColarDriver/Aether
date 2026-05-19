"""Helpers for provider-hosted web search tools."""

from __future__ import annotations

from typing import Any

from aether.tools.base import ToolDescriptor


WEB_SEARCH_TOOL_NAME = "web_search"


def is_web_search_tool(tool: ToolDescriptor) -> bool:
    return tool.name == WEB_SEARCH_TOOL_NAME


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for source in sources:
        url = str(source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(source.get("title") or "").strip() or url
        out.append({"title": title, "url": url})
    return out


def append_sources_section(content: str, sources: list[dict[str, Any]]) -> str:
    deduped = dedupe_sources(sources)
    if not deduped:
        return content
    if "sources:" in content.lower():
        return content

    head = content.rstrip()
    lines = [head, "", "Sources:"] if head else ["Sources:"]
    for source in deduped[:10]:
        lines.append(f"- [{source['title']}]({source['url']})")
    return "\n".join(lines).rstrip()


__all__ = [
    "WEB_SEARCH_TOOL_NAME",
    "append_sources_section",
    "dedupe_sources",
    "is_web_search_tool",
]
