from __future__ import annotations

from dataclasses import asdict

from aether.services.tools import ToolService
from aether.tools.base import ToolDescriptor


class _Registry:
    def __init__(self, descriptors: list[ToolDescriptor]) -> None:
        self._descriptors = descriptors

    def list_descriptors(self) -> list[ToolDescriptor]:
        return self._descriptors


def test_tool_service_lists_builtin_descriptors() -> None:
    catalog = ToolService().list_tools()
    names = {tool.name for tool in catalog.tools}

    assert "read_file" in names
    assert "shell" in names
    assert "web_search" in names
    assert catalog.tools == sorted(catalog.tools, key=lambda item: item.name)


def test_tool_grouping_is_deterministic() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry(
            [
                ToolDescriptor(name="shell"),
                ToolDescriptor(name="read_file"),
                ToolDescriptor(name="memory_read"),
                ToolDescriptor(name="custom_tool"),
            ]
        )
    )

    groups = service.list_groups()

    assert [(group.name, [tool.name for tool in group.tools]) for group in groups] == [
        ("filesystem", ["read_file"]),
        ("shell", ["shell"]),
        ("memory", ["memory_read"]),
        ("other", ["custom_tool"]),
    ]


def test_tool_summary_is_display_safe_and_lookup_works() -> None:
    service = ToolService(
        registry_factory=lambda: _Registry(
            [
                ToolDescriptor(
                    name="read_file",
                    description="Read a file",
                    parameters={"type": "object"},
                    required=["path"],
                )
            ]
        )
    )

    summary = service.get_tool("read_file")

    assert summary is not None
    assert asdict(summary) == {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object"},
        "required": ["path"],
        "enabled": True,
    }
    assert "executor" not in repr(summary).lower()
    assert service.get_tool("missing") is None
