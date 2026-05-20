from __future__ import annotations

from pathlib import Path

from aether.runtime.tools.capabilities import (
    capabilities_for_tool_name,
    extract_tool_scope_path,
    is_parallel_safe_tool,
)


def test_builtin_read_tools_are_parallel_safe() -> None:
    for name in ("read_file", "list_dir", "grep", "glob", "web_fetch", "web_search", "task_output"):
        caps = capabilities_for_tool_name(name)
        assert caps.read_only is True
        assert caps.parallel_safe is True


def test_mutating_tools_are_unsafe() -> None:
    for name in ("write_file", "file_edit", "notebook_edit"):
        caps = capabilities_for_tool_name(name)
        assert caps.mutates_files is True
        assert caps.requires_permission is True
        assert caps.parallel_safe is False


def test_shell_and_interactive_tools_are_unsafe() -> None:
    assert capabilities_for_tool_name("shell").parallel_safe is False
    ask = capabilities_for_tool_name("ask_user_question")
    assert ask.interactive is True
    assert ask.parallel_safe is False


def test_unknown_external_tool_is_unsafe() -> None:
    caps = capabilities_for_tool_name("mcp__external__unknown")
    assert caps.parallel_safe is False
    assert caps.read_only is False
    assert caps.mutates_files is False


def test_parallel_safe_helper_normalizes_names() -> None:
    assert is_parallel_safe_tool("read-file") is True
    assert is_parallel_safe_tool("Write-File") is False


def test_scope_path_normalized_against_cwd() -> None:
    cwd = Path("/repo/project")
    assert extract_tool_scope_path("read_file", {"path": "src/app.py"}, cwd) == Path(
        "/repo/project/src/app.py"
    )


def test_scope_path_supports_common_argument_names() -> None:
    cwd = Path("/repo/project")
    assert extract_tool_scope_path("read_file", {"file_path": "a.py"}, cwd) == Path(
        "/repo/project/a.py"
    )
    assert extract_tool_scope_path("file_edit", {"target": "a.py"}, cwd) == Path(
        "/repo/project/a.py"
    )


def test_missing_or_unknown_path_scope_returns_none() -> None:
    cwd = Path("/repo/project")
    assert extract_tool_scope_path("read_file", {}, cwd) is None
    assert extract_tool_scope_path("web_fetch", {"url": "https://example.com"}, cwd) is None


def test_cheap_classification_is_compatible_with_default_refund_names() -> None:
    assert capabilities_for_tool_name("todo_write").cheap is True
    assert capabilities_for_tool_name("memory_read").cheap is True
    assert capabilities_for_tool_name("read_file").cheap is False
