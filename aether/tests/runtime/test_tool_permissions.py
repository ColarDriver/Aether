from __future__ import annotations

import unittest

from aether.runtime.core.contracts import ToolCall
from aether.runtime.tools.tool_permissions import (
    ToolPermissionMode,
    ToolPermissionPreview,
    ToolPermissionRequest,
    ToolPermissionRule,
    build_session_rule_for_request,
    find_matching_rule,
    is_dangerous_tool,
    make_permission_request,
)


class ToolPermissionPolicyTests(unittest.TestCase):
    def test_write_tools_are_dangerous(self) -> None:
        self.assertTrue(is_dangerous_tool("file_edit"))
        self.assertTrue(is_dangerous_tool("write_file"))
        self.assertTrue(is_dangerous_tool("shell"))
        self.assertFalse(is_dangerous_tool("todo_write"))
        self.assertFalse(is_dangerous_tool("read_file"))

    def test_session_rule_matches_preview_path(self) -> None:
        request = make_permission_request(
            ToolCall(id="c1", name="file_edit", arguments={"path": "app.py"}),
            session_id="s1",
            preview=ToolPermissionPreview(title="Edit file", path="/tmp/app.py"),
        )
        rule = ToolPermissionRule(
            tool_name="file_edit",
            behavior=ToolPermissionMode.ALLOW,
            path_prefix="/tmp/app.py",
        )

        self.assertEqual(find_matching_rule(request, [rule]), rule)

    def test_deny_rule_overrides_allow_rule(self) -> None:
        request = ToolPermissionRequest(
            session_id="s1",
            tool_call_id="c1",
            tool_name="write_file",
            arguments={"path": "/tmp/a"},
            category="write",
            risk="write",
            preview=ToolPermissionPreview(title="Write", path="/tmp/a"),
        )
        allow = ToolPermissionRule(
            tool_name="write_file",
            behavior=ToolPermissionMode.ALLOW,
            path_prefix="/tmp/a",
        )
        deny = ToolPermissionRule(
            tool_name="write_file",
            behavior=ToolPermissionMode.DENY,
            path_prefix="/tmp/a",
        )

        self.assertEqual(find_matching_rule(request, [allow, deny]), deny)

    def test_build_session_rule_uses_shell_prefix(self) -> None:
        request = make_permission_request(
            ToolCall(
                id="c1",
                name="shell",
                arguments={"command": "pytest tests && echo done"},
            ),
            session_id="s1",
            preview=ToolPermissionPreview(
                title="Run command",
                command="pytest tests && echo done",
            ),
        )

        rule = build_session_rule_for_request(request)

        self.assertEqual(rule.tool_name, "shell")
        self.assertEqual(rule.command_prefix, "pytest tests")
        self.assertIsNone(rule.path_prefix)


if __name__ == "__main__":
    unittest.main()
