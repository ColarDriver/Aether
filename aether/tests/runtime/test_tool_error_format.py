from __future__ import annotations

import json
import unittest

from aether.runtime.tools.tool_error_format import (
    FormattedToolError,
    SchemaIssue,
    format_invalid_tool_args_error,
    format_schema_error,
    format_schema_error_from_issues,
    format_unknown_tool_error,
)


class ToolErrorFormatTests(unittest.TestCase):
    def test_invalid_json_error_includes_position_snippet_and_hint(self) -> None:
        raw = '{"path": "/tmp/x" "bad": 1}'
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            formatted = format_invalid_tool_args_error("read_file", exc, raw)
        else:  # pragma: no cover - defensive
            self.fail("fixture unexpectedly parsed")

        self.assertIsInstance(formatted, FormattedToolError)
        self.assertEqual(formatted.category, "json_syntax")
        self.assertEqual(formatted.line, 1)
        self.assertIsNotNone(formatted.column)
        self.assertIn("Invalid JSON arguments for tool `read_file`", formatted.text)
        self.assertIn("The parser stopped after:", formatted.text)
        self.assertIn("Hint:", formatted.text)

    def test_schema_error_reports_missing_required_parameter(self) -> None:
        formatted = format_schema_error(
            "read_file",
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            {},
        )

        self.assertEqual(formatted.category, "schema_missing")
        self.assertIn("The required parameter `path` is missing", formatted.text)

    def test_schema_error_reports_unexpected_parameter_when_schema_is_strict(self) -> None:
        formatted = format_schema_error(
            "read_file",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            {"path": "/tmp/x", "extra": True},
        )

        self.assertEqual(formatted.category, "schema_unexpected")
        self.assertIn("An unexpected parameter `extra` was provided", formatted.text)

    def test_schema_error_allows_unexpected_parameter_when_schema_explicitly_allows_it(self) -> None:
        formatted = format_schema_error(
            "flex",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": True,
            },
            {"path": "/tmp/x", "extra": True},
        )

        self.assertEqual(formatted.category, "schema_unknown")

    def test_schema_error_reports_type_mismatch(self) -> None:
        formatted = format_schema_error(
            "read_file",
            {"type": "object", "properties": {"offset": {"type": "integer"}}},
            {"offset": "10"},
        )

        self.assertEqual(formatted.category, "schema_type_mismatch")
        self.assertIn(
            "The parameter `offset` type is expected as `integer` but provided as `string`",
            formatted.text,
        )

    def test_schema_error_from_issues_preserves_dominant_category_order(self) -> None:
        formatted = format_schema_error_from_issues(
            "tool",
            [
                SchemaIssue(code="type_mismatch", path="count", expected="integer", received="string"),
                SchemaIssue(code="missing", path="path"),
            ],
        )

        self.assertEqual(formatted.category, "schema_missing")
        self.assertIn("following 2 issues", formatted.text)

    def test_unknown_tool_error_includes_suggestions_and_available_tools(self) -> None:
        formatted = format_unknown_tool_error("readfile", ["read_file", "write_file", "shell"])

        self.assertEqual(formatted.category, "unknown_tool")
        self.assertIn("Unknown tool", formatted.text)
        self.assertIn("`read_file`", formatted.text)
        self.assertIn("Available tools:", formatted.text)


if __name__ == "__main__":
    unittest.main()
