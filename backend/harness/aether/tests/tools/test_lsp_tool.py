"""Tests for ``aether.tools.builtins.lsp.LSPTool``.

Sprint 3.5 / PR-3 (PR 3.5.9).

We cover the tool's argument validation, manager-resolution chain,
result formatting and config gates with a stub LSP manager + client.
End-to-end coverage with a real LSP server is out of scope for unit
tests (it would force pylsp on the CI image); :class:`LSPClient`
already has direct subprocess coverage.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.tools.builtins.lsp import LSPTool


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Optional[float]]] = []
        self.next_response: Any = None
        self.next_exception: Optional[Exception] = None
        self.is_running = True

    def request(self, method: str, params: Any, *, timeout: float | None = None) -> Any:
        self.calls.append((method, params, timeout))
        if self.next_exception is not None:
            raise self.next_exception
        return self.next_response


class _FakeManager:
    def __init__(self, client: Optional[_FakeClient]) -> None:
        self._client = client

    def get_client_for(self, _file: Path) -> Optional[_FakeClient]:
        return self._client

    def any_initialized_client(self) -> Optional[_FakeClient]:
        return self._client


def _ctx(*, config: Optional[EngineConfig] = None, manager: Any = None) -> TurnContext:
    metadata: dict[str, Any] = {"_engine_config": config or EngineConfig()}
    if manager is not None:
        metadata["_lsp_manager"] = manager
    return TurnContext(session_id="ses-lsp", iteration=0, metadata=metadata)


def _make_file(suffix: str = ".py") -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="lsp-tool-"))
    p = tmp / f"sample{suffix}"
    p.write_text("def hello():\n    return 1\n", encoding="utf-8")
    return p


class ValidationTests(unittest.TestCase):
    def test_a1_unknown_operation_returns_error(self) -> None:
        tool = LSPTool(manager=_FakeManager(_FakeClient()))
        out = tool.execute(ToolCall(id="c1", name="lsp", arguments={"operation": "fooBar"}), _ctx())
        self.assertTrue(out.is_error)
        self.assertIn("unknown LSP operation", out.content)

    def test_a2_disabled_via_config(self) -> None:
        cfg = EngineConfig(lsp_tool_enabled=False)
        tool = LSPTool(manager=_FakeManager(_FakeClient()))
        out = tool.execute(
            ToolCall(id="c2", name="lsp", arguments={"operation": "hover", "filePath": "x.py"}),
            _ctx(config=cfg),
        )
        self.assertTrue(out.is_error)
        self.assertIn("disabled", out.content)

    def test_a3_workspace_symbol_requires_query(self) -> None:
        tool = LSPTool(manager=_FakeManager(_FakeClient()))
        out = tool.execute(
            ToolCall(id="c3", name="lsp", arguments={"operation": "workspaceSymbol"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("query", out.content)

    def test_a4_positional_op_requires_filePath(self) -> None:
        tool = LSPTool(manager=_FakeManager(_FakeClient()))
        out = tool.execute(
            ToolCall(id="c4", name="lsp", arguments={"operation": "hover"}),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("filePath", out.content)

    def test_a5_missing_file_returns_error(self) -> None:
        tool = LSPTool(manager=_FakeManager(_FakeClient()))
        out = tool.execute(
            ToolCall(
                id="c5",
                name="lsp",
                arguments={"operation": "hover", "filePath": "/nope/does/not/exist.py"},
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("file not found", out.content)

    def test_a6_missing_manager_for_supported_file(self) -> None:
        path = _make_file()
        tool = LSPTool(manager=_FakeManager(None))
        out = tool.execute(
            ToolCall(
                id="c6",
                name="lsp",
                arguments={"operation": "hover", "filePath": str(path)},
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertTrue(
            "no LSP server available" in out.content or "LSP unavailable" in out.content,
            out.content,
        )
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_a7_call_hierarchy_op_requires_item(self) -> None:
        client = _FakeClient()
        path = _make_file()
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c7",
                name="lsp",
                arguments={"operation": "incomingCalls", "filePath": str(path)},
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("'item'", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_a8_invalid_line_or_character_returns_error(self) -> None:
        client = _FakeClient()
        path = _make_file()
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c8",
                name="lsp",
                arguments={
                    "operation": "hover",
                    "filePath": str(path),
                    "line": "abc",
                    "character": 1,
                },
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("integers", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)


class HappyPathTests(unittest.TestCase):
    def test_b1_hover_renders_markdown(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_response = {
            "contents": {"kind": "markdown", "value": "**hello** function"}
        }
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "hover",
                    "filePath": str(path),
                    "line": 1,
                    "character": 5,
                },
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertIn("hello", out.content)
        # Position is 0-based on the wire — verify translation.
        method, params, _ = client.calls[0]
        self.assertEqual(method, "textDocument/hover")
        self.assertEqual(params["position"], {"line": 0, "character": 4})
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_b2_definition_renders_locations(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_response = [
            {
                "uri": path.as_uri(),
                "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
            }
        ]
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "goToDefinition",
                    "filePath": str(path),
                    "line": 1,
                    "character": 1,
                },
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertIn(":1:1", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_b3_document_symbol_walks_tree(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_response = [
            {
                "name": "Foo",
                "kind": 5,
                "range": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "bar",
                        "kind": 6,
                        "range": {"start": {"line": 1, "character": 4}},
                    }
                ],
            }
        ]
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={"operation": "documentSymbol", "filePath": str(path)},
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertIn("Foo", out.content)
        self.assertIn("bar", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_b4_workspace_symbol_uses_any_client(self) -> None:
        client = _FakeClient()
        client.next_response = [
            {
                "name": "BigClass",
                "kind": 5,
                "location": {
                    "uri": "file:///x.py",
                    "range": {"start": {"line": 9, "character": 0}},
                },
            }
        ]
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={"operation": "workspaceSymbol", "query": "Big"},
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error, out.content)
        self.assertIn("BigClass", out.content)
        method, params, _ = client.calls[0]
        self.assertEqual(method, "workspace/symbol")
        self.assertEqual(params, {"query": "Big"})

    def test_b5_workspace_symbol_without_seeded_client_errors(self) -> None:
        tool = LSPTool(manager=_FakeManager(None))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={"operation": "workspaceSymbol", "query": "Big"},
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("seed", out.content.lower())

    def test_b6_no_results_renders_helpful_marker(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_response = []
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "findReferences",
                    "filePath": str(path),
                    "line": 1,
                    "character": 1,
                },
            ),
            _ctx(),
        )
        self.assertFalse(out.is_error)
        self.assertIn("no results", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)


class ErrorPropagationTests(unittest.TestCase):
    def test_c1_timeout_is_surfaced_clearly(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_exception = TimeoutError("slow")
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "hover",
                    "filePath": str(path),
                    "line": 1,
                    "character": 1,
                },
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("timed out", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_c2_generic_error_is_wrapped(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_exception = RuntimeError("kaboom")
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "hover",
                    "filePath": str(path),
                    "line": 1,
                    "character": 1,
                },
            ),
            _ctx(),
        )
        self.assertTrue(out.is_error)
        self.assertIn("kaboom", out.content)
        shutil.rmtree(path.parent, ignore_errors=True)

    def test_c3_metadata_carries_operation(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_response = {"contents": "x"}
        tool = LSPTool(manager=_FakeManager(client))
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "hover",
                    "filePath": str(path),
                    "line": 1,
                    "character": 1,
                },
            ),
            _ctx(),
        )
        self.assertEqual(out.metadata.get("operation"), "hover")
        self.assertEqual(out.metadata.get("file_path"), str(path))
        shutil.rmtree(path.parent, ignore_errors=True)


class ManagerResolutionTests(unittest.TestCase):
    def test_d1_metadata_manager_is_used_when_no_constructor_arg(self) -> None:
        path = _make_file()
        client = _FakeClient()
        client.next_response = {"contents": "y"}
        tool = LSPTool()
        out = tool.execute(
            ToolCall(
                id="c1",
                name="lsp",
                arguments={
                    "operation": "hover",
                    "filePath": str(path),
                    "line": 1,
                    "character": 1,
                },
            ),
            _ctx(manager=_FakeManager(client)),
        )
        self.assertFalse(out.is_error, out.content)
        shutil.rmtree(path.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
