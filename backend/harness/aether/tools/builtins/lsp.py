"""Built-in ``lsp`` tool — Sprint 3.5 / PR-3 (PR 3.5.9).

Exposes 9 Language Server Protocol operations to the model:

* ``goToDefinition`` — symbol -> defining location
* ``findReferences`` — symbol -> all references
* ``hover`` — symbol -> documentation / signature
* ``documentSymbol`` — file -> outline (classes, functions, ...)
* ``workspaceSymbol`` — workspace-wide symbol search by name
* ``goToImplementation`` — interface / abstract -> implementations
* ``prepareCallHierarchy`` — symbol -> a call-hierarchy item
* ``incomingCalls`` — call-hierarchy item -> who calls it
* ``outgoingCalls`` — call-hierarchy item -> what it calls

Manager resolution order mirrors the other v3.5 tools:

1. Constructor injection (``LSPTool(manager=...)``).
2. ``context.metadata['_lsp_manager']`` (set by
   :meth:`AgentEngine._prepare_turn_entry` when a manager is configured
   on the engine).
3. Lazy default — build a fresh :class:`LSPManager` rooted at
   ``Path.cwd()``.  Subsequent calls reuse the lazy instance via
   ``self._manager``.

When no language server binary is on ``PATH`` the manager returns
``None`` and the tool surfaces a clear "no LSP server available"
message that points the model at grep / read_file as a fallback.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.resources.lsp_manager import LSPManager
from aether.runtime.resources.lsp_servers import (
    EXT_TO_LANG,
    LANGUAGE_SERVERS,
    language_for,
)
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool

logger = logging.getLogger(__name__)


__all__ = ["LSPTool"]


_OPERATIONS: frozenset[str] = frozenset(
    {
        "goToDefinition",
        "findReferences",
        "hover",
        "documentSymbol",
        "workspaceSymbol",
        "goToImplementation",
        "prepareCallHierarchy",
        "incomingCalls",
        "outgoingCalls",
    }
)


_OP_TO_METHOD: Dict[str, str] = {
    "goToDefinition": "textDocument/definition",
    "findReferences": "textDocument/references",
    "hover": "textDocument/hover",
    "documentSymbol": "textDocument/documentSymbol",
    "workspaceSymbol": "workspace/symbol",
    "goToImplementation": "textDocument/implementation",
    "prepareCallHierarchy": "textDocument/prepareCallHierarchy",
    "incomingCalls": "callHierarchy/incomingCalls",
    "outgoingCalls": "callHierarchy/outgoingCalls",
}


# Operations that take a ``{textDocument, position}`` body.
_POSITIONAL_OPS: frozenset[str] = frozenset(
    {
        "goToDefinition",
        "findReferences",
        "hover",
        "documentSymbol",
        "goToImplementation",
        "prepareCallHierarchy",
    }
)


class LSPTool(ToolExecutor):
    """Wrap LSPManager + LSPClient for model-driven semantic queries."""

    NAME = "lsp"
    MAX_RESULT_CHARS = 40_000

    def __init__(
        self,
        manager: Optional[LSPManager] = None,
        *,
        default_cwd: Optional[Path] = None,
    ) -> None:
        self._manager = manager
        self._default_cwd = default_cwd
        self._manager_lock = threading.Lock()
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Run a Language Server Protocol query for semantic code "
                "navigation. Supports goToDefinition, findReferences, "
                "hover, documentSymbol, workspaceSymbol, "
                "goToImplementation, prepareCallHierarchy, "
                "incomingCalls and outgoingCalls. Falls back gracefully "
                "with a clear error when the matching language server "
                "binary is not installed; use grep / read_file in that "
                "case."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": sorted(_OPERATIONS),
                    },
                    "filePath": {
                        "type": "string",
                        "description": (
                            "Absolute or workspace-relative path. Required for "
                            "every op except workspaceSymbol."
                        ),
                    },
                    "line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based line number.",
                    },
                    "character": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based character (column) number.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query string for workspaceSymbol.",
                    },
                    "item": {
                        "type": "object",
                        "description": (
                            "Call-hierarchy item returned by a prior "
                            "prepareCallHierarchy call; required for "
                            "incomingCalls / outgoingCalls."
                        ),
                    },
                },
                "required": ["operation"],
            },
            required=["operation"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    # ------------------------------------------------------------- execute

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        op = args.get("operation")
        if op not in _OPERATIONS:
            return _error(
                call,
                f"unknown LSP operation: {op!r}; supported: {sorted(_OPERATIONS)}",
            )

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "lsp_tool_enabled", True)):
            return _error(call, "LSP tool is disabled by configuration")

        timeout = float(getattr(config, "lsp_request_timeout_seconds", 15.0))

        manager = self._resolve_manager(context, config)

        if op == "workspaceSymbol":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return _error(call, "'query' is required and must be non-empty for workspaceSymbol")
            client = manager.any_initialized_client() if manager else None
            if client is None:
                return _error(
                    call,
                    "workspaceSymbol requires at least one initialised "
                    "LSP server; call a positional op first to seed one.",
                )
            try:
                result = client.request("workspace/symbol", {"query": query}, timeout=timeout)
            except TimeoutError:
                return _error(call, f"LSP {op} timed out after {timeout:.0f}s")
            except Exception as exc:
                return _error(call, f"LSP {op} failed: {exc}")
            body = self._format_workspace_symbols(query, result)
            content = maybe_spill_for_tool(
                body,
                call=call,
                context=context,
                max_chars=self.MAX_RESULT_CHARS,
                extension="md",
                full_lines=body.count("\n") + 1,
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=False,
                metadata={"operation": op, "query": query},
            )

        # Positional ops + call-hierarchy follow-ups.
        file_path_str = args.get("filePath")
        if not isinstance(file_path_str, str) or not file_path_str.strip():
            return _error(call, f"'filePath' is required for {op}")
        file_path = self._resolve_path(file_path_str.strip())
        if not file_path.exists():
            return _error(call, f"file not found: {file_path}")
        if not file_path.is_file():
            return _error(call, f"path is not a regular file: {file_path}")

        if manager is None:
            return _error(
                call,
                f"LSP unavailable: no manager configured for {file_path.suffix} files. "
                "Install an LSP server (e.g. pylsp / typescript-language-server) and "
                "retry, or fall back to grep / read_file.",
            )
        client = manager.get_client_for(file_path)
        if client is None:
            lang = language_for(file_path) or "unknown"
            candidates = LANGUAGE_SERVERS.get(lang, [])
            install_hint = (
                ", ".join(" ".join(c) for c in candidates)
                if candidates
                else "(no bundled candidates for this language)"
            )
            return _error(
                call,
                f"no LSP server available for {file_path.suffix} files (language={lang!r}). "
                f"Install one of: {install_hint}. "
                "Use grep / read_file as a fallback.",
            )

        if op in ("incomingCalls", "outgoingCalls"):
            item = args.get("item")
            if not isinstance(item, dict):
                return _error(
                    call,
                    f"'item' (a call-hierarchy item from prepareCallHierarchy) is required for {op}",
                )
            method = _OP_TO_METHOD[op]
            try:
                result = client.request(method, {"item": item}, timeout=timeout)
            except TimeoutError:
                return _error(call, f"LSP {op} timed out after {timeout:.0f}s")
            except Exception as exc:
                return _error(call, f"LSP {op} failed: {exc}")
            body = self._format_call_hierarchy_calls(op, result)
        else:
            line_arg = args.get("line", 1)
            char_arg = args.get("character", 1)
            try:
                line = max(1, int(line_arg))
                character = max(1, int(char_arg))
            except (TypeError, ValueError):
                return _error(call, "'line' and 'character' must be integers ≥ 1")
            params: Dict[str, Any] = {
                "textDocument": {"uri": file_path.resolve().as_uri()},
            }
            if op != "documentSymbol":
                params["position"] = {"line": line - 1, "character": character - 1}
            method = _OP_TO_METHOD[op]
            try:
                result = client.request(method, params, timeout=timeout)
            except TimeoutError:
                return _error(call, f"LSP {op} timed out after {timeout:.0f}s")
            except Exception as exc:
                return _error(call, f"LSP {op} failed: {exc}")
            body = self._format_positional_result(op, result, file_path=file_path)

        content = maybe_spill_for_tool(
            body,
            call=call,
            context=context,
            max_chars=self.MAX_RESULT_CHARS,
            extension="md",
            full_lines=body.count("\n") + 1,
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=False,
            metadata={"operation": op, "file_path": str(file_path)},
        )

    # --------------------------------------------------------- helpers

    def _resolve_manager(
        self, context: TurnContext, config: Any
    ) -> Optional[LSPManager]:
        if self._manager is not None:
            return self._manager
        injected = context.metadata.get("_lsp_manager") if context.metadata else None
        # Duck-type: any object exposing ``get_client_for`` works.
        # The engine guarantees the value is an :class:`LSPManager`
        # instance (key lives in ``_METADATA_INTERNAL_KEYS``), but
        # accepting structural matches keeps test stubs easy to wire.
        if injected is not None and hasattr(injected, "get_client_for"):
            return injected  # type: ignore[return-value]
        with self._manager_lock:
            if self._manager is None:
                root = self._default_cwd or Path.cwd()
                init_timeout = float(
                    getattr(config, "lsp_initialization_timeout_seconds", 10.0)
                )
                overrides = getattr(config, "lsp_server_overrides", None)
                self._manager = LSPManager(
                    project_root=root,
                    init_timeout=init_timeout,
                    overrides=overrides,
                )
            return self._manager

    def _resolve_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            base = self._default_cwd or Path.cwd()
            path = (base / path).resolve()
        return path

    @staticmethod
    def _location_to_str(loc: Any) -> str:
        if not isinstance(loc, dict):
            return str(loc)
        uri = loc.get("uri") or ""
        rng = loc.get("range") or {}
        start = rng.get("start") or {}
        line = (start.get("line") or 0) + 1
        ch = (start.get("character") or 0) + 1
        # Strip the file:// URI scheme so the model sees a plain path.
        if uri.startswith("file://"):
            uri = uri[len("file://") :]
        return f"{uri}:{line}:{ch}"

    @staticmethod
    def _walk_symbols(
        symbols: list[Any], depth: int, lines: list[str]
    ) -> None:
        for sym in symbols:
            if not isinstance(sym, dict):
                continue
            indent = "  " * depth
            name = sym.get("name", "?")
            kind = sym.get("kind", "?")
            detail = sym.get("detail")
            label = f"{indent}- {name} (kind={kind})"
            if detail:
                label += f" — {detail}"
            rng = sym.get("range") or sym.get("selectionRange") or {}
            start = rng.get("start") or {}
            line = (start.get("line") or 0) + 1
            ch = (start.get("character") or 0) + 1
            label += f"  [{line}:{ch}]"
            lines.append(label)
            children = sym.get("children")
            if isinstance(children, list):
                LSPTool._walk_symbols(children, depth + 1, lines)

    def _format_positional_result(
        self, op: str, result: Any, *, file_path: Path
    ) -> str:
        header = f"# LSP {op} for {file_path}\n"
        if result is None or (isinstance(result, list) and not result):
            return header + "\n_(no results)_\n"

        if op == "hover":
            contents = result.get("contents") if isinstance(result, dict) else None
            text = self._render_hover(contents)
            return header + "\n" + text + "\n"

        if op == "documentSymbol":
            lines: list[str] = []
            if isinstance(result, list):
                self._walk_symbols(result, 0, lines)
            return header + "\n" + ("\n".join(lines) or "_(no symbols)_") + "\n"

        if op == "prepareCallHierarchy":
            if isinstance(result, list):
                lines = []
                for i, item in enumerate(result, 1):
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name", "?")
                    kind = item.get("kind", "?")
                    uri = item.get("uri", "")
                    if uri.startswith("file://"):
                        uri = uri[len("file://") :]
                    rng = item.get("selectionRange") or item.get("range") or {}
                    start = rng.get("start") or {}
                    line = (start.get("line") or 0) + 1
                    ch = (start.get("character") or 0) + 1
                    lines.append(f"{i}. {name} (kind={kind})  {uri}:{line}:{ch}")
                return header + "\n" + ("\n".join(lines) or "_(no items)_") + "\n"
            return header + "\n_(unexpected response)_\n"

        # goToDefinition / findReferences / goToImplementation
        locations: list[Any] = []
        if isinstance(result, list):
            locations = result
        elif isinstance(result, dict):
            locations = [result]
        lines = [f"- {self._location_to_str(loc)}" for loc in locations]
        return header + "\n" + ("\n".join(lines) or "_(no locations)_") + "\n"

    def _format_workspace_symbols(self, query: str, result: Any) -> str:
        header = f"# LSP workspaceSymbol query={query!r}\n"
        if not isinstance(result, list) or not result:
            return header + "\n_(no symbols)_\n"
        lines = []
        for sym in result:
            if not isinstance(sym, dict):
                continue
            name = sym.get("name", "?")
            kind = sym.get("kind", "?")
            container = sym.get("containerName") or ""
            loc = sym.get("location") or {}
            location_str = self._location_to_str(loc)
            label = f"- {name} (kind={kind})"
            if container:
                label += f" in {container}"
            label += f"  {location_str}"
            lines.append(label)
        return header + "\n" + "\n".join(lines) + "\n"

    def _format_call_hierarchy_calls(self, op: str, result: Any) -> str:
        header = f"# LSP {op}\n"
        if not isinstance(result, list) or not result:
            return header + "\n_(no calls)_\n"
        lines = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            target = entry.get("from") if op == "incomingCalls" else entry.get("to")
            if not isinstance(target, dict):
                continue
            name = target.get("name", "?")
            kind = target.get("kind", "?")
            uri = target.get("uri", "")
            if uri.startswith("file://"):
                uri = uri[len("file://") :]
            rng = target.get("selectionRange") or target.get("range") or {}
            start = rng.get("start") or {}
            line = (start.get("line") or 0) + 1
            ch = (start.get("character") or 0) + 1
            ranges = entry.get("fromRanges") or []
            range_count = len(ranges) if isinstance(ranges, list) else 0
            label = f"- {name} (kind={kind})  {uri}:{line}:{ch}"
            if range_count:
                label += f"  ({range_count} call site{'s' if range_count != 1 else ''})"
            lines.append(label)
        return header + "\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _render_hover(contents: Any) -> str:
        if contents is None:
            return "_(no hover content)_"
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            value = contents.get("value")
            if isinstance(value, str):
                return value
            return str(contents)
        if isinstance(contents, list):
            chunks = []
            for entry in contents:
                if isinstance(entry, str):
                    chunks.append(entry)
                elif isinstance(entry, dict):
                    val = entry.get("value")
                    if isinstance(val, str):
                        chunks.append(val)
            return "\n\n".join(chunks) if chunks else "_(no hover content)_"
        return str(contents)


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )
