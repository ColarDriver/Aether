"""Built-in ``notebook_edit`` tool — Sprint 3.5 / PR 3.5.4.

Structured cell-level editing for Jupyter ``.ipynb`` files.  Treating
notebooks as opaque JSON via ``write_file`` is fragile (one stray
escape destroys the file); a dedicated tool understands the cell
schema, lets the model reference cells by ``cell_id`` or ``cell_idx``,
and performs three operations:

* ``replace`` \u2014 update an existing cell's source (and optionally its
  type).
* ``insert`` \u2014 add a new cell after the referenced one (or at index 0
  if no reference is given).
* ``delete`` \u2014 remove a cell.

The tool only mutates ``cells``; ``metadata`` / ``nbformat`` /
``nbformat_minor`` round-trip unchanged so a notebook re-opened in
Jupyter continues to render with its original kernelspec, language
hints, etc.

Cell ``source`` is stored as a list-of-line-strings (Jupyter's native
shape).  Models pass plain text via ``new_source`` and we normalise
it to the list form on the way in.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.tool_result_storage import DEFAULT_SPILL_ROOT
from aether.tools.base import ToolDescriptor, ToolExecutor


_VALID_CELL_TYPES = frozenset({"code", "markdown"})
_VALID_EDIT_MODES = frozenset({"replace", "insert", "delete"})


class NotebookEditTool(ToolExecutor):
    """Edit a Jupyter notebook at the cell level."""

    def __init__(self, *, default_cwd: Path | None = None) -> None:
        self.default_cwd = default_cwd
        self._descriptor = ToolDescriptor(
            name="notebook_edit",
            description=(
                "Edit a Jupyter notebook (.ipynb) one cell at a time. "
                "``edit_mode=replace`` updates the cell at ``cell_id`` (or "
                "``cell_idx``); ``edit_mode=insert`` adds a new cell after "
                "the referenced cell (or at the head if no reference is "
                "given) \u2014 ``cell_type`` defaults to 'code' for inserts; "
                "``edit_mode=delete`` removes the referenced cell. The "
                "notebook's ``metadata``, ``nbformat`` and other top-level "
                "fields are preserved."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": (
                            "Path to the .ipynb file (absolute or relative "
                            "to cwd)."
                        ),
                    },
                    "edit_mode": {
                        "type": "string",
                        "enum": sorted(_VALID_EDIT_MODES),
                        "description": (
                            "Operation to perform. Defaults to 'replace'."
                        ),
                    },
                    "cell_id": {
                        "type": "string",
                        "description": (
                            "Cell id for replace/delete. For insert, the new "
                            "cell is inserted AFTER the referenced cell."
                        ),
                    },
                    "cell_idx": {
                        "type": "integer",
                        "description": (
                            "Zero-based cell index \u2014 alternative to "
                            "cell_id when the model already knows the position."
                        ),
                    },
                    "new_source": {
                        "type": "string",
                        "description": (
                            "New cell content (required for replace/insert)."
                        ),
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": sorted(_VALID_CELL_TYPES),
                        "description": (
                            "Cell type. Required for insert (defaults to "
                            "'code'); on replace, omit to keep the existing "
                            "type."
                        ),
                    },
                },
                "required": ["notebook_path", "edit_mode"],
            },
            required=["notebook_path", "edit_mode"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args = call.arguments or {}
        raw_path = args.get("notebook_path")
        if not raw_path:
            return _error(call, "'notebook_path' must be a non-empty string")
        edit_mode = args.get("edit_mode", "replace")
        if edit_mode not in _VALID_EDIT_MODES:
            return _error(
                call,
                (
                    f"unknown edit_mode: {edit_mode!r} \u2014 must be one of "
                    f"{sorted(_VALID_EDIT_MODES)}"
                ),
            )

        path = self._resolve_path(raw_path)

        if self._is_under_spill_root(path):
            return _error(
                call,
                f"refusing to edit cached spill file: {path}",
                metadata={"path": str(path)},
            )

        if path.suffix != ".ipynb":
            return _error(
                call,
                f"not a notebook file (expected .ipynb): {path}",
                metadata={"path": str(path)},
            )
        if not path.exists():
            return _error(
                call, f"file not found: {path}", metadata={"path": str(path)}
            )
        if path.is_dir():
            return _error(
                call,
                f"path is a directory (cannot edit): {path}",
                metadata={"path": str(path)},
            )

        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            return _error(call, f"could not read {path}: {exc}")
        except json.JSONDecodeError as exc:
            return _error(
                call,
                f"invalid JSON in notebook: {exc}",
                metadata={"path": str(path)},
            )

        if not isinstance(notebook, dict):
            return _error(
                call,
                "notebook root is not a JSON object",
                metadata={"path": str(path)},
            )
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            return _error(
                call,
                "notebook has no ``cells`` list",
                metadata={"path": str(path)},
            )

        cell_id = args.get("cell_id")
        cell_idx_raw = args.get("cell_idx")

        # ``insert`` is the only mode where missing references are
        # legal (\u2192 prepend at idx 0).  replace/delete must locate a
        # specific cell.
        if edit_mode in {"replace", "delete"} and cell_id is None and cell_idx_raw is None:
            return _error(
                call,
                "'cell_id' or 'cell_idx' is required for replace/delete",
            )

        idx = self._locate_cell(cells, cell_id=cell_id, cell_idx=cell_idx_raw)
        if isinstance(idx, str):  # error message
            return _error(call, idx, metadata={"path": str(path)})

        if edit_mode == "replace":
            new_source = args.get("new_source")
            if not isinstance(new_source, str):
                return _error(call, "'new_source' is required for replace")
            cells[idx]["source"] = self._normalize_source(new_source)
            new_type = args.get("cell_type")
            if new_type is not None:
                if new_type not in _VALID_CELL_TYPES:
                    return _error(
                        call,
                        f"invalid cell_type: {new_type!r} \u2014 must be "
                        f"one of {sorted(_VALID_CELL_TYPES)}",
                    )
                cells[idx]["cell_type"] = new_type
            existing_id = cells[idx].get("id", "?")
            summary = (
                f"replaced cell at index {idx} (id={existing_id}, "
                f"type={cells[idx].get('cell_type', '?')})"
            )
        elif edit_mode == "insert":
            new_source = args.get("new_source")
            if not isinstance(new_source, str):
                return _error(call, "'new_source' is required for insert")
            new_type = args.get("cell_type") or "code"
            if new_type not in _VALID_CELL_TYPES:
                return _error(
                    call,
                    f"invalid cell_type: {new_type!r} \u2014 must be "
                    f"one of {sorted(_VALID_CELL_TYPES)}",
                )
            new_cell = self._build_new_cell(source=new_source, cell_type=new_type)
            insert_at = idx + 1 if idx >= 0 else 0
            cells.insert(insert_at, new_cell)
            summary = (
                f"inserted new {new_type} cell at index {insert_at} "
                f"(id={new_cell['id']})"
            )
        else:  # delete \u2014 already validated mode is in _VALID_EDIT_MODES
            removed = cells.pop(idx)
            summary = (
                f"deleted cell at index {idx} "
                f"(id={removed.get('id', '?')}, "
                f"type={removed.get('cell_type', '?')})"
            )

        notebook["cells"] = cells
        try:
            # ``indent=1`` matches Jupyter's default save format and
            # keeps diffs sane in version control.
            path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            return _error(call, f"could not write {path}: {exc}")

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=summary,
            is_error=False,
            metadata={
                "path": str(path),
                "edit_mode": edit_mode,
                "cell_count": len(cells),
            },
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, raw: Any) -> Path:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()

    @staticmethod
    def _is_under_spill_root(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            return False
        try:
            resolved.relative_to(DEFAULT_SPILL_ROOT.resolve())
            return True
        except (ValueError, OSError, RuntimeError):
            return False

    @staticmethod
    def _locate_cell(
        cells: List[Dict[str, Any]],
        *,
        cell_id: Optional[str],
        cell_idx: Optional[Any],
    ) -> int | str:
        """Return the resolved cell index, or an error message string.

        ``cell_idx`` wins when both are supplied (model is being more
        precise); on conflict between idx and id we prefer the explicit
        positional argument over a string lookup.
        """
        if cell_idx is not None:
            try:
                idx = int(cell_idx)
            except (TypeError, ValueError):
                return f"cell_idx must be an integer, got {cell_idx!r}"
            if idx < 0 or idx >= len(cells):
                return (
                    f"cell_idx {idx} out of range (notebook has {len(cells)} cells)"
                )
            return idx
        if cell_id is not None:
            for i, cell in enumerate(cells):
                if isinstance(cell, dict) and cell.get("id") == cell_id:
                    return i
            return f"cell_id {cell_id!r} not found in notebook"
        # No reference \u2014 caller (insert mode) treats -1 as "insert at 0".
        return -1

    @staticmethod
    def _normalize_source(text: str) -> List[str]:
        """Convert plain text to Jupyter's source list-of-strings shape.

        Jupyter stores ``source`` as a list of lines, each line ending
        with ``\\n`` except (optionally) the last.  Empty source is
        ``[]``, not ``[""]`` \u2014 matches the convention used by every
        notebook produced by the JupyterLab UI.
        """
        if not text:
            return []
        return text.splitlines(keepends=True)

    @staticmethod
    def _build_new_cell(*, source: str, cell_type: str) -> Dict[str, Any]:
        cell: Dict[str, Any] = {
            "cell_type": cell_type,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": NotebookEditTool._normalize_source(source),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return cell


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["NotebookEditTool"]
