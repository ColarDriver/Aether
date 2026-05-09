"""Sprint 3.5 / PR 3.5.4 \u2014 ``notebook_edit`` tool coverage.

Pins:

* arg validation (notebook_path / edit_mode required, enums respected),
* reference resolution (cell_id, cell_idx, both, neither),
* replace / insert / delete operations leave correct cell list,
* notebook top-level metadata round-trips unchanged,
* file format errors (non-.ipynb, missing, dir, invalid JSON, no cells),
* cell_type coercion (insert defaults to 'code'),
* source normalisation (string \u2192 list-of-lines with line terminators),
* spill-root protection.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aether.runtime.contracts import ToolCall, TurnContext
from aether.runtime.tool_result_storage import DEFAULT_SPILL_ROOT
from aether.tools.builtins.notebook_edit import NotebookEditTool


def _ctx() -> TurnContext:
    return TurnContext(session_id="notebook-edit-tests", iteration=1)


def _call(**args) -> ToolCall:
    return ToolCall(
        id="call-notebook-edit", name="notebook_edit", arguments=args
    )


def _make_notebook(path: Path, *, cells: list[dict] | None = None) -> None:
    nb = {
        "cells": cells if cells is not None else [
            {
                "cell_type": "code",
                "id": "c1",
                "metadata": {"tags": ["x"]},
                "execution_count": None,
                "outputs": [],
                "source": ["print('hi')"],
            },
            {
                "cell_type": "markdown",
                "id": "c2",
                "metadata": {},
                "source": ["# heading"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ArgValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotebookEditTool()

    def test_missing_notebook_path(self) -> None:
        result = self.tool.execute(_call(edit_mode="replace"), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("'notebook_path'", result.content)

    def test_invalid_edit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(notebook_path=str(p), edit_mode="purge"), _ctx()
            )
        self.assertTrue(result.is_error)
        self.assertIn("unknown edit_mode", result.content)

    def test_replace_requires_a_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    new_source="x",
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("required for replace/delete", result.content)

    def test_replace_requires_new_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c1",
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("'new_source'", result.content)


class ReferenceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotebookEditTool()

    def test_locate_by_cell_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c2",
                    new_source="# new heading",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertFalse(result.is_error)
        self.assertEqual(nb["cells"][1]["source"], ["# new heading"])

    def test_locate_by_cell_idx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_idx=0,
                    new_source="print('replaced')",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertFalse(result.is_error)
        self.assertEqual(nb["cells"][0]["source"], ["print('replaced')"])

    def test_unknown_cell_id_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_id="does-not-exist",
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_out_of_range_cell_idx_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_idx=42,
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("out of range", result.content)

    def test_cell_idx_overrides_cell_id_when_both_supplied(self) -> None:
        # When both are supplied we trust the explicit positional one.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c2",  # would be index 1
                    cell_idx=0,
                    new_source="x = 1",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertFalse(result.is_error)
        self.assertEqual(nb["cells"][0]["source"], ["x = 1"])
        # c2 must be untouched.
        self.assertEqual(nb["cells"][1]["source"], ["# heading"])


class ReplaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotebookEditTool()

    def test_replace_preserves_existing_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c2",
                    new_source="# different heading",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertEqual(nb["cells"][1]["cell_type"], "markdown")

    def test_replace_can_change_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c2",
                    new_source="print('was markdown')",
                    cell_type="code",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertEqual(nb["cells"][1]["cell_type"], "code")

    def test_replace_normalises_multiline_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c1",
                    new_source="a = 1\nb = 2\nc = 3",
                ),
                _ctx(),
            )
            src = _load(p)["cells"][0]["source"]
        # Jupyter convention: list-of-lines with trailing \n on all but
        # optionally the last.
        self.assertEqual(src, ["a = 1\n", "b = 2\n", "c = 3"])

    def test_replace_invalid_cell_type_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c1",
                    new_source="x",
                    cell_type="raw",
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("invalid cell_type", result.content)


class InsertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotebookEditTool()

    def test_insert_after_referenced_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="insert",
                    cell_id="c1",
                    new_source="print('inserted')",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertEqual(len(nb["cells"]), 3)
        self.assertEqual(nb["cells"][1]["source"], ["print('inserted')"])
        # c2 shifts to index 2.
        self.assertEqual(nb["cells"][2]["id"], "c2")

    def test_insert_at_head_when_no_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="insert",
                    new_source="# preamble",
                    cell_type="markdown",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertEqual(nb["cells"][0]["source"], ["# preamble"])
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        # Original cells follow.
        self.assertEqual(nb["cells"][1]["id"], "c1")

    def test_insert_default_cell_type_is_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="insert",
                    new_source="print('ok')",
                ),
                _ctx(),
            )
            cell = _load(p)["cells"][0]
        self.assertEqual(cell["cell_type"], "code")
        # Code cells get execution_count + outputs scaffolding.
        self.assertIsNone(cell["execution_count"])
        self.assertEqual(cell["outputs"], [])

    def test_insert_assigns_unique_cell_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="insert",
                    new_source="x",
                ),
                _ctx(),
            )
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="insert",
                    new_source="y",
                ),
                _ctx(),
            )
            ids = [c["id"] for c in _load(p)["cells"][:2]]
        self.assertEqual(len(set(ids)), 2, ids)


class DeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotebookEditTool()

    def test_delete_removes_referenced_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_id="c1",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertEqual(len(nb["cells"]), 1)
        self.assertEqual(nb["cells"][0]["id"], "c2")


class TopLevelMetadataPreservedTests(unittest.TestCase):
    def test_kernelspec_and_nbformat_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.ipynb"
            _make_notebook(p)
            tool = NotebookEditTool()
            tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="replace",
                    cell_id="c1",
                    new_source="changed",
                ),
                _ctx(),
            )
            nb = _load(p)
        self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(nb["nbformat_minor"], 5)


class FileFormatErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotebookEditTool()

    def test_non_ipynb_extension_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "n.txt"
            p.write_text("{}")
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_idx=0,
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn(".ipynb", result.content)

    def test_missing_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.ipynb"
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_idx=0,
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_invalid_json_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "broken.ipynb"
            p.write_text("not-json")
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_idx=0,
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("invalid JSON", result.content)

    def test_no_cells_list_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "weird.ipynb"
            p.write_text(json.dumps({"metadata": {}, "nbformat": 4}))
            result = self.tool.execute(
                _call(
                    notebook_path=str(p),
                    edit_mode="delete",
                    cell_idx=0,
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("no ``cells`` list", result.content)


class SpillRootProtectionTests(unittest.TestCase):
    def test_refuses_to_edit_notebooks_under_spill_root(self) -> None:
        spill_session_dir = DEFAULT_SPILL_ROOT / "_notebook_edit_test"
        spill_session_dir.mkdir(parents=True, exist_ok=True)
        target = spill_session_dir / "fake.ipynb"
        try:
            _make_notebook(target)
            tool = NotebookEditTool()
            result = tool.execute(
                _call(
                    notebook_path=str(target),
                    edit_mode="delete",
                    cell_idx=0,
                ),
                _ctx(),
            )
            self.assertTrue(result.is_error)
            self.assertIn("spill", result.content)
        finally:
            try:
                target.unlink()
                spill_session_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
