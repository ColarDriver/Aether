"""Built-in ``file_edit`` tool.

Local search/replace editing on a file with claude-code-style
``unique-match`` semantics: ``old_string`` must appear exactly once
in the file unless ``replace_all=True``.  Output is a compact unified
diff so the model can verify the change without re-reading the file.

Why this exists
---------------
Without ``file_edit`` the only way to modify an existing file is
``write_file``, which means the model has to:

    1. ``read_file`` (consumes ~N tokens),
    2. echo the entire file back with the change (~2N tokens),
    3. ``write_file`` to overwrite (~N tokens).

For a small change in a 200-line file that's >800 line-tokens of work
the LLM could spend on actual reasoning.  ``file_edit`` reduces it to
``read_file`` plus a tiny structured patch, and the unique-match
guard means the model physically can't smear an unintended change
across multiple lookalike sites.

Safety constraints
------------------
* Refuses to edit files inside the spill root
  (``~/.aether/tool_results``) so the model can't accidentally
  corrupt its own paged-out tool-result cache.
* 1 GiB stat-size ceiling matches claude-code's
  ``MAX_EDIT_FILE_SIZE`` \u2014 anything bigger is almost certainly the
  wrong tool and should fall back to ``shell sed``.
* ``old_string == new_string`` is rejected as a no-op so a confused
  model doesn't think it succeeded.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.tool_permissions import ToolPermissionPreview
from aether.runtime.tools.tool_result_storage import DEFAULT_SPILL_ROOT
from aether.tools.base import ToolDescriptor, ToolExecutor


# Matches claude-code; well beyond any text file we'd realistically
# read into a Python ``str``.
_MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024  # 1 GiB
# Diff lines past this threshold get truncated with a "... N more diff
# lines elided ..." footer.  80 lines is generous enough that any
# single-site edit fits, but tight enough that ``replace_all`` over a
# huge file doesn't dump 5000 diff lines into context.
_DIFF_PREVIEW_LINES = 80


@dataclass(slots=True, frozen=True)
class FileEditPlan:
    path: Path
    original: str
    modified: str
    change_count: int
    replace_all: bool


class FileEditTool(ToolExecutor):
    """In-place ``old_string`` -> ``new_string`` substitution."""

    def __init__(self, *, default_cwd: Path | None = None) -> None:
        self.default_cwd = default_cwd
        self._descriptor = ToolDescriptor(
            name="file_edit",
            description=(
                "Edit a file in place by replacing ``old_string`` with "
                "``new_string``. By default ``old_string`` must appear "
                "exactly once in the file \u2014 include enough surrounding "
                "context (3-5 lines before/after the actual change) to make "
                "it unique. Set ``replace_all=true`` for global rename-style "
                "substitutions. Use ``write_file`` instead when creating a "
                "brand-new file or rewriting the whole content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file (absolute or relative to cwd)."
                        ),
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact substring to replace. Must appear exactly "
                            "once unless replace_all=true. Preserve all "
                            "whitespace and indentation \u2014 the match is "
                            "byte-for-byte."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement substring.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "When true, replace every occurrence of "
                            "old_string. Use only for variable / symbol "
                            "renames where every occurrence should change "
                            "identically."
                        ),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            required=["path", "old_string", "new_string"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def build_permission_preview(
        self,
        call: ToolCall,
        context: TurnContext,
    ) -> ToolPermissionPreview | ToolResult:
        plan = self.plan_edit(call)
        if isinstance(plan, ToolResult):
            return plan
        context.metadata.setdefault("_tool_permission_preview_plans", {})[call.id] = plan
        diff_text = self._build_diff(
            path=plan.path,
            original=plan.original,
            modified=plan.modified,
        )
        # Defense in depth: plan_edit already refuses old==new, but if
        # difflib still produces an empty diff (e.g., whitespace-only
        # tweak collapsed by n=2 context), fall back to a body line so
        # the modal does not dump raw JSON.
        body: str | None = None
        if not diff_text:
            plural = "s" if plan.change_count != 1 else ""
            body = (
                f"Edit {plan.path}\n"
                f"{plan.change_count} change{plural} · "
                f"{len(plan.original.encode('utf-8'))} -> "
                f"{len(plan.modified.encode('utf-8'))} bytes"
            )
        return ToolPermissionPreview(
            title="Edit file",
            subtitle=str(plan.path),
            path=str(plan.path),
            diff=diff_text or None,
            body=body,
            metadata={
                "change_count": plan.change_count,
                "replace_all": plan.replace_all,
                "bytes_before": len(plan.original.encode("utf-8")),
                "bytes_after": len(plan.modified.encode("utf-8")),
            },
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        plan = self._plan_from_context(call, context)
        if plan is None:
            planned = self.plan_edit(call)
            if isinstance(planned, ToolResult):
                return planned
            plan = planned
        else:
            try:
                current = plan.path.read_text(encoding="utf-8")
            except OSError as exc:
                return _error(call, f"could not re-read {plan.path}: {exc}")
            except UnicodeDecodeError as exc:
                return _error(
                    call,
                    f"file is no longer valid UTF-8 (cannot edit safely): {exc}",
                    metadata={"path": str(plan.path)},
                )
            if current != plan.original:
                return _error(
                    call,
                    "file changed after permission preview; retry the edit",
                    metadata={"path": str(plan.path), "stale_preview": True},
                )

        try:
            plan.path.write_text(plan.modified, encoding="utf-8")
        except OSError as exc:
            return _error(call, f"could not write {plan.path}: {exc}")

        summary = self._build_diff_summary(
            path=plan.path,
            original=plan.original,
            modified=plan.modified,
            change_count=plan.change_count,
        )
        diff_text, lines_added, lines_removed, hunks = self._build_diff_with_stats(
            path=plan.path,
            original=plan.original,
            modified=plan.modified,
        )
        result_metadata: dict[str, Any] = {
            "path": str(plan.path),
            "change_count": plan.change_count,
            "replace_all": plan.replace_all,
            "bytes_before": len(plan.original.encode("utf-8")),
            "bytes_after": len(plan.modified.encode("utf-8")),
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "hunks": hunks,
        }
        if diff_text and len(diff_text) < 4096:
            result_metadata["diff"] = diff_text
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=summary,
            is_error=False,
            metadata=result_metadata,
        )

    def plan_edit(self, call: ToolCall) -> FileEditPlan | ToolResult:
        args = call.arguments or {}
        raw_path = args.get("path")
        if not raw_path:
            return _error(call, "'path' must be a non-empty string")

        old_string = args.get("old_string")
        new_string = args.get("new_string")
        if not isinstance(old_string, str) or not old_string:
            return _error(call, "'old_string' must be a non-empty string")
        if not isinstance(new_string, str):
            return _error(call, "'new_string' must be a string")
        replace_all = bool(args.get("replace_all") or False)

        path = self._resolve_path(raw_path)

        if self._is_under_spill_root(path):
            return _error(
                call,
                f"refusing to edit cached spill file: {path}",
                metadata={"path": str(path)},
            )

        if not path.exists():
            return _error(
                call,
                (
                    f"file not found: {path}. Use ``write_file`` if you "
                    "intend to create a new file."
                ),
                metadata={"path": str(path)},
            )
        if path.is_dir():
            return _error(
                call,
                f"path is a directory (cannot edit): {path}",
                metadata={"path": str(path)},
            )

        try:
            stat = path.stat()
        except OSError as exc:
            return _error(call, f"could not stat {path}: {exc}")
        if stat.st_size > _MAX_EDIT_FILE_SIZE:
            return _error(
                call,
                (
                    f"file too large to edit ({stat.st_size} bytes > 1 GiB cap). "
                    "Use a ``shell`` command (sed/awk) for files this size."
                ),
                metadata={"path": str(path), "size_bytes": stat.st_size},
            )

        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            return _error(call, f"could not read {path}: {exc}")
        except UnicodeDecodeError as exc:
            return _error(
                call,
                f"file is not valid UTF-8 (cannot edit safely): {exc}",
                metadata={"path": str(path)},
            )

        if old_string == new_string:
            return _error(
                call,
                "old_string and new_string are identical \u2014 no change applied",
                metadata={"path": str(path)},
            )

        occurrences = original.count(old_string)
        if occurrences == 0:
            return _error(
                call,
                (
                    f"old_string not found in {path}. Read the file first and "
                    "ensure old_string is an exact match (preserve all "
                    "whitespace, indentation, and line endings)."
                ),
                metadata={"path": str(path)},
            )
        if not replace_all and occurrences > 1:
            return _error(
                call,
                (
                    f"old_string matches {occurrences} places in {path}. Either "
                    "expand old_string with more surrounding context to make "
                    "it unique, or set replace_all=true if every match should "
                    "change identically."
                ),
                metadata={"path": str(path), "occurrences": occurrences},
            )

        if replace_all:
            modified = original.replace(old_string, new_string)
            change_count = occurrences
        else:
            modified = original.replace(old_string, new_string, 1)
            change_count = 1

        return FileEditPlan(
            path=path,
            original=original,
            modified=modified,
            change_count=change_count,
            replace_all=replace_all,
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
    def _build_diff(
        *,
        path: Path,
        original: str,
        modified: str,
    ) -> str:
        text, _added, _removed, _hunks = FileEditTool._build_diff_with_stats(
            path=path,
            original=original,
            modified=modified,
        )
        return text

    @staticmethod
    def _build_diff_with_stats(
        *,
        path: Path,
        original: str,
        modified: str,
    ) -> tuple[str, int, int, int]:
        raw_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path),
                n=2,
            )
        )
        added = sum(
            1 for line in raw_lines if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in raw_lines if line.startswith("-") and not line.startswith("---")
        )
        hunks = sum(1 for line in raw_lines if line.startswith("@@"))
        if len(raw_lines) > _DIFF_PREVIEW_LINES:
            elided = len(raw_lines) - _DIFF_PREVIEW_LINES
            raw_lines = raw_lines[:_DIFF_PREVIEW_LINES] + [
                f"\n... ({elided} more diff lines elided) ...\n"
            ]
        return "".join(raw_lines), added, removed, hunks

    @classmethod
    def _build_diff_summary(
        cls,
        *,
        path: Path,
        original: str,
        modified: str,
        change_count: int,
    ) -> str:
        plural = "s" if change_count != 1 else ""
        header = f"edited {path} ({change_count} change{plural})\n"
        return header + cls._build_diff(
            path=path,
            original=original,
            modified=modified,
        )

    @staticmethod
    def _plan_from_context(call: ToolCall, context: TurnContext) -> FileEditPlan | None:
        plans = context.metadata.get("_tool_permission_preview_plans")
        if not isinstance(plans, dict):
            return None
        plan = plans.pop(call.id, None)
        return plan if isinstance(plan, FileEditPlan) else None


def _error(
    call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["FileEditPlan", "FileEditTool"]
