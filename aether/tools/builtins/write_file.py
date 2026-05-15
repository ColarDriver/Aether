"""Built-in ``write_file`` tool — atomically write text to a file.

This is the simplest possible writer: it overwrites the destination
with the given content (creating parent directories as needed) and
returns a hash + byte count so callers can prove they wrote what they
intended.  Edits / patches belong in a future ``edit_file`` /
``str_replace`` tool — keeping this one purely a *create-or-overwrite*
operation matches the JSON schema sent to the model and avoids subtle
"partial write succeeded" failure modes.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.tools.tool_permissions import ToolPermissionPreview
from aether.tools.base import ToolDescriptor, ToolExecutor


_MAX_BYTES = 1024 * 1024
_DIFF_PREVIEW_LINES = 120


@dataclass(slots=True, frozen=True)
class WriteFilePlan:
    path: Path
    existed: bool
    old_content: str
    new_content: str
    size_bytes: int


class WriteFileTool(ToolExecutor):
    """Atomically write a text file (create or overwrite)."""

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        max_bytes: int = _MAX_BYTES,
    ) -> None:
        self.default_cwd = default_cwd
        self.max_bytes = max_bytes
        self._descriptor = ToolDescriptor(
            name="write_file",
            description=(
                "Create or overwrite a text file with the given content. "
                "Parent directories are created as needed. The write is "
                "atomic (writes to a temp file in the target directory, "
                "then renames). Returns the SHA-256 hash and byte count "
                "of the written content. Cap is "
                f"{self.max_bytes // 1024} KB."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination path (absolute or relative to cwd).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full UTF-8 text to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
            required=["path", "content"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def build_permission_preview(
        self,
        call: ToolCall,
        context: TurnContext,
    ) -> ToolPermissionPreview | ToolResult:
        plan = self.plan_write(call)
        if isinstance(plan, ToolResult):
            return plan
        # No-op short circuit: overwriting a file with identical content
        # should not prompt the user. Mirrors file_edit's old==new rejection.
        if plan.existed and plan.old_content == plan.new_content:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"no-op: {plan.path} already has identical content",
                is_error=False,
                metadata={
                    "path": str(plan.path),
                    "no_op": True,
                    "size_bytes": plan.size_bytes,
                    "existed": True,
                    "bytes_before": plan.size_bytes,
                    "bytes_after": plan.size_bytes,
                    "lines_added": 0,
                    "lines_removed": 0,
                    "hunks": 0,
                },
            )
        context.metadata.setdefault("_tool_permission_preview_plans", {})[call.id] = plan
        diff_text = self._build_diff(plan)
        body = None if diff_text else self._fallback_body(plan)
        return ToolPermissionPreview(
            title="Overwrite file" if plan.existed else "Create file",
            subtitle=str(plan.path),
            path=str(plan.path),
            diff=diff_text or None,
            body=body,
            metadata={
                "existed": plan.existed,
                "size_bytes": plan.size_bytes,
                "parent_exists": plan.path.parent.exists(),
                "old_bytes": len(plan.old_content.encode("utf-8")),
                "new_bytes": len(plan.new_content.encode("utf-8")),
            },
        )

    @staticmethod
    def _fallback_body(plan: WriteFilePlan) -> str:
        verb = "Overwrite" if plan.existed else "Create"
        line_count = plan.new_content.count("\n") + (0 if not plan.new_content else 1)
        return (
            f"{verb} {plan.path}\n"
            f"size: {plan.size_bytes} bytes ({line_count} lines)"
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        plan = self._plan_from_context(call, context)
        if plan is None:
            planned = self.plan_write(call)
            if isinstance(planned, ToolResult):
                return planned
            plan = planned
        else:
            stale = self._check_stale_preview(plan)
            if stale is not None:
                return _error(
                    call,
                    stale,
                    metadata={"path": str(plan.path), "stale_preview": True},
                )
        tracker = context.metadata.get("_diagnostic_tracker")
        if isinstance(tracker, DiagnosticTracker) and tracker.enabled:
            tracker.before_file_edited(plan.path)
        return self._apply_plan(call, plan)

    def plan_write(self, call: ToolCall) -> WriteFilePlan | ToolResult:
        args = call.arguments or {}
        raw_path = args.get("path")
        content = args.get("content")
        if not raw_path:
            return _error(call, "'path' must be a non-empty string")
        if content is None:
            return _error(call, "'content' is required (use empty string for an empty file)")
        if not isinstance(content, str):
            return _error(call, "'content' must be a string")

        encoded = content.encode("utf-8")
        if len(encoded) > self.max_bytes:
            return _error(
                call,
                (
                    f"content too large: {len(encoded)} bytes > {self.max_bytes} byte cap. "
                    "Split the write across multiple calls or use `shell` for streaming writes."
                ),
                metadata={"size_bytes": len(encoded)},
            )

        path = self._resolve_path(raw_path)
        if path.exists() and path.is_dir():
            return _error(
                call,
                f"path is a directory (cannot overwrite): {path}",
                metadata={"path": str(path)},
            )
        existed = path.exists()
        old_content = ""
        if existed:
            try:
                old_content = path.read_text(encoding="utf-8")
            except OSError as exc:
                return _error(
                    call,
                    f"could not read existing {path}: {exc}",
                    metadata={"path": str(path)},
                )
            except UnicodeDecodeError as exc:
                return _error(
                    call,
                    f"existing file is not valid UTF-8 (cannot preview safely): {exc}",
                    metadata={"path": str(path)},
                )
        return WriteFilePlan(
            path=path,
            existed=existed,
            old_content=old_content,
            new_content=content,
            size_bytes=len(encoded),
        )

    def _apply_plan(self, call: ToolCall, plan: WriteFilePlan) -> ToolResult:
        path = plan.path
        content = plan.new_content
        encoded = content.encode("utf-8")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return _error(call, f"could not create parent directory: {exc}", metadata={"path": str(path)})

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(encoded)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)
        except OSError as exc:
            return _error(call, f"could not write {path}: {exc}", metadata={"path": str(path)})

        digest = hashlib.sha256(encoded).hexdigest()

        verb = "overwrote" if plan.existed else "created"
        body = (
            f"{verb} {path}\n"
            f"size: {len(encoded)} bytes ({content.count(chr(10)) + (0 if not content else 1)} lines)\n"
            f"sha256: {digest}"
        )

        diff_text, lines_added, lines_removed, hunks = self._build_diff_with_stats(plan)
        result_metadata: dict[str, Any] = {
            "path": str(path),
            "edited_paths": [str(path)],
            "size_bytes": len(encoded),
            "sha256": digest,
            "existed": plan.existed,
            "bytes_before": len(plan.old_content.encode("utf-8")),
            "bytes_after": len(encoded),
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "hunks": hunks,
        }
        if diff_text and len(diff_text) < 4096:
            result_metadata["diff"] = diff_text
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=body,
            is_error=False,
            metadata=result_metadata,
        )

    def _resolve_path(self, raw: Any) -> Path:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()

    @staticmethod
    def _build_diff(plan: WriteFilePlan) -> str:
        text, _added, _removed, _hunks = WriteFileTool._build_diff_with_stats(plan)
        return text

    @staticmethod
    def _build_diff_with_stats(plan: WriteFilePlan) -> tuple[str, int, int, int]:
        raw_lines = list(
            difflib.unified_diff(
                plan.old_content.splitlines(keepends=True),
                plan.new_content.splitlines(keepends=True),
                fromfile=str(plan.path) if plan.existed else "/dev/null",
                tofile=str(plan.path),
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

    @staticmethod
    def _plan_from_context(call: ToolCall, context: TurnContext) -> WriteFilePlan | None:
        plans = context.metadata.get("_tool_permission_preview_plans")
        if not isinstance(plans, dict):
            return None
        plan = plans.pop(call.id, None)
        return plan if isinstance(plan, WriteFilePlan) else None

    @staticmethod
    def _check_stale_preview(plan: WriteFilePlan) -> str | None:
        if plan.existed:
            if not plan.path.exists():
                return "file changed after permission preview; retry the write"
            try:
                current = plan.path.read_text(encoding="utf-8")
            except OSError as exc:
                return f"could not re-read {plan.path}: {exc}"
            except UnicodeDecodeError as exc:
                return f"file is no longer valid UTF-8 (cannot write safely): {exc}"
            if current != plan.old_content:
                return "file changed after permission preview; retry the write"
        elif plan.path.exists():
            return "file appeared after permission preview; retry the write"
        return None


def _error(call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["WriteFilePlan", "WriteFileTool"]
