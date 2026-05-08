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

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from aether.runtime.contracts import ToolCall, ToolResult, TurnContext
from aether.tools.base import ToolDescriptor, ToolExecutor


_MAX_BYTES = 1024 * 1024


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

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
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
        existed = path.exists()

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

        verb = "overwrote" if existed else "created"
        body = (
            f"{verb} {path}\n"
            f"size: {len(encoded)} bytes ({content.count(chr(10)) + (0 if not content else 1)} lines)\n"
            f"sha256: {digest}"
        )

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=body,
            is_error=False,
            metadata={
                "path": str(path),
                "size_bytes": len(encoded),
                "sha256": digest,
                "existed": existed,
            },
        )

    def _resolve_path(self, raw: Any) -> Path:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute() and self.default_cwd is not None:
            return (self.default_cwd / candidate).resolve()
        return candidate.resolve()


def _error(call: ToolCall, message: str, *, metadata: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"error: {message}",
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["WriteFileTool"]
