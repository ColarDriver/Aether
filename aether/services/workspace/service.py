"""Safe local workspace browsing service."""

from __future__ import annotations

import mimetypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable, Sequence
import uuid

from aether.services.common import ServiceConflictError, ServiceExecutionError, ServiceNotFoundError, ServiceUnavailableError, ServiceValidationError
from aether.services.workspace.contracts import (
    WorkspaceChange,
    WorkspaceChangeActionResult,
    WorkspaceChangeList,
    WorkspaceChangeVerificationResult,
    WorkspaceCheckpoint,
    WorkspaceCheckpointFile,
    WorkspaceCheckpointList,
    WorkspaceEntry,
    WorkspaceFile,
    WorkspaceGitDiff,
    WorkspaceGitFile,
    WorkspaceGitStatus,
    WorkspaceRootInfo,
    WorkspaceSearchResult,
    WorkspaceTree,
)

_DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_BINARY_EXTENSIONS = {
    ".7z",
    ".avif",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".icns",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".odt",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".pyc",
    ".so",
    ".sqlite",
    ".svg",
    ".tar",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
_IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".webp",
}
_LANGUAGE_BY_EXTENSION = {
    ".css": "css",
    ".bmp": "image",
    ".gif": "image",
    ".html": "html",
    ".jpeg": "image",
    ".jpg": "image",
    ".json": "json",
    ".md": "markdown",
    ".png": "image",
    ".py": "python",
    ".sh": "shell",
    ".svg": "svg",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".yaml": "yaml",
    ".webp": "image",
    ".yml": "yaml",
}


class WorkspaceService:
    def __init__(self, *, root: str | Path | None = None, max_file_bytes: int = 200_000) -> None:
        self._root = self._validate_root(root if root is not None else _default_workspace_root())
        self._max_file_bytes = max(1, int(max_file_bytes))

    @property
    def root(self) -> Path:
        return self._root

    def root_info(self, *, recent_roots: Sequence[str] | None = None) -> WorkspaceRootInfo:
        git_root = self._git_root(raise_unavailable=False)
        return WorkspaceRootInfo(
            root=str(self._root),
            name=self._root.name or str(self._root),
            exists=self._root.exists(),
            readable=os.access(self._root, os.R_OK | os.X_OK),
            git_root=str(git_root) if git_root is not None else None,
            is_git=git_root is not None,
            recent_roots=_normalized_recent_roots(recent_roots or [], current=self._root),
        )

    def switch_root(self, path: str | Path, *, recent_roots: Sequence[str] | None = None) -> WorkspaceRootInfo:
        self._root = self._validate_root(path)
        return self.root_info(recent_roots=recent_roots)

    def tree(self, path: str = "") -> WorkspaceTree:
        directory = self._resolve(path)
        if not directory.exists() or not directory.is_dir():
            raise ServiceNotFoundError(f"workspace directory not found: {path or '.'}", details={"path": path})
        entries = [self._entry(child) for child in self._iter_children(directory)]
        relative = _relative_path(self._root, directory)
        parent = None
        if directory != self._root:
            parent = _relative_path(self._root, directory.parent)
        return WorkspaceTree(root=str(self._root), path=relative, parent_path=parent, entries=entries)

    def read_file(self, path: str) -> WorkspaceFile:
        file_path = self._resolve(path)
        if not file_path.exists() or not file_path.is_file():
            raise ServiceNotFoundError(f"workspace file not found: {path}", details={"path": path})
        if _is_binary_path(file_path):
            stat = file_path.stat()
            return WorkspaceFile(
                root=str(self._root),
                path=_relative_path(self._root, file_path),
                name=file_path.name,
                content="",
                size_bytes=stat.st_size,
                updated_at=stat.st_mtime,
                language=_language_for_path(file_path),
                mime_type=_mime_type_for_path(file_path),
                binary=True,
            )
        stat = file_path.stat()
        raw = file_path.read_bytes()[: self._max_file_bytes + 1]
        truncated = len(raw) > self._max_file_bytes
        if truncated:
            raw = raw[: self._max_file_bytes]
        if b"\x00" in raw:
            return WorkspaceFile(
                root=str(self._root),
                path=_relative_path(self._root, file_path),
                name=file_path.name,
                content="",
                size_bytes=stat.st_size,
                updated_at=stat.st_mtime,
                language=_language_for_path(file_path),
                mime_type=_mime_type_for_path(file_path),
                truncated=truncated,
                binary=True,
            )
        return WorkspaceFile(
            root=str(self._root),
            path=_relative_path(self._root, file_path),
            name=file_path.name,
            content=raw.decode("utf-8", errors="replace"),
            size_bytes=stat.st_size,
            updated_at=stat.st_mtime,
            language=_language_for_path(file_path),
            mime_type=_mime_type_for_path(file_path),
            truncated=truncated,
        )

    def raw_file_path(self, path: str) -> Path:
        file_path = self._resolve(path)
        if not file_path.exists() or not file_path.is_file():
            raise ServiceNotFoundError(f"workspace file not found: {path}", details={"path": path})
        return file_path

    def mime_type(self, path: str) -> str:
        return _mime_type_for_path(self.raw_file_path(path))

    def write_file(self, path: str, content: str) -> WorkspaceFile:
        file_path = self._resolve(path)
        if not file_path.exists() or not file_path.is_file():
            raise ServiceNotFoundError(f"workspace file not found: {path}", details={"path": path})
        current = self.read_file(path)
        if current.binary or _is_image_path(file_path):
            raise ServiceValidationError("workspace binary or image files cannot be edited", details={"path": path})
        if current.truncated:
            raise ServiceValidationError("workspace truncated files cannot be edited", details={"path": path})
        encoded = str(content).encode("utf-8")
        if len(encoded) > self._max_file_bytes:
            raise ServiceValidationError(
                "workspace file exceeds editable size limit",
                details={"path": path, "max_file_bytes": self._max_file_bytes},
            )
        file_path.write_bytes(encoded)
        return self.read_file(path)

    def create_file(self, path: str, content: str = "") -> WorkspaceFile:
        file_path = self._resolve_mutation_target(path)
        if file_path.exists():
            raise ServiceConflictError("workspace path already exists", details={"path": path})
        parent = file_path.parent
        if not parent.exists() or not parent.is_dir():
            raise ServiceNotFoundError("workspace parent directory not found", details={"path": path})
        encoded = str(content).encode("utf-8")
        if len(encoded) > self._max_file_bytes:
            raise ServiceValidationError(
                "workspace file exceeds editable size limit",
                details={"path": path, "max_file_bytes": self._max_file_bytes},
            )
        file_path.write_bytes(encoded)
        return self.read_file(_relative_path(self._root, file_path))

    def create_directory(self, path: str) -> WorkspaceEntry:
        directory = self._resolve_mutation_target(path)
        if directory.exists():
            raise ServiceConflictError("workspace path already exists", details={"path": path})
        parent = directory.parent
        if not parent.exists() or not parent.is_dir():
            raise ServiceNotFoundError("workspace parent directory not found", details={"path": path})
        directory.mkdir()
        return self._entry(directory)

    def rename_path(self, path: str, new_path: str) -> WorkspaceEntry:
        source = self._resolve_mutation_target(path)
        target = self._resolve_mutation_target(new_path)
        if not source.exists():
            raise ServiceNotFoundError("workspace path not found", details={"path": path})
        if target.exists():
            raise ServiceConflictError("workspace destination already exists", details={"path": new_path})
        if not target.parent.exists() or not target.parent.is_dir():
            raise ServiceNotFoundError("workspace destination parent directory not found", details={"path": new_path})
        if source.is_dir():
            try:
                target.relative_to(source)
            except ValueError:
                pass
            else:
                raise ServiceValidationError("workspace directory cannot be moved into itself", details={"path": path, "new_path": new_path})
        source.rename(target)
        return self._entry(target)

    def delete_path(self, path: str, *, recursive: bool = False) -> None:
        target = self._resolve_mutation_target(path)
        if not target.exists():
            raise ServiceNotFoundError("workspace path not found", details={"path": path})
        if target.is_dir():
            children = list(self._iter_children(target))
            if children and not recursive:
                raise ServiceConflictError("workspace directory is not empty", details={"path": path})
            if recursive:
                self._ensure_no_excluded_descendants(target)
                shutil.rmtree(target)
            else:
                target.rmdir()
            return
        target.unlink()

    def search(self, query: str, *, limit: int = 100) -> WorkspaceSearchResult:
        normalized_query = query.strip().lower()
        normalized_limit = max(1, min(int(limit or 100), 500))
        if not normalized_query:
            return WorkspaceSearchResult(root=str(self._root), query=query, entries=[])
        matches: list[WorkspaceEntry] = []
        for path in self._walk_files_and_dirs():
            relative = _relative_path(self._root, path).lower()
            if normalized_query not in relative:
                continue
            matches.append(self._entry(path))
            if len(matches) >= normalized_limit:
                break
        return WorkspaceSearchResult(root=str(self._root), query=query, entries=matches)

    def git_status(self) -> WorkspaceGitStatus:
        git_root = self._git_root(raise_unavailable=False)
        if git_root is None:
            return WorkspaceGitStatus(
                root=str(self._root),
                git_root=None,
                available=False,
                clean=True,
                message="Workspace is not inside a git repository.",
            )

        branch = self._git_output(["rev-parse", "--abbrev-ref", "HEAD"], allow_failure=True)
        if branch == "HEAD":
            branch = self._git_output(["rev-parse", "--short", "HEAD"], allow_failure=True) or "detached"
        upstream = self._git_output(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], allow_failure=True)
        ahead = 0
        behind = 0
        if upstream:
            counts = self._git_output(["rev-list", "--left-right", "--count", "HEAD...@{u}"], allow_failure=True)
            if counts:
                parts = counts.split()
                if len(parts) == 2:
                    ahead = _parse_int(parts[0])
                    behind = _parse_int(parts[1])

        status_output = self._git_output(["status", "--porcelain=v1", "--untracked-files=all", "--", "."], allow_failure=False)
        files = [
            file
            for line in status_output.splitlines()
            if (file := self._git_file_from_status_line(line, git_root)) is not None
        ]
        return WorkspaceGitStatus(
            root=str(self._root),
            git_root=str(git_root),
            available=True,
            branch=branch or None,
            upstream=upstream or None,
            ahead=ahead,
            behind=behind,
            clean=len(files) == 0,
            files=files,
            message=None,
        )

    def git_diff(self, path: str | None = None, *, staged: bool = False, max_bytes: int = 400_000) -> WorkspaceGitDiff:
        git_root = self._git_root(raise_unavailable=True)
        assert git_root is not None
        pathspec = "."
        normalized_path: str | None = None
        if path is not None and str(path).strip():
            target = self._resolve_mutation_target(str(path))
            normalized_path = _relative_path(self._root, target)
            pathspec = _relative_path(git_root, target)
            status = self.git_status()
            file_status = next((item for item in status.files if item.path == normalized_path), None)
            if file_status and file_status.untracked and not staged:
                diff = self._untracked_file_diff(target, normalized_path, max_bytes=max_bytes)
                return WorkspaceGitDiff(
                    root=str(self._root),
                    path=normalized_path,
                    diff=diff.text,
                    staged=False,
                    truncated=diff.truncated,
                )
        args = ["diff", "--no-ext-diff"]
        if staged:
            args.append("--cached")
        args.extend(["--", pathspec])
        diff_result = self._git_output(args, allow_failure=False, max_bytes=max_bytes + 1)
        truncated = len(diff_result.encode("utf-8", errors="replace")) > max_bytes
        if truncated:
            diff_result = diff_result.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="replace")
        return WorkspaceGitDiff(
            root=str(self._root),
            path=normalized_path,
            diff=diff_result,
            staged=staged,
            truncated=truncated,
        )

    def git_restore(self, path: str) -> WorkspaceGitStatus:
        git_root = self._git_root(raise_unavailable=True)
        assert git_root is not None
        target = self._resolve_mutation_target(path)
        normalized_path = _relative_path(self._root, target)
        pathspec = _relative_path(git_root, target)
        status = self.git_status()
        file_status = next((item for item in status.files if item.path == normalized_path), None)
        if file_status is None:
            raise ServiceValidationError("workspace path has no git changes", details={"path": normalized_path})
        if file_status.untracked:
            if target.is_dir():
                self._ensure_no_excluded_descendants(target)
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            return self.git_status()
        self._run_git(["restore", "--staged", "--worktree", "--", pathspec], check=True)
        return self.git_status()

    def changes(self) -> WorkspaceChangeList:
        status = self.git_status()
        if not status.available:
            return WorkspaceChangeList(
                root=status.root,
                git_root=status.git_root,
                available=False,
                message=status.message,
            )
        accepted = self._load_accepted_changes()
        checkpoint_paths = {
            file.path
            for checkpoint in self.list_checkpoints().checkpoints[:1]
            for file in checkpoint.files
        }
        changes: list[WorkspaceChange] = []
        for file in status.files:
            current_hash = self._workspace_path_hash(file.path)
            diff_stats = self._diff_stats_for_path(file.path)
            accepted_hash = accepted.get(file.path, {}).get("hash")
            target = self._resolve_mutation_target(file.path)
            changes.append(
                WorkspaceChange(
                    change_id=file.path,
                    path=file.path,
                    status=file.status,
                    source="git",
                    staged=file.staged,
                    unstaged=file.unstaged,
                    untracked=file.untracked,
                    binary=target.exists() and target.is_file() and _is_binary_path(target),
                    accepted=bool(accepted_hash and accepted_hash == current_hash),
                    checkpoint_available=file.path in checkpoint_paths,
                    additions=diff_stats.additions,
                    removals=diff_stats.removals,
                    hunks=diff_stats.hunks,
                    current_hash=current_hash,
                )
            )
        return WorkspaceChangeList(
            root=status.root,
            git_root=status.git_root,
            available=True,
            changes=changes,
        )

    def accept_changes(self, paths: Sequence[str]) -> WorkspaceChangeActionResult:
        requested_paths = self._normalize_change_paths(paths)
        accepted = self._load_accepted_changes()
        now = time.time()
        for path in requested_paths:
            accepted[path] = {
                "hash": self._workspace_path_hash(path),
                "accepted_at": now,
            }
        self._save_accepted_changes(accepted)
        status = self.git_status()
        return WorkspaceChangeActionResult(
            root=str(self._root),
            action="accepted",
            paths=requested_paths,
            status=status,
            message="Accepted " + str(len(requested_paths)) + " workspace change" + ("" if len(requested_paths) == 1 else "s") + ".",
        )

    def reject_changes(
        self,
        paths: Sequence[str],
        *,
        checkpoint_id: str | None = None,
        expected_hashes: dict[str, str] | None = None,
    ) -> WorkspaceChangeActionResult:
        requested_paths = self._normalize_change_paths(paths)
        expected_hashes = expected_hashes or {}
        for path in requested_paths:
            expected = expected_hashes.get(path)
            current = self._workspace_path_hash(path)
            if expected and expected != current:
                raise ServiceConflictError(
                    "workspace change conflict",
                    details={
                        "path": path,
                        "expected_hash": expected,
                        "current_hash": current,
                    },
                )
        if checkpoint_id:
            status = self.restore_paths_since_checkpoint(checkpoint_id, requested_paths)
        else:
            for path in requested_paths:
                self.git_restore(path)
            status = self.git_status()
        self._clear_accepted_changes(requested_paths)
        return WorkspaceChangeActionResult(
            root=str(self._root),
            action="rejected",
            paths=requested_paths,
            checkpoint_id=checkpoint_id,
            status=status,
            message="Rejected " + str(len(requested_paths)) + " workspace change" + ("" if len(requested_paths) == 1 else "s") + ".",
        )

    def verify_changes(
        self,
        paths: Sequence[str],
        *,
        command: Sequence[str] | None = None,
        timeout_seconds: float = 120.0,
    ) -> WorkspaceChangeVerificationResult:
        requested_paths = self._normalize_change_paths(paths)
        if command is None:
            return WorkspaceChangeVerificationResult(
                root=str(self._root),
                paths=requested_paths,
                status="skipped",
                message="No verification command was provided.",
            )
        command_list = _normalize_command(command)
        try:
            completed = subprocess.run(
                command_list,
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return WorkspaceChangeVerificationResult(
                root=str(self._root),
                paths=requested_paths,
                status="timeout",
                command=command_list,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                message="Verification command timed out.",
            )
        except OSError as exc:
            raise ServiceExecutionError(
                "verification command failed to start",
                details={"command": command_list, "error": str(exc)},
            ) from exc
        return WorkspaceChangeVerificationResult(
            root=str(self._root),
            paths=requested_paths,
            status="passed" if completed.returncode == 0 else "failed",
            command=command_list,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            message="Verification passed." if completed.returncode == 0 else "Verification failed.",
        )

    def list_checkpoints(self) -> WorkspaceCheckpointList:
        root = self._checkpoint_root()
        checkpoints: list[WorkspaceCheckpoint] = []
        if root.exists():
            for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
                checkpoint = self._read_checkpoint_manifest(manifest_path)
                if checkpoint is not None:
                    checkpoints.append(checkpoint)
        return WorkspaceCheckpointList(root=str(self._root), checkpoints=checkpoints)

    def create_checkpoint(self, *, label: str | None = None) -> WorkspaceCheckpoint:
        status = self.git_status()
        if not status.available:
            raise ServiceUnavailableError(status.message or "Workspace is not inside a git repository.")
        checkpoint_id = time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        checkpoint_dir = self._checkpoint_root() / checkpoint_id
        files_dir = checkpoint_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=False)

        checkpoint_files: list[WorkspaceCheckpointFile] = []
        for changed in status.files:
            target = self._resolve_mutation_target(changed.path)
            stored = files_dir / changed.path
            if not target.exists():
                checkpoint_files.append(WorkspaceCheckpointFile(path=changed.path, exists=False))
                continue
            if not target.is_file():
                continue
            raw = target.read_bytes()
            stored.parent.mkdir(parents=True, exist_ok=True)
            stored.write_bytes(raw)
            checkpoint_files.append(
                WorkspaceCheckpointFile(
                    path=changed.path,
                    exists=True,
                    size_bytes=len(raw),
                    binary=_looks_binary(raw) or _is_binary_path(target),
                )
            )

        checkpoint = WorkspaceCheckpoint(
            checkpoint_id=checkpoint_id,
            label=label.strip() if isinstance(label, str) and label.strip() else None,
            created_at=time.time(),
            root=str(self._root),
            files=checkpoint_files,
        )
        self._write_checkpoint_manifest(checkpoint_dir / "manifest.json", checkpoint)
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> WorkspaceCheckpoint:
        normalized = _require_checkpoint_id(checkpoint_id)
        checkpoint_dir = self._checkpoint_root() / normalized
        manifest_path = checkpoint_dir / "manifest.json"
        checkpoint = self._read_checkpoint_manifest(manifest_path)
        if checkpoint is None:
            raise ServiceNotFoundError("workspace checkpoint not found", details={"checkpoint_id": normalized})
        files_dir = checkpoint_dir / "files"
        for file in checkpoint.files:
            target = self._resolve_mutation_target(file.path)
            if not file.exists:
                if target.exists():
                    if target.is_dir():
                        self._ensure_no_excluded_descendants(target)
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                continue
            stored = files_dir / file.path
            if not stored.exists() or not stored.is_file():
                raise ServiceNotFoundError(
                    "workspace checkpoint file is missing",
                    details={"checkpoint_id": normalized, "path": file.path},
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stored.read_bytes())
        return checkpoint

    def restore_paths_since_checkpoint(self, checkpoint_id: str, paths: Sequence[str]) -> WorkspaceGitStatus:
        normalized = _require_checkpoint_id(checkpoint_id)
        checkpoint_dir = self._checkpoint_root() / normalized
        manifest_path = checkpoint_dir / "manifest.json"
        checkpoint = self._read_checkpoint_manifest(manifest_path)
        if checkpoint is None:
            raise ServiceNotFoundError("workspace checkpoint not found", details={"checkpoint_id": normalized})
        requested_paths = _unique_paths(
            _relative_path(self._root, self._resolve_mutation_target(path))
            for path in paths
            if isinstance(path, str) and path.strip()
        )
        if not requested_paths:
            raise ServiceValidationError("at least one workspace path is required", details={"paths": list(paths)})

        checkpoint_files = {file.path: file for file in checkpoint.files}
        for path in requested_paths:
            checkpoint_file = checkpoint_files.get(path)
            if checkpoint_file is not None:
                self._restore_checkpoint_file(checkpoint_dir, normalized, checkpoint_file)
                continue
            try:
                self.git_restore(path)
            except ServiceValidationError as exc:
                if "no git changes" not in str(exc):
                    raise
        return self.git_status()

    def _normalize_change_paths(self, paths: Sequence[str]) -> list[str]:
        normalized = _unique_paths(
            _relative_path(self._root, self._resolve_mutation_target(path))
            for path in paths
            if isinstance(path, str) and path.strip()
        )
        if not normalized:
            raise ServiceValidationError("at least one workspace path is required", details={"paths": list(paths)})
        status = self.git_status()
        changed_paths = {file.path for file in status.files}
        missing = [path for path in normalized if path not in changed_paths]
        if missing:
            raise ServiceValidationError("workspace path has no tracked changes", details={"paths": missing})
        return normalized

    def _resolve(self, path: str) -> Path:
        value = str(path or "").strip().replace("\\", "/")
        if value in {"", "."}:
            return self._root
        candidate = (self._root / value).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise ServiceValidationError("workspace path escapes root", details={"path": path}) from exc
        if _has_excluded_part(candidate, self._root):
            raise ServiceValidationError("workspace path is excluded", details={"path": path})
        return candidate

    def _validate_root(self, path: str | Path) -> Path:
        try:
            root = Path(path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ServiceValidationError("workspace root is invalid", details={"path": str(path)}) from exc
        if not root.exists():
            raise ServiceNotFoundError("workspace root not found", details={"path": str(path)})
        if not root.is_dir():
            raise ServiceValidationError("workspace root must be a directory", details={"path": str(path)})
        if not os.access(root, os.R_OK | os.X_OK):
            raise ServiceValidationError("workspace root is not readable", details={"path": str(root)})
        return root

    def _resolve_mutation_target(self, path: str) -> Path:
        value = str(path or "").strip().replace("\\", "/")
        if value in {"", "."} or value.endswith("/"):
            raise ServiceValidationError("workspace path must name a file or directory", details={"path": path})
        return self._resolve(value)

    def _ensure_no_excluded_descendants(self, directory: Path) -> None:
        for child in directory.rglob("*"):
            if _has_excluded_part(child, self._root):
                raise ServiceValidationError("workspace directory contains excluded paths", details={"path": _relative_path(self._root, child)})

    def _iter_children(self, directory: Path) -> Iterable[Path]:
        try:
            children = [child for child in directory.iterdir() if not _is_excluded_child(child)]
        except OSError as exc:
            raise ServiceValidationError("workspace directory is not readable", details={"path": str(directory)}) from exc
        return sorted(children, key=lambda child: (not child.is_dir(), child.name.lower()))

    def _walk_files_and_dirs(self) -> Iterable[Path]:
        if not self._root.exists():
            return []
        out: list[Path] = []
        for path in self._root.rglob("*"):
            if _has_excluded_part(path, self._root):
                continue
            out.append(path)
        return sorted(out, key=lambda item: _relative_path(self._root, item).lower())

    def _entry(self, path: Path) -> WorkspaceEntry:
        stat = path.stat()
        return WorkspaceEntry(
            path=_relative_path(self._root, path),
            name=path.name,
            kind="directory" if path.is_dir() else "file",
            size_bytes=None if path.is_dir() else stat.st_size,
            updated_at=stat.st_mtime,
        )

    def _git_root(self, *, raise_unavailable: bool) -> Path | None:
        try:
            result = self._run_git(["rev-parse", "--show-toplevel"], check=True)
        except ServiceErrorCompat as exc:
            if raise_unavailable:
                raise ServiceUnavailableError("Workspace is not inside a git repository.") from exc.error
            return None
        root = Path(result.stdout.strip()).resolve()
        try:
            self._root.relative_to(root)
        except ValueError:
            if raise_unavailable:
                raise ServiceUnavailableError("Workspace git root does not contain the configured workspace root.")
            return None
        return root

    def _git_output(self, args: list[str], *, allow_failure: bool, max_bytes: int | None = None) -> str:
        try:
            result = self._run_git(args, check=not allow_failure)
        except ServiceErrorCompat as exc:
            if allow_failure:
                return ""
            raise exc.error
        output = result.stdout
        if max_bytes is not None and len(output.encode("utf-8", errors="replace")) > max_bytes:
            return output.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="replace")
        return output.strip("\n")

    def _run_git(self, args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(self._root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise ServiceErrorCompat(ServiceUnavailableError("git executable is not available.")) from exc
        except subprocess.TimeoutExpired as exc:
            raise ServiceErrorCompat(ServiceExecutionError("git command timed out.", details={"args": args})) from exc
        if check and result.returncode != 0:
            message = (result.stderr or result.stdout or "git command failed").strip()
            raise ServiceErrorCompat(
                ServiceExecutionError(message, details={"args": args, "returncode": result.returncode})
            )
        return result

    def _git_file_from_status_line(self, line: str, git_root: Path) -> WorkspaceGitFile | None:
        if len(line) < 4:
            return None
        status = line[:2]
        raw_path = line[3:].strip()
        if " -> " in raw_path:
            raw_path = raw_path.rsplit(" -> ", 1)[1]
        raw_path = raw_path.strip('"')
        full_path = (git_root / raw_path).resolve()
        try:
            workspace_path = _relative_path(self._root, full_path)
        except ValueError:
            return None
        index_status = status[0]
        worktree_status = status[1]
        return WorkspaceGitFile(
            path=workspace_path,
            status=_git_status_label(index_status, worktree_status),
            index_status=index_status,
            worktree_status=worktree_status,
            staged=index_status not in {" ", "?"},
            unstaged=worktree_status not in {" ", "?"},
            untracked=status == "??",
        )

    def _untracked_file_diff(self, target: Path, path: str, *, max_bytes: int) -> "_DiffText":
        if not target.exists() or not target.is_file():
            return _DiffText(text="", truncated=False)
        raw = target.read_bytes()
        truncated = len(raw) > max_bytes
        raw = raw[:max_bytes]
        if _looks_binary(raw) or _is_binary_path(target):
            return _DiffText(
                text="\n".join(
                    [
                        f"diff --git a/{path} b/{path}",
                        "new file mode 100644",
                        "--- /dev/null",
                        f"+++ b/{path}",
                        "@@ -0,0 +1 @@",
                        "+[binary file]",
                    ]
                ),
                truncated=truncated,
            )
        content = raw.decode("utf-8", errors="replace")
        lines = content.splitlines()
        header = [
            f"diff --git a/{path} b/{path}",
            "new file mode 100644",
            "--- /dev/null",
            f"+++ b/{path}",
            f"@@ -0,0 +1,{max(1, len(lines))} @@",
        ]
        return _DiffText(text="\n".join([*header, *("+" + line for line in lines)]), truncated=truncated)

    def _checkpoint_root(self) -> Path:
        base = Path(os.environ.get("AETHER_HOME", str(Path.home() / ".aether"))).expanduser()
        return base / "workspace_checkpoints" / _checkpoint_root_key(self._root)

    def _change_state_path(self) -> Path:
        base = Path(os.environ.get("AETHER_HOME", str(Path.home() / ".aether"))).expanduser()
        return base / "workspace_changes" / _checkpoint_root_key(self._root) / "accepted.json"

    def _workspace_path_hash(self, path: str) -> str | None:
        target = self._resolve_mutation_target(path)
        if not target.exists() or not target.is_file():
            return None
        digest = hashlib.sha256()
        with target.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _diff_stats_for_path(self, path: str) -> "_DiffStats":
        try:
            diff = self.git_diff(path=path).diff
        except Exception:
            return _DiffStats()
        additions = 0
        removals = 0
        hunks = 0
        for line in diff.splitlines():
            if line.startswith("@@"):
                hunks += 1
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                removals += 1
        return _DiffStats(additions=additions, removals=removals, hunks=hunks)

    def _load_accepted_changes(self) -> dict[str, dict[str, object]]:
        path = self._change_state_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        changes = payload.get("accepted")
        if not isinstance(changes, dict):
            return {}
        return {
            key: value
            for key, value in changes.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def _save_accepted_changes(self, changes: dict[str, dict[str, object]]) -> None:
        path = self._change_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "accepted": changes}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _clear_accepted_changes(self, paths: Sequence[str]) -> None:
        accepted = self._load_accepted_changes()
        changed = False
        for path in paths:
            changed = accepted.pop(path, None) is not None or changed
        if changed:
            self._save_accepted_changes(accepted)

    def _restore_checkpoint_file(
        self,
        checkpoint_dir: Path,
        checkpoint_id: str,
        file: WorkspaceCheckpointFile,
    ) -> None:
        target = self._resolve_mutation_target(file.path)
        if not file.exists:
            if target.exists():
                if target.is_dir():
                    self._ensure_no_excluded_descendants(target)
                    shutil.rmtree(target)
                else:
                    target.unlink()
            return
        stored = checkpoint_dir / "files" / file.path
        if not stored.exists() or not stored.is_file():
            raise ServiceNotFoundError(
                "workspace checkpoint file is missing",
                details={"checkpoint_id": checkpoint_id, "path": file.path},
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stored.read_bytes())

    def _read_checkpoint_manifest(self, manifest_path: Path) -> WorkspaceCheckpoint | None:
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        files = []
        for item in payload.get("files", []):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            files.append(
                WorkspaceCheckpointFile(
                    path=item["path"],
                    exists=bool(item.get("exists")),
                    size_bytes=_parse_int(item.get("size_bytes")),
                    binary=bool(item.get("binary")),
                )
            )
        checkpoint_id = payload.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            checkpoint_id = manifest_path.parent.name
        label = payload.get("label")
        return WorkspaceCheckpoint(
            checkpoint_id=checkpoint_id,
            label=label if isinstance(label, str) and label else None,
            created_at=float(payload.get("created_at") or 0),
            root=str(payload.get("root") or self._root),
            files=files,
        )

    def _write_checkpoint_manifest(self, manifest_path: Path, checkpoint: WorkspaceCheckpoint) -> None:
        manifest_path.write_text(json.dumps(asdict(checkpoint), indent=2, ensure_ascii=False), encoding="utf-8")


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class _DiffText:
    text: str
    truncated: bool = False


@dataclass(slots=True)
class _DiffStats:
    additions: int = 0
    removals: int = 0
    hunks: int = 0


class ServiceErrorCompat(Exception):
    def __init__(self, error: ServiceUnavailableError | ServiceExecutionError) -> None:
        super().__init__(error.message)
        self.error = error


def _relative_path(root: Path, path: Path) -> str:
    if path == root:
        return ""
    return path.relative_to(root).as_posix()


def _has_excluded_part(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in _DEFAULT_EXCLUDES or part.startswith(".") and part not in {".env.example"} for part in parts)


def _is_excluded_child(path: Path) -> bool:
    return path.name in _DEFAULT_EXCLUDES or (path.name.startswith(".") and path.name not in {".env.example"})


def _is_binary_path(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _mime_type_for_path(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() in _IMAGE_EXTENSIONS:
        return "image/" + path.suffix.lower().lstrip(".")
    return "application/octet-stream" if _is_binary_path(path) else "text/plain"


def _language_for_path(path: Path) -> str:
    return _LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "text")


def _git_status_label(index_status: str, worktree_status: str) -> str:
    if index_status == "?" and worktree_status == "?":
        return "untracked"
    if "U" in {index_status, worktree_status}:
        return "conflict"
    if index_status == "A" or worktree_status == "A":
        return "added"
    if index_status == "D" or worktree_status == "D":
        return "deleted"
    if index_status == "R" or worktree_status == "R":
        return "renamed"
    if index_status == "C" or worktree_status == "C":
        return "copied"
    if index_status == "M" or worktree_status == "M":
        return "modified"
    return "changed"


def _looks_binary(raw: bytes) -> bool:
    return b"\x00" in raw


def _parse_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _require_checkpoint_id(value: str) -> str:
    text = str(value or "").strip()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ServiceValidationError("checkpoint_id is invalid", details={"checkpoint_id": value})
    return text


def _normalize_command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)):
        raise ServiceValidationError("verification command must be an array of strings")
    out = [part.strip() for part in command if isinstance(part, str) and part.strip()]
    if not out:
        raise ServiceValidationError("verification command must not be empty")
    return out


def _unique_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _normalized_recent_roots(paths: Sequence[str], *, current: Path) -> list[str]:
    seen: set[str] = {str(current)}
    out = [str(current)]
    for path in paths:
        if not isinstance(path, str) or not path.strip():
            continue
        try:
            resolved = str(Path(path).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out[:12]


def _checkpoint_root_key(root: Path) -> str:
    safe = root.as_posix().strip("/").replace("/", "__").replace(":", "_")
    return safe or "workspace"


__all__ = ["WorkspaceService"]
