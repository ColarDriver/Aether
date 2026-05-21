"""Safe local workspace browsing service."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from aether.services.common import ServiceNotFoundError, ServiceValidationError
from aether.services.workspace.contracts import (
    WorkspaceEntry,
    WorkspaceFile,
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
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".sqlite",
    ".tar",
    ".webp",
    ".zip",
}
_LANGUAGE_BY_EXTENSION = {
    ".css": "css",
    ".html": "html",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".sh": "shell",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".yaml": "yaml",
    ".yml": "yaml",
}


class WorkspaceService:
    def __init__(self, *, root: str | Path | None = None, max_file_bytes: int = 200_000) -> None:
        self._root = Path(root).expanduser().resolve() if root is not None else _default_workspace_root()
        self._max_file_bytes = max(1, int(max_file_bytes))

    @property
    def root(self) -> Path:
        return self._root

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
            truncated=truncated,
        )

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


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _language_for_path(path: Path) -> str:
    return _LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "text")


__all__ = ["WorkspaceService"]
