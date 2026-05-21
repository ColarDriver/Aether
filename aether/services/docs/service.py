"""Local project documentation service."""

from __future__ import annotations

from pathlib import Path

from aether.services.common import ServiceNotFoundError, ServiceValidationError
from aether.services.docs.contracts import DocContent, DocIndex, DocSummary


class DocsService:
    def __init__(self, *, docs_root: str | Path | None = None) -> None:
        self._docs_root = Path(docs_root).expanduser().resolve() if docs_root is not None else _default_docs_root()

    @property
    def docs_root(self) -> Path:
        return self._docs_root

    def index(self) -> DocIndex:
        documents = [self._summary(path) for path in self._iter_markdown_files()]
        default_path = _select_default_path(documents)
        return DocIndex(root=str(self._docs_root), default_path=default_path, documents=documents)

    def read(self, doc_path: str) -> DocContent:
        path = self._resolve_doc_path(doc_path)
        if not path.exists() or not path.is_file():
            raise ServiceNotFoundError(f"documentation file not found: {doc_path}", details={"path": doc_path})
        if path.suffix.lower() != ".md":
            raise ServiceValidationError("only markdown documentation files are readable", details={"path": doc_path})
        content = path.read_text(encoding="utf-8", errors="replace")
        stat = path.stat()
        return DocContent(
            path=_relative_doc_path(self._docs_root, path),
            title=_extract_title(content, path),
            content=content,
            size_bytes=stat.st_size,
            updated_at=stat.st_mtime,
        )

    def _iter_markdown_files(self) -> list[Path]:
        if not self._docs_root.exists() or not self._docs_root.is_dir():
            return []
        files: list[Path] = []
        for path in self._docs_root.rglob("*.md"):
            if any(part.startswith(".") for part in path.relative_to(self._docs_root).parts):
                continue
            if path.is_file():
                files.append(path)
        return sorted(files, key=lambda item: _relative_doc_path(self._docs_root, item))

    def _summary(self, path: Path) -> DocSummary:
        stat = path.stat()
        preview = path.read_text(encoding="utf-8", errors="replace")[:4096]
        return DocSummary(
            path=_relative_doc_path(self._docs_root, path),
            title=_extract_title(preview, path),
            size_bytes=stat.st_size,
            updated_at=stat.st_mtime,
        )

    def _resolve_doc_path(self, doc_path: str) -> Path:
        value = str(doc_path or "").strip().replace("\\", "/")
        if not value:
            raise ServiceValidationError("documentation path is required")
        candidate = (self._docs_root / value).resolve()
        try:
            candidate.relative_to(self._docs_root)
        except ValueError as exc:
            raise ServiceValidationError("documentation path escapes docs root", details={"path": doc_path}) from exc
        return candidate


def _default_docs_root() -> Path:
    source_root = Path(__file__).resolve().parents[3]
    candidate = source_root / "docs"
    if candidate.exists():
        return candidate
    return (Path.cwd() / "docs").resolve()


def _relative_doc_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _extract_title(content: str, path: Path) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _select_default_path(documents: list[DocSummary]) -> str | None:
    if not documents:
        return None
    preferred_names = ("README.md", "00_overview.md")
    for name in preferred_names:
        for document in documents:
            if document.path == name or document.path.endswith("/" + name):
                return document.path
    return documents[0].path


__all__ = ["DocsService"]
