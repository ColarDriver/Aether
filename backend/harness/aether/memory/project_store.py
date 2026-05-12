"""Auditable markdown-backed project memory store."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .budget import estimate_text_tokens
from .contracts import MemoryKind, MemoryScope
from .frontmatter import parse_frontmatter_documents, render_frontmatter_document
from .sanitize import scan_secrets


DEFAULT_PROJECT_MEMORY_TOPICS = ("architecture", "workflows", "decisions", "pitfalls")
PROJECT_MEMORY_INDEX_VERSION = 1

_TOPIC_TITLES = {
    "architecture": "Architecture",
    "workflows": "Workflows",
    "decisions": "Decisions",
    "pitfalls": "Pitfalls",
}


@dataclass(slots=True, frozen=True)
class ProjectMemoryEntry:
    """One durable project memory entry."""

    id: str
    topic: str
    text: str
    kind: MemoryKind = MemoryKind.PROJECT_FACT
    scope: MemoryScope = MemoryScope.PROJECT
    created_at: str = ""
    updated_at: str = ""
    source_session: str | None = None
    confidence: str = "medium"
    tags: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProjectMemoryWriteResult:
    """Structured result for explicit project memory writes."""

    success: bool
    entry_id: str | None = None
    path: Path | None = None
    error: str | None = None
    redacted: bool = False
    secret_kinds: tuple[str, ...] = ()


class ProjectMemoryStoreError(RuntimeError):
    """Raised when the project memory store cannot complete an operation."""


class ProjectMemoryStore:
    """Markdown topic-file backed project memory store."""

    def __init__(
        self,
        working_directory: str | Path,
        *,
        store_root: str | Path | None = None,
        home_root: str | Path | None = None,
        topics: Iterable[str] = DEFAULT_PROJECT_MEMORY_TOPICS,
    ) -> None:
        self.working_directory = canonical_workdir(working_directory)
        self.project_hash = project_memory_hash(self.working_directory)
        self._explicit_root = store_root is not None
        self._home_root = (
            Path(home_root).expanduser()
            if home_root is not None
            else Path.home() / ".aether" / "projects" / "memory"
        )
        self.root = (
            Path(store_root).expanduser()
            if store_root is not None
            else self._home_root / self.project_hash
        )
        self.topics = tuple(_normalise_topic(topic) for topic in topics)
        self._lock = threading.RLock()

    @property
    def memory_md_path(self) -> Path:
        return self.root / "MEMORY.md"

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def topics_dir(self) -> Path:
        return self.root / "topics"

    def topic_path(self, topic: str) -> Path:
        return self.topics_dir / f"{_normalise_topic(topic)}.md"

    def initialize(self) -> Path:
        """Create the store layout on disk."""

        with self._lock:
            self._ensure_layout(self.root)
            return self.root

    def write_entry(
        self,
        *,
        topic: str,
        text: str,
        kind: MemoryKind | str = MemoryKind.PROJECT_FACT,
        entry_id: str | None = None,
        source_session: str | None = None,
        confidence: str = "medium",
        tags: Iterable[str] = (),
        paths: Iterable[str] = (),
        reject_secrets: bool = True,
    ) -> ProjectMemoryWriteResult:
        """Append or replace one explicitly-approved project memory entry."""

        self.initialize()
        topic_name = _normalise_topic(topic)
        if topic_name not in self.topics:
            return ProjectMemoryWriteResult(success=False, error="unknown_topic")

        scan = scan_secrets(text)
        if scan.redacted and reject_secrets:
            return ProjectMemoryWriteResult(
                success=False,
                error="secret_detected",
                redacted=True,
                secret_kinds=tuple(sorted({finding.kind for finding in scan.findings})),
            )

        now = _utc_now()
        normalized_kind = _normalise_kind(kind)
        try:
            with self._store_file_lock():
                existing = list(self.read_entries(topic=topic_name, sanitize=False))
                final_id = entry_id or self._next_entry_id_from_entries(
                    normalized_kind,
                    existing,
                )
                entry = ProjectMemoryEntry(
                    id=final_id,
                    topic=topic_name,
                    text=scan.redacted_text,
                    kind=normalized_kind,
                    created_at=now,
                    updated_at=now,
                    source_session=source_session,
                    confidence=confidence,
                    tags=tuple(str(tag) for tag in tags if str(tag).strip()),
                    paths=tuple(str(path) for path in paths if str(path).strip()),
                    redacted=scan.redacted,
                    metadata={"redacted": True} if scan.redacted else {},
                )
                next_entries: list[ProjectMemoryEntry] = []
                for old_entry in existing:
                    if old_entry.id == entry.id:
                        entry = replace(
                            entry,
                            created_at=old_entry.created_at or entry.created_at,
                        )
                    else:
                        next_entries.append(old_entry)
                next_entries.append(entry)
                self._write_topic_entries(topic_name, next_entries)
                self._rebuild_index_unlocked()
        except (OSError, ProjectMemoryStoreError) as exc:
            return ProjectMemoryWriteResult(success=False, error=str(exc))

        return ProjectMemoryWriteResult(
            success=True,
            entry_id=entry.id,
            path=self.topic_path(topic_name),
            redacted=entry.redacted,
            secret_kinds=tuple(sorted({finding.kind for finding in scan.findings})),
        )

    def read_entries(
        self,
        *,
        topic: str | None = None,
        sanitize: bool = True,
    ) -> tuple[ProjectMemoryEntry, ...]:
        """Read entries from markdown topic files."""

        self.initialize()
        topics = (topic,) if topic else self.topics
        entries: list[ProjectMemoryEntry] = []
        for raw_topic in topics:
            topic_name = _normalise_topic(raw_topic)
            path = self.topic_path(topic_name)
            if not path.exists():
                continue
            entries.extend(_parse_topic_entries(path, topic_name, sanitize=sanitize))
        return tuple(entries)

    def load_index(self) -> dict[str, Any]:
        """Load ``index.json``, rebuilding it from markdown if corrupted."""

        self.initialize()
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("version") == PROJECT_MEMORY_INDEX_VERSION:
                entries = raw.get("entries")
                if isinstance(entries, list):
                    return raw
        except (OSError, json.JSONDecodeError):
            pass
        return self.rebuild_index()

    def rebuild_index(self) -> dict[str, Any]:
        """Rebuild ``index.json`` from topic markdown files."""

        self.initialize()
        try:
            with self._store_file_lock():
                return self._rebuild_index_unlocked()
        except ProjectMemoryStoreError:
            if self.index_path.exists():
                try:
                    raw = json.loads(self.index_path.read_text(encoding="utf-8"))
                    return raw if isinstance(raw, dict) else _empty_index()
                except (OSError, json.JSONDecodeError):
                    return _empty_index()
            return _empty_index()

    def _ensure_layout(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        topics_dir = root / "topics"
        topics_dir.mkdir(parents=True, exist_ok=True)
        memory_md = root / "MEMORY.md"
        if not memory_md.exists():
            _write_text_atomic(memory_md, _render_memory_md(self.topics))
        for topic in self.topics:
            path = topics_dir / f"{topic}.md"
            if not path.exists():
                title = _TOPIC_TITLES.get(topic, topic.replace("_", " ").title())
                _write_text_atomic(path, f"# {title}\n\n")
        if not (root / "index.json").exists():
            _write_text_atomic(root / "index.json", _json_dumps(_empty_index()))

    @contextmanager
    def _store_file_lock(self) -> Iterator[None]:
        lock_path = self.root / ".lock"
        fd: int | None = None
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            yield
        except FileExistsError as exc:
            raise ProjectMemoryStoreError("lock_unavailable") from exc
        finally:
            if fd is not None:
                os.close(fd)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _write_topic_entries(self, topic: str, entries: list[ProjectMemoryEntry]) -> None:
        title = _TOPIC_TITLES.get(topic, topic.replace("_", " ").title())
        rendered = [f"# {title}\n\n"]
        for entry in entries:
            rendered.append(_render_entry(entry))
            rendered.append("\n")
        _write_text_atomic(self.topic_path(topic), "".join(rendered))

    def _rebuild_index_unlocked(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for entry in self.read_entries(sanitize=True):
            entries.append(
                {
                    "id": entry.id,
                    "topic": entry.topic,
                    "path": f"topics/{entry.topic}.md",
                    "scope": entry.scope.value,
                    "kind": entry.kind.value,
                    "tags": list(entry.tags),
                    "paths": list(entry.paths),
                    "updated_at": entry.updated_at,
                    "token_estimate": estimate_text_tokens(entry.text),
                    "redacted": entry.redacted,
                }
            )
        index = {
            "version": PROJECT_MEMORY_INDEX_VERSION,
            "updated_at": _utc_now(),
            "entries": entries,
        }
        _write_text_atomic(self.index_path, _json_dumps(index))
        return index

    def _next_entry_id_from_entries(
        self,
        kind: MemoryKind,
        entries: Iterable[ProjectMemoryEntry],
    ) -> str:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        prefix = f"project-{kind.value}-{date}"
        existing = {
            entry.id
            for entry in entries
            if entry.id.startswith(prefix)
        }
        counter = 1
        while True:
            candidate = f"{prefix}-{counter:03d}"
            if candidate not in existing:
                return candidate
            counter += 1


def canonical_workdir(path: str | Path) -> Path:
    """Return a canonical project path for hashing and store selection."""

    return Path(path).expanduser().resolve()


def project_memory_hash(path: str | Path) -> str:
    """Stable short hash for a canonical working directory."""

    canonical = str(canonical_workdir(path))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _parse_topic_entries(
    path: Path,
    topic: str,
    *,
    sanitize: bool,
) -> tuple[ProjectMemoryEntry, ...]:
    documents = parse_frontmatter_documents(path.read_text(encoding="utf-8"))
    entries: list[ProjectMemoryEntry] = []
    for document in documents:
        metadata = dict(document.metadata)
        text = document.body
        redacted = bool(metadata.get("redacted"))
        if sanitize:
            scan = scan_secrets(text)
            text = scan.redacted_text
            redacted = redacted or scan.redacted
            if scan.redacted:
                metadata["redacted"] = True
        entries.append(
            ProjectMemoryEntry(
                id=str(metadata.get("id") or f"project-memory-{uuid.uuid4().hex[:12]}"),
                topic=str(metadata.get("topic") or topic),
                text=text,
                scope=_normalise_scope(metadata.get("scope")),
                kind=_normalise_kind(metadata.get("kind")),
                created_at=str(metadata.get("created_at") or ""),
                updated_at=str(metadata.get("updated_at") or ""),
                source_session=_optional_str(metadata.get("source_session")),
                confidence=str(metadata.get("confidence") or "medium"),
                tags=tuple(str(item) for item in _as_list(metadata.get("tags"))),
                paths=tuple(str(item) for item in _as_list(metadata.get("paths"))),
                redacted=redacted,
                metadata=metadata,
            )
        )
    return tuple(entries)


def _render_entry(entry: ProjectMemoryEntry) -> str:
    metadata: dict[str, Any] = {
        "id": entry.id,
        "topic": entry.topic,
        "scope": entry.scope.value,
        "kind": entry.kind.value,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "source_session": entry.source_session,
        "confidence": entry.confidence,
        "tags": list(entry.tags),
        "paths": list(entry.paths),
    }
    if entry.redacted:
        metadata["redacted"] = True
    return render_frontmatter_document(metadata, entry.text)


def _render_memory_md(topics: tuple[str, ...]) -> str:
    lines = [
        "# Aether Project Memory",
        "",
        "This file indexes durable project memory used by Aether.",
        "",
        "## Topics",
        "",
    ]
    for topic in topics:
        title = _TOPIC_TITLES.get(topic, topic.replace("_", " ").title())
        lines.append(f"- [{title}](./topics/{topic}.md)")
    lines.append("")
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _empty_index() -> dict[str, Any]:
    return {
        "version": PROJECT_MEMORY_INDEX_VERSION,
        "updated_at": _utc_now(),
        "entries": [],
    }


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _normalise_topic(topic: str) -> str:
    cleaned = str(topic).strip().lower().replace("-", "_")
    if not cleaned or not all(char.isalnum() or char == "_" for char in cleaned):
        raise ValueError("memory topic must contain only letters, numbers, hyphen, or underscore")
    return cleaned


def _normalise_kind(value: MemoryKind | str | Any) -> MemoryKind:
    if isinstance(value, MemoryKind):
        return value
    try:
        return MemoryKind(str(value).strip().lower())
    except ValueError:
        return MemoryKind.PROJECT_FACT


def _normalise_scope(value: MemoryScope | str | Any) -> MemoryScope:
    if isinstance(value, MemoryScope):
        return value
    try:
        return MemoryScope(str(value).strip().lower())
    except ValueError:
        return MemoryScope.PROJECT


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_PROJECT_MEMORY_TOPICS",
    "PROJECT_MEMORY_INDEX_VERSION",
    "ProjectMemoryEntry",
    "ProjectMemoryStore",
    "ProjectMemoryStoreError",
    "ProjectMemoryWriteResult",
    "canonical_workdir",
    "project_memory_hash",
]
