"""Workspace browser service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    path: str
    name: str
    kind: str
    size_bytes: int | None = None
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceTree:
    root: str
    path: str
    parent_path: str | None
    entries: list[WorkspaceEntry] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WorkspaceRootInfo:
    root: str
    name: str
    exists: bool
    readable: bool
    git_root: str | None = None
    is_git: bool = False
    recent_roots: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    root: str
    path: str
    name: str
    content: str
    size_bytes: int
    updated_at: float
    language: str
    mime_type: str | None = None
    truncated: bool = False
    binary: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceSearchResult:
    root: str
    query: str
    entries: list[WorkspaceEntry] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WorkspaceGitFile:
    path: str
    status: str
    index_status: str
    worktree_status: str
    staged: bool = False
    unstaged: bool = False
    untracked: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceGitStatus:
    root: str
    git_root: str | None
    available: bool
    branch: str | None = None
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    clean: bool = True
    files: list[WorkspaceGitFile] = field(default_factory=list)
    message: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceGitDiff:
    root: str
    path: str | None
    diff: str
    staged: bool = False
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceCheckpointFile:
    path: str
    exists: bool
    size_bytes: int = 0
    binary: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceCheckpoint:
    checkpoint_id: str
    label: str | None
    created_at: float
    root: str
    files: list[WorkspaceCheckpointFile] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WorkspaceCheckpointList:
    root: str
    checkpoints: list[WorkspaceCheckpoint] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WorkspaceChange:
    change_id: str
    path: str
    status: str
    source: str
    staged: bool = False
    unstaged: bool = False
    untracked: bool = False
    binary: bool = False
    accepted: bool = False
    rejected: bool = False
    conflict: bool = False
    checkpoint_available: bool = False
    additions: int = 0
    removals: int = 0
    hunks: int = 0
    current_hash: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceChangeList:
    root: str
    git_root: str | None
    available: bool
    changes: list[WorkspaceChange] = field(default_factory=list)
    message: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceChangeActionResult:
    root: str
    action: str
    paths: list[str]
    status: WorkspaceGitStatus
    checkpoint_id: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceChangeVerificationResult:
    root: str
    paths: list[str]
    status: str
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str | None = None


__all__ = [
    "WorkspaceChange",
    "WorkspaceChangeActionResult",
    "WorkspaceChangeList",
    "WorkspaceChangeVerificationResult",
    "WorkspaceCheckpoint",
    "WorkspaceCheckpointFile",
    "WorkspaceCheckpointList",
    "WorkspaceEntry",
    "WorkspaceFile",
    "WorkspaceGitDiff",
    "WorkspaceGitFile",
    "WorkspaceGitStatus",
    "WorkspaceRootInfo",
    "WorkspaceSearchResult",
    "WorkspaceTree",
]
