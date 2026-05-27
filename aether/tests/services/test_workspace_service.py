from __future__ import annotations

import subprocess

import pytest

from aether.services.common import ServiceConflictError, ServiceNotFoundError, ServiceValidationError
from aether.services.workspace import WorkspaceService


def test_workspace_service_lists_and_reads_text_files(tmp_path) -> None:
    root = tmp_path / "project"
    src = root / "src"
    src.mkdir(parents=True)
    (src / "app.ts").write_text("export const value = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n", encoding="utf-8")

    service = WorkspaceService(root=root)

    tree = service.tree()
    assert tree.root == str(root.resolve())
    assert [entry.name for entry in tree.entries] == ["src", "README.md"]

    nested = service.tree("src")
    assert nested.parent_path == ""
    assert nested.entries[0].path == "src/app.ts"

    file = service.read_file("src/app.ts")
    assert file.path == "src/app.ts"
    assert file.language == "typescript"
    assert file.content == "export const value = 1\n"
    assert file.binary is False


def test_workspace_service_searches_and_blocks_unsafe_paths(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "visible.py").write_text("print('ok')\n", encoding="utf-8")
    hidden = root / ".git"
    hidden.mkdir()
    (hidden / "config").write_text("secret", encoding="utf-8")
    service = WorkspaceService(root=root)

    result = service.search("visible")
    assert [entry.path for entry in result.entries] == ["visible.py"]

    with pytest.raises(ServiceValidationError):
        service.tree("../outside")

    with pytest.raises(ServiceValidationError):
        service.read_file(".git/config")

    with pytest.raises(ServiceValidationError):
        service.read_file(".env")

    with pytest.raises(ServiceNotFoundError):
        service.read_file("missing.py")


def test_workspace_service_marks_binary_and_truncates_large_text(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "image.png").write_bytes(b"\x89PNG\r\n")
    (root / "large.txt").write_text("abcdef", encoding="utf-8")
    service = WorkspaceService(root=root, max_file_bytes=3)

    image = service.read_file("image.png")
    assert image.binary is True
    assert image.content == ""
    assert image.mime_type == "image/png"
    assert service.mime_type("image.png") == "image/png"
    assert service.raw_file_path("image.png").name == "image.png"
    with pytest.raises(ServiceValidationError):
        service.write_file("image.png", "not an image")

    large = service.read_file("large.txt")
    assert large.truncated is True
    assert large.content == "abc"



def test_workspace_service_creates_renames_and_deletes_paths(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    service = WorkspaceService(root=root)

    created_dir = service.create_directory("src")
    assert created_dir.path == "src"
    assert (root / "src").is_dir()

    created_file = service.create_file("src/app.py", "print(1)\n")
    assert created_file.path == "src/app.py"
    assert created_file.content == "print(1)\n"

    renamed_file = service.rename_path("src/app.py", "src/main.py")
    assert renamed_file.path == "src/main.py"
    assert not (root / "src" / "app.py").exists()
    assert (root / "src" / "main.py").read_text(encoding="utf-8") == "print(1)\n"

    with pytest.raises(ServiceConflictError):
        service.create_file("src/main.py")

    with pytest.raises(ServiceConflictError):
        service.delete_path("src")

    service.delete_path("src", recursive=True)
    assert not (root / "src").exists()


def test_workspace_service_mutations_block_root_escape_and_excluded_paths(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    service = WorkspaceService(root=root)

    with pytest.raises(ServiceValidationError):
        service.create_file("../escape.txt")

    with pytest.raises(ServiceValidationError):
        service.create_directory(".git/hooks")

    with pytest.raises(ServiceValidationError):
        service.delete_path("")

    with pytest.raises(ServiceNotFoundError):
        service.rename_path("missing.txt", "next.txt")


def test_workspace_service_reports_git_status_diff_restore_and_checkpoints(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Aether Test")
    (root / "app.py").write_text("print('old')\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "initial")

    (root / "app.py").write_text("print('new')\n", encoding="utf-8")
    (root / "notes.txt").write_text("scratch\n", encoding="utf-8")
    service = WorkspaceService(root=root)

    status = service.git_status()
    assert status.available is True
    assert status.clean is False
    assert {file.path: file.status for file in status.files} == {
        "app.py": "modified",
        "notes.txt": "untracked",
    }

    diff = service.git_diff("app.py")
    assert "-print('old')" in diff.diff
    assert "+print('new')" in diff.diff

    untracked = service.git_diff("notes.txt")
    assert "new file mode" in untracked.diff
    assert "+scratch" in untracked.diff

    checkpoint = service.create_checkpoint(label="before restore")
    assert checkpoint.label == "before restore"
    assert {file.path for file in checkpoint.files} == {"app.py", "notes.txt"}
    assert service.list_checkpoints().checkpoints[0].checkpoint_id == checkpoint.checkpoint_id

    restored_status = service.git_restore("app.py")
    assert "app.py" not in {file.path for file in restored_status.files}
    assert (root / "app.py").read_text(encoding="utf-8") == "print('old')\n"

    service.restore_checkpoint(checkpoint.checkpoint_id)
    assert (root / "app.py").read_text(encoding="utf-8") == "print('new')\n"
    assert (root / "notes.txt").read_text(encoding="utf-8") == "scratch\n"


def test_workspace_service_restores_paths_since_checkpoint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Aether Test")
    (root / "app.py").write_text("print('old')\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "initial")

    (root / "app.py").write_text("print('pre-run user edit')\n", encoding="utf-8")
    (root / "scratch.txt").write_text("pre-run scratch\n", encoding="utf-8")
    service = WorkspaceService(root=root)
    checkpoint = service.create_checkpoint(label="before agent run")

    (root / "app.py").write_text("print('agent edit')\n", encoding="utf-8")
    (root / "scratch.txt").write_text("agent scratch edit\n", encoding="utf-8")
    (root / "new.py").write_text("created by agent\n", encoding="utf-8")

    status = service.restore_paths_since_checkpoint(
        checkpoint.checkpoint_id,
        ["app.py", "scratch.txt", "new.py"],
    )

    assert (root / "app.py").read_text(encoding="utf-8") == "print('pre-run user edit')\n"
    assert (root / "scratch.txt").read_text(encoding="utf-8") == "pre-run scratch\n"
    assert not (root / "new.py").exists()
    assert {file.path for file in status.files} == {"app.py", "scratch.txt"}


def test_workspace_service_accepts_and_rejects_changes_with_conflict_guard(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Aether Test")
    (root / "app.py").write_text("print('old')\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "initial")
    (root / "app.py").write_text("print('accepted')\n", encoding="utf-8")
    service = WorkspaceService(root=root)

    changes = service.changes()
    assert changes.available is True
    assert changes.changes[0].path == "app.py"
    assert changes.changes[0].additions == 1
    current_hash = changes.changes[0].current_hash

    accepted = service.accept_changes(["app.py"])
    assert accepted.action == "accepted"
    assert service.changes().changes[0].accepted is True

    (root / "app.py").write_text("print('manual edit after render')\n", encoding="utf-8")
    with pytest.raises(ServiceConflictError):
        service.reject_changes(["app.py"], expected_hashes={"app.py": current_hash or ""})

    rejected = service.reject_changes(["app.py"])
    assert rejected.action == "rejected"
    assert (root / "app.py").read_text(encoding="utf-8") == "print('old')\n"


def test_workspace_service_git_status_handles_non_git_workspace(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    service = WorkspaceService(root=root)

    status = service.git_status()

    assert status.available is False
    assert status.clean is True
    assert "not inside a git repository" in (status.message or "")


def _git(root, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
