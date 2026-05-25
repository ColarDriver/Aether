from __future__ import annotations

import pytest

from aether.services.common import ServiceNotFoundError, ServiceValidationError
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

