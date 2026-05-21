from __future__ import annotations

import pytest

from aether.services.common import ServiceNotFoundError, ServiceValidationError
from aether.services.docs import DocsService


def test_docs_service_indexes_and_reads_markdown(tmp_path) -> None:
    docs = tmp_path / "docs"
    sprint = docs / "sprint"
    sprint.mkdir(parents=True)
    overview = sprint / "00_overview.md"
    overview.write_text("# Sprint Overview\n\nDetails", encoding="utf-8")
    readme = docs / "README.md"
    readme.write_text("# Root Docs\n", encoding="utf-8")

    service = DocsService(docs_root=docs)

    index = service.index()
    assert index.root == str(docs.resolve())
    assert index.default_path == "README.md"
    assert [item.path for item in index.documents] == ["README.md", "sprint/00_overview.md"]
    assert index.documents[1].title == "Sprint Overview"

    content = service.read("sprint/00_overview.md")
    assert content.path == "sprint/00_overview.md"
    assert content.title == "Sprint Overview"
    assert "Details" in content.content


def test_docs_service_rejects_missing_and_escaped_paths(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    service = DocsService(docs_root=docs)

    with pytest.raises(ServiceNotFoundError):
        service.read("missing.md")

    with pytest.raises(ServiceValidationError):
        service.read("../secret.md")
