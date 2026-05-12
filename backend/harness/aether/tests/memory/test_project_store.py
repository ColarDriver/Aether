from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from aether.memory.frontmatter import (
    parse_frontmatter_documents,
    render_frontmatter_document,
)
from aether.memory.project_store import (
    ProjectMemoryStore,
    ProjectMemoryStoreError,
    project_memory_hash,
)
from aether.memory.sanitize import scan_secrets


@pytest.fixture()
def store(tmp_path: Path) -> ProjectMemoryStore:
    return ProjectMemoryStore(tmp_path, store_root=tmp_path / ".aether" / "memory")


def test_project_store_initializes_memory_md_and_topics(store: ProjectMemoryStore) -> None:
    root = store.initialize()

    assert (root / "MEMORY.md").exists()
    assert (root / "index.json").exists()
    assert (root / "topics").is_dir()
    for topic in ("architecture", "workflows", "decisions", "pitfalls"):
        assert (root / "topics" / f"{topic}.md").exists()

    content = (root / "MEMORY.md").read_text(encoding="utf-8")
    assert "Aether Project Memory" in content
    assert "[Architecture](./topics/architecture.md)" in content


def test_project_store_appends_frontmatter_entry(store: ProjectMemoryStore) -> None:
    result = store.write_entry(
        topic="decisions",
        text="Memory injection is outbound-only.",
        kind="decision",
        tags=("memory",),
        paths=("backend/harness/aether/agents/core/agent.py",),
    )

    assert result.success
    assert result.entry_id is not None
    assert result.path is not None

    entries = store.read_entries(topic="decisions")
    assert len(entries) == 1
    assert entries[0].text == "Memory injection is outbound-only."
    assert entries[0].tags == ("memory",)
    assert entries[0].paths == ("backend/harness/aether/agents/core/agent.py",)


def test_index_rebuilds_from_markdown_when_corrupted(store: ProjectMemoryStore) -> None:
    store.write_entry(topic="decisions", text="Use markdown store.")

    store.index_path.write_text("CORRUPTED", encoding="utf-8")

    index = store.load_index()
    assert index["version"] == 1
    assert len(index["entries"]) == 1
    assert index["entries"][0]["topic"] == "decisions"


def test_project_hash_is_stable_for_same_workdir(tmp_path: Path) -> None:
    h1 = project_memory_hash(tmp_path)
    h2 = project_memory_hash(tmp_path)
    h3 = project_memory_hash(str(tmp_path))

    assert h1 == h2 == h3
    assert len(h1) == 16


def test_store_falls_back_to_home_when_project_not_writable(tmp_path: Path) -> None:
    project_dir = tmp_path / "readonly_project"
    project_dir.mkdir()
    home_dir = tmp_path / "fake_home"

    store = ProjectMemoryStore(
        project_dir,
        home_root=home_dir,
    )

    original_ensure = store._ensure_layout

    def _fail_on_project(root: Path) -> None:
        if root == store._project_root:
            raise OSError("read-only filesystem")
        original_ensure(root)

    with mock.patch.object(store, "_ensure_layout", side_effect=_fail_on_project):
        root = store.initialize()

    assert str(home_dir) in str(root)
    assert (root / "MEMORY.md").exists()


def test_atomic_write_does_not_leave_partial_topic_file(store: ProjectMemoryStore) -> None:
    store.write_entry(topic="decisions", text="Original entry.")

    original = store.topic_path("decisions").read_text(encoding="utf-8")

    with mock.patch("os.replace", side_effect=OSError("disk full")):
        result = store.write_entry(topic="decisions", text="Should fail atomically.")

    assert not result.success
    current = store.topic_path("decisions").read_text(encoding="utf-8")
    assert current == original


def test_secret_like_content_is_rejected_or_redacted(store: ProjectMemoryStore) -> None:
    cases = [
        "API key is sk-abc123def456ghi789jkl012",
        "Token: ghp_abcdefghijklmnopqrstuvwxyz",
        "Bot token: xoxb-123-456-abcdefghijklmnopq",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIE...base64...\n-----END RSA PRIVATE KEY-----",
        "AWS_SECRET_KEY=AKIAIOSFODNN7EXAMPLE",
    ]
    for text in cases:
        result = store.write_entry(
            topic="pitfalls",
            text=text,
            reject_secrets=True,
        )
        assert not result.success, f"Should reject: {text[:40]}"
        assert result.error == "secret_detected"


def test_frontmatter_roundtrip() -> None:
    metadata = {
        "id": "project-decision-20260512-001",
        "scope": "project",
        "kind": "decision",
        "confidence": "high",
        "tags": ["memory", "compaction"],
        "paths": ["backend/harness/aether/agents/core/agent.py"],
    }
    body = "Memory injection is outbound-only."

    rendered = render_frontmatter_document(metadata, body)
    parsed = parse_frontmatter_documents(rendered)

    assert len(parsed) == 1
    doc = parsed[0]
    assert doc.metadata["id"] == "project-decision-20260512-001"
    assert doc.metadata["scope"] == "project"
    assert doc.metadata["tags"] == ["memory", "compaction"]
    assert doc.body == body


def test_sanitize_redacts_multiple_secret_types() -> None:
    text = (
        "OpenAI key: sk-abc123def456ghi789jkl012\n"
        "GitHub token: ghp_abcdefghijklmnopqrstuvwxyz\n"
        "Safe content here.\n"
    )
    result = scan_secrets(text)

    assert result.redacted
    assert len(result.findings) == 2
    assert "sk-abc123" not in result.redacted_text
    assert "ghp_abcdef" not in result.redacted_text
    assert "Safe content here." in result.redacted_text
    kinds = {f.kind for f in result.findings}
    assert "openai_api_key" in kinds
    assert "github_token" in kinds


def test_write_entry_updates_existing_by_id(store: ProjectMemoryStore) -> None:
    r1 = store.write_entry(
        topic="decisions",
        text="First version.",
        entry_id="test-entry-001",
    )
    assert r1.success

    r2 = store.write_entry(
        topic="decisions",
        text="Updated version.",
        entry_id="test-entry-001",
    )
    assert r2.success

    entries = store.read_entries(topic="decisions")
    assert len(entries) == 1
    assert entries[0].text == "Updated version."
    assert entries[0].id == "test-entry-001"


def test_read_entries_sanitizes_by_default(store: ProjectMemoryStore) -> None:
    store.write_entry(
        topic="pitfalls",
        text="Safe content only.",
    )

    topic_path = store.topic_path("pitfalls")
    raw = topic_path.read_text(encoding="utf-8")
    raw += "\n---\nid: injected-secret\n---\n\nDo not share sk-abc123def456ghi789jkl012\n"
    topic_path.write_text(raw, encoding="utf-8")

    entries = store.read_entries(topic="pitfalls", sanitize=True)
    secret_entry = [e for e in entries if e.id == "injected-secret"]
    assert len(secret_entry) == 1
    assert "sk-abc123" not in secret_entry[0].text
    assert "[REDACTED:" in secret_entry[0].text
    assert secret_entry[0].redacted
