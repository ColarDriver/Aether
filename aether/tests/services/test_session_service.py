from __future__ import annotations

from dataclasses import asdict

import pytest

from aether.cli.sessions import SessionRecord, load_session, save_session, session_file
from aether.runtime.session.plan_artifact import read_plan, write_plan
from aether.runtime.session.session_state import SessionMode, clear_mode, get_mode, set_mode
from aether.services.common import ServiceConflictError, ServiceNotFoundError
from aether.services.sessions import (
    SessionCreateRequest,
    SessionDeleteRequest,
    SessionExportRequest,
    SessionRenameRequest,
    SessionResumeRequest,
    SessionService,
    SessionUpdateRequest,
)


def _service(tmp_path, monkeypatch: pytest.MonkeyPatch) -> SessionService:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    return SessionService(session_dir=tmp_path / "sessions")


def test_create_clears_stale_plan_state_and_sets_current(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    set_mode("abc12345", SessionMode.PLAN)
    write_plan("abc12345", "old plan")

    info = service.create(
        SessionCreateRequest(
            session_id="abc12345",
            provider="openai",
            model="gpt-5",
        )
    )

    assert info.session_id == "abc12345"
    assert info.mode == "agent"
    assert get_mode("abc12345") == "agent"
    assert read_plan("abc12345") is None
    assert service.current() is not None

    clear_mode("abc12345")


def test_list_resume_update_delete_and_export(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    service.create(SessionCreateRequest(session_id="sess-a", provider="openai", model="gpt-5"))
    service.create(SessionCreateRequest(session_id="sess-b", provider="claude", model="sonnet"))

    listed = service.list(limit=1)
    assert len(listed.sessions) == 1

    resumed = service.resume(SessionResumeRequest("sess-a"))
    assert resumed.session_id == "sess-a"
    assert resumed.info.provider == "openai"

    updated = service.update(
        SessionUpdateRequest(
            session_id="sess-a",
            provider="codex",
            model="gpt-5.4",
            base_url="https://example.invalid",
            update_base_url=True,
            system_prompt="system",
            update_system_prompt=True,
        )
    )
    assert updated.provider == "codex"
    assert updated.model == "gpt-5.4"
    assert updated.base_url == "https://example.invalid"
    assert updated.system_prompt == "system"

    exported = service.export(SessionExportRequest("sess-a"))
    assert exported.session_id == "sess-a"
    assert exported.data["provider"] == "codex"

    set_mode("sess-a", SessionMode.PLAN)
    write_plan("sess-a", "# plan\n")

    assert session_file("sess-a", base=tmp_path / "sessions").is_file()
    assert service.delete(SessionDeleteRequest("sess-a")) is True
    assert get_mode("sess-a") == "agent"
    assert read_plan("sess-a") is None
    assert not session_file("sess-a", base=tmp_path / "sessions").exists()
    assert not (tmp_path / "sessions" / ".deleted").exists()
    assert service.delete(SessionDeleteRequest("sess-a")) is False
    with pytest.raises(ServiceNotFoundError):
        service.resume(SessionResumeRequest("sess-a"))


def test_delete_removes_storage_file_and_late_run_persist_cannot_recreate_it(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    service.create(SessionCreateRequest(session_id="sess-a", provider="openai", model="gpt-5"))

    assert session_file("sess-a", base=tmp_path / "sessions").is_file()
    assert service.delete(SessionDeleteRequest("sess-a")) is True
    assert not session_file("sess-a", base=tmp_path / "sessions").exists()
    assert service.list().sessions == []

    with pytest.raises(ServiceNotFoundError):
        service.persist_run_result("sess-a", messages=[{"role": "assistant", "content": "done"}])
    assert not session_file("sess-a", base=tmp_path / "sessions").exists()


def test_delete_removes_legacy_filename_with_matching_embedded_session_id(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    record = SessionRecord.new(session_id="sess-legacy", provider="openai", model="gpt-5")
    saved = save_session(record, base=tmp_path / "sessions")
    legacy_path = tmp_path / "sessions" / "legacy-name.json"
    saved.rename(legacy_path)

    assert service.delete(SessionDeleteRequest("sess-legacy")) is True
    assert not legacy_path.exists()
    assert service.list().sessions == []
    with pytest.raises(ServiceNotFoundError):
        service.resume(SessionResumeRequest("sess-legacy"))


def test_resume_unique_prefix_and_ambiguous_prefix(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    service.create(SessionCreateRequest(session_id="abcdef", provider="openai", model="gpt-5"))
    service.create(SessionCreateRequest(session_id="abcxyz", provider="openai", model="gpt-5"))
    service.create(SessionCreateRequest(session_id="unique", provider="openai", model="gpt-5"))

    assert service.resume(SessionResumeRequest("uniq")).session_id == "unique"
    with pytest.raises(ServiceConflictError):
        service.resume(SessionResumeRequest("abc"))


def test_rename_rejects_existing_target(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    service.create(SessionCreateRequest(session_id="old", provider="openai", model="gpt-5"))
    service.create(SessionCreateRequest(session_id="new", provider="openai", model="gpt-5"))

    with pytest.raises(ServiceConflictError):
        service.rename(SessionRenameRequest(session_id="old", new_session_id="new"))

    renamed = service.rename(SessionRenameRequest(session_id="old", new_session_id="renamed"))
    assert renamed.session_id == "renamed"


def test_transcript_normalizes_messages_and_malformed_tool_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    record = SessionRecord.new(session_id="with-msgs", provider="openai", model="gpt-5")
    record.messages = [
        {"role": "weird", "content": "coerced"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":"},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "grep", "arguments": {"pattern": "x"}},
                },
            ],
        },
        {
            "role": "tool",
            "content": "failed",
            "tool_call_id": "call-1",
            "is_error": True,
            "metadata": {"kind": "test"},
        },
    ]
    save_session(record, base=tmp_path / "sessions")

    transcript = service.transcript("with-msgs")

    assert transcript[0].role == "user"
    assert transcript[0].text == "coerced"
    assert transcript[1].tool_calls[0].arguments == {"__raw__": "{\"path\":"}
    assert transcript[1].tool_calls[1].arguments == {"pattern": "x"}
    assert transcript[2].is_error is True
    assert transcript[2].metadata == {"kind": "test"}
    assert asdict(transcript[2])["tool_call_id"] == "call-1"


def test_transcript_extracts_text_and_attachments_from_content_parts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    record = SessionRecord.new(session_id="with-parts", provider="openai", model="gpt-5")
    record.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
                {
                    "type": "file",
                    "name": "app.py",
                    "path": "src/app.py",
                },
            ],
            "metadata": {
                "displayAttachments": [
                    {"type": "file", "name": "notes.md", "path": "docs/notes.md", "lineStart": 2}
                ]
            },
        }
    ]
    save_session(record, base=tmp_path / "sessions")

    transcript = service.transcript("with-parts")

    assert transcript[0].text == "inspect this image"
    assert [attachment.type for attachment in transcript[0].attachments] == ["file", "image", "file"]
    assert transcript[0].attachments[0].name == "notes.md"
    assert transcript[0].attachments[0].line_start == 2
    assert transcript[0].attachments[1].data == "data:image/png;base64,abc"
    assert transcript[0].attachments[2].path == "src/app.py"



def test_search_and_detail_include_matching_session_messages(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, monkeypatch)
    record = SessionRecord.new(session_id="alpha-session", provider="openai", model="gpt-5")
    record.messages = [
        {"role": "user", "content": "please inspect auth flow"},
        {"role": "assistant", "content": "done"},
    ]
    record.first_user_message = "please inspect auth flow"
    save_session(record, base=tmp_path / "sessions")
    service.create(SessionCreateRequest(session_id="beta-session", provider="codex", model="gpt-5.4"))

    search = service.search("auth")
    assert [item.session_id for item in search.sessions] == ["alpha-session"]

    detail = service.detail("alpha")
    assert detail.session_id == "alpha-session"
    assert detail.messages[0].text == "please inspect auth flow"
