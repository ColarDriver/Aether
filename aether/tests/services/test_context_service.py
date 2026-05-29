from __future__ import annotations

from aether.cli.sessions import SessionRecord, load_session, save_session
from aether.runtime.context import CompressionResult
from aether.services.context import ContextCompressRequest, ContextEstimateRequest, ContextService
from aether.services.sessions import SessionService


class _FakeCompressionService:
    def compress(self, request):
        return CompressionResult(
            messages=[{"role": "user", "content": "compressed summary"}],
            status="compressed",
            metadata={
                "status": "compressed",
                "source_message_count": len(request.messages),
                "result_message_count": 1,
                "source_tokens": 100,
                "result_tokens": 25,
            },
        )


def test_context_status_persists_with_session_record(tmp_path) -> None:
    sessions = SessionService(session_dir=tmp_path)
    record = SessionRecord.new(session_id="ctx", provider="openai", model="gpt-5.4")
    record.messages = [{"role": "user", "content": f"message {index}"} for index in range(6)]
    save_session(record, base=tmp_path)
    service = ContextService(
        session_service=sessions,
        compression_service_factory=lambda _record: _FakeCompressionService(),
    )

    result = service.compress(ContextCompressRequest(session_id="ctx", focus="auth"))

    assert result.status == "compressed"
    assert result.compression_count == 1
    saved = load_session("ctx", base=tmp_path)
    assert saved is not None
    assert saved.metadata["context_status"]["compression_count"] == 1
    assert saved.metadata["context_status"]["last_compression"]["source_message_count"] == 6

    restored = ContextService(session_service=sessions).status("ctx")
    assert restored.compression_count == 1
    assert restored.status == "compressed"
    assert restored.last_compression is not None
    assert restored.last_compression["result_message_count"] == 1


def test_skipped_context_status_persists_without_compressing(tmp_path) -> None:
    sessions = SessionService(session_dir=tmp_path)
    record = SessionRecord.new(session_id="short", provider="openai", model="gpt-5.4")
    record.messages = [{"role": "user", "content": "short"}]
    save_session(record, base=tmp_path)
    service = ContextService(session_service=sessions)

    result = service.compress(ContextCompressRequest(session_id="short", focus="auth"))

    assert result.status == "skipped"
    saved = load_session("short", base=tmp_path)
    assert saved is not None
    assert saved.metadata["context_status"]["status"] == "skipped"
    assert ContextService(session_service=sessions).status("short").status == "skipped"


def test_context_status_and_estimate_include_breakdown(tmp_path) -> None:
    sessions = SessionService(session_dir=tmp_path)
    record = SessionRecord.new(session_id="estimate", provider="openai", model="gpt-5.4")
    record.system_prompt = "You are a coding assistant."
    record.messages = [
        {"role": "user", "content": "Read this file."},
        {"role": "tool", "content": "Large tool output"},
    ]
    save_session(record, base=tmp_path)
    service = ContextService(session_service=sessions)

    status = service.status("estimate")
    estimate = service.estimate(
        ContextEstimateRequest(
            session_id="estimate",
            draft="Now summarize it.",
            attachments=[{"content": "attached file content"}],
        )
    )

    assert status.provider == "openai"
    assert status.model == "gpt-5.4"
    assert status.prompt_tokens > 0
    assert status.context_window == 128_000
    assert status.pressure_level == "low"
    assert {item.label for item in status.breakdown} >= {"System prompt", "Transcript", "Tool results"}
    assert estimate.message_count == 3
    assert estimate.context_window == 128_000
    assert estimate.attachment_tokens > 0
    assert estimate.prompt_tokens > status.prompt_tokens


def test_context_estimate_uses_model_window_for_pressure(tmp_path) -> None:
    sessions = SessionService(session_dir=tmp_path)
    record = SessionRecord.new(session_id="pressure", provider="custom", model="tiny")
    save_session(record, base=tmp_path)
    service = ContextService(
        session_service=sessions,
        model_window_resolver=lambda _provider, _model: 100,
    )

    estimate = service.estimate(ContextEstimateRequest(session_id="pressure", draft="x" * 500))

    assert estimate.context_window == 100
    assert estimate.prompt_tokens > estimate.context_window
    assert estimate.pressure_level == "critical"
    assert estimate.next_action == "blocked"


def test_unknown_model_window_remains_unknown(tmp_path) -> None:
    sessions = SessionService(session_dir=tmp_path)
    record = SessionRecord.new(session_id="unknown-window", provider="openai", model="not-in-catalog")
    save_session(record, base=tmp_path)

    status = ContextService(session_service=sessions).status("unknown-window")

    assert status.context_window is None
    assert status.pressure_level == "unknown"
    assert status.next_action == "none"
