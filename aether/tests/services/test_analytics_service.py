from __future__ import annotations

from aether.cli.sessions import SessionRecord, save_session
from aether.services.analytics import AnalyticsService
from aether.services.sessions import SessionService


def test_analytics_report_aggregates_session_usage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    session_dir = tmp_path / "sessions"
    record = SessionRecord.new(session_id="analytics-session", provider="codex", model="gpt-5.4")
    record.first_user_message = "inspect analytics"
    record.messages = [
        {"role": "user", "content": "inspect analytics"},
        {
            "role": "assistant",
            "content": "done",
            "metadata": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_tokens": 2,
                    "reasoning_tokens": 3,
                }
            },
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": {}}}
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]
    save_session(record, base=session_dir)

    service = AnalyticsService(session_service=SessionService(session_dir=session_dir))

    report = service.report(days=30, session_limit=5)

    assert report.summary.session_count == 1
    assert report.summary.message_count == 3
    assert report.summary.assistant_message_count == 1
    assert report.summary.tool_call_count == 1
    assert report.summary.usage.input_tokens == 10
    assert report.summary.usage.output_tokens == 5
    assert report.summary.usage.cache_read_tokens == 2
    assert report.summary.usage.reasoning_tokens == 3
    assert report.summary.usage.total_tokens == 20
    assert report.models[0].provider == "codex"
    assert report.models[0].model == "gpt-5.4"
    assert report.daily[0].sessions == 1
    assert report.top_sessions[0].session_id == "analytics-session"


def test_analytics_report_accepts_openai_token_usage_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AETHER_HOME", str(tmp_path))
    session_dir = tmp_path / "sessions"
    record = SessionRecord.new(session_id="openai-session", provider="openai", model="gpt-5")
    record.messages = [
        {"role": "assistant", "content": "done", "metadata": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}},
    ]
    save_session(record, base=session_dir)

    report = AnalyticsService(session_service=SessionService(session_dir=session_dir)).report()

    assert report.summary.usage.input_tokens == 7
    assert report.summary.usage.output_tokens == 3
    assert report.summary.usage.total_tokens == 10
