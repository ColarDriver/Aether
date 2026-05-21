"""Local session analytics service."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from aether.services.analytics.contracts import (
    AnalyticsDailyEntry,
    AnalyticsModelEntry,
    AnalyticsReport,
    AnalyticsSessionEntry,
    AnalyticsSummary,
    TokenUsageSummary,
)
from aether.services.sessions import SessionService, TranscriptMessage


class AnalyticsService:
    def __init__(self, *, session_service: SessionService | None = None) -> None:
        self._sessions = session_service or SessionService()

    def report(self, *, days: int = 30, session_limit: int = 20) -> AnalyticsReport:
        normalized_days = max(1, int(days or 30))
        normalized_limit = max(1, int(session_limit or 20))
        cutoff = datetime.now(timezone.utc) - timedelta(days=normalized_days)
        session_infos = self._sessions.list(limit=None).sessions
        included = [item for item in session_infos if _epoch_to_datetime(item.updated_at) >= cutoff]

        daily: dict[str, _Bucket] = defaultdict(_Bucket)
        models: dict[tuple[str, str], _Bucket] = defaultdict(_Bucket)
        session_entries: list[AnalyticsSessionEntry] = []
        summary_bucket = _Bucket()

        for info in included:
            messages = self._load_messages(info.session_id)
            usage = _usage_for_messages(messages)
            tool_calls = _tool_calls_for_messages(messages)
            message_count = len(messages)
            assistant_count = sum(1 for message in messages if message.role == "assistant")

            summary_bucket.sessions += 1
            summary_bucket.messages += message_count
            summary_bucket.assistant_messages += assistant_count
            summary_bucket.tool_calls += tool_calls
            summary_bucket.usage = _add_usage(summary_bucket.usage, usage)

            day = _day_key(info.updated_at)
            daily_bucket = daily[day]
            daily_bucket.sessions += 1
            daily_bucket.messages += message_count
            daily_bucket.tool_calls += tool_calls
            daily_bucket.usage = _add_usage(daily_bucket.usage, usage)

            model_key = (info.provider or "unknown", info.model or "unknown")
            model_bucket = models[model_key]
            model_bucket.sessions += 1
            model_bucket.messages += message_count
            model_bucket.tool_calls += tool_calls
            model_bucket.usage = _add_usage(model_bucket.usage, usage)

            session_entries.append(
                AnalyticsSessionEntry(
                    session_id=info.session_id,
                    summary=info.summary,
                    provider=info.provider,
                    model=info.model,
                    updated_at=info.updated_at,
                    messages=message_count,
                    tool_calls=tool_calls,
                    usage=usage,
                )
            )

        return AnalyticsReport(
            days=normalized_days,
            summary=AnalyticsSummary(
                session_count=summary_bucket.sessions,
                message_count=summary_bucket.messages,
                assistant_message_count=summary_bucket.assistant_messages,
                tool_call_count=summary_bucket.tool_calls,
                usage=summary_bucket.usage,
            ),
            daily=[
                AnalyticsDailyEntry(
                    day=day,
                    sessions=bucket.sessions,
                    messages=bucket.messages,
                    tool_calls=bucket.tool_calls,
                    usage=bucket.usage,
                )
                for day, bucket in sorted(daily.items())
            ],
            models=[
                AnalyticsModelEntry(
                    provider=provider,
                    model=model,
                    sessions=bucket.sessions,
                    messages=bucket.messages,
                    tool_calls=bucket.tool_calls,
                    usage=bucket.usage,
                )
                for (provider, model), bucket in sorted(
                    models.items(),
                    key=lambda item: item[1].usage.total_tokens,
                    reverse=True,
                )
            ],
            top_sessions=sorted(
                session_entries,
                key=lambda item: (item.usage.total_tokens, item.updated_at),
                reverse=True,
            )[:normalized_limit],
        )

    def _load_messages(self, session_id: str) -> list[TranscriptMessage]:
        try:
            return self._sessions.transcript(session_id)
        except Exception:
            return []


class _Bucket:
    def __init__(self) -> None:
        self.sessions = 0
        self.messages = 0
        self.assistant_messages = 0
        self.tool_calls = 0
        self.usage = TokenUsageSummary()


def _usage_for_messages(messages: list[TranscriptMessage]) -> TokenUsageSummary:
    total = TokenUsageSummary()
    for message in messages:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        total = _add_usage(total, _usage_from_metadata(metadata))
    return total


def _usage_from_metadata(metadata: dict[str, Any]) -> TokenUsageSummary:
    raw = metadata.get("usage")
    if not isinstance(raw, dict):
        raw = metadata.get("token_usage")
    if not isinstance(raw, dict):
        raw = metadata
    input_tokens = _int_value(raw.get("input_tokens"))
    output_tokens = _int_value(raw.get("output_tokens"))
    cache_read_tokens = _int_value(raw.get("cache_read_tokens")) + _int_value(raw.get("cache_read_input_tokens"))
    cache_write_tokens = _int_value(raw.get("cache_write_tokens")) + _int_value(raw.get("cache_creation_input_tokens"))
    reasoning_tokens = _int_value(raw.get("reasoning_tokens"))
    prompt_tokens = _int_value(raw.get("prompt_tokens"))
    completion_tokens = _int_value(raw.get("completion_tokens"))
    if input_tokens == 0 and prompt_tokens > 0:
        input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    if output_tokens == 0 and completion_tokens > 0:
        output_tokens = completion_tokens
    total_tokens = _int_value(raw.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens + cache_read_tokens + cache_write_tokens + reasoning_tokens
    return TokenUsageSummary(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def _add_usage(left: TokenUsageSummary, right: TokenUsageSummary) -> TokenUsageSummary:
    return replace(
        left,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
        cache_write_tokens=left.cache_write_tokens + right.cache_write_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _tool_calls_for_messages(messages: list[TranscriptMessage]) -> int:
    seen: set[str] = set()
    fallback = 0
    for message in messages:
        for tool_call in message.tool_calls:
            seen.add(tool_call.id)
        if message.role == "tool" and message.tool_call_id and message.tool_call_id not in seen:
            seen.add(message.tool_call_id)
        elif message.role == "tool" and not message.tool_call_id:
            fallback += 1
    return len(seen) + fallback


def _day_key(epoch: float) -> str:
    return _epoch_to_datetime(epoch).date().isoformat()


def _epoch_to_datetime(epoch: float) -> datetime:
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(float(value)))
        except ValueError:
            return 0
    return 0


__all__ = ["AnalyticsService"]
