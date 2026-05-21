"""Analytics service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TokenUsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AnalyticsSummary:
    session_count: int = 0
    message_count: int = 0
    assistant_message_count: int = 0
    tool_call_count: int = 0
    usage: TokenUsageSummary = field(default_factory=TokenUsageSummary)


@dataclass(frozen=True, slots=True)
class AnalyticsDailyEntry:
    day: str
    sessions: int = 0
    messages: int = 0
    tool_calls: int = 0
    usage: TokenUsageSummary = field(default_factory=TokenUsageSummary)


@dataclass(frozen=True, slots=True)
class AnalyticsModelEntry:
    provider: str
    model: str
    sessions: int = 0
    messages: int = 0
    tool_calls: int = 0
    usage: TokenUsageSummary = field(default_factory=TokenUsageSummary)


@dataclass(frozen=True, slots=True)
class AnalyticsSessionEntry:
    session_id: str
    summary: str | None
    provider: str
    model: str
    updated_at: float
    messages: int = 0
    tool_calls: int = 0
    usage: TokenUsageSummary = field(default_factory=TokenUsageSummary)


@dataclass(frozen=True, slots=True)
class AnalyticsReport:
    days: int
    summary: AnalyticsSummary
    daily: list[AnalyticsDailyEntry] = field(default_factory=list)
    models: list[AnalyticsModelEntry] = field(default_factory=list)
    top_sessions: list[AnalyticsSessionEntry] = field(default_factory=list)


__all__ = [
    "AnalyticsDailyEntry",
    "AnalyticsModelEntry",
    "AnalyticsReport",
    "AnalyticsSessionEntry",
    "AnalyticsSummary",
    "TokenUsageSummary",
]
