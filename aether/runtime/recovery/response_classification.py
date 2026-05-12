"""Classify empty provider responses for recovery decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aether.runtime.core.contracts import NormalizedResponse


class EmptyKind(str, Enum):
    NOT_EMPTY = "not_empty"
    LEGITIMATE_END_TURN = "legitimate_end_turn"
    THINKING_ONLY = "thinking_only"
    BUG_EMPTY = "bug_empty"


@dataclass(frozen=True)
class ResponseClassification:
    kind: EmptyKind
    has_thinking: bool = False
    has_streamed_partial: bool = False
    visible_text_chars: int = 0
    raw_stop_reason: str | None = None
    raw_finish_reason: str | None = None

    @property
    def is_recoverable(self) -> bool:
        return self.kind == EmptyKind.BUG_EMPTY

    @property
    def is_success(self) -> bool:
        return self.kind in {
            EmptyKind.NOT_EMPTY,
            EmptyKind.LEGITIMATE_END_TURN,
            EmptyKind.THINKING_ONLY,
        }


_DELIBERATE_STOPS: frozenset[str] = frozenset(
    {
        "end_turn",
        "stop",
        "stop_sequence",
        "completed",
        "finish",
    }
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_TAG_RE = re.compile(
    r"<reasoning>.*?</reasoning>",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking_tags(text: str) -> str:
    """Remove provider-visible thinking markup from ``text``."""
    if not text:
        return ""
    text = _THINK_TAG_RE.sub("", text)
    return _REASONING_TAG_RE.sub("", text)


def is_legitimate_empty(
    response: NormalizedResponse,
    *,
    streamed_assistant_text: str = "",
) -> ResponseClassification:
    """Classify ``response`` as visible, deliberate empty, thinking-only, or bug."""
    visible = strip_thinking_tags(response.content or "").strip()
    visible_chars = len(visible)
    has_thinking = _detect_thinking_in_response(response)
    has_partial = bool(strip_thinking_tags(streamed_assistant_text).strip())
    finish = (response.finish_reason or "").lower() if response.finish_reason else None
    stop = _extract_stop_reason(response)

    if visible:
        return ResponseClassification(
            kind=EmptyKind.NOT_EMPTY,
            has_thinking=has_thinking,
            has_streamed_partial=has_partial,
            visible_text_chars=visible_chars,
            raw_stop_reason=stop,
            raw_finish_reason=finish,
        )

    deliberate = (stop or "") in _DELIBERATE_STOPS or (finish or "") in _DELIBERATE_STOPS

    if has_thinking and not has_partial:
        return ResponseClassification(
            kind=EmptyKind.THINKING_ONLY,
            has_thinking=True,
            has_streamed_partial=False,
            raw_stop_reason=stop,
            raw_finish_reason=finish,
        )

    if deliberate and not has_partial:
        return ResponseClassification(
            kind=EmptyKind.LEGITIMATE_END_TURN,
            has_thinking=has_thinking,
            has_streamed_partial=False,
            raw_stop_reason=stop,
            raw_finish_reason=finish,
        )

    return ResponseClassification(
        kind=EmptyKind.BUG_EMPTY,
        has_thinking=has_thinking,
        has_streamed_partial=has_partial,
        raw_stop_reason=stop,
        raw_finish_reason=finish,
    )


def _detect_thinking_in_response(response: NormalizedResponse) -> bool:
    md: dict[str, Any] = getattr(response, "metadata", None) or {}
    if md.get("reasoning_content") or md.get("reasoning_details"):
        return True
    content = response.content or ""
    lowered = content.lower()
    return "<think>" in lowered or "<reasoning>" in lowered


def _extract_stop_reason(response: NormalizedResponse) -> str | None:
    md = getattr(response, "metadata", None) or {}
    raw = md.get("stop_reason")
    return str(raw).lower() if raw else None


__all__ = [
    "EmptyKind",
    "ResponseClassification",
    "is_legitimate_empty",
    "strip_thinking_tags",
]
