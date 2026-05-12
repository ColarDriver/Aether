"""Session/task-scoped memory provider.

This provider intentionally stays local and heuristic-only.  It records small
task snapshots from canonical transcript messages and returns budgetable memory
blocks on later turns.  It does not write project or user memory.
"""

from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from .budget import estimate_text_tokens, trim_text_to_token_budget
from .contracts import (
    MemoryBlock,
    MemoryBundle,
    MemoryKind,
    MemoryQuery,
    MemoryScope,
)


_SESSION_TASK_KEY = "__session__"
_MAX_LIST_ITEMS = 8
_MAX_FINDINGS = 6
_MAX_ITEM_CHARS = 240

_CONSTRAINT_MARKERS = (
    "must",
    "should",
    "need to",
    "make sure",
    "do not",
    "don't",
    "never",
    "always",
    "only",
    "require",
    "要求",
    "必须",
    "需要",
    "一定",
    "不要",
    "不能",
    "禁止",
    "默认",
    "保证",
    "确保",
    "严格",
)
_DECISION_MARKERS = (
    "decided",
    "we will",
    "implemented",
    "use ",
    "采用",
    "决定",
    "已经",
    "实现",
    "选择",
)
_QUESTION_MARKERS = (
    "?",
    "？",
    "need clarification",
    "blocked",
    "open question",
    "需要确认",
    "待确认",
)


@dataclass(slots=True)
class TaskMemorySnapshot:
    """Small cross-turn memory for one session/task."""

    session_id: str
    task_id: str | None
    goal: str | None = None
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recent_findings: list[str] = field(default_factory=list)
    updated_at: str | None = None
    token_estimate: int = 0

    def is_empty(self) -> bool:
        return not any(
            (
                self.goal,
                self.constraints,
                self.decisions,
                self.open_questions,
                self.recent_findings,
            )
        )

    def refresh_token_estimate(self) -> None:
        self.token_estimate = estimate_text_tokens(render_task_snapshot_text(self))

    def clone(self) -> "TaskMemorySnapshot":
        return copy.deepcopy(self)


class TaskMemoryProvider:
    """In-process task/session memory provider."""

    def __init__(self, *, session_runtime: Any | None = None) -> None:
        self._lock = threading.RLock()
        self._session_runtime = session_runtime
        self._snapshots: dict[str, dict[str, TaskMemorySnapshot]] = {}

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        snapshot = self.snapshot_for(query.session_id, query.task_id)
        if snapshot is None or snapshot.is_empty():
            return MemoryBundle.skipped("no_candidates")
        blocks = _snapshot_to_blocks(snapshot, token_budget=query.token_budget)
        if not blocks:
            return MemoryBundle.skipped("no_relevant_blocks")
        return MemoryBundle.from_blocks(blocks)

    def observe_turn(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        del metadata
        updates = _extract_updates(messages)
        if not updates.has_updates:
            return
        with self._lock:
            self._merge_updates(self._get_or_create(session_id, task_id), updates)
            self._merge_updates(self._get_or_create(session_id, None), updates)

    def before_compaction(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        self.observe_turn(
            session_id=session_id,
            task_id=task_id,
            messages=messages,
            metadata=metadata,
        )
        with self._lock:
            store = self._store_for_session(session_id)
            for key in (self._task_key(task_id), self._task_key(None)):
                snapshot = store.get(key)
                if snapshot is not None:
                    _compact_snapshot(snapshot)

    def snapshot_for(
        self,
        session_id: str,
        task_id: str | None = None,
    ) -> TaskMemorySnapshot | None:
        with self._lock:
            store = self._store_for_session(session_id)
            snapshot = store.get(self._task_key(task_id))
            if snapshot is None and task_id is not None:
                snapshot = store.get(self._task_key(None))
            return snapshot.clone() if snapshot is not None else None

    def clear(self, session_id: str, task_id: str | None = None) -> None:
        with self._lock:
            if self._session_runtime is None and task_id is None:
                self._snapshots.pop(session_id, None)
                return
            store = self._store_for_session(session_id)
            if task_id is None:
                store.clear()
            else:
                store.pop(self._task_key(task_id), None)

    def _get_or_create(self, session_id: str, task_id: str | None) -> TaskMemorySnapshot:
        store = self._store_for_session(session_id)
        store_key = self._task_key(task_id)
        snapshot = store.get(store_key)
        if snapshot is None:
            snapshot = TaskMemorySnapshot(session_id=session_id, task_id=task_id)
            store[store_key] = snapshot
        return snapshot

    def _merge_updates(self, snapshot: TaskMemorySnapshot, updates: "_TaskMemoryUpdates") -> None:
        if updates.goal:
            snapshot.goal = updates.goal
        _merge_unique(snapshot.constraints, updates.constraints, limit=_MAX_LIST_ITEMS)
        _merge_unique(snapshot.decisions, updates.decisions, limit=_MAX_LIST_ITEMS)
        _merge_unique(snapshot.open_questions, updates.open_questions, limit=_MAX_LIST_ITEMS)
        _merge_unique(snapshot.recent_findings, updates.recent_findings, limit=_MAX_FINDINGS)
        snapshot.updated_at = _utc_now()
        snapshot.refresh_token_estimate()

    @staticmethod
    def _task_key(task_id: str | None) -> str:
        return task_id or _SESSION_TASK_KEY

    def _store_for_session(self, session_id: str) -> dict[str, TaskMemorySnapshot]:
        if self._session_runtime is None:
            return self._snapshots.setdefault(session_id, {})
        state = self._session_runtime.get(session_id)
        store = getattr(state, "task_memory_snapshots", None)
        if not isinstance(store, dict):
            store = {}
            setattr(state, "task_memory_snapshots", store)
        return store


@dataclass(slots=True)
class _TaskMemoryUpdates:
    goal: str | None = None
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recent_findings: list[str] = field(default_factory=list)

    @property
    def has_updates(self) -> bool:
        return bool(
            self.goal
            or self.constraints
            or self.decisions
            or self.open_questions
            or self.recent_findings
        )


def render_task_snapshot_text(snapshot: TaskMemorySnapshot) -> str:
    """Render a task snapshot as concise plain text."""

    lines: list[str] = []
    if snapshot.goal:
        lines.append(f"Current goal: {snapshot.goal}")
    _extend_section(lines, "Constraints", snapshot.constraints)
    _extend_section(lines, "Decisions", snapshot.decisions)
    _extend_section(lines, "Open questions", snapshot.open_questions)
    _extend_section(lines, "Recent findings", snapshot.recent_findings)
    return "\n".join(lines)


def _snapshot_to_blocks(
    snapshot: TaskMemorySnapshot,
    *,
    token_budget: int,
) -> tuple[MemoryBlock, ...]:
    budget = token_budget if token_budget > 0 else 1200
    blocks: list[MemoryBlock] = []
    remaining = budget

    def add_block(kind: MemoryKind, text: str, relevance: float, suffix: str) -> None:
        nonlocal remaining
        if remaining <= 0 or not text.strip():
            return
        trimmed = trim_text_to_token_budget(text, min(remaining, 500))
        token_estimate = estimate_text_tokens(trimmed)
        if token_estimate <= 0 or token_estimate > remaining:
            return
        blocks.append(
            MemoryBlock(
                id=f"task-memory:{snapshot.session_id}:{snapshot.task_id or _SESSION_TASK_KEY}:{suffix}",
                scope=MemoryScope.TASK if snapshot.task_id else MemoryScope.SESSION,
                kind=kind,
                text=trimmed,
                source=f"task:{snapshot.task_id or snapshot.session_id}",
                token_estimate=token_estimate,
                relevance=relevance,
                confidence="medium",
                updated_at=snapshot.updated_at,
            )
        )
        remaining -= token_estimate

    if snapshot.goal:
        add_block(
            MemoryKind.TASK_STATE,
            f"Current goal: {snapshot.goal}",
            100.0,
            "goal",
        )
    if snapshot.constraints:
        add_block(
            MemoryKind.CONSTRAINT,
            _format_section("Constraints", snapshot.constraints),
            90.0,
            "constraints",
        )
    if snapshot.decisions:
        add_block(
            MemoryKind.DECISION,
            _format_section("Decisions", snapshot.decisions),
            80.0,
            "decisions",
        )
    if snapshot.open_questions:
        add_block(
            MemoryKind.WARNING,
            _format_section("Open questions", snapshot.open_questions),
            70.0,
            "questions",
        )
    if snapshot.recent_findings:
        add_block(
            MemoryKind.REFERENCE,
            _format_section("Recent findings", snapshot.recent_findings),
            60.0,
            "findings",
        )
    return tuple(blocks)


def _extract_updates(messages: list[dict[str, Any]]) -> _TaskMemoryUpdates:
    updates = _TaskMemoryUpdates()
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        text = _message_text(message)
        if not text:
            continue
        if role == "user":
            updates.goal = _truncate_item(text)
            constraints = _extract_marked_sentences(text, _CONSTRAINT_MARKERS)
            _merge_unique(updates.constraints, constraints, limit=_MAX_LIST_ITEMS)
            questions = _extract_marked_sentences(text, _QUESTION_MARKERS)
            _merge_unique(updates.open_questions, questions, limit=_MAX_LIST_ITEMS)
        elif role == "assistant":
            decisions = _extract_marked_sentences(text, _DECISION_MARKERS)
            _merge_unique(updates.decisions, decisions, limit=_MAX_LIST_ITEMS)
            questions = _extract_marked_sentences(text, _QUESTION_MARKERS)
            _merge_unique(updates.open_questions, questions, limit=_MAX_LIST_ITEMS)
        elif role == "tool":
            finding = _tool_finding(message, text)
            if finding:
                _merge_unique(updates.recent_findings, [finding], limit=_MAX_FINDINGS)
    return updates


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return _clean_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
        return _clean_text("\n".join(parts))
    return _clean_text(str(content))


def _tool_finding(message: dict[str, Any], text: str) -> str | None:
    name = str(message.get("name") or "tool")
    if not text.strip():
        return None
    prefix = f"{name}: "
    return _truncate_item(prefix + text)


def _extract_marked_sentences(text: str, markers: Iterable[str]) -> list[str]:
    lowered = text.lower()
    if not any(marker.lower() in lowered for marker in markers):
        return []
    sentences = _split_sentences(text)
    matches: list[str] = []
    for sentence in sentences:
        lowered_sentence = sentence.lower()
        if any(marker.lower() in lowered_sentence for marker in markers):
            matches.append(_truncate_item(sentence))
    return matches[:3]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    cleaned = [_clean_text(part) for part in parts if _clean_text(part)]
    return cleaned or [_clean_text(text)]


def _merge_unique(target: list[str], values: Iterable[str], *, limit: int) -> None:
    seen = {_normalise_item(item) for item in target}
    for value in values:
        item = _truncate_item(value)
        key = _normalise_item(item)
        if not item or key in seen:
            continue
        target.append(item)
        seen.add(key)
    del target[:-limit]


def _compact_snapshot(snapshot: TaskMemorySnapshot) -> None:
    snapshot.constraints[:] = snapshot.constraints[-_MAX_LIST_ITEMS:]
    snapshot.decisions[:] = snapshot.decisions[-_MAX_LIST_ITEMS:]
    snapshot.open_questions[:] = snapshot.open_questions[-_MAX_LIST_ITEMS:]
    snapshot.recent_findings[:] = snapshot.recent_findings[-_MAX_FINDINGS:]
    snapshot.refresh_token_estimate()


def _extend_section(lines: list[str], title: str, values: list[str]) -> None:
    if values:
        lines.append(_format_section(title, values))


def _format_section(title: str, values: list[str]) -> str:
    rendered = "\n".join(f"- {value}" for value in values)
    return f"{title}:\n{rendered}"


def _truncate_item(text: str) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= _MAX_ITEM_CHARS:
        return cleaned
    return cleaned[: _MAX_ITEM_CHARS - 3].rstrip() + "..."


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalise_item(text: str) -> str:
    return _clean_text(text).casefold()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "TaskMemoryProvider",
    "TaskMemorySnapshot",
    "render_task_snapshot_text",
]
