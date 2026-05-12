"""Deterministic memory retrieval, scoring, and composite provider.

No vector DB, no embeddings — keyword/path/tag matching with explicit
weights.  Designed to be stable, explainable, and fast.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Iterable

from .budget import estimate_text_tokens, pack_memory_blocks
from .contracts import (
    MemoryBlock,
    MemoryBundle,
    MemoryKind,
    MemoryMode,
    MemoryQuery,
    MemoryScope,
    scopes_for_mode,
)
from .task import TaskMemoryProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (from design doc)
# ---------------------------------------------------------------------------

SCOPE_WEIGHT_TASK = 4.0
SCOPE_WEIGHT_PROJECT = 2.5
PATH_MATCH_BOOST = 3.0
TAG_MATCH_BOOST = 1.5
KEYWORD_OVERLAP_BOOST = 0.2
CONFIDENCE_HIGH_BOOST = 1.0
STALE_PENALTY = -1.0
STALE_DAYS_THRESHOLD = 90
LARGE_BLOCK_PENALTY = -2.0
LARGE_BLOCK_TOKEN_THRESHOLD = 500

SCORE_INJECTION_THRESHOLD = 1.5
MAX_INJECTED_BLOCKS = 8
MAX_PROJECT_BLOCKS = 5
BLOCK_TOKEN_MAX = 500
RETRIEVAL_TIMEOUT_MS = 2000

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "must",
    "to", "of", "in", "for", "on", "at", "by", "with", "from",
    "as", "into", "about", "it", "its", "this", "that", "and",
    "or", "but", "not", "if", "then", "so", "no", "yes",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "what", "which", "who", "how", "when", "where",
    "just", "also", "very", "too", "only", "up", "out",
})

_TOPIC_TAGS = frozenset({
    "architecture", "workflows", "decisions", "pitfalls",
    "memory", "permission", "compaction", "recovery",
    "tool", "agent", "model", "config", "runtime",
    "session", "task", "project", "auth", "security",
})

_WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


# ---------------------------------------------------------------------------
# Query features
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QueryFeatures:
    keywords: set[str] = field(default_factory=set)
    path_segments: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    tool_names: set[str] = field(default_factory=set)
    mode: MemoryMode = MemoryMode.PROJECT


def extract_query_features(query: MemoryQuery) -> QueryFeatures:
    features = QueryFeatures(mode=query.mode)

    _extract_keywords_from_text(query.user_message, features.keywords)
    for msg in query.recent_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role in ("user", "assistant"):
            _extract_keywords_from_text(_msg_text(msg), features.keywords)
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name:
                features.tool_names.add(name.lower())

    for path in query.active_files:
        _extract_path_segments(path, features.path_segments)
    if query.working_directory:
        _extract_path_segments(query.working_directory, features.path_segments)

    for kw in list(features.keywords):
        if kw in _TOPIC_TAGS:
            features.tags.add(kw)

    return features


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_block(block: MemoryBlock, features: QueryFeatures) -> float:
    score = 0.0

    if block.scope in (MemoryScope.TASK, MemoryScope.SESSION):
        score += SCOPE_WEIGHT_TASK
    elif block.scope == MemoryScope.PROJECT:
        score += SCOPE_WEIGHT_PROJECT

    if features.path_segments and block.metadata.get("paths"):
        block_paths: set[str] = set()
        for p in block.metadata["paths"]:
            _extract_path_segments(str(p), block_paths)
        if features.path_segments & block_paths:
            score += PATH_MATCH_BOOST

    if features.tags and block.tags:
        tag_set = {t.lower() for t in block.tags}
        if features.tags & tag_set:
            score += TAG_MATCH_BOOST

    block_words = set(_WORD_RE.findall(block.text.lower())) - _STOPWORDS
    overlap = features.keywords & block_words
    score += len(overlap) * KEYWORD_OVERLAP_BOOST

    conf = str(block.metadata.get("confidence") or block.confidence).lower()
    if conf == "high":
        score += CONFIDENCE_HIGH_BOOST

    if block.updated_at:
        days = _days_since(block.updated_at)
        if days is not None and days > STALE_DAYS_THRESHOLD:
            score += STALE_PENALTY

    if block.token_estimate > LARGE_BLOCK_TOKEN_THRESHOLD:
        score += LARGE_BLOCK_PENALTY

    return score


# ---------------------------------------------------------------------------
# Project candidate recall
# ---------------------------------------------------------------------------

def recall_project_candidates(
    store: Any,
    features: QueryFeatures,
) -> list[MemoryBlock]:
    try:
        index = store.load_index()
    except Exception:
        logger.debug("Project index load failed, attempting rebuild")
        try:
            index = store.rebuild_index()
        except Exception:
            logger.debug("Project index rebuild also failed, skipping project recall")
            return []

    entries_list = index.get("entries") or []
    if not entries_list:
        return []

    hit_ids: set[str] = set()
    for entry in entries_list:
        if not isinstance(entry, dict):
            continue
        entry_tags = {str(t).lower() for t in (entry.get("tags") or [])}
        entry_paths: set[str] = set()
        for p in entry.get("paths") or []:
            _extract_path_segments(str(p), entry_paths)

        if features.tags & entry_tags:
            hit_ids.add(entry["id"])
            continue
        if features.path_segments & entry_paths:
            hit_ids.add(entry["id"])
            continue
        entry_topic = str(entry.get("topic") or "").lower()
        if entry_topic in features.tags:
            hit_ids.add(entry["id"])
            continue
        entry_kind = str(entry.get("kind") or "").lower()
        if entry_kind in features.keywords:
            hit_ids.add(entry["id"])

    if not hit_ids:
        return []

    try:
        all_entries = store.read_entries(sanitize=True)
    except Exception:
        logger.debug("Failed to read project entries")
        return []

    blocks: list[MemoryBlock] = []
    for entry in all_entries:
        if entry.id not in hit_ids:
            continue
        if not entry.text.strip():
            continue
        token_est = estimate_text_tokens(entry.text)
        if token_est <= 0:
            continue
        blocks.append(MemoryBlock(
            id=f"project:{entry.topic}:{entry.id}",
            scope=MemoryScope.PROJECT,
            kind=entry.kind,
            text=entry.text,
            source=f"project:{entry.topic}",
            token_estimate=token_est,
            relevance=0.0,
            confidence=entry.confidence,
            created_at=entry.created_at or None,
            updated_at=entry.updated_at or None,
            tags=entry.tags,
            metadata={
                "paths": list(entry.paths),
                "confidence": entry.confidence,
                "redacted": entry.redacted,
            },
        ))
    return blocks


# ---------------------------------------------------------------------------
# RetrievalMemoryProvider
# ---------------------------------------------------------------------------

class RetrievalMemoryProvider:
    """Composite MemoryProvider: task memory + project store + ranking."""

    def __init__(
        self,
        *,
        session_runtime: Any | None = None,
        project_store: Any | None = None,
    ) -> None:
        self._task_provider = TaskMemoryProvider(session_runtime=session_runtime)
        self._project_store = project_store

    @property
    def task_provider(self) -> TaskMemoryProvider:
        return self._task_provider

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        features = extract_query_features(query)
        allowed_scopes = set(scopes_for_mode(query.mode))

        candidates: list[MemoryBlock] = []

        if allowed_scopes & {MemoryScope.TASK, MemoryScope.SESSION}:
            task_bundle = self._task_provider.retrieve(query)
            for block in task_bundle.blocks:
                if block.scope in allowed_scopes:
                    candidates.append(block)

        if MemoryScope.PROJECT in allowed_scopes and self._project_store is not None:
            try:
                project_blocks = recall_project_candidates(self._project_store, features)
                candidates.extend(project_blocks[:MAX_PROJECT_BLOCKS])
            except Exception:
                logger.debug("Project candidate recall failed", exc_info=True)

        if not candidates:
            return MemoryBundle.skipped("no_candidates")

        scored: list[tuple[float, MemoryBlock]] = []
        for block in candidates:
            s = score_block(block, features)
            if s >= SCORE_INJECTION_THRESHOLD:
                scored.append((s, block))

        if not scored:
            return MemoryBundle.skipped("no_relevant_blocks")

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:MAX_INJECTED_BLOCKS]

        ranked_blocks = [
            replace(block, relevance=s) for s, block in top
        ]

        return pack_memory_blocks(
            ranked_blocks,
            token_budget=query.token_budget,
            block_token_max=BLOCK_TOKEN_MAX,
        )

    def observe_turn(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        self._task_provider.observe_turn(
            session_id=session_id,
            task_id=task_id,
            messages=messages,
            metadata=metadata,
        )

    def before_compaction(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        self._task_provider.before_compaction(
            session_id=session_id,
            task_id=task_id,
            messages=messages,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords_from_text(text: str, target: set[str]) -> None:
    if not text:
        return
    for word in _WORD_RE.findall(text.lower()):
        if word not in _STOPWORDS and len(word) >= 3:
            target.add(word)


def _extract_path_segments(path: str, target: set[str]) -> None:
    if not path:
        return
    parts = PurePosixPath(path.replace("\\", "/")).parts
    for part in parts:
        cleaned = part.strip("/").lower()
        if cleaned and len(cleaned) >= 2 and cleaned != ".":
            target.add(cleaned)
    if len(parts) >= 2:
        target.add("/".join(parts[-2:]).lower())
    filename = PurePosixPath(path.replace("\\", "/")).name
    if filename and "." in filename:
        target.add(filename.lower())


def _msg_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(parts)
    return str(content)


def _days_since(iso_str: str) -> int | None:
    try:
        dt_str = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0, delta.days)
    except (ValueError, TypeError):
        return None


__all__ = [
    "BLOCK_TOKEN_MAX",
    "MAX_INJECTED_BLOCKS",
    "MAX_PROJECT_BLOCKS",
    "QueryFeatures",
    "SCORE_INJECTION_THRESHOLD",
    "RetrievalMemoryProvider",
    "extract_query_features",
    "recall_project_candidates",
    "score_block",
]
