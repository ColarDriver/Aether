"""Helpers for request-time memory injection metadata."""

from __future__ import annotations

from typing import Any

from .contracts import MemoryBundle


def default_memory_metadata(*, enabled: bool, mode: str) -> dict[str, Any]:
    """Return the stable per-turn memory metadata shape."""

    return {
        "enabled": bool(enabled),
        "mode": mode,
        "retrieval_ms": 0.0,
        "candidate_count": 0,
        "injected_count": 0,
        "injected_tokens": 0,
        "scopes": [],
        "skipped_reason": None,
        "write_count": 0,
        "error": None,
    }


def metadata_from_bundle(
    bundle: MemoryBundle,
    *,
    enabled: bool,
    mode: str,
    injected_count: int | None = None,
    injected_tokens: int | None = None,
    candidate_count: int | None = None,
    skipped_reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build stable metadata from a retrieved or skipped memory bundle."""

    blocks = tuple(bundle.blocks)
    return {
        "enabled": bool(enabled),
        "mode": mode,
        "retrieval_ms": float(bundle.latency_ms),
        "candidate_count": len(blocks) if candidate_count is None else int(candidate_count),
        "injected_count": len(blocks) if injected_count is None else int(injected_count),
        "injected_tokens": bundle.token_estimate
        if injected_tokens is None
        else int(injected_tokens),
        "scopes": sorted({block.scope.value for block in blocks}),
        "skipped_reason": skipped_reason
        if skipped_reason is not None
        else bundle.skipped_reason,
        "write_count": 0,
        "error": error or (bundle.provider_errors[0] if bundle.provider_errors else None),
    }


def append_memory_context(existing_user_context: str | None, memory_context: str) -> str:
    """Append memory context to hook user-context text."""

    cleaned = memory_context.strip()
    if not cleaned:
        return existing_user_context or ""
    if not existing_user_context or not existing_user_context.strip():
        return cleaned
    return f"{existing_user_context.rstrip()}\n\n{cleaned}"
