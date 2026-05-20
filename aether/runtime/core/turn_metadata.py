"""TurnContext.metadata key contract and snapshot helpers.

This module is intentionally small: Sprint 13.1 centralises the string
contract without changing engine behaviour.  The public turn snapshot keeps
the historical semantics of ``AgentEngine._build_result``: copy every
metadata key except live runtime helpers / accumulators.
"""

from __future__ import annotations

from typing import Any, Final, TypeVar, cast

from aether.runtime.core.contracts import TurnContext
from aether.runtime.session.session_runtime import (
    TURN_KEY_CODEX_ACK_RETRIES,
    TURN_KEY_EMPTY_RECOVERY_LAST_STEP,
    TURN_KEY_EMPTY_RESPONSE_RETRIES,
    TURN_KEY_INVALID_JSON_RETRIES,
    TURN_KEY_PHANTOM_TOOL_RETRIES,
    TURN_KEY_PHANTOM_TOOL_SYNTHESIZED,
    TURN_KEY_POST_TOOL_EMPTY_RETRIED,
    TURN_KEY_PROVIDER_ERROR_RETRIES,
    TURN_KEY_STREAMED_ASSISTANT_TEXT,
    TURN_KEY_THINKING_PREFILL_RETRIES,
    TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES,
    TURN_RETRY_COUNTER_KEYS,
)

T = TypeVar("T")


RUNTIME_REF_KEYS: Final[frozenset[str]] = frozenset(
    {
        "_agent_type_registry",
        "_approval_prompter",
        "_browser_manager",
        "_compaction_pipeline",
        "_diagnostic_tracker",
        "_engine_config",
        "_interrupt_signal",
        "_iteration_budget_obj",
        "_lsp_manager",
        "_parent_agent",
        "_project_memory_store",
        "_skill_catalog",
        "_subagent_manager",
        "_task_store",
        "_tool_permission_prompter",
    }
)


INTERNAL_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "usage_accumulator",
        "_tool_permission_preview_plans",
        "_compaction_in_progress",
        "_api_request_attempt_count",
        "_failed_request_dump_written",
        "turn_start_idx",
        "_task_resource_handles",
        "_task_resource_keys",
        "_task_cleanup_done",
        "_schema_sanitized_tool_descriptors",
        "_schema_sanitizer_retry_attempted",
        "_image_shrink_retry_attempted",
        "_loop_state_callback",
    }
) | RUNTIME_REF_KEYS


PUBLIC_STABLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_calls",
        "compaction",
        "empty_recovery",
        "exit",
        "interrupt",
        "iteration_budget",
        "memory",
        "pending_steer",
        "reasoning",
        "recovery",
        "request",
        "request_dump",
        "resource_cleanup",
        "runtime",
        "tool_errors",
        "tool_permissions",
        "trajectory",
        "turn",
        "usage",
    }
)


TURN_METADATA_DEFAULTS: Final[dict[str, Any]] = {
    TURN_KEY_EMPTY_RESPONSE_RETRIES: 0,
    TURN_KEY_PROVIDER_ERROR_RETRIES: 0,
    TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES: 0,
    TURN_KEY_INVALID_JSON_RETRIES: 0,
    TURN_KEY_PHANTOM_TOOL_RETRIES: 0,
    TURN_KEY_PHANTOM_TOOL_SYNTHESIZED: 0,
    TURN_KEY_THINKING_PREFILL_RETRIES: 0,
    TURN_KEY_CODEX_ACK_RETRIES: 0,
    TURN_KEY_STREAMED_ASSISTANT_TEXT: "",
    TURN_KEY_POST_TOOL_EMPTY_RETRIED: False,
    TURN_KEY_EMPTY_RECOVERY_LAST_STEP: "",
}


def init_turn_retry_counters(metadata: dict[str, Any]) -> None:
    """Initialise per-turn retry and empty-response state.

    Callers pass the live metadata dict and this function mutates it in place,
    matching the old ``metadata.update({...})`` behaviour in ``AgentEngine``.
    """

    metadata.update(TURN_METADATA_DEFAULTS)


def set_runtime_ref(context: TurnContext, key: str, value: Any) -> None:
    """Store a live runtime object on ``context.metadata``.

    The key must be listed in ``RUNTIME_REF_KEYS`` so future callers do not add
    a live object that can leak into ``EngineResult.metadata["turn"]``.
    """

    if key not in RUNTIME_REF_KEYS:
        raise KeyError(f"unknown runtime metadata ref key: {key}")
    context.metadata[key] = value


def get_runtime_ref(
    context: TurnContext,
    key: str,
    expected_type: type[T] | None = None,
) -> T | Any | None:
    """Read a live runtime object from ``context.metadata``.

    When ``expected_type`` is provided, return ``None`` instead of a value with
    the wrong runtime type.  This keeps call sites concise without hiding the
    key contract.
    """

    if key not in RUNTIME_REF_KEYS:
        raise KeyError(f"unknown runtime metadata ref key: {key}")
    value = context.metadata.get(key)
    if expected_type is not None:
        return cast(T | None, value if isinstance(value, expected_type) else None)
    return value


def public_turn_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the public ``EngineResult.metadata["turn"]`` snapshot.

    The function deliberately preserves the historical shallow-copy semantics:
    non-internal ad-hoc observability values are kept as-is, while known live
    objects / accumulators are filtered out.
    """

    return {k: v for k, v in metadata.items() if k not in INTERNAL_METADATA_KEYS}


def sanitize_metadata_value(value: Any) -> Any:
    """Compatibility hook for future JSON-friendly metadata conversion.

    PR 13.1 does not apply extra conversion to avoid changing public turn
    metadata.  The function exists so later decomposition PRs have a single
    place to extend sanitisation if a concrete leak is found.
    """

    return value


__all__ = [
    "INTERNAL_METADATA_KEYS",
    "PUBLIC_STABLE_KEYS",
    "RUNTIME_REF_KEYS",
    "TURN_METADATA_DEFAULTS",
    "TURN_RETRY_COUNTER_KEYS",
    "get_runtime_ref",
    "init_turn_retry_counters",
    "public_turn_metadata",
    "sanitize_metadata_value",
    "set_runtime_ref",
]
