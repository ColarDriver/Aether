"""Per-session runtime state with thread-safe registry.

This module exists to hold mutable state that lives **across multiple turns of
a single session** (e.g. nudge cadence counters, cached system prompt) without
storing it as instance attributes on ``AgentEngine``.

Why this matters
----------------
``AgentEngine`` is designed to be a single instance that serves many sessions
concurrently — ``InterruptController`` and ``SessionStore`` are already keyed
by ``session_id``.  Storing per-session counters on the engine instance
(``self._memory_nudge_counter``, etc.) creates a cross-session leak: thread A
running session_a can have its counter incremented by thread B running a
completely different session_b, causing nudge events to fire at the wrong
time or, worse, retry counters to be silently reset mid-retry.

``SessionRuntimeRegistry`` solves this by partitioning state per
``session_id`` behind a single lock.  All access is mediated through
``get(session_id)``.

Lifecycle scopes (important — pick the right home for your state):

- ``SessionRuntimeState`` (this module)
    Lives across turns within one session.  Examples: nudge counters,
    cached system prompt for prefix-cache stability.

- ``TurnContext.metadata`` (see ``runtime/contracts.py``)
    Lives across iterations within one turn.  Examples: per-turn retry
    counters such as ``empty_response_retries`` / ``provider_error_retries``.
    These keys are documented in ``TURN_METADATA_KEYS`` below so that
    Sprint 1+ work can rely on them.

- Module-level singletons or DI services
    Lives across all sessions (e.g. ``InterruptController``,
    ``SessionStore``).  Use only for things that genuinely have no
    per-session identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Final


# ---------------------------------------------------------------------------
# TurnContext.metadata key contract
# ---------------------------------------------------------------------------
# These string constants document the standard keys that the run-loop and
# its helpers (Sprint 0 → Sprint 5) read/write inside ``TurnContext.metadata``.
# Centralising them here prevents typos and makes it discoverable where each
# piece of per-turn state lives.
#
# Convention: only fields whose lifetime is exactly "one turn" belong here.
# Cross-turn fields belong on ``SessionRuntimeState`` below.

# Per-turn retry counters (Sprint 0).  Reset at the top of every turn.
TURN_KEY_EMPTY_RESPONSE_RETRIES: Final[str] = "empty_response_retries"
TURN_KEY_PROVIDER_ERROR_RETRIES: Final[str] = "provider_error_retries"

# Sprint 1 / PR 1.3 — per-turn retry counters for the truncated tool-call
# detector.  Both reset to 0 at turn entry; they live on
# ``TurnContext.metadata`` because their lifetime is exactly one turn and
# concurrent sessions sharing the engine instance must not interfere.
#
# - ``truncated_tool_call_retries``: how many times we have re-issued the
#   provider call after seeing finish_reason="length" + tool_calls (without
#   appending the broken response to history).
# - ``invalid_json_retries``: how many times we have re-issued the call
#   because the tool_call arguments did not parse as JSON but did not look
#   truncated (so the model just made a JSON formatting mistake).
TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES: Final[str] = "truncated_tool_call_retries"
TURN_KEY_INVALID_JSON_RETRIES: Final[str] = "invalid_json_retries"
# Sprint 1.5 / phantom-tool recovery: bumped each time the engine
# detects the model wrote a tool intent in prose (\u0060\u0060\u0060bash …,
# ``<function=NAME>`` …, ``<invoke>`` …) but did not populate the
# structured ``tool_calls`` field, and we sent a corrective user
# message asking it to retry with proper tool_calls.  Bounded by
# ``EngineConfig.max_phantom_tool_retries`` so a model that
# consistently refuses tool_calls exits with PHANTOM_TOOL_INTENT
# instead of looping forever.
TURN_KEY_PHANTOM_TOOL_RETRIES: Final[str] = "phantom_tool_retries"
# Sprint 1.5 / P0-9: bumped each time the engine *successfully*
# turns prose-style tool intent into structured ``ToolCall``s and
# dispatches them inline (see ``synthesize_tool_calls_from_phantom``
# + the run-loop synthesis branch).  Distinct from
# ``TURN_KEY_PHANTOM_TOOL_RETRIES`` because synthesis keeps the loop
# moving forward; only un-synthesizable phantoms burn a corrective-
# message retry.  Surfaced in the UI as "↻ synthesized N call(s)
# from prose" so users see the model misbehaved without the loud
# warning.
TURN_KEY_PHANTOM_TOOL_SYNTHESIZED: Final[str] = "phantom_tool_synthesized"

# Sprint 4 / PR 4.1: empty-response degradation counters and state keys.
TURN_KEY_THINKING_PREFILL_RETRIES: Final[str] = "thinking_prefill_retries"
TURN_KEY_POST_TOOL_EMPTY_RETRIED: Final[str] = "post_tool_empty_retried"
TURN_KEY_CODEX_ACK_RETRIES: Final[str] = "codex_intermediate_ack_retries"
TURN_KEY_STREAMED_ASSISTANT_TEXT: Final[str] = "streamed_assistant_text"
TURN_KEY_TRUNCATED_RESPONSE_PREFIX: Final[str] = "truncated_response_prefix"
TURN_KEY_EMPTY_RECOVERY_LAST_STEP: Final[str] = "empty_recovery_last_step"

# Set of all turn-scoped retry-counter keys.  Helpers iterate over this set
# when initialising/resetting per-turn state so adding a new counter only
# requires touching this constant + the consumer site.
TURN_RETRY_COUNTER_KEYS: Final[frozenset[str]] = frozenset(
    {
        TURN_KEY_EMPTY_RESPONSE_RETRIES,
        TURN_KEY_PROVIDER_ERROR_RETRIES,
        TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES,
        TURN_KEY_INVALID_JSON_RETRIES,
        TURN_KEY_PHANTOM_TOOL_RETRIES,
        TURN_KEY_PHANTOM_TOOL_SYNTHESIZED,
        TURN_KEY_THINKING_PREFILL_RETRIES,
        TURN_KEY_CODEX_ACK_RETRIES,
    }
)


@dataclass(slots=True)
class SessionRuntimeState:
    """Mutable state for one session, persisted across turns.

    Instances live inside ``SessionRuntimeRegistry`` and are shared across
    all turns of the same session.  They are **not** thread-safe by themselves;
    the registry's lock guards reads/writes during ``get()``.  Mutations that
    happen inside ``run_loop`` are single-threaded per session (one turn at
    a time), so the engine reads/writes fields directly without holding the
    registry lock — the only contended access is ``get()``/``discard()``.
    """

    # Memory review cadence.  Incremented once per turn at the top of
    # ``_apply_turn_nudges``; when it reaches the configured interval
    # the engine emits ``context.metadata["should_review_memory"] = True``
    # and resets the counter to 0.
    memory_nudge_counter: int = 0

    # Skill review cadence.  Incremented every time the model emits a
    # tool-call (in ``_register_skill_nudge``) and likewise reset when
    # it triggers.  The two counters use different cadence semantics:
    # memory = per-turn, skill = per-tool-call iteration.
    skill_nudge_counter: int = 0

    # Sprint 4 / PR 4.1: cross-turn housekeeping fallback snapshot.
    # Written at the end of a turn by the agent and read by the next
    # turn's empty-response recovery path.
    last_assistant_text_with_tools: str = ""
    last_assistant_tools_all_housekeeping: bool = False

    # Sprint 7: in-memory, session-scoped tool permission rules.
    # Stored here rather than on ``TurnContext.metadata`` because
    # "allow this path/command for this session" must survive across
    # turns but must not leak into other sessions or persist to disk.
    tool_permission_rules: list[Any] = field(default_factory=list)

    # Sprint 8 / PR 8.3: task/session memory snapshots.  The concrete
    # ``TaskMemorySnapshot`` type lives in ``aether.memory.task``; this field
    # stays typed as ``Any`` to keep the runtime package independent from the
    # memory provider implementation.  Keys are task ids, plus ``"__session__"``
    # for the session-level fallback snapshot.
    task_memory_snapshots: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionRuntimeRegistry:
    """Thread-safe per-session ``SessionRuntimeState`` lookup.

    Mirrors the design of ``InterruptController`` / ``InMemorySessionStore``
    — a single ``RLock`` guards a ``dict[session_id, state]`` map.  Returns
    a live (mutable) reference; callers do not need to ``put`` modifications
    back.

    Example::

        registry = SessionRuntimeRegistry()
        state = registry.get("session-abc")  # creates if missing
        state.memory_nudge_counter += 1
        # No put-back needed; ``state`` is the same object the registry holds.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._states: Dict[str, SessionRuntimeState] = {}

    def get(self, session_id: str) -> SessionRuntimeState:
        """Return the state for ``session_id``, creating it on first access."""
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = SessionRuntimeState()
                self._states[session_id] = state
            return state

    def discard(self, session_id: str) -> None:
        """Drop the entry for ``session_id`` if present.

        Used when a session is permanently terminated (e.g. ``/reset``,
        long inactivity GC).  Safe to call for unknown ids.
        """
        with self._lock:
            self._states.pop(session_id, None)

    def reset(self) -> None:
        """Clear all per-session state.  Primarily useful in tests."""
        with self._lock:
            self._states.clear()

    def __len__(self) -> int:  # pragma: no cover - debug aid
        with self._lock:
            return len(self._states)


__all__ = [
    "SessionRuntimeState",
    "SessionRuntimeRegistry",
    "TURN_KEY_EMPTY_RESPONSE_RETRIES",
    "TURN_KEY_PROVIDER_ERROR_RETRIES",
    "TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES",
    "TURN_KEY_INVALID_JSON_RETRIES",
    "TURN_KEY_PHANTOM_TOOL_RETRIES",
    "TURN_KEY_PHANTOM_TOOL_SYNTHESIZED",
    "TURN_KEY_THINKING_PREFILL_RETRIES",
    "TURN_KEY_POST_TOOL_EMPTY_RETRIED",
    "TURN_KEY_CODEX_ACK_RETRIES",
    "TURN_KEY_STREAMED_ASSISTANT_TEXT",
    "TURN_KEY_TRUNCATED_RESPONSE_PREFIX",
    "TURN_KEY_EMPTY_RECOVERY_LAST_STEP",
    "TURN_RETRY_COUNTER_KEYS",
]
