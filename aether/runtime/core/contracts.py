"""Shared contracts for the Aether loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from aether.config.schema import ModelCallConfig

if TYPE_CHECKING:
    from aether.runtime.control.interrupt_signal import InterruptSignal


class LoopState(str, Enum):
    INIT = "INIT"
    PREPARE = "PREPARE"
    PRE_LLM = "PRE_LLM"
    LLM_CALL = "LLM_CALL"
    POST_LLM = "POST_LLM"
    TOOL_DISPATCH = "TOOL_DISPATCH"
    TOOL_EXECUTE = "TOOL_EXECUTE"
    CHECK_EXIT = "CHECK_EXIT"
    FINALIZE = "FINALIZE"
    DONE = "DONE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ExitReason(str, Enum):
    TEXT_RESPONSE = "TEXT_RESPONSE"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    INTERRUPTED = "INTERRUPTED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    MIDDLEWARE_ERROR = "MIDDLEWARE_ERROR"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    # Response-shape validation failed even after the recovery layer's
    # retry budget was exhausted. Distinct from
    # PROVIDER_ERROR so observers can tell "the API itself broke" apart
    # from "the API kept returning malformed bodies".
    RESPONSE_INVALID = "RESPONSE_INVALID"
    # Response hit the model's output budget but the engine stitched a
    # continuation and returned the full text.
    LENGTH_RECOVERED = "LENGTH_RECOVERED"
    # Response hit the output budget repeatedly and the engine had to
    # stop after rollback or partial return.
    LENGTH_EXHAUSTED = "LENGTH_EXHAUSTED"
    # Assistant tool-call payload was cut off mid-stream, either because
    # ``finish_reason="length"`` arrived alongside ``tool_calls`` or
    # because the JSON arguments never terminated with ``}`` or ``]``.
    # Distinct from ``LENGTH_EXHAUSTED`` so observers can tell "the
    # model asked for a tool but never finished the arguments" apart
    # from "the model exhausted its output budget on prose".
    TOOL_CALL_TRUNCATED = "TOOL_CALL_TRUNCATED"
    # The model wrote a tool intent in prose instead of populating the
    # structured ``tool_calls`` field, and repeated repair attempts
    # still never produced a valid tool invocation. Distinct from
    # ``TEXT_RESPONSE`` because the turn is broken rather than complete.
    PHANTOM_TOOL_INTENT = "PHANTOM_TOOL_INTENT"
    # Provider returned a rate-limit failure and the recovery strategy
    # gave up. Distinct from ``PROVIDER_ERROR`` so observability can
    # surface throttling explicitly.
    RATE_LIMITED = "RATE_LIMITED"
    # Server rejected the request because the prompt exceeded the
    # model's context window and the recovery path could not proceed.
    CONTEXT_EXHAUSTED = "CONTEXT_EXHAUSTED"
    # The request body itself is too large regardless of the model's
    # context window.
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    # The compaction pipeline ran but still could not reduce the prompt
    # under the target threshold. Distinct from
    # ``CONTEXT_EXHAUSTED`` so observability can tell "compression was
    # attempted and still failed".
    COMPRESSION_EXHAUSTED = "COMPRESSION_EXHAUSTED"
    # The fallback chain ran out of providers while the last attempt
    # was still failing.
    FALLBACK_EXHAUSTED = "FALLBACK_EXHAUSTED"
    # The model emitted unrepairable tool names for
    # ``invalid_tool_max_retries`` iterations in a row. Distinct from a
    # tool runtime failure.
    INVALID_TOOL_REPEATED = "INVALID_TOOL_REPEATED"
    # Empty-response recovery terminals.
    PARTIAL_STREAM_RECOVERY = "PARTIAL_STREAM_RECOVERY"
    FALLBACK_PRIOR_TURN_CONTENT = "FALLBACK_PRIOR_TURN_CONTENT"
    POST_TOOL_NUDGE = "POST_TOOL_NUDGE"




StreamDeltaCallback = Callable[[str], None]

# Silent count-only streaming callback.
#
# Some providers spend many seconds emitting tool-call argument JSON
# fragments (``delta.tool_calls.function.arguments`` for OpenAI-style
# APIs, ``input_json_delta`` for Anthropic) before any visible body
# content is produced.  Those fragments must NOT render in the visible
# transcript — they're internal control plane — but they DO represent
# real model output that should drive the activity bar's "↓ N tokens"
# counter.  This callback is the count-only sibling of
# ``StreamDeltaCallback``: providers forward each non-visible chunk
# here so the UI estimator advances even during long tool-only turns,
# mirroring ``claude-code`` semantics where every delta type increases
# the live token count regardless of visibility.
StreamSilentCallback = Callable[[str], None]

class EngineStatus(str, Enum):
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    MAX_ITERATIONS = "MAX_ITERATIONS"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedResponse:
    content: str | None = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    # Providers that expose reasoning should store it in metadata under
    # ``reasoning_content`` or ``reasoning_details``. The response
    # classifier reads those keys; there is intentionally no top-level
    # ``reasoning`` field in this contract.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnContext:
    session_id: str
    iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    turn_id: str | None = None
    interrupt_signal: "InterruptSignal | None" = None


@dataclass(slots=True)
class EngineRequest:
    session_id: str
    user_message: str | None = None
    system_message: str | None = None
    stream_callback: StreamDeltaCallback | None = None
    # Optional count-only sibling of ``stream_callback`` — providers
    # forward non-visible streaming chunks (tool-arg JSON fragments,
    # signatures, redacted thinking) here so the UI's token estimator
    # keeps ticking during long tool-only turns without polluting the
    # visible body.  See :data:`StreamSilentCallback` for rationale.
    stream_silent_callback: StreamSilentCallback | None = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model_config: ModelCallConfig = field(default_factory=ModelCallConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Interactive prompter for plan-mode approval and
    # ``AskUserQuestionTool``. Optional: when ``None``
    # the corresponding tools degrade to a clear "non-interactive"
    # error without crashing.  Untyped (``Any``) to avoid a hard import
    # of ``aether.cli.approval_prompter`` from the runtime layer; it
    # is duck-typed against the ``Prompter`` protocol.
    approval_prompter: Any = None
    # Engine-level permission prompter for dangerous tools. Separate
    # from ``approval_prompter`` because this is a dispatch gate owned
    # by the engine, not a tool-internal interaction helper.
    tool_permission_prompter: Any = None
    interrupt_signal: "InterruptSignal | None" = None


@dataclass(slots=True)
class EngineResult:
    """Result of one ``AgentEngine.run_loop`` invocation.

    Most fields are self-explanatory. ``metadata`` follows a stable,
    additive schema for the top-level keys below:

    * ``request`` — snapshot of the originating ``EngineRequest``.
    * ``turn`` — flat snapshot of ``TurnContext.metadata`` (excluding a
      small set of internal-only accumulator objects, which are exposed
      in their normalised form at the top level instead).
    * ``runtime`` — per-turn retry counters surfaced for observability.
    * ``usage`` — ``CanonicalUsage.to_dict()``: input/output/cache_read/
      cache_write/reasoning + derived prompt/completion/total totals.
      Always present (zero-valued when no LLM call ran).
    * ``api_calls`` — int, number of provider ``generate()`` calls in
      this turn.
    * ``pending_steer`` — unconsumed async user guidance text, or ``None``.
      Populated when ``/steer`` arrived but no tool-result boundary
      existed to inject it safely.
    * ``memory`` — ``{enabled, mode, retrieval_ms, candidate_count,
      injected_count, injected_tokens, scopes, skipped_reason, write_count,
      error}``. Populated for transient memory retrieval and injection
      observability. Memory content is not included here.
    * ``trajectory`` — ``{saved, path, error}``. Populated when optional
      trajectory persistence is enabled.
    * ``resource_cleanup`` — ``{completed, interrupted, acquired, released,
      errors}``. Populated after task-scoped cleanup runs.
    * ``iteration_budget`` — ``{used, max, remaining, grace_consumed}``.
      Filled from ``IterationBudget``; ``max`` mirrors
      ``EngineConfig.max_iterations`` so downstream footers can render
      the bound.
    * ``exit`` — ``{reason, last_msg_role, stuck_after_tool}``.  Auxiliary
      diagnostic to the canonical ``exit_reason`` enum.
    * ``reasoning`` — ``{last_reasoning}``. Reserved shape; populated by
      reasoning-block extraction.
    * ``compaction`` — ``{tier1_spilled_count, tier2_snipped_count,
      tier3_cleared_count, tier4_collapse_segments,
      tier5_summaries_generated}``. Reserved shape; the five-tier
      compaction pipeline increments these as it fires.

    Other keys present in ``metadata`` (``phantom_synth_count``,
    ``tool_names_repaired``, ``recovery_decisions``, etc.) are
    **non-stable ad-hoc** observability fields — useful for
    debugging but consumers SHOULD only depend on the keys listed
    above for production behaviour.

    Underscore-prefixed keys (``_active_tool_call`` etc.) are strictly
    internal engine state and MUST NOT be consumed externally.
    """

    session_id: str
    status: EngineStatus
    exit_reason: ExitReason
    messages: List[Dict[str, Any]]
    iterations: int
    final_response: str | None = None
    error: str | None = None
    task_id: str | None = None
    turn_id: str | None = None
    system_prompt: str | None = None
    streamed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
