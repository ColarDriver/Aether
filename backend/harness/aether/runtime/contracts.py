"""Shared contracts for the Aether loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from aether.config.schema import ModelCallConfig

if TYPE_CHECKING:
    from aether.runtime.interrupt_signal import InterruptSignal


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
    # Sprint 1 / PR 1.1: response-shape validation failed even after the
    # recovery layer's retry budget was exhausted.  Distinct from
    # PROVIDER_ERROR so observers can tell "the API itself broke" apart
    # from "the API kept returning malformed bodies".
    RESPONSE_INVALID = "RESPONSE_INVALID"
    # Sprint 1 / PR 1.2: response hit the model's output budget but we
    # successfully stitched a continuation and returned the full text.
    LENGTH_RECOVERED = "LENGTH_RECOVERED"
    # Sprint 1 / PR 1.2: response hit the output budget repeatedly and we
    # had to stop after rollback / partial return.
    LENGTH_EXHAUSTED = "LENGTH_EXHAUSTED"
    # Sprint 1 / PR 1.3: assistant tool-call payload was cut off mid-stream
    # (either because finish_reason="length" arrived alongside tool_calls
    # or because the JSON arguments did not terminate with "}" / "]").  We
    # refused to dispatch the truncated call and either retried once or
    # surfaced a partial turn.  Distinct from LENGTH_EXHAUSTED so observers
    # can branch on "the model asked for a tool but never finished writing
    # the arguments" vs "the model exhausted its output budget on prose".
    TOOL_CALL_TRUNCATED = "TOOL_CALL_TRUNCATED"
    # Sprint 1.5 / phantom-tool recovery: model wrote shell commands or
    # ``<function=NAME>`` / ``<invoke>`` markers in prose instead of
    # populating the structured ``tool_calls`` field, and the loop sent
    # corrective messages for ``max_phantom_tool_retries`` turns without
    # ever getting back a properly structured invocation.  Distinct from
    # TEXT_RESPONSE because the model *intended* to invoke a tool — the
    # turn is broken, not just complete.  Surfaced to the UI so the
    # user sees a clear "the model never got around to actually running
    # anything" footer instead of a misleading green checkmark.
    PHANTOM_TOOL_INTENT = "PHANTOM_TOOL_INTENT"
    # Sprint 2 / PR 2.2: provider returned 429 (or a 4xx body that
    # classified to ``rate_limit``) and the recovery strategy gave up
    # — usually because the fallback chain is exhausted *and* the
    # rate-limit budget is too long for an interactive turn.  Distinct
    # from PROVIDER_ERROR so observability can surface "throttled"
    # specifically (operators tend to react differently to 429-class
    # failures than to generic 5xx).
    RATE_LIMITED = "RATE_LIMITED"
    # Sprint 2 / PR 2.2: server rejected the request because the prompt
    # exceeds the model's context window.  Surfaced when the recovery
    # strategy decided compression was needed but no compressor is
    # configured (Sprint 3 will wire compression in; until then this
    # exits cleanly so the user knows *why* the call failed instead of
    # silently retrying forever).
    CONTEXT_EXHAUSTED = "CONTEXT_EXHAUSTED"
    # Sprint 2 / PR 2.2: HTTP 413 (or message-pattern-matched
    # equivalent) — the request body itself is too big regardless of
    # the model's context window.  Same fallthrough rationale as
    # CONTEXT_EXHAUSTED above; Sprint 3 compression will remove this
    # being a user-visible terminal in the common case.
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    # Sprint 3 / PR 3.4: the compaction pipeline ran but could not
    # reduce the prompt under the target threshold.  Distinct from
    # CONTEXT_EXHAUSTED so observability can tell "we tried all enabled
    # tiers" apart from the legacy "compression unavailable" terminal.
    COMPRESSION_EXHAUSTED = "COMPRESSION_EXHAUSTED"
    # Sprint 2 / PR 2.2: the fallback chain ran out of providers while
    # the last attempt was still failing.  Distinct from PROVIDER_ERROR
    # so operators can tell "we tried everyone and they all rejected
    # us" apart from "single provider blew up".
    FALLBACK_EXHAUSTED = "FALLBACK_EXHAUSTED"
    # Sprint 2 / PR 2.3 — model emitted unrepairable tool names (typos
    # / hallucinated tools) for ``invalid_tool_max_retries`` iterations
    # in a row.  We surface a distinct terminal so the UI / observability
    # can tell "the model never figured out which tool to call" apart
    # from "the tool ran but failed".
    INVALID_TOOL_REPEATED = "INVALID_TOOL_REPEATED"
    # Sprint 4 / PR 4.2: empty-response recovery terminals.
    PARTIAL_STREAM_RECOVERY = "PARTIAL_STREAM_RECOVERY"
    FALLBACK_PRIOR_TURN_CONTENT = "FALLBACK_PRIOR_TURN_CONTENT"
    POST_TOOL_NUDGE = "POST_TOOL_NUDGE"




StreamDeltaCallback = Callable[[str], None]

# Sprint 3 / PR 3.1 — silent (count-only) streaming callback.
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
# mirroring claude-code's ``onUpdateLength`` semantics where every
# delta type increases the live token count regardless of visibility.
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
    # ``reasoning_content`` or ``reasoning_details``.  Sprint 4's response
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
    # Sprint 3.5 / PR 3.5.7 — interactive prompter for plan-mode
    # approval and ``AskUserQuestionTool``.  Optional: when ``None``
    # the corresponding tools degrade to a clear "non-interactive"
    # error without crashing.  Untyped (``Any``) to avoid a hard import
    # of ``aether.cli.approval_prompter`` from the runtime layer; it
    # is duck-typed against the ``Prompter`` protocol.
    approval_prompter: Any = None
    interrupt_signal: "InterruptSignal | None" = None


@dataclass(slots=True)
class EngineResult:
    """Result of one ``AgentEngine.run_loop`` invocation.

    Most fields are self-explanatory.  ``metadata`` deserves a contract:

    Sprint 3 / PR 3.1 — Stable ``metadata`` v1 schema (additive only).
    The following top-level keys are part of the public API and will not
    be renamed or have their value shape narrowed without a major version
    bump:

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
      Sprint 5.2 populates this when `/steer` arrived but no tool result
      boundary existed to inject it safely.
    * ``trajectory`` — ``{saved, path, error}``.  Sprint 5.5 populates
      this when optional trajectory persistence is enabled.
    * ``resource_cleanup`` — ``{completed, interrupted, acquired, released,
      errors}``.  Sprint 5.6 populates this after task-scoped cleanup runs.
    * ``iteration_budget`` — ``{used, max, remaining, grace_consumed}``.
      Filled with structured data by PR 3.2 (IterationBudget); PR 3.1
      surfaces ``max == EngineConfig.max_iterations`` so downstream
      footers can already render the bound.
    * ``exit`` — ``{reason, last_msg_role, stuck_after_tool}``.  Auxiliary
      diagnostic to the canonical ``exit_reason`` enum.
    * ``reasoning`` — ``{last_reasoning}``.  Reserved shape; populated by
      Sprint 5 reasoning-block extraction.
    * ``compaction`` — ``{tier1_spilled_count, tier2_snipped_count,
      tier3_cleared_count, tier4_collapse_segments,
      tier5_summaries_generated}``.  Reserved shape; PR 3.3..3.7
      five-tier compaction pipeline increments these as it fires.

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
