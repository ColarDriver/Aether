"""Engine-side recovery strategies for ``ProviderInvocationError``.

Background
----------
Retry and backoff logic used to live inside
``OpenAICompatibleModel.generate`` itself as a hand-rolled loop with
blocking ``time.sleep`` calls. The provider layer is now single-shot
and raises structured ``ProviderInvocationError`` instances instead.
This module is the engine-layer counterpart that decides what to do
when such an error escapes the provider.

Design
------
A ``RecoveryStrategy`` is queried with an ``AttemptState`` (how many tries
have we already made? how long have we waited?) and an error, and returns
a ``RecoveryDecision``:

* ``retry=True``  → engine waits ``wait_seconds`` (interruptibly), then
                    re-issues ``provider.generate(...)``.
* ``retry=False`` → engine falls through to the existing middleware
                    ``on_error`` path so middleware can convert the failure
                    into a user-facing assistant response.

The wait is **interrupt-aware**: if the session is interrupted while the
strategy is sleeping, ``wait_interruptible`` returns ``False`` and the
engine treats this as a normal interruption (no further retries).

Built-in strategies:

* ``NoRetryStrategy``       — strategy that always gives up (used for
                              tests / scripted providers / when retries
                              are explicitly undesired).
* ``GenericBackoffStrategy``— exponential backoff for the well-known
                              retriable status codes (429 / 5xx /
                              network errors), respecting any
                              ``Retry-After`` hint surfaced by the
                              provider.  This is the **default** wired
                              into ``AgentEngine`` so behaviour parity
                              with the previous in-provider retry loop
                              is preserved.

The strategy surface also supports fallback-provider routing, context
compression, credential rotation, and similar recovery hints.

Why not just bring back the old in-provider retry?
--------------------------------------------------
The old design had three concrete failures:

1. ``time.sleep`` blocked the engine — interrupts could not preempt a
   waiting retry.  This module's ``wait_interruptible`` polls the
   ``InterruptController`` so every wait is preemptable.
2. The engine could not see error metadata (status_code,
   retry_after_seconds) until **after** all retries exhausted, so
   features like context-overflow → compression were impossible.
3. Retry counters were buried inside provider state, invisible to
   external observability and tests.

Strategy implementations live here, in the engine layer, so they can
read ``ProviderInvocationError`` fields directly and update
``TurnContext.metadata`` for downstream observability.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from aether.runtime.recovery.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
)
from aether.runtime.recovery.provider_errors import ProviderInvocationError, ResponseInvalidError
from aether.runtime.control.interrupt_signal import InterruptSignal

if TYPE_CHECKING:  # pragma: no cover
    from aether.runtime.core.contracts import TurnContext
    from aether.runtime.control.interrupts import InterruptController


# HTTP status codes the generic strategy will retry by default.  Mirrors the
# previous ``OpenAICompatibleModel`` set so behavior stays aligned.
DEFAULT_RETRIABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 529})

# Hard upper bound on a single wait — even if the server asks for 10 minutes
# we cap so an honestly buggy ``Retry-After`` cannot stall an interactive
# session indefinitely.  Interrupts are still respected during the wait.
DEFAULT_MAX_WAIT_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# Decision / state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AttemptState:
    """Mutable state describing the retry history for a single LLM call.

    The engine creates one ``AttemptState`` per ``provider.generate`` call
    site.  It's mutated **in-place** as retries happen — this keeps the
    strategy interface free of return-value plumbing for state.

    Field semantics:

    * ``attempt`` — 1-indexed attempt number.  ``1`` means the first try
      has just failed (we are deciding whether to make a *second* try).
    * ``total_wait_seconds`` — sum of all waits so far.  Strategies can
      use this to enforce a global wall-clock budget if desired.
    * ``errors`` — chronological log of every error observed for this
      call.  Useful for debug logs and for letting strategies inspect
      the trajectory (e.g. "first 429, then network error → still
      retriable").
    """

    attempt: int = 0
    total_wait_seconds: float = 0.0
    errors: list[ProviderInvocationError] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class RecoveryDecision:
    """Engine-actionable verdict from a ``RecoveryStrategy``.

    Core control flow:

    * ``retry=True``  → engine sleeps for ``wait_seconds`` (interruptible),
                        then re-issues ``provider.generate``.
    * ``retry=False`` → engine stops trying and falls through to the
                        middleware ``on_error`` pipeline.

    Additional orthogonal flags:

    * ``activate_fallback`` — if ``True`` AND the engine has a
      ``FallbackChain`` with a remaining slot AND
      ``EngineConfig.fallback_chain_enabled`` is ``True``, the engine
      rotates to the next provider before the next attempt.  Failed
      activations (chain exhausted, fallback disabled) degrade to the
      base ``retry`` / give-up behaviour.
    * ``compress_context`` — informational hint that the next attempt
      should run after the engine compresses the prompt history.
      Strategies surface this hint and the engine maps it to
      ``ExitReason.CONTEXT_EXHAUSTED`` / ``PAYLOAD_TOO_LARGE`` when
      no compressor is available.
    * ``strip_thinking`` — informational hint for the
      ``thinking_signature`` recovery path: the next attempt should
      drop ``reasoning_details`` from the assistant message stream.
      The engine records this hint in
      ``context.metadata['recovery_decisions']``.
    * ``classified_reason`` — the ``FailoverReason.value`` string that
      led to this decision.  Always set when the decision originated
      from a classifier-aware strategy; ``None`` when the decision was
      produced by a hand-coded path (e.g. interrupt-aware wait).

    The two new flags are *additive*: ``retry=True`` paired with
    ``activate_fallback=True`` means "rotate first, then immediately
    retry on the new provider".  ``retry=True`` paired with
    ``compress_context=True`` means "compress history first, then
    retry".  When the engine cannot honour a hint (chain exhausted,
    no compressor), it degrades the decision rather than ignoring it
    silently — see ``_invoke_provider_with_recovery`` in agent.py.

    ``reason`` is a short tag (``"backoff"`` / ``"give-up:non-retriable"`` /
    ``"rate-limit:fallback"`` / ``"context-overflow:compress-required"``,
    etc.) recorded in ``TurnContext.metadata['recovery_decisions']`` for
    observability.
    """

    retry: bool
    wait_seconds: float = 0.0
    reason: str = ""
    activate_fallback: bool = False
    compress_context: bool = False
    strip_thinking: bool = False
    classified_reason: Optional[str] = None

    @classmethod
    def give_up(
        cls,
        reason: str,
        *,
        activate_fallback: bool = False,
        compress_context: bool = False,
        classified_reason: Optional[str] = None,
    ) -> "RecoveryDecision":
        return cls(
            retry=False,
            wait_seconds=0.0,
            reason=f"give-up:{reason}",
            activate_fallback=activate_fallback,
            compress_context=compress_context,
            classified_reason=classified_reason,
        )

    @classmethod
    def retry_after(
        cls,
        seconds: float,
        reason: str = "backoff",
        *,
        activate_fallback: bool = False,
        compress_context: bool = False,
        strip_thinking: bool = False,
        classified_reason: Optional[str] = None,
    ) -> "RecoveryDecision":
        return cls(
            retry=True,
            wait_seconds=max(0.0, seconds),
            reason=reason,
            activate_fallback=activate_fallback,
            compress_context=compress_context,
            strip_thinking=strip_thinking,
            classified_reason=classified_reason,
        )


# ---------------------------------------------------------------------------
# Strategy interface + built-in implementations
# ---------------------------------------------------------------------------


class RecoveryStrategy(ABC):
    """Abstract decision-maker for provider invocation errors."""

    @abstractmethod
    def decide(
        self,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
        context: "TurnContext",
    ) -> RecoveryDecision:
        """Return whether to retry the provider call after ``error``.

        ``attempt_state.attempt`` reflects how many attempts have already
        failed including the current ``error``.  Strategies should use it
        to enforce budgets.
        """


class NoRetryStrategy(RecoveryStrategy):
    """Always give up.  Equivalent to disabling the recovery layer."""

    def decide(
        self,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
        context: "TurnContext",
    ) -> RecoveryDecision:
        return RecoveryDecision.give_up("no-retry-policy")


@dataclass(slots=True)
class GenericBackoffStrategy(RecoveryStrategy):
    """Exponential backoff for retriable HTTP status codes / network errors.

    Matches the provider retry semantics the engine expects by default:
    * Retriable on ``429``, ``500``, ``502``, ``503``, ``504``, ``529`` and
      any ``is_network_error=True`` failure.
    * Up to ``max_attempts`` total tries (so ``max_attempts - 1`` retries).
    * Backoff is ``base_wait_seconds * 2 ** (attempt - 1)``, with the
      provider's ``retry_after_seconds`` taking precedence when present.
    * Each individual wait is clamped to ``max_wait_seconds``.

    Non-retriable errors (4xx other than 429, malformed-JSON marked as
    network error but non-retriable status, etc.) result in immediate
    give-up so the middleware path can build a user-facing message.
    """

    max_attempts: int = 3
    base_wait_seconds: float = 2.0
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS
    retriable_status_codes: frozenset[int] = DEFAULT_RETRIABLE_STATUS_CODES

    def decide(
        self,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
        context: "TurnContext",
    ) -> RecoveryDecision:
        if attempt_state.attempt >= self.max_attempts:
            return RecoveryDecision.give_up("budget-exhausted")

        if not self._is_retriable(error):
            return RecoveryDecision.give_up("non-retriable")

        # Server-supplied hint always wins over our exponential schedule.
        # Cap to ``max_wait_seconds`` so a misbehaving 10-minute hint
        # cannot stall an interactive session.
        if error.retry_after_seconds is not None:
            wait = min(self.max_wait_seconds, float(error.retry_after_seconds))
            return RecoveryDecision.retry_after(wait, reason="retry-after-header")

        # 1st failure → wait base; 2nd → 2*base; 3rd → 4*base; ...
        wait = self.base_wait_seconds * (1 << max(0, attempt_state.attempt - 1))
        wait = min(self.max_wait_seconds, wait)
        return RecoveryDecision.retry_after(wait, reason="exponential-backoff")

    def _is_retriable(self, error: ProviderInvocationError) -> bool:
        if error.is_network_error:
            return True
        # Keep ``ResponseInvalidError`` retriable here so this strategy
        # remains a safe drop-in for tests and scripted callers that do
        # not use classifier-aware recovery.
        if isinstance(error, ResponseInvalidError):
            return True
        return error.status_code in self.retriable_status_codes


# ---------------------------------------------------------------------------
# Classifier-aware composite strategy
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassifiedRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy that branches on ``classify_api_error`` output.

    This is the production default. Every ``ProviderInvocationError``
    is first run through ``classify_api_error`` and the resulting
    ``FailoverReason`` selects a recovery shape:

    +------------------------+--------+----------+----------+----------+
    | reason                 | retry  | fallback | compress | strip-thk|
    +========================+========+==========+==========+==========+
    | rate_limit             |  yes   |   yes*   |    no    |    no    |
    | billing                |   no   |   yes    |    no    |    no    |
    | overloaded             |  yes   |   no     |    no    |    no    |
    | server_error           |  yes   |   no     |    no    |    no    |
    | timeout                |  yes   |   no     |    no    |    no    |
    | context_overflow       |  yes   |   no     |   yes    |    no    |
    | payload_too_large      |  yes   |   no     |   yes    |    no    |
    | image_too_large        |  yes   |   no     |    no    |    no    |
    | llama_cpp_grammar_pat. |  yes   |   no     |    no    |    no    |
    | thinking_signature     |  yes   |   no     |    no    |   yes    |
    | long_context_tier      |  yes   |   no     |   yes    |    no    |
    | response_invalid       |  yes   |   yes    |    no    |    no    |
    | stream_stalled         |  yes   |   no     |    no    |    no    |
    | model_not_found        |   no   |   yes    |    no    |    no    |
    | auth                   |   no   |   yes    |    no    |    no    |
    | format_error           |   no   |   yes    |    no    |    no    |
    | unknown                | backoff|   no     |    no    |    no    |
    +------------------------+--------+----------+----------+----------+

    \\* ``rate_limit`` only requests fallback when the strategy detects a
    long ``Retry-After`` (configurable via
    ``rate_limit_fallback_threshold_seconds``).  Short waits (sub-30 s
    by default) just sleep on the same provider — rotating after a
    1-second hiccup is more disruptive than the wait itself.

    Attempt budget
    --------------
    The strategy tracks per-attempt retry budgets in ``AttemptState``
    just like ``GenericBackoffStrategy``.  ``max_attempts`` here counts
    *all* attempts on the current ``AgentEngine._invoke_provider_with_recovery``
    invocation — including attempts that succeed only after a fallback
    rotation.  When the engine rotates the chain it asks the strategy
    to ``reset_attempt_state`` (see ``agent.py``) so the new provider
    gets a fresh budget; without that reset, a 5-provider chain would
    burn its entire budget on the first slot.

    Wait calculation
    ----------------
    * ``rate_limit``: ``Retry-After`` header preferred, capped to
      ``max_wait_seconds``.  When the wait would exceed
      ``rate_limit_fallback_threshold_seconds`` we set
      ``activate_fallback=True`` so the engine tries another provider
      immediately.
    * Generic retriable: exponential backoff ``base_wait_seconds *
      2**(attempt-1)``, capped to ``max_wait_seconds``.

    Composability
    -------------
    The strategy is a single class rather than a chain of mini
    strategies.  Each ``FailoverReason`` maps to one branch in
    ``_dispatch`` — adding a reason or refining a hint is a localised
    edit, and the dispatch table in this docstring is the single
    source of truth. Per-reason strategies can be extracted later if
    they ever need to coexist with separate credential-pool or
    OAuth-refresh logic. Until then the inline branches keep the code
    path obvious.
    """

    max_attempts: int = 5
    base_wait_seconds: float = 2.0
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS
    rate_limit_fallback_threshold_seconds: float = 30.0

    def decide(
        self,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
        context: "TurnContext",
    ) -> RecoveryDecision:
        # Always classify, even if we'll delegate.  The reason string
        # is part of the recovery breadcrumb every consumer reads —
        # surfacing ``classified_reason`` on the decision means the
        # engine layer can record it without re-classifying.
        classified = classify_api_error(
            error,
            provider=str(context.metadata.get("active_provider_name") or ""),
            model=str(context.metadata.get("active_model") or ""),
            approx_tokens=int(context.metadata.get("approx_tokens") or 0),
            context_length=int(context.metadata.get("context_length") or 200_000),
            num_messages=int(context.metadata.get("num_messages") or 0),
        )
        # Stash the classification for downstream consumers / tests.
        # We deliberately overwrite each call so the most recent
        # decision drives observability — the recovery_decisions trail
        # already keeps the historical timeline.
        context.metadata["last_classified_reason"] = classified.reason.value

        if attempt_state.attempt >= self.max_attempts:
            return RecoveryDecision.give_up(
                "budget-exhausted",
                activate_fallback=classified.should_fallback,
                compress_context=classified.should_compress,
                classified_reason=classified.reason.value,
            )

        return self._dispatch(classified, error, attempt_state)

    def _dispatch(
        self,
        classified: ClassifiedError,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
    ) -> RecoveryDecision:
        reason = classified.reason

        # ── Hard-stop reasons (no retry on same payload) ───────────
        if reason in (FailoverReason.billing, FailoverReason.auth, FailoverReason.format_error):
            return RecoveryDecision.give_up(
                f"non-retriable:{reason.value}",
                activate_fallback=classified.should_fallback,
                classified_reason=reason.value,
            )

        if reason is FailoverReason.model_not_found:
            return RecoveryDecision.give_up(
                "non-retriable:model_not_found",
                activate_fallback=True,
                classified_reason=reason.value,
            )

        # ── Rate limit: server-supplied wait + fallback when long ───
        # ``>=`` semantics on purpose: ``threshold=0`` means "always
        # rotate on rate_limit", which is the ergonomic knob for
        # operators who'd rather burn fallback budget than block on
        # any throttle.  Strict ``>`` would force callers to pass
        # ``-1.0`` which is awkward and easy to forget.
        if reason is FailoverReason.rate_limit:
            wait = self._wait_for_rate_limit(error, attempt_state)
            should_fallback = wait >= self.rate_limit_fallback_threshold_seconds
            return RecoveryDecision.retry_after(
                wait,
                reason="rate-limit:fallback" if should_fallback else "rate-limit:wait",
                activate_fallback=should_fallback,
                classified_reason=reason.value,
            )

        # ── Response invalid: eager fallback ───────────────────────
        if reason is FailoverReason.response_invalid:
            return RecoveryDecision.retry_after(
                0.0,
                reason="response-invalid:fallback",
                activate_fallback=True,
                classified_reason=reason.value,
            )

        # ── Stream stalled: provider has self-disabled streaming;
        #     immediately retry on the same provider (non-streaming).
        if reason is FailoverReason.stream_stalled:
            return RecoveryDecision.retry_after(
                0.0,
                reason="stream-stalled:disable-streaming",
                classified_reason=reason.value,
            )

        # ── Context overflow: signal compression hint to the engine ─
        if reason in (FailoverReason.context_overflow, FailoverReason.long_context_tier):
            return RecoveryDecision.retry_after(
                0.0,
                reason=f"{reason.value}:compress-required",
                compress_context=True,
                classified_reason=reason.value,
            )

        if reason is FailoverReason.payload_too_large:
            return RecoveryDecision.retry_after(
                0.0,
                reason="payload-too-large:compress-required",
                compress_context=True,
                classified_reason=reason.value,
            )

        # ── Image too large: shrink inline base64 image payloads, retry once
        if reason is FailoverReason.image_too_large:
            return RecoveryDecision.retry_after(
                0.0,
                reason="image-too-large:shrink-and-retry",
                classified_reason=reason.value,
            )

        # ── llama.cpp grammar: strip incompatible schema keywords, retry
        if reason is FailoverReason.llama_cpp_grammar_pattern:
            return RecoveryDecision.retry_after(
                0.0,
                reason="schema-sanitizer:strip-pattern-format",
                classified_reason=reason.value,
            )

        # ── Thinking signature: strip reasoning, retry once ───────
        if reason is FailoverReason.thinking_signature:
            return RecoveryDecision.retry_after(
                0.0,
                reason="thinking-signature:strip-and-retry",
                strip_thinking=True,
                classified_reason=reason.value,
            )

        # ── Overloaded / server / timeout: exponential backoff ────
        if reason in (
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        ):
            wait = self._exponential_wait(attempt_state)
            return RecoveryDecision.retry_after(
                wait,
                reason=f"{reason.value}:backoff",
                classified_reason=reason.value,
            )

        # ── Unknown: exponential backoff retry ────────────────────
        # The ``unknown`` bucket means "the classifier could not pin
        # a reason" — usually a generic 5xx body or a fresh provider
        # error shape we haven't taught the matcher about yet.  Per
        # the classifier contract this should retry with backoff
        # (``retryable=True`` on the ``ClassifiedError`` itself), not
        # give up. Delegating to ``GenericBackoffStrategy``
        # would mis-classify ``ProviderInvocationError()`` (no status,
        # no network flag) as non-retriable and surface a hard failure
        # for what is most likely a transient gateway hiccup.
        wait = self._exponential_wait(attempt_state)
        return RecoveryDecision.retry_after(
            wait,
            reason="unknown:backoff",
            classified_reason=reason.value,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(
        self,
        error: ProviderInvocationError,
        attempt_state: AttemptState,
    ) -> float:
        if error.retry_after_seconds is not None:
            return min(self.max_wait_seconds, float(error.retry_after_seconds))
        return self._exponential_wait(attempt_state)

    def _exponential_wait(self, attempt_state: AttemptState) -> float:
        wait = self.base_wait_seconds * (1 << max(0, attempt_state.attempt - 1))
        return min(self.max_wait_seconds, wait)


# ---------------------------------------------------------------------------
# Wait helper
# ---------------------------------------------------------------------------


def wait_interruptible(
    seconds: float,
    *,
    interrupt_controller: "InterruptController | None",
    session_id: str,
    poll_interval: float = 0.1,
    interrupt_signal: InterruptSignal | None = None,
) -> bool:
    """Sleep up to ``seconds`` and return early when interrupted."""
    if interrupt_signal is None and interrupt_controller is not None:
        interrupt_signal = interrupt_controller.signal_for(session_id)
    if seconds <= 0:
        return interrupt_signal is None or not interrupt_signal.is_aborted()
    if interrupt_signal is not None:
        return not interrupt_signal.wait(seconds)

    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(poll_interval, remaining))


__all__ = [
    "AttemptState",
    "RecoveryDecision",
    "RecoveryStrategy",
    "NoRetryStrategy",
    "GenericBackoffStrategy",
    "ClassifiedRecoveryStrategy",
    "wait_interruptible",
    "DEFAULT_RETRIABLE_STATUS_CODES",
    "DEFAULT_MAX_WAIT_SECONDS",
]
