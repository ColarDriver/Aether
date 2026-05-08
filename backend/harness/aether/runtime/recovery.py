"""Engine-side recovery strategies for ``ProviderInvocationError``.

Background
----------
Until Sprint 0, retry/backoff logic for the LLM provider lived inside
``OpenAICompatibleModel.generate`` itself (a hand-rolled loop with blocking
``time.sleep``).  PR 0.2 collapsed that into a *single-shot* provider that
raises a structured ``ProviderInvocationError``.  This module is the
engine-layer counterpart that decides what to do when such an error
escapes the provider.

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

Sprint 0 only ships:

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

Future PRs will add fallback-provider routing, context compression,
credential rotation, etc., as additional ``RecoveryStrategy``
implementations composed via ``CompositeRecoveryStrategy``.

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
from typing import TYPE_CHECKING

from aether.runtime.provider_errors import ProviderInvocationError, ResponseInvalidError

if TYPE_CHECKING:  # pragma: no cover
    from aether.runtime.contracts import TurnContext
    from aether.runtime.interrupts import InterruptController


# HTTP status codes the generic strategy will retry by default.  Mirrors the
# pre-Sprint-0 ``OpenAICompatibleModel`` set so behaviour parity is preserved.
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

    * ``retry=True``  → engine sleeps for ``wait_seconds`` (interruptible),
                        then re-issues ``provider.generate``.
    * ``retry=False`` → engine stops trying and falls through to the
                        middleware ``on_error`` pipeline.

    ``reason`` is a short tag (``"backoff"``, ``"give-up:non-retriable"``,
    etc.) recorded in ``TurnContext.metadata['recovery_decisions']`` for
    observability.
    """

    retry: bool
    wait_seconds: float = 0.0
    reason: str = ""

    @classmethod
    def give_up(cls, reason: str) -> "RecoveryDecision":
        return cls(retry=False, wait_seconds=0.0, reason=f"give-up:{reason}")

    @classmethod
    def retry_after(cls, seconds: float, reason: str = "backoff") -> "RecoveryDecision":
        return cls(retry=True, wait_seconds=max(0.0, seconds), reason=reason)


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

    Matches the pre-Sprint-0 in-provider retry semantics:
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
        # Sprint 1 stop-gap: invalid-response errors are retried generically
        # so the engine doesn't immediately fail when a single 200-OK body is
        # malformed.  Sprint 2 will replace this with a dedicated
        # ResponseInvalidStrategy that does eager fallback to the next
        # provider — at that point this isinstance check goes away.
        if isinstance(error, ResponseInvalidError):
            return True
        return error.status_code in self.retriable_status_codes


# ---------------------------------------------------------------------------
# Wait helper
# ---------------------------------------------------------------------------


def wait_interruptible(
    seconds: float,
    *,
    interrupt_controller: "InterruptController | None",
    session_id: str,
    poll_interval: float = 0.1,
) -> bool:
    """Sleep up to ``seconds``, polling ``interrupt_controller`` periodically.

    Returns ``True`` if the full duration elapsed normally, ``False`` if an
    interrupt was raised against ``session_id`` part-way through.

    Implementation note: we use ``time.monotonic()`` and a coarse
    ``poll_interval`` (default 100ms) rather than ``threading.Event`` so
    we don't have to plumb a per-session event through the interrupt
    controller.  The 100ms poll is more than fast enough for any human-
    facing interruption — the user's perceived latency is dominated by
    the network round-trip we're waiting on, not by the polling cadence.
    """
    if seconds <= 0:
        return interrupt_controller is None or not interrupt_controller.is_interrupted(session_id)

    deadline = time.monotonic() + seconds
    while True:
        if interrupt_controller is not None and interrupt_controller.is_interrupted(session_id):
            return False
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
    "wait_interruptible",
    "DEFAULT_RETRIABLE_STATUS_CODES",
    "DEFAULT_MAX_WAIT_SECONDS",
]
