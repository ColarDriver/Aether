"""Per-turn iteration budget with cheap-tool refund + grace call.

Sprint 3 / PR 3.2 — replaces the nine scattered
``iterations >= self.config.max_iterations`` checks in
:mod:`aether.agents.core.agent` with a single structured object that
also models two new behaviours:

* **Cheap-tool refund** — turns spent entirely on bookkeeping tools
  (``update_todo``, ``memory_write``, ``session_search`` and similar
  zero-IO operations) refund their consumed slot so the model gets
  the same effective headroom for "real" work as a turn that doesn't
  use them at all.  Without this, every ``update_todo`` call eats
  one of the user's iteration budget for free.
* **Grace call** — once the budget is exhausted, the engine is
  allowed exactly one more LLM round (no tools, no streaming) to
  generate a summary of what got done.  Avoids the previous
  pathological UX where the user saw ``done · 0.0s`` with an empty
  ``final_response`` after a long tool-using turn.

Lifecycle (mirrors :func:`AgentEngine.run_loop`):

    budget = IterationBudget(max_total=config.max_iterations)
    while budget.consume():
        # ... LLM call + tool dispatch ...
        if all(tool is cheap for tool in tool_calls):
            budget.refund()

    if budget.exhausted and budget.grace_call():
        # one summary call permitted

The dataclass is **per-turn, per-session**.  Multiple concurrent
turns each get their own instance — never share one across sessions
or you reintroduce the cross-session-leak family of bugs Sprint 0
closed.

Serialisation: ``to_dict()`` returns a JSON-friendly snapshot suitable
for ``EngineResult.metadata['iteration_budget']`` (PR 3.1 reserved
that slot; this PR fills it with structured fields).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IterationBudget:
    """Track remaining iterations for one turn.

    Counters
    --------
    used:
        Net iterations consumed (``consume_count - refund_count``).
        Equivalent to the legacy ``iterations`` local variable in
        :func:`AgentEngine.run_loop`.
    grace_consumed:
        Whether the one-shot grace round has already been granted.
        Set by :meth:`grace_call`; remains ``False`` for any turn
        that never exhausted the budget.
    consume_count / refund_count:
        Raw counters for observability.  ``consume_count`` is the
        number of LLM rounds that actually attempted real work;
        ``refund_count`` measures how many of those turned out to be
        bookkeeping-only (and therefore got their slot back).  The
        ratio is a rough indicator of how much of the user's budget
        was spent on cheap-tool churn vs. substantive iteration.
    """

    max_total: int
    used: int = 0
    grace_consumed: bool = False
    consume_count: int = 0
    refund_count: int = 0

    @property
    def remaining(self) -> int:
        """Iterations still available before the budget exhausts.

        Always ``>= 0`` — saturates at zero rather than going
        negative when ``used`` somehow exceeds ``max_total`` (which
        shouldn't happen, but a defensive ``max`` here means a stale
        snapshot can never confuse downstream UI rendering).
        """
        return max(0, self.max_total - self.used)

    @property
    def exhausted(self) -> bool:
        """``True`` when no further :meth:`consume` calls will succeed."""
        return self.used >= self.max_total

    def consume(self) -> bool:
        """Try to consume one iteration slot.

        Returns ``True`` if a slot was reserved (caller proceeds with
        the iteration), ``False`` if the budget is already exhausted
        (caller should break out of the loop).  Idempotent on a
        single failure: repeated calls after exhaustion keep
        returning ``False`` without further state change.
        """
        if self.exhausted:
            return False
        self.used += 1
        self.consume_count += 1
        return True

    def refund(self) -> None:
        """Cancel the most recent :meth:`consume` (cheap-tool path).

        No-op when ``used == 0`` — the engine should never refund
        more than it consumed, but this defensive guard keeps a
        misuse from underflowing into negative territory.  Increments
        ``refund_count`` so observability sees the cumulative number
        of bookkeeping-only iterations even after the slots they
        occupied are reclaimed.
        """
        if self.used > 0:
            self.used -= 1
            self.refund_count += 1

    def grace_call(self) -> bool:
        """Permit one extra iteration after exhaustion (summary path).

        Returns ``True`` the first time it is called and ``False``
        every time after — the contract is "exactly one grace round
        per turn, ever".  ``grace_consumed`` flips to ``True`` on
        the granting call so :func:`AgentEngine._handle_max_iterations`
        can guard against accidental double-summary.

        Grace deliberately does **not** modify ``used`` or
        ``max_total``: it is a one-time bonus outside the regular
        budget accounting, so :meth:`exhausted` keeps reporting
        ``True`` even after grace is granted.  The summary path is
        a side-channel; the loop itself stays terminated.
        """
        if self.grace_consumed:
            return False
        self.grace_consumed = True
        return True

    def to_dict(self) -> dict[str, int | bool]:
        """Serialise to a JSON-friendly dict for ``EngineResult.metadata``.

        Field shape is the stable v1 schema PR 3.1 reserved under
        ``metadata['iteration_budget']``.  Adding fields here is
        backwards-compatible (consumers ignore unknown keys);
        renaming or removing fields is a breaking change to the
        public engine contract — coordinate with downstream
        observability + CLI footer changes.
        """
        return {
            "used": self.used,
            "max": self.max_total,
            "remaining": self.remaining,
            "grace_consumed": self.grace_consumed,
            "consume_count": self.consume_count,
            "refund_count": self.refund_count,
        }
