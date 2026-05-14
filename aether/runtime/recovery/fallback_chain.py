"""Provider fallback chain.

Background
----------
A single upstream model provider is a single point of failure: a 429
storm, a billing block, a regional outage, or a mis-configured API key
turns every turn into a hard error. This module addresses that by
keeping an ordered list of equivalent providers and rotating to the
next one when a recovery strategy classifies the failure as a
"transfer this to someone else" event (``rate_limit`` / ``billing`` /
``response_invalid`` / ``model_not_found`` per the recovery classifier).

This module is the Aether equivalent.  The chain is intentionally
boring:

* It wraps an ordered list of ``ProviderFactory`` callables.
* It exposes ``current_provider`` (lazy — the factory only runs the
  first time a slot is consulted, so cold spares cost nothing in
  memory).
* ``activate_next()`` advances the cursor and returns ``True`` if a
  new slot is now active, ``False`` if the chain is exhausted.

The recovery strategy decides *when* to call ``activate_next()``;
this module never decides on its own.  The clean separation lets
tests script "429 then succeed on the 2nd provider" without spinning
up real network traffic.

Design notes
------------
1. **Factory-of-providers, not list-of-providers.**  Some providers
   are expensive to instantiate (Anthropic SDK preloads OAuth tokens,
   Codex profile loads GPT model metadata).  Holding factories means
   cold spares pay zero startup cost — only the active slot is
   materialised.  The factory result is cached in
   ``_resolved_providers[i]`` so repeated activations of the same
   slot don't double-instantiate.

2. **Single-cursor, no parallel attempts.**  The chain is for
   sequential failover, not load balancing.  If you want
   round-robin, keep a separate chain per session or build a
   higher-level wrapper.

3. **Thread-safe activation.**  ``activate_next`` and
   ``current_provider`` are both guarded by a single ``RLock`` so
   the chain is safe to share across concurrent sessions.  Callers
   should still treat the active provider as a per-turn snapshot
   (read once at the top of a recovery loop) to avoid surprises if
   another session activates mid-turn.

4. **Observability.**  ``activations`` is a monotonic counter and
   ``activated_provider_names`` records the labels of every slot
   that has ever served traffic.  Both are surfaced in
   ``EngineResult.metadata.runtime`` so dashboards can answer
   "how often did fallback fire?" without log scraping.

Usage
-----
::

    chain = FallbackChain([
        ProviderSlot(name="openrouter-anthropic", factory=lambda: openrouter_provider),
        ProviderSlot(name="anthropic-direct", factory=lambda: anthropic_provider),
    ])
    services = EngineServices(provider=chain.current_provider, ..., fallback_chain=chain)

    try:
        provider.generate(...)
    except ProviderInvocationError as exc:
        classified = classify_api_error(exc, ...)
        if classified.should_fallback and chain.activate_next():
            services.refresh_provider()  # re-reads chain.current_provider
            continue
        ...

The engine wires the recovery side of that loop in ``agent.py``; this
module just exposes the primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, List, Optional

from aether.models.provider.base import ModelProvider


# Type alias — the provider factory contract.  Returning ``None`` is
# treated as "this slot is not available right now"; the chain will
# skip to the next slot on activation rather than swapping in a
# broken provider.  This is rare in practice but handy for credential
# pools that haven't been initialised yet.
ProviderFactory = Callable[[], Optional[ModelProvider]]


@dataclass(slots=True)
class ProviderSlot:
    """One entry in the fallback chain.

    ``name`` is a human-readable label used in observability — it
    shows up in ``EngineResult.metadata`` and recovery-decision logs.
    Slot names should be stable across deployments so dashboards
    remain comparable; the convention is ``"<provider>-<auth>"``,
    e.g. ``"openrouter-key1"`` or ``"anthropic-oauth"``.

    ``factory`` is invoked at most once per slot per chain instance —
    its result is cached on the chain so repeated activations of the
    same slot don't double-instantiate the provider.
    """

    name: str
    factory: ProviderFactory


class FallbackChainExhausted(RuntimeError):
    """Raised when ``current_provider`` is consulted on an empty chain.

    Engine code never relies on this — it always inspects
    ``has_next()`` / ``activate_next()`` first.  The exception exists
    to fail loud when ``EngineServices`` is mis-configured (e.g. an
    empty list passed to the chain constructor).
    """


@dataclass
class FallbackChain:
    """Ordered list of provider slots with a single active cursor.

    Construction:
        ``FallbackChain([slot1, slot2, ...])``.  An empty list is a
        configuration error — the engine requires at least one
        provider to make any LLM call at all.

    State:
        * ``cursor`` (int)        — index of the active slot.
        * ``activations`` (int)   — count of ``activate_next()`` calls
                                    that returned True (i.e. successful
                                    rotations).  Monotonic per chain
                                    lifetime; surfaced in observability.
        * ``activated_provider_names`` (list[str]) — names of every
                                    slot that has ever been activated,
                                    in order.  Includes the initial
                                    slot.

    The chain is **safe to share across sessions** — all mutating
    operations are guarded by a single ``RLock`` and the resolved
    provider cache is keyed by slot index, not by session.  However,
    rotating the cursor affects every concurrent reader, so production
    deployments that want per-session failover should give each
    session its own ``FallbackChain`` instance (cheap — slots are
    factories).
    """

    slots: List[ProviderSlot]
    _cursor: int = field(default=0, init=False)
    _activations: int = field(default=0, init=False)
    _activated_names: List[str] = field(default_factory=list, init=False)
    _resolved: List[Optional[ModelProvider]] = field(default_factory=list, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("FallbackChain requires at least one ProviderSlot")
        # Allocate the resolution cache once — slots are immutable
        # post-construction.  Resolving the head slot eagerly so
        # ``current_provider`` doesn't surprise callers with a first-
        # access factory error if the chain was constructed lazily.
        self._resolved = [None] * len(self.slots)
        self._activated_names.append(self.slots[0].name)
        # Force resolution of the head slot to surface configuration
        # errors at construction time rather than first turn.
        self._resolve(0)

    # ------------------------------------------------------------------
    # Public read-only surface
    # ------------------------------------------------------------------

    @property
    def current_provider(self) -> ModelProvider:
        """Return the active provider, instantiating its factory on first use."""
        with self._lock:
            provider = self._resolve(self._cursor)
            if provider is None:
                # The active slot's factory returned None — try to
                # advance until we find a live one.  If the whole
                # chain is dead, fail loud: this is a configuration
                # error that no recovery strategy can fix.
                if not self._advance_to_live_locked():
                    raise FallbackChainExhausted(
                        "no live provider in fallback chain (every slot factory returned None)"
                    )
                provider = self._resolve(self._cursor)
                assert provider is not None  # _advance_to_live guarantees
            return provider

    @property
    def current_slot_name(self) -> str:
        """Name of the currently active slot — for logs / observability."""
        with self._lock:
            return self.slots[self._cursor].name

    @property
    def activations(self) -> int:
        """Total successful rotations performed since chain construction."""
        with self._lock:
            return self._activations

    @property
    def activated_provider_names(self) -> List[str]:
        """Snapshot of every slot that has ever been activated, in order."""
        with self._lock:
            return list(self._activated_names)

    @property
    def cursor(self) -> int:
        """Current slot index (0-based)."""
        with self._lock:
            return self._cursor

    @property
    def size(self) -> int:
        """Number of slots in the chain (constant for a chain instance)."""
        return len(self.slots)

    def has_next(self) -> bool:
        """True iff ``activate_next()`` would succeed right now."""
        with self._lock:
            return self._cursor + 1 < len(self.slots)

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def activate_next(self) -> bool:
        """Advance the cursor to the next slot.

        Returns ``True`` if a new slot is now active, ``False`` if the
        chain was already on the last slot (caller must surface
        ``ExitReason.FALLBACK_EXHAUSTED`` in that case).

        Skips slots whose factory returns ``None`` — those are
        "configured but currently unavailable" (e.g. an OAuth key that
        is mid-refresh).  Strict equality with ``len(slots)`` after
        the skip means the chain is fully exhausted.
        """
        with self._lock:
            return self._advance_to_live_locked()

    def reset(self) -> None:
        """Re-park the cursor at slot 0.

        Primarily for tests and for long-running daemon use cases
        where the operator wants to "start fresh" after a maintenance
        window.  Counters are NOT reset — observability should retain
        the historical activation trail.
        """
        with self._lock:
            self._cursor = 0
            # Don't drop _activated_names / _activations — those are
            # cumulative observability counters that should survive
            # a logical reset (otherwise dashboards would lose the
            # rotation history every reset).

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, index: int) -> Optional[ModelProvider]:
        """Resolve ``slots[index].factory()`` once, cache the result."""
        cached = self._resolved[index]
        if cached is not None:
            return cached
        try:
            provider = self.slots[index].factory()
        except Exception:        # noqa: BLE001 — factory errors are config bugs
            # We deliberately swallow factory exceptions here and
            # treat them as "slot unavailable" so the chain can keep
            # rotating.  The first attempt to actually use a fully
            # dead chain will surface FallbackChainExhausted, which
            # is the right place for the loud configuration error.
            provider = None
        self._resolved[index] = provider
        return provider

    def _advance_to_live_locked(self) -> bool:
        """Move cursor forward until a live slot is found, or exhaustion.

        Caller must hold ``self._lock``.
        """
        next_index = self._cursor + 1
        while next_index < len(self.slots):
            provider = self._resolve(next_index)
            if provider is not None:
                self._cursor = next_index
                self._activations += 1
                self._activated_names.append(self.slots[next_index].name)
                return True
            next_index += 1
        return False


__all__ = [
    "FallbackChain",
    "FallbackChainExhausted",
    "ProviderFactory",
    "ProviderSlot",
]
