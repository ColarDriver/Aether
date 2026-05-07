"""Structured exceptions raised by ``ModelProvider`` implementations.

This module defines the *only* exception type that ``ModelProvider.generate``
implementations should raise to signal a transport/HTTP failure to the engine.
Any provider-internal exception (httpx.HTTPStatusError, httpx.TransportError,
TimeoutError, etc.) must be wrapped into a ``ProviderInvocationError`` before
escaping the provider boundary.

Why this matters
----------------
Until Sprint 0, ``OpenAICompatibleModel`` did its own retry loop with blocking
``time.sleep`` calls.  This meant the engine layer could not:

* Cancel during retry waits (interrupts blocked by ``time.sleep``).
* Switch to a fallback provider (engine never sees the error until retry is
  exhausted).
* Trigger context compression on 413 / context-overflow errors.
* Apply different recovery strategies based on error class (rate-limit vs
  auth vs network drop vs transient 5xx).

By moving retry control up to the engine, ``ModelProvider`` becomes a
**single-shot** abstraction: it makes one network call and returns either a
``NormalizedResponse`` or a structured error.  All retry policy is owned by
``runtime/recovery.py`` (``RecoveryStrategy`` chain — added in PR 0.3).

Field semantics
---------------
* ``raw`` — the original underlying exception (for logging / debugging).
* ``status_code`` — HTTP status if the request actually got a response from
  the server, otherwise ``None``.
* ``retry_after_seconds`` — parsed from the ``Retry-After`` / ``Retry-After-Ms``
  response header.  Engine layer SHOULD wait at least this long before
  re-attempting (capped to a sane maximum upstream).  ``None`` means "no
  hint, use default backoff".
* ``body_summary`` — short truncated copy of the response body / error text
  used for error classification (Sprint 2 ``ErrorClassifier`` reads this).
* ``is_network_error`` — ``True`` for transport-level failures (DNS, TLS,
  connection reset, read timeout) where ``status_code`` is unavailable.
  Distinguishing this from "server returned 5xx" matters for retry policy:
  network errors are usually safer to retry, while a 4xx body explicitly
  rejects the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderInvocationError(Exception):
    """Single structured exception raised by ``ModelProvider.generate``.

    Subclassing ``Exception`` keeps it usable in ``except Exception`` blocks,
    but engine code SHOULD catch it explicitly so it can read the structured
    fields below.  The string form (used by ``str(exc)``) collapses the most
    relevant fields into a one-line summary suitable for logs.
    """

    raw: BaseException | None = None
    status_code: int | None = None
    retry_after_seconds: float | None = None
    body_summary: str | None = None
    is_network_error: bool = False
    # Free-form metadata bag — providers can attach extra hints (e.g. the
    # endpoint path that was hit, the request id, etc.) for the engine /
    # recovery strategies to consume.  Engine code should treat unknown keys
    # as opaque.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        # Initialise Exception with a stable summary so str(exc) is useful
        # without having to introspect the dataclass fields.
        Exception.__init__(self, self._build_message())

    def _build_message(self) -> str:
        if self.is_network_error:
            head = "provider network error"
        elif self.status_code is not None:
            head = f"provider HTTP {self.status_code}"
        else:
            head = "provider error"

        body = self.body_summary or (str(self.raw) if self.raw is not None else "")
        if body:
            head = f"{head}: {body}"
        if self.retry_after_seconds is not None:
            head = f"{head} (retry-after={self.retry_after_seconds:.2f}s)"
        return head

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._build_message()


__all__ = ["ProviderInvocationError"]
