"""Structured exceptions raised by ``ModelProvider`` implementations.

This module defines the *only* exception type that ``ModelProvider.generate``
implementations should raise to signal a transport/HTTP failure to the engine.
Any provider-internal exception (httpx.HTTPStatusError, httpx.TransportError,
TimeoutError, etc.) must be wrapped into a ``ProviderInvocationError`` before
escaping the provider boundary.

Why this matters
----------------
``OpenAICompatibleModel`` used to do its own retry loop with blocking
``time.sleep`` calls. This meant the engine layer could not:

* Cancel during retry waits (interrupts blocked by ``time.sleep``).
* Switch to a fallback provider (engine never sees the error until retry is
  exhausted).
* Trigger context compression on 413 / context-overflow errors.
* Apply different recovery strategies based on error class (rate-limit vs
  auth vs network drop vs transient 5xx).

By moving retry control up to the engine, ``ModelProvider`` becomes a
**single-shot** abstraction: it makes one network call and returns either a
``NormalizedResponse`` or a structured error.  All retry policy is owned by
``runtime/recovery.py``.

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
  used for error classification.
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


@dataclass
class StreamStallError(ProviderInvocationError):
    """Raised when an SSE stream produces no events for too long.

    Added so the engine's recovery layer can distinguish "the network
    got cut" from "the stream is hung mid-way". A stall is
    semantically a network-level failure (the server is alive enough to keep
    the TCP connection open but never sends another byte), so we set
    ``is_network_error=True`` on the parent. ``GenericBackoffStrategy``
    therefore retries it like any other transport error — but providers also
    use this signal to flip their own ``_disable_streaming`` flag and fall
    back to non-streaming for the rest of the session.

    ``stalled_after_seconds`` records how long the wait was before we gave
    up, so log messages and observability can pinpoint slow backends.
    """

    stalled_after_seconds: float = 0.0

    def __post_init__(self) -> None:  # pragma: no cover - inherits behaviour
        # Ensure the base-class invariants hold even when callers forget to
        # set them explicitly.  is_network_error must be True for the
        # recovery layer to retry; status_code is None because we never got
        # a response status.
        self.is_network_error = True
        self.status_code = None
        super().__post_init__()


@dataclass
class ResponseInvalidError(ProviderInvocationError):
    """Raised when a provider returns HTTP 200 but the response shape is malformed.

    Covers the "OpenRouter returned an `error` field in a 200 body" /
    "choices is empty" / "message is null" class of failures.
    The engine catches this from ``provider.validate_response`` and treats
    it like any other ``ProviderInvocationError``. We deliberately mark
    this as ``is_network_error=True`` so the recovery layer can pick it
    up through the generic retry path when the classifier-aware path is
    not in use.

    ``validation_errors`` is the human-readable list returned by
    ``ModelProvider.validate_response``.  It propagates into the recovery-
    decision log so future strategies can branch on the specific defect.
    """

    validation_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:  # pragma: no cover - inherits behaviour
        # Classify as network-error-equivalent so the generic retry path
        # picks it up when the caller is not using classifier-aware
        # recovery.
        self.is_network_error = True
        # Body summary defaults to "; ".join(validation_errors) so logs are
        # immediately useful without callers having to set it manually.
        if not self.body_summary and self.validation_errors:
            self.body_summary = "invalid response: " + "; ".join(self.validation_errors)
        super().__post_init__()


__all__ = [
    "ProviderInvocationError",
    "StreamStallError",
    "ResponseInvalidError",
]
