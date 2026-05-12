"""Sprint 2 / PR 2.1 — structured API-error classification for the engine.

Background
----------
Sprint 0 introduced ``ProviderInvocationError`` as the single structured
exception that providers raise.  Sprint 0 / PR 0.3 then wired a
``RecoveryStrategy`` chain that today only knows two concepts: "retry with
backoff" or "give up".  That single-axis decision is too coarse for the
real-world failure modes the engine has to recover from:

* A 429 with a `Retry-After: 5` header should retry **and** rotate the
  failed credential / hand the next attempt to a fallback provider, not
  just sleep.
* A 413 / context-overflow should trigger **context compression**
  (Sprint 3) before any retry — retrying without compression just burns
  the same payload again.
* An OpenRouter HTTP-200 with an ``error`` body (Sprint 1's
  ``ResponseInvalidError``) should **eagerly fall back** to the next
  provider rather than backoff retry on the same gateway.
* A `400 "thinking_signature invalid"` from Anthropic must strip
  ``reasoning_details`` from the message stream before retrying — the
  next attempt without that fix is guaranteed to fail.

The classifier turns each failure into a ``ClassifiedError`` with a
single ``FailoverReason`` plus four orthogonal recovery hints
(``retryable`` / ``should_compress`` / ``should_rotate_credential`` /
``should_fallback``).  Sprint 2 / PR 2.2 will introduce the strategy
chain that consumes these hints; this PR is **pure classification** so
its tests can be exhaustive and the strategy work can layer on top
without re-deriving anything.

Design notes
------------
1. **Reason taxonomy mirrors hermes-agent's `agent/error_classifier.py`.**
   The two engines share an observability vocabulary so dashboards,
   logs, and operator runbooks read the same regardless of which
   engine is generating the failure.  We deliberately drop a couple of
   hermes-specific reasons that have no Aether-side recovery action
   yet (`oauth_long_context_beta_forbidden`) — these will return at
   the same time as the
   matching strategies in later PRs.  When they come back the only
   change required here is adding the enum entry and the pattern
   matcher; the recovery hints already model the right action shape.

2. **Status-code priority over message text.**  An HTTP status is a
   server-issued classification; it is higher signal than free-text
   pattern matching.  We only fall back to ``_classify_by_message``
   when the status is missing (which happens for transport errors and
   for some SDKs that swallow the status).

3. **Pattern lists are flat & frozen.**  Easy to grep, easy to extend
   with regression cases.  Each pattern carries a comment with a
   real-world provider example so reviewers can tell at a glance why a
   string is on the list.

4. **No I/O, no logging side effects.**  The classifier is a pure
   function of (exception, context tuple).  This makes it cheap to
   unit-test against fixture exceptions and lets the engine call it
   at any point in the recovery loop without worrying about ordering
   side effects with logger handlers.

5. **Adapter-friendly.**  Although the engine currently sees only
   ``ProviderInvocationError``, the classifier accepts a generic
   ``Exception`` so it can be reused if/when a future provider raises
   a non-Aether SDK exception class directly (e.g. a ``LiteLLMError``
   or an ``openai.APIError`` leaked through a thin wrapper).
   ``_extract_status_code`` walks ``__cause__`` chains for exactly
   that case.

Usage
-----
::

    from aether.runtime.recovery.error_classifier import classify_api_error

    classified = classify_api_error(
        exc,
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        approx_tokens=180_000,
        context_length=200_000,
        num_messages=80,
    )
    if classified.should_fallback:
        chain.activate_next()
    elif classified.should_compress:
        compressor.compress()
    elif classified.retryable:
        ...

Each downstream consumer (Sprint 2 PR 2.2 strategies, Sprint 3
compression) should branch on ``classified.reason`` first and read the
boolean hints only as defensive defaults.  The reason is the contract;
the hints are convenience.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from aether.runtime.recovery.provider_errors import (
    ProviderInvocationError,
    ResponseInvalidError,
    StreamStallError,
)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class FailoverReason(str, enum.Enum):
    """Structured cause of an API failure — drives recovery strategy.

    String values match ``hermes-agent``'s enum exactly so cross-engine
    log correlation tools don't need a translation table.  When extending
    this enum, mirror the value naming convention (snake_case, lowercase,
    no provider name in the value).
    """

    # ── Authentication / authorization ───────────────────────────────
    auth = "auth"
    """Transient auth failure (401/403). Recovery: rotate credential,
    refresh token, or fall back to next provider before retry."""

    # ── Billing / quota ──────────────────────────────────────────────
    billing = "billing"
    """402 or confirmed credit exhaustion. Recovery: rotate credential
    immediately; do NOT retry on the same key."""

    rate_limit = "rate_limit"
    """429 or quota throttling. Recovery: honour ``Retry-After`` if
    present, rotate credential / fall back, then retry."""

    # ── Server-side ──────────────────────────────────────────────────
    overloaded = "overloaded"
    """503 / 529 — provider overloaded. Recovery: backoff retry."""

    server_error = "server_error"
    """500 / 502 — internal server error. Recovery: backoff retry."""

    # ── Transport ────────────────────────────────────────────────────
    timeout = "timeout"
    """Connect/read timeout / SSL hiccup / mid-stream disconnect on a
    small session. Recovery: rebuild client, retry."""

    # ── Context / payload ────────────────────────────────────────────
    context_overflow = "context_overflow"
    """Server rejected request because the prompt exceeds the model's
    context window.  Recovery: compress + retry; do NOT just retry."""

    payload_too_large = "payload_too_large"
    """413 — request body itself is too big (independent of context
    window).  Recovery: compress + retry."""

    image_too_large = "image_too_large"
    """Provider rejected an oversized inline image payload.
    Recovery: shrink base64 image blocks on an API-call copy and retry once."""

    # ── Model ────────────────────────────────────────────────────────
    model_not_found = "model_not_found"
    """404 / "invalid model" — the requested model does not exist on
    this provider.  Recovery: try the next provider in the fallback
    chain (the same model may be served elsewhere)."""

    # ── Request format ───────────────────────────────────────────────
    format_error = "format_error"
    """Generic non-recoverable 4xx (other than the well-known buckets
    above).  Recovery: fall back; if exhausted, abort."""

    llama_cpp_grammar_pattern = "llama_cpp_grammar_pattern"
    """llama.cpp/local backend rejected unsupported JSON schema keywords.
    Recovery: strip ``pattern`` / ``format`` from copied tool schemas
    and retry once."""

    # ── Provider-specific ────────────────────────────────────────────
    thinking_signature = "thinking_signature"
    """Anthropic-style 400 "invalid signature" on a reasoning block.
    Recovery: strip ``reasoning_details`` from the message stream and
    retry once on the same provider."""

    long_context_tier = "long_context_tier"
    """Anthropic 429 "extra usage" tier-gate on long-context requests.
    Recovery: cap context to the standard tier (~200k) and compress."""

    # ── Response-shape (Sprint 1 / PR 1.1 hand-off) ──────────────────
    response_invalid = "response_invalid"
    """Provider returned HTTP 200 but the body shape was malformed
    (empty choices, embedded ``error`` field, missing message).
    Recovery: eager fall back to next provider — generic backoff just
    re-asks the same broken gateway."""

    # ── Stream-specific (Sprint 1 / PR 1.1 hand-off) ─────────────────
    stream_stalled = "stream_stalled"
    """SSE stream produced no events for the watchdog interval.
    Recovery: retry on the same provider but force the non-streaming
    path (the provider's ``_disable_streaming`` flag is set on its way
    out, so this is implicit)."""

    # ── Catch-all ────────────────────────────────────────────────────
    unknown = "unknown"
    """Unclassifiable failure.  Recovery: backoff retry; if budget is
    exhausted, abort."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassifiedError:
    """Structured verdict from ``classify_api_error``.

    The four boolean hints (``retryable`` / ``should_compress`` /
    ``should_rotate_credential`` / ``should_fallback``) are
    **orthogonal** — a single failure can warrant any combination
    (e.g. a 429 is ``retryable=True, should_rotate_credential=True,
    should_fallback=True`` because the strategy can either wait,
    rotate to another key, or hand off to another provider).  The
    consumer (recovery strategy) decides the priority.

    ``status_code`` / ``message`` / ``error_context`` carry the bits
    of the source exception that downstream consumers most often need
    to surface in logs and observability — the strategy layer should
    not have to re-walk the original exception object.
    """

    reason: FailoverReason
    status_code: Optional[int] = None
    provider: str = ""
    model: str = ""
    message: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)

    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        """True for any auth-class failure.  Convenience for strategies."""
        return self.reason is FailoverReason.auth


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------
#
# All pattern tables here are **lower-case substring matches** against
# a normalised error_msg.  Substring is intentionally permissive —
# providers reformat their bodies often (e.g. JSON envelope vs plain
# text) and full-string regex matching breaks every quarter.  The
# trade-off: false positives are easier to write a regression test for
# than false negatives.
#
# When adding a new pattern, include a comment with the **provider name
# + paraphrased real example** so reviewers know which provider it
# came from and can tell whether a refactor will break it.

_BILLING_PATTERNS: tuple[str, ...] = (
    "insufficient credits",          # OpenRouter
    "insufficient_quota",            # OpenAI
    "insufficient balance",          # DeepSeek
    "credit balance",                # OpenRouter
    "credits have been exhausted",   # OpenRouter
    "top up your credits",           # OpenRouter
    "payment required",              # generic 402
    "billing hard limit",            # OpenAI
    "exceeded your current quota",   # OpenAI
    "account is deactivated",        # OpenAI
    "plan does not include",         # Anthropic OAuth
)

_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",                  # OpenAI / Anthropic
    "please retry after",
    "resource_exhausted",            # Google Vertex
    "rate increased too quickly",    # Alibaba DashScope
    "throttlingexception",           # AWS Bedrock
    "too many concurrent requests",
    "servicequotaexceededexception", # AWS Bedrock
)

_USAGE_LIMIT_PATTERNS: tuple[str, ...] = (
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
)

_USAGE_LIMIT_TRANSIENT_SIGNALS: tuple[str, ...] = (
    "try again",
    "retry",
    "resets at",
    "reset in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
)

_PAYLOAD_TOO_LARGE_PATTERNS: tuple[str, ...] = (
    "request entity too large",
    "payload too large",
    "error code: 413",
)

_IMAGE_TOO_LARGE_PATTERNS: tuple[str, ...] = (
    "image_too_large",
    "image too large",
    "image is too large",
    "image exceeds",
    "image size",
    "base64 image",
    "image payload",
)

_CONTEXT_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    "max_tokens",
    "maximum number of tokens",
    "exceeds the max_model_len",     # vLLM
    "max_model_len",                 # vLLM
    "prompt length",                 # vLLM
    "input is too long",             # AWS Bedrock
    "maximum model length",
    "context length exceeded",       # Ollama
    "truncating input",              # Ollama
    "slot context",                  # llama.cpp
    "n_ctx_slot",                    # llama.cpp
    "超过最大长度",                   # Chinese providers (Alibaba/Baidu)
    "上下文长度",
    "max input token",
    "input token",
    "exceeds the maximum number of input tokens",
)

_MODEL_NOT_FOUND_PATTERNS: tuple[str, ...] = (
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
)

_AUTH_PATTERNS: tuple[str, ...] = (
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
)

# SSL transient error patterns.  Distinct from generic disconnect
# patterns because we want retry but NOT compression for SSL alerts.
# Strings are duplicated in space and underscore variants because
# OpenSSL formats codes both ways across versions.
_SSL_TRANSIENT_PATTERNS: tuple[str, ...] = (
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    "[ssl:",
)

# Server-disconnect patterns — when present without a status code
# AND the session is large, classify as context_overflow (the
# gateway closed the connection rather than returning a structured
# error for the oversized request).  Otherwise treat as transport
# timeout.
_SERVER_DISCONNECT_PATTERNS: tuple[str, ...] = (
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
)

_LLAMA_CPP_GRAMMAR_CORE_PATTERNS: tuple[str, ...] = (
    "pattern",
    "format",
)

_LLAMA_CPP_GRAMMAR_CONTEXT_PATTERNS: tuple[str, ...] = (
    "json schema",
    "schema",
    "unsupported",
    "not supported",
    "compile",
    "failed",
    "invalid",
    "llama.cpp",
    "llama-server",
    "gbnf",
)

# Transport error type names — matched against ``type(exc).__name__``
# so the classifier still works when an SDK re-raises a transport
# error wrapped in a generic Exception (and the chained __cause__ is
# the real httpx / openai / anthropic SDK exception).
_TRANSPORT_ERROR_TYPES: frozenset[str] = frozenset({
    "ReadTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "ConnectError",
    "RemoteProtocolError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
    "TimeoutError",
    "ReadError",
    "ServerDisconnectedError",
    "SSLError",
    "SSLZeroReturnError",
    "SSLWantReadError",
    "SSLWantWriteError",
    "SSLEOFError",
    "SSLSyscallError",
    "APIConnectionError",
    "APITimeoutError",
})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200_000,
    num_messages: int = 0,
) -> ClassifiedError:
    """Classify an arbitrary API exception into a recovery-aware verdict.

    Priority pipeline (mirrors hermes-agent ``classify_api_error``):

    0.  Aether-native subclasses of ``ProviderInvocationError`` get
        first crack — they carry the most specific signal.  Today
        that's ``StreamStallError`` (→ ``stream_stalled``) and
        ``ResponseInvalidError`` (→ ``response_invalid``).  Both
        unconditionally request fallback because the next attempt
        on the same provider is unlikely to succeed.
    1.  Provider-specific patterns checked first because they require
        narrow status_code + message combinations and would be hidden
        by the generic 400/429 buckets if checked later.
    2.  HTTP status code dispatch with message-aware refinement
        (e.g. 429 + "long context" → ``long_context_tier``).
    3.  Free-text message classification as a fallback when no status
        is available (transport errors).
    4.  SSL/TLS transient patterns *before* the disconnect check —
        an SSL hiccup on a large session is still a transport error,
        not a context overflow.
    5.  Server-disconnect + large-session heuristic for context
        overflow (gateways often drop the connection instead of
        returning a structured error).
    6.  Transport heuristics by exception type (`__class__.__name__`
        in ``_TRANSPORT_ERROR_TYPES``).
    7.  Fallback: ``unknown`` (retryable with backoff).

    Args
    ----
    error
        Exception from the failed API call.  Most often a
        ``ProviderInvocationError`` raised by ``ModelProvider.generate``,
        but the function accepts any ``Exception`` for forward-
        compatibility with thin wrappers around third-party SDKs.
    provider
        Active provider name (``"openrouter"``, ``"anthropic"``, …).
        Used only as metadata on the result; never as a classification
        signal.  This is intentional: error messages should be
        recognisable regardless of which gateway is in front.
    model
        Active model slug.  Same intent as ``provider`` — metadata
        only, no classification side effects.
    approx_tokens, context_length, num_messages
        Used **only** by the disconnect-on-large-session heuristic
        and by the generic-400 fallback to context_overflow.  Default
        values are conservative — passing 0 means "unknown size, do
        not infer context overflow from a generic disconnect".

    Returns
    -------
    ClassifiedError
        Always returns a verdict.  The pipeline is total: it never
        raises and never returns ``None``.
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)

    # Combine raw str(error), structured body message, and OpenRouter
    # metadata.raw (which wraps upstream provider errors as a JSON
    # string buried two levels deep — a real-world Anthropic 400
    # context-overflow is only visible after parsing it).
    error_msg = _build_normalised_message(error, body)
    provider_lower = provider.strip().lower()
    model_lower = model.strip().lower()

    def _result(reason: FailoverReason, **overrides: Any) -> ClassifiedError:
        defaults: Dict[str, Any] = {
            "reason": reason,
            "status_code": status_code,
            "provider": provider,
            "model": model,
            "message": _extract_message(error, body),
            "error_context": _build_error_context(error, status_code, error_type),
        }
        defaults.update(overrides)
        return ClassifiedError(**defaults)

    # ── 0. Aether-native subclasses (highest priority) ───────────────
    # ResponseInvalidError + StreamStallError carry signal that no
    # message/status sniffing can replicate.

    if isinstance(error, ResponseInvalidError):
        return _result(
            FailoverReason.response_invalid,
            retryable=True,
            should_fallback=True,
        )
    if isinstance(error, StreamStallError):
        return _result(
            FailoverReason.stream_stalled,
            retryable=True,
            should_fallback=False,  # provider self-disabled streaming; same gateway is fine
        )

    # ── 1. Provider-specific patterns (highest priority among generic) ──

    if (
        status_code == 400
        and "signature" in error_msg
        and "thinking" in error_msg
    ):
        return _result(
            FailoverReason.thinking_signature,
            retryable=True,
            should_compress=False,
        )

    if (
        status_code == 429
        and "extra usage" in error_msg
        and "long context" in error_msg
    ):
        return _result(
            FailoverReason.long_context_tier,
            retryable=True,
            should_compress=True,
        )

    if _is_llama_cpp_grammar_pattern_error(error_msg):
        return _result(
            FailoverReason.llama_cpp_grammar_pattern,
            retryable=True,
            should_fallback=False,
        )

    if _is_image_too_large_error(error_msg):
        return _result(
            FailoverReason.image_too_large,
            retryable=True,
            should_fallback=False,
        )

    # ── 2. Structured body error_code (highest signal) ───────────────
    # Checked BEFORE status-code dispatch because a structured code in
    # the body is unambiguous (provider-issued classification) while an
    # HTTP status alone often needs message-pattern refinement.  Doing
    # this in the other order would mean a 400 carrying
    # ``error.code=context_length_exceeded`` falls into the generic
    # ``_classify_400`` branch which only consults *message text* — and
    # the message often just says "too long" with no recognisable
    # pattern, mis-classifying as ``format_error``.

    if error_code:
        classified = _classify_by_error_code(error_code, _result)
        if classified is not None:
            return classified

    # ── 3. HTTP status code classification ───────────────────────────

    if status_code is not None:
        classified = _classify_by_status(
            status_code,
            error_msg,
            error_code,
            body,
            provider=provider_lower,
            model=model_lower,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=_result,
        )
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ─────────────────

    classified = _classify_by_message(error_msg, _result)
    if classified is not None:
        return classified

    # ── 5. SSL/TLS transient → retry as timeout ──────────────────────

    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ──────

    if any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS) and status_code is None:
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256_000 and (approx_tokens > 120_000 or num_messages > 200)
        )
        if is_large:
            return _result(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7. Transport heuristics by exception type ────────────────────

    if (
        error_type in _TRANSPORT_ERROR_TYPES
        or isinstance(error, (TimeoutError, ConnectionError, OSError))
        or _is_network_provider_error(error)
    ):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 8. Fallback ──────────────────────────────────────────────────

    return _result(FailoverReason.unknown, retryable=True)


# ---------------------------------------------------------------------------
# Status-code dispatcher
# ---------------------------------------------------------------------------


def _classify_by_status(
    status_code: int,
    error_msg: str,
    error_code: str,
    body: Dict[str, Any],
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int,
    result_fn: Callable[..., ClassifiedError],
) -> Optional[ClassifiedError]:
    """Classify by HTTP status code with refinement from message text."""

    if status_code == 401:
        # Not retryable on the same key — caller (recovery strategy)
        # rotates credential / falls back; if those fail it surfaces
        # the auth failure terminally.
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        # OpenRouter "key limit exceeded" returns 403 with a billing
        # message — treat as billing rather than auth so the fallback
        # path picks a different key.
        if "key limit exceeded" in error_msg or "spending limit" in error_msg:
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return result_fn(
                FailoverReason.model_not_found,
                retryable=False,
                should_fallback=True,
            )
        # Generic 404 — could be a misconfigured base_url for a local
        # llama.cpp / Ollama server; surfacing model_not_found there
        # is misleading.  Classify as unknown so the strategy reports
        # the real error.
        return result_fn(FailoverReason.unknown, retryable=True)

    if status_code == 413:
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    if status_code == 429:
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 400:
        return _classify_400(
            error_msg,
            error_code,
            body,
            provider=provider,
            model=model,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in (500, 502):
        return result_fn(FailoverReason.server_error, retryable=True)

    if status_code in (503, 529):
        return result_fn(FailoverReason.overloaded, retryable=True)

    if 400 <= status_code < 500:
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


def _classify_402(
    error_msg: str,
    result_fn: Callable[..., ClassifiedError],
) -> ClassifiedError:
    """Disambiguate 402 — billing exhaustion vs transient quota."""
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)

    if has_usage_limit and has_transient_signal:
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    return result_fn(
        FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _classify_400(
    error_msg: str,
    error_code: str,
    body: Dict[str, Any],
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int,
    result_fn: Callable[..., ClassifiedError],
) -> ClassifiedError:
    """400 fan-out — context_overflow / model_not_found / rate_limit / billing / format."""

    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Some providers return rate_limit / billing as 400 instead of 429/402.
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Generic 400 + large session → probable context overflow.  Some
    # providers (notably Anthropic via aggregators) return a bare
    # "Error" message when context is too large; the heuristic catches
    # that case so we still get compression rather than a hard abort.
    body_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            body_msg = str(err_obj.get("message") or "").strip().lower()
        if not body_msg:
            body_msg = str(body.get("message") or "").strip().lower()
    is_generic = len(body_msg) < 30 or body_msg in ("error", "")
    is_large = approx_tokens > context_length * 0.4 or (
        context_length <= 256_000 and (approx_tokens > 80_000 or num_messages > 80)
    )

    if is_generic and is_large:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    return result_fn(
        FailoverReason.format_error,
        retryable=False,
        should_fallback=True,
    )


# ---------------------------------------------------------------------------
# Body-code dispatcher
# ---------------------------------------------------------------------------


def _classify_by_error_code(
    error_code: str,
    result_fn: Callable[..., ClassifiedError],
) -> Optional[ClassifiedError]:
    """Classify by structured ``error.code`` string from the response body."""
    code = error_code.lower()

    if code in {"resource_exhausted", "throttled", "rate_limit_exceeded"}:
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
        )
    if code in {"insufficient_quota", "billing_not_active", "payment_required"}:
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if code in {"model_not_found", "model_not_available", "invalid_model"}:
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )
    if code in {"context_length_exceeded", "max_tokens_exceeded"}:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )
    return None


# ---------------------------------------------------------------------------
# Free-text message dispatcher (no status code available)
# ---------------------------------------------------------------------------


def _classify_by_message(
    error_msg: str,
    result_fn: Callable[..., ClassifiedError],
) -> Optional[ClassifiedError]:
    """Pattern-match the message text when no status_code is present."""

    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    if any(p in error_msg for p in _USAGE_LIMIT_PATTERNS):
        if any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS):
            return result_fn(
                FailoverReason.rate_limit,
                retryable=True,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    if any(p in error_msg for p in _AUTH_PATTERNS):
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    return None


def _is_image_too_large_error(error_msg: str) -> bool:
    """Return True for provider errors that specifically reject image size."""

    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return True
    if ("image" in error_msg or "base64" in error_msg) and "5 mb" in error_msg:
        return True
    if "image" in error_msg and "too large" in error_msg:
        return True
    return False


def _is_llama_cpp_grammar_pattern_error(error_msg: str) -> bool:
    """Return True for local backend grammar errors caused by schema keywords."""

    if "grammar" not in error_msg:
        return False
    if not any(p in error_msg for p in _LLAMA_CPP_GRAMMAR_CORE_PATTERNS):
        return False
    return any(p in error_msg for p in _LLAMA_CPP_GRAMMAR_CONTEXT_PATTERNS)


# ---------------------------------------------------------------------------
# Helpers — exception introspection
# ---------------------------------------------------------------------------


def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk ``__cause__`` chain looking for an HTTP status attribute.

    Aether's own ``ProviderInvocationError`` exposes ``status_code``
    directly, but the function still walks the cause chain so it
    works against unwrapped third-party SDK exceptions a future
    provider might surface.
    """
    current: BaseException | None = error
    for _ in range(5):
        if current is None:
            break
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        # Some SDKs (httpx ResponseStatusError, openai APIStatusError)
        # use ``status`` instead.
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        nxt = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return None


def _extract_error_body(error: Exception) -> Dict[str, Any]:
    """Try several conventions to recover the parsed error body.

    ``ProviderInvocationError`` keeps the body summary as a string; we
    try to JSON-parse it because most provider error bodies are JSON.
    Falls back to ``response.json()`` on attached ``response`` objects
    for SDK-style exceptions.
    """
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body
    summary = getattr(error, "body_summary", None)
    if isinstance(summary, str) and summary.strip().startswith(("{", "[")):
        try:
            parsed = json.loads(summary)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    response = getattr(error, "response", None)
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                return parsed
        except Exception:        # noqa: BLE001 — defensive: response.json() is unreliable
            pass
    return {}


def _extract_error_code(body: Dict[str, Any]) -> str:
    """Pull ``error.code`` / ``error.type`` / top-level ``code`` out of body."""
    if not body:
        return ""
    error_obj = body.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and code.strip():
            return code.strip()
    code = body.get("code") or body.get("error_code") or ""
    if isinstance(code, (str, int)):
        return str(code).strip()
    return ""


def _extract_message(error: Exception, body: Dict[str, Any]) -> str:
    """Best human-readable message — body-derived first, then ``str(error)``."""
    if body:
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            msg = error_obj.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
        msg = body.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
    return str(error)[:500]


def _build_normalised_message(error: Exception, body: Dict[str, Any]) -> str:
    """Compose a single lowercase string searched by every pattern matcher.

    Why a *combined* message?  Because providers like OpenRouter
    routinely wrap the original upstream error inside multiple JSON
    layers (``error.message`` is generic, the real signal is in
    ``error.metadata.raw`` parsed as JSON).  Pattern-matching only
    against ``str(error)`` misses those cases.
    """
    raw_msg = str(error).lower()
    body_msg = ""
    metadata_msg = ""

    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            body_msg = str(err_obj.get("message") or "").lower()
            meta = err_obj.get("metadata", {})
            if isinstance(meta, dict):
                raw_json = meta.get("raw") or ""
                if isinstance(raw_json, str) and raw_json.strip():
                    try:
                        inner = json.loads(raw_json)
                        if isinstance(inner, dict):
                            inner_err = inner.get("error", {})
                            if isinstance(inner_err, dict):
                                metadata_msg = str(inner_err.get("message") or "").lower()
                    except (json.JSONDecodeError, TypeError):
                        pass
        if not body_msg:
            body_msg = str(body.get("message") or "").lower()

    parts = [raw_msg]
    if body_msg and body_msg not in raw_msg:
        parts.append(body_msg)
    if metadata_msg and metadata_msg not in raw_msg and metadata_msg not in body_msg:
        parts.append(metadata_msg)
    return " ".join(parts)


def _build_error_context(
    error: Exception,
    status_code: Optional[int],
    error_type: str,
) -> Dict[str, Any]:
    """Snapshot the parts of the exception observers want to see in logs."""
    ctx: Dict[str, Any] = {
        "exception_type": error_type,
        "status_code": status_code,
    }
    is_network_error = getattr(error, "is_network_error", None)
    if isinstance(is_network_error, bool):
        ctx["is_network_error"] = is_network_error
    retry_after = getattr(error, "retry_after_seconds", None)
    if isinstance(retry_after, (int, float)):
        ctx["retry_after_seconds"] = float(retry_after)
    metadata = getattr(error, "metadata", None)
    if isinstance(metadata, dict):
        # Only forward small JSON-serialisable bits — avoid leaking the
        # full underlying SDK exception object into observability.
        for key in ("url", "method", "phase"):
            val = metadata.get(key)
            if isinstance(val, (str, int, float, bool)):
                ctx[key] = val
    return ctx


def _is_network_provider_error(error: Exception) -> bool:
    """True when a ``ProviderInvocationError`` is flagged as transport-level.

    Aether providers raise ``ProviderInvocationError(is_network_error=True)``
    for DNS / TLS / connection-reset / read-timeout failures.  The
    classifier should treat those as ``timeout`` even when the type
    name (``ProviderInvocationError``) is not in
    ``_TRANSPORT_ERROR_TYPES``.
    """
    if not isinstance(error, ProviderInvocationError):
        return False
    return bool(getattr(error, "is_network_error", False))


__all__ = [
    "FailoverReason",
    "ClassifiedError",
    "classify_api_error",
]
