"""LLM error handling middleware for Aether runtime."""

from __future__ import annotations

import logging
import os
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Any

from aether.agents.middlewares.common import RuntimeMiddlewareBase
from aether.runtime.contracts import LoopState, NormalizedResponse, TurnContext

logger = logging.getLogger(__name__)

_RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_BUSY_PATTERNS = (
    "server busy",
    "temporarily unavailable",
    "try again later",
    "please retry",
    "please try again",
    "overloaded",
    "high demand",
    "rate limit",
    "负载较高",
    "服务繁忙",
    "稍后重试",
    "请稍后重试",
)
_QUOTA_PATTERNS = (
    "insufficient_quota",
    "quota",
    "billing",
    "credit",
    "payment",
    "余额不足",
    "超出限额",
    "额度不足",
    "欠费",
)
_AUTH_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "permission",
    "forbidden",
    "access denied",
    "无权",
    "未授权",
)


class LLMErrorHandlingMiddleware(RuntimeMiddlewareBase):
    """Classify provider failures and surface graceful fallback responses."""

    def __init__(
        self,
        *,
        circuit_failure_threshold: int | None = None,
        circuit_recovery_timeout_sec: int | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger=logger_ or logger)

        env_threshold = _env_int("AETHER_CIRCUIT_FAILURE_THRESHOLD")
        env_timeout = _env_int("AETHER_CIRCUIT_RECOVERY_TIMEOUT_SEC")

        self.circuit_failure_threshold = (
            circuit_failure_threshold
            if circuit_failure_threshold is not None
            else (env_threshold if env_threshold is not None else 5)
        )
        self.circuit_recovery_timeout_sec = (
            circuit_recovery_timeout_sec
            if circuit_recovery_timeout_sec is not None
            else (env_timeout if env_timeout is not None else 60)
        )

        self._circuit_lock = threading.Lock()
        self._circuit_failure_count = 0
        self._circuit_open_until = 0.0
        self._circuit_state = "closed"
        self._circuit_probe_in_flight = False

    def before_llm(self, messages: list[dict], context: TurnContext) -> list[dict]:
        if self._check_circuit():
            context.metadata["llm_pre_response"] = NormalizedResponse(
                content=self._build_circuit_breaker_message(),
                finish_reason="error",
                metadata={"error_reason": "circuit_open"},
            )
            self._append_event(
                context,
                {
                    "type": "llm_circuit_open",
                    "recovery_timeout_sec": self.circuit_recovery_timeout_sec,
                },
            )
        return messages

    def after_llm(self, response: NormalizedResponse, context: TurnContext) -> NormalizedResponse:
        self._record_success()
        return response

    def on_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        if state != LoopState.LLM_CALL:
            return

        retriable, reason = self._classify_error(error)
        if retriable:
            self._record_failure()

        detail = _extract_error_detail(error)
        self.logger.warning(
            "LLM call failed: reason=%s retriable=%s detail=%s",
            reason,
            retriable,
            detail,
            exc_info=error,
        )

        context.metadata["llm_error_response"] = NormalizedResponse(
            content=self._build_user_message(error, reason),
            finish_reason="error",
            metadata={
                "error_reason": reason,
                "error_detail": detail,
                "status_code": _extract_status_code(error),
                "retry_after_ms": _extract_retry_after_ms(error),
            },
        )
        self._append_event(
            context,
            {
                "type": "llm_error",
                "reason": reason,
                "retriable": retriable,
                "detail": detail,
            },
        )

    def _check_circuit(self) -> bool:
        """Return True when the circuit is open and provider calls should be skipped."""
        with self._circuit_lock:
            now = time.time()

            if self._circuit_state == "open":
                if now < self._circuit_open_until:
                    return True
                self._circuit_state = "half_open"
                self._circuit_probe_in_flight = False

            if self._circuit_state == "half_open":
                if self._circuit_probe_in_flight:
                    return True
                self._circuit_probe_in_flight = True
                return False

            return False

    def _record_success(self) -> None:
        with self._circuit_lock:
            if self._circuit_state != "closed" or self._circuit_failure_count > 0:
                self.logger.info("Circuit breaker reset (Closed). LLM service recovered.")
            self._circuit_failure_count = 0
            self._circuit_open_until = 0.0
            self._circuit_state = "closed"
            self._circuit_probe_in_flight = False

    def _record_failure(self) -> None:
        with self._circuit_lock:
            if self._circuit_state == "half_open":
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                self._circuit_state = "open"
                self._circuit_probe_in_flight = False
                self.logger.error(
                    "Circuit breaker probe failed (Open). Will probe again after %ds.",
                    self.circuit_recovery_timeout_sec,
                )
                return

            self._circuit_failure_count += 1
            if self._circuit_failure_count >= self.circuit_failure_threshold:
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                if self._circuit_state != "open":
                    self._circuit_state = "open"
                    self._circuit_probe_in_flight = False
                    self.logger.error(
                        "Circuit breaker tripped (Open). Threshold reached (%d). Will probe after %ds.",
                        self.circuit_failure_threshold,
                        self.circuit_recovery_timeout_sec,
                    )

    def _classify_error(self, exc: BaseException) -> tuple[bool, str]:
        detail = _extract_error_detail(exc)
        lowered = detail.lower()
        error_code = _extract_error_code(exc)
        status_code = _extract_status_code(exc)

        if _matches_any(lowered, _QUOTA_PATTERNS) or _matches_any(str(error_code).lower(), _QUOTA_PATTERNS):
            return False, "quota"
        if _matches_any(lowered, _AUTH_PATTERNS):
            return False, "auth"

        exc_name = exc.__class__.__name__
        if exc_name in {
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ReadError",
            "RemoteProtocolError",
            "ConnectError",
            "ReadTimeout",
        }:
            return True, "transient"
        if status_code in _RETRIABLE_STATUS_CODES:
            return True, "transient"
        if _matches_any(lowered, _BUSY_PATTERNS):
            return True, "busy"

        return False, "generic"

    @staticmethod
    def _build_circuit_breaker_message() -> str:
        return (
            "The configured LLM provider is currently unavailable due to continuous failures. "
            "Circuit breaker is engaged to protect the system. Please wait a moment before trying again."
        )

    @staticmethod
    def _build_user_message(exc: BaseException, reason: str) -> str:
        detail = _extract_error_detail(exc)
        if reason == "quota":
            return (
                "The configured LLM provider rejected the request because the account is out of quota, "
                "billing is unavailable, or usage is restricted. Please fix the provider account and try again."
            )
        if reason == "auth":
            return (
                "The configured LLM provider rejected the request because authentication or access is invalid. "
                "Please check the provider credentials and try again."
            )
        if reason in {"busy", "transient"}:
            return (
                "The configured LLM provider is temporarily unavailable after multiple retries. "
                "Please wait a moment and continue the conversation."
            )
        return f"LLM request failed: {detail}"


def _matches_any(detail: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in detail for pattern in patterns)


def _extract_error_code(exc: BaseException) -> Any:
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            return value

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if value not in (None, ""):
                    return value
    return None


def _extract_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _extract_retry_after_ms(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = None
    header_name = ""
    for key in ("retry-after-ms", "Retry-After-Ms", "retry-after", "Retry-After"):
        header_name = key
        if hasattr(headers, "get"):
            raw = headers.get(key)
        if raw:
            break
    if not raw:
        return None

    try:
        multiplier = 1 if "ms" in header_name.lower() else 1000
        return max(0, int(float(raw) * multiplier))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            delta = target.timestamp() - time.time()
            return max(0, int(delta * 1000))
        except (TypeError, ValueError, OverflowError):
            return None


def _extract_error_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return exc.__class__.__name__


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
