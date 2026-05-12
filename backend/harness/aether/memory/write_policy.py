"""Write policy for durable memory operations."""

from __future__ import annotations

from dataclasses import dataclass

from .sanitize import scan_secrets

MAX_MEMORY_TEXT_CHARS = 2000


@dataclass(slots=True, frozen=True)
class WritePolicyResult:
    allowed: bool
    reason: str
    redacted: bool = False


def check_write_policy(
    *,
    scope: str,
    text: str,
    mode: str,
    auto_write_enabled: bool,
    reason: str | None,
) -> WritePolicyResult:
    if not reason or not reason.strip():
        return WritePolicyResult(allowed=False, reason="reason_required")

    if scope == "user" and mode != "personal_assistant":
        return WritePolicyResult(allowed=False, reason="user_scope_denied")

    if len(text) > MAX_MEMORY_TEXT_CHARS:
        return WritePolicyResult(allowed=False, reason="text_too_long")

    if not auto_write_enabled:
        return WritePolicyResult(allowed=False, reason="auto_write_disabled")

    scan = scan_secrets(text)
    if scan.redacted:
        return WritePolicyResult(allowed=False, reason="secret_detected", redacted=True)

    return WritePolicyResult(allowed=True, reason="allowed")


__all__ = [
    "MAX_MEMORY_TEXT_CHARS",
    "WritePolicyResult",
    "check_write_policy",
]
