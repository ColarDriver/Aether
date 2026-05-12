"""Secret detection and redaction for durable memory."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SecretFinding:
    """A secret-like pattern found in candidate memory text."""

    kind: str
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class SecretScanResult:
    """Result of scanning text for secret-like content."""

    findings: tuple[SecretFinding, ...]
    redacted_text: str

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b")),
    ("github_fine_grained_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack_bot_token", re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "env_secret",
        re.compile(
            r"(?im)^\s*[A-Z0-9_]*(?:TOKEN|SECRET|API_KEY|PASSWORD)[A-Z0-9_]*\s*=\s*[^\s#]+"
        ),
    ),
)


def scan_secrets(text: str) -> SecretScanResult:
    """Return redacted text and metadata for common secret-like patterns."""

    findings: list[SecretFinding] = []
    for kind, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(SecretFinding(kind=kind, start=match.start(), end=match.end()))
    if not findings:
        return SecretScanResult(findings=(), redacted_text=text)

    merged = _merge_overlapping(findings)
    redacted_parts: list[str] = []
    cursor = 0
    for finding in merged:
        redacted_parts.append(text[cursor:finding.start])
        redacted_parts.append(f"[REDACTED:{finding.kind}]")
        cursor = finding.end
    redacted_parts.append(text[cursor:])
    return SecretScanResult(findings=tuple(merged), redacted_text="".join(redacted_parts))


def redact_secrets(text: str) -> str:
    """Return text with secret-like content redacted."""

    return scan_secrets(text).redacted_text


def _merge_overlapping(findings: list[SecretFinding]) -> list[SecretFinding]:
    ordered = sorted(findings, key=lambda item: (item.start, item.end))
    merged: list[SecretFinding] = []
    for finding in ordered:
        if not merged or finding.start > merged[-1].end:
            merged.append(finding)
            continue
        previous = merged[-1]
        merged[-1] = SecretFinding(
            kind=previous.kind,
            start=previous.start,
            end=max(previous.end, finding.end),
        )
    return merged


__all__ = [
    "SecretFinding",
    "SecretScanResult",
    "redact_secrets",
    "scan_secrets",
]
