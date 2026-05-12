"""Guardrail provider contracts for Aether runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class GuardrailReason:
    code: str
    message: str


@dataclass(slots=True)
class GuardrailDecision:
    allow: bool
    policy_id: str | None = None
    reasons: list[GuardrailReason] = field(default_factory=list)


@dataclass(slots=True)
class GuardrailRequest:
    tool_name: str
    tool_input: dict[str, Any]
    agent_id: str | None = None
    session_id: str | None = None
    iteration: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class GuardrailProvider(ABC):
    """Evaluate whether a tool call should be allowed."""

    @abstractmethod
    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        raise NotImplementedError

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class AllowAllGuardrailProvider(GuardrailProvider):
    """Default provider that permits every tool call."""

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:  # noqa: ARG002
        return GuardrailDecision(allow=True, policy_id="allow_all")


class AllowlistGuardrailProvider(GuardrailProvider):
    """Allow only tool names in an explicit allowlist."""

    def __init__(self, allowed_tools: list[str], *, policy_id: str = "allowlist") -> None:
        self.allowed_tools = {name.strip() for name in allowed_tools if name and name.strip()}
        self.policy_id = policy_id

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name in self.allowed_tools:
            return GuardrailDecision(allow=True, policy_id=self.policy_id)
        return GuardrailDecision(
            allow=False,
            policy_id=self.policy_id,
            reasons=[
                GuardrailReason(
                    code="oap.denied",
                    message=f"Tool '{request.tool_name}' is not in the configured allowlist.",
                )
            ],
        )
