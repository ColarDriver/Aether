"""Response finalization decisions for AgentEngine turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    TurnContext,
)

ResponseFinalizationAction = Literal["finalize", "continue", "dispatch_synthesized"]


@dataclass(slots=True)
class ResponseFinalizationInput:
    response: NormalizedResponse
    messages: list[dict[str, Any]]
    context: TurnContext
    request: EngineRequest


@dataclass(slots=True)
class ResponseFinalizationResult:
    action: ResponseFinalizationAction
    messages: list[dict[str, Any]]
    final_response: str | None = None
    error_text: str | None = None
    exit_reason: ExitReason | None = None
    synthesized_response: NormalizedResponse | None = None


class ResponseFinalizationAdapter(Protocol):
    def maybe_recover_phantom_tool_intent(
        self,
        *,
        response_to_store: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> str: ...

    def finalize_empty_response(
        self,
        *,
        response: NormalizedResponse,
        response_to_store: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
        request: EngineRequest,
        phantom_outcome: str,
        prefix: str,
    ) -> Any: ...

    def is_continue_loop_finalization(self, finalized: Any) -> bool: ...


class LegacyResponseFinalizationAdapter:
    """Delegate finalization details to existing AgentEngine helpers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def maybe_recover_phantom_tool_intent(self, **kwargs: Any) -> str:
        return self._engine._maybe_recover_phantom_tool_intent(**kwargs)  # noqa: SLF001

    def finalize_empty_response(self, **kwargs: Any) -> Any:
        return self._engine._finalize_empty_response(**kwargs)  # noqa: SLF001

    def is_continue_loop_finalization(self, finalized: Any) -> bool:
        return self._engine._is_continue_loop_finalization(finalized)  # noqa: SLF001


class ResponseFinalizationController:
    """Finalize no-tool-call provider responses or request loop recovery."""

    def __init__(self, *, adapter: ResponseFinalizationAdapter) -> None:
        self._adapter = adapter

    def finalize(self, finalization: ResponseFinalizationInput) -> ResponseFinalizationResult:
        response = finalization.response
        messages = finalization.messages
        context = finalization.context

        prefix_parts = context.metadata.pop("truncated_response_prefix_parts", None)
        prefix = (
            " ".join(
                part.strip()
                for part in prefix_parts
                if isinstance(part, str) and part.strip()
            )
            if isinstance(prefix_parts, list)
            else ""
        )
        suffix = (response.content or "").strip()
        combined_content = (
            (prefix + " " + suffix).strip()
            if prefix and suffix
            else (prefix or suffix)
        )
        response_to_store = response
        if combined_content != (response.content or ""):
            response_to_store = NormalizedResponse(
                content=combined_content,
                tool_calls=list(response.tool_calls),
                finish_reason=response.finish_reason,
                metadata=dict(response.metadata),
            )

        phantom_outcome = self._adapter.maybe_recover_phantom_tool_intent(
            response_to_store=response_to_store,
            messages=messages,
            context=context,
        )
        if phantom_outcome == "synthesized":
            return ResponseFinalizationResult(
                action="dispatch_synthesized",
                messages=messages,
                synthesized_response=response_to_store,
            )

        if phantom_outcome == "retry":
            return ResponseFinalizationResult(action="continue", messages=messages)

        finalized = self._adapter.finalize_empty_response(
            response=response,
            response_to_store=response_to_store,
            messages=messages,
            context=context,
            request=finalization.request,
            phantom_outcome=phantom_outcome,
            prefix=prefix,
        )
        if self._adapter.is_continue_loop_finalization(finalized):
            return ResponseFinalizationResult(action="continue", messages=messages)

        return ResponseFinalizationResult(
            action="finalize",
            messages=messages,
            final_response=(
                finalized.final_response
                if isinstance(finalized.final_response, str)
                else None
            ),
            exit_reason=finalized.exit_reason or ExitReason.EMPTY_RESPONSE,
            error_text=finalized.error_text,
        )


__all__ = [
    "LegacyResponseFinalizationAdapter",
    "ResponseFinalizationAction",
    "ResponseFinalizationAdapter",
    "ResponseFinalizationController",
    "ResponseFinalizationInput",
    "ResponseFinalizationResult",
]
