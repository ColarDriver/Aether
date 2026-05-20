"""Length and tool-call argument repair decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    TurnContext,
)

ResponseRepairAction = Literal["proceed", "continue", "finalize"]


@dataclass(slots=True)
class ResponseRepairInput:
    response: NormalizedResponse
    messages: list[dict[str, Any]]
    request: EngineRequest
    context: TurnContext


@dataclass(slots=True)
class ResponseRepairResult:
    action: ResponseRepairAction
    messages: list[dict[str, Any]]
    final_response: str | None = None
    exit_reason: ExitReason | None = None
    error_text: str | None = None


class ResponseRepairAdapter(Protocol):
    def handle_length_with_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> Any: ...

    def handle_length_finish_reason(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        request: EngineRequest,
        context: TurnContext,
    ) -> Any: ...

    def validate_tool_call_arguments(
        self,
        *,
        response: NormalizedResponse,
        messages: list[dict[str, Any]],
        context: TurnContext,
    ) -> Any: ...

    def get_messages_up_to_last_assistant(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def extract_visible_text(self, content: str) -> str: ...

    def apply_pending_steer_to_tool_results(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        start_idx: int,
        context: TurnContext,
    ) -> None: ...


class LegacyResponseRepairAdapter:
    """Delegate repair details to existing AgentEngine helpers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def handle_length_with_tool_calls(self, **kwargs: Any) -> Any:
        return self._engine._handle_length_with_tool_calls(**kwargs)  # noqa: SLF001

    def handle_length_finish_reason(self, **kwargs: Any) -> Any:
        return self._engine._handle_length_finish_reason(**kwargs)  # noqa: SLF001

    def validate_tool_call_arguments(self, **kwargs: Any) -> Any:
        return self._engine._validate_tool_call_arguments(**kwargs)  # noqa: SLF001

    def get_messages_up_to_last_assistant(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._engine._get_messages_up_to_last_assistant(messages)  # noqa: SLF001

    def extract_visible_text(self, content: str) -> str:
        return self._engine._extract_visible_text(content)  # noqa: SLF001

    def apply_pending_steer_to_tool_results(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        self._engine._apply_pending_steer_to_tool_results(messages, **kwargs)  # noqa: SLF001


class ResponseRepairController:
    """Handle length continuations and malformed tool-call arguments."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        adapter: ResponseRepairAdapter,
    ) -> None:
        self._config = config
        self._adapter = adapter

    def repair(self, repair_input: ResponseRepairInput) -> ResponseRepairResult:
        response = repair_input.response
        messages = repair_input.messages
        context = repair_input.context

        length_text = (response.content or "").strip()
        if response.finish_reason == "length" and (
            response.tool_calls or length_text
        ):
            if (
                response.tool_calls
                and getattr(self._config, "truncated_tool_call_detection_enabled", True)
            ):
                handled = self._adapter.handle_length_with_tool_calls(
                    response=response,
                    messages=messages,
                    context=context,
                )
            else:
                handled = self._adapter.handle_length_finish_reason(
                    response=response,
                    messages=messages,
                    request=repair_input.request,
                    context=context,
                )
            return self._from_length_outcome(handled)

        if not response.tool_calls:
            return ResponseRepairResult(action="proceed", messages=messages)

        if not getattr(self._config, "truncated_tool_call_detection_enabled", True):
            return ResponseRepairResult(action="proceed", messages=messages)

        validation = self._adapter.validate_tool_call_arguments(
            response=response,
            messages=messages,
            context=context,
        )
        if validation.action == "retry":
            return ResponseRepairResult(action="continue", messages=messages)
        if validation.action == "truncated":
            rollback = self._adapter.get_messages_up_to_last_assistant(messages)
            visible_text = self._adapter.extract_visible_text(response.content or "")
            if visible_text:
                prefix_parts = context.metadata.setdefault(
                    "truncated_response_prefix_parts",
                    [],
                )
                if isinstance(prefix_parts, list):
                    prefix_parts.append(visible_text)
            context.metadata["partial"] = True
            context.metadata.setdefault("length_exit_reason", "tool_call_truncated")
            return ResponseRepairResult(
                action="finalize",
                messages=rollback,
                final_response=visible_text or None,
                exit_reason=ExitReason.TOOL_CALL_TRUNCATED,
            )
        if validation.action == "inject_error":
            tool_result_start_idx = len(messages)
            messages.extend(validation.injection_messages)
            self._adapter.apply_pending_steer_to_tool_results(
                messages,
                session_id=repair_input.request.session_id,
                start_idx=tool_result_start_idx,
                context=context,
            )
            return ResponseRepairResult(action="continue", messages=messages)

        return ResponseRepairResult(action="proceed", messages=messages)

    @staticmethod
    def _from_length_outcome(handled: Any) -> ResponseRepairResult:
        if handled.action == "continue":
            return ResponseRepairResult(action="continue", messages=handled.messages)
        if handled.action == "finalize":
            return ResponseRepairResult(
                action="finalize",
                messages=handled.messages,
                final_response=handled.final_response,
                exit_reason=handled.exit_reason,
            )
        return ResponseRepairResult(action="proceed", messages=handled.messages)


__all__ = [
    "LegacyResponseRepairAdapter",
    "ResponseRepairAction",
    "ResponseRepairAdapter",
    "ResponseRepairController",
    "ResponseRepairInput",
    "ResponseRepairResult",
]
