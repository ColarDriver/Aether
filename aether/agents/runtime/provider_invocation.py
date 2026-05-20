"""Provider invocation controller for AgentEngine turns."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.core.services import EngineServices
from aether.runtime.observability.usage import CanonicalUsage, normalize_usage
from aether.runtime.recovery.provider_errors import ResponseInvalidError
from aether.services.compact import estimate_messages_tokens
from aether.tools.base import ToolDescriptor


ProviderSuccessCallback = Callable[[ModelProvider, TurnContext], None]


@dataclass(slots=True)
class ProviderInvocationRequest:
    request: EngineRequest
    canonical_messages: list[dict[str, Any]]
    prepared_messages: list[dict[str, Any]]
    tools: list[ToolDescriptor]
    call_config: ModelCallConfig
    context: TurnContext
    stream_callback: StreamDeltaCallback | None = None
    stream_silent_callback: StreamSilentCallback | None = None
    on_valid_response: ProviderSuccessCallback | None = None


@dataclass(slots=True)
class ProviderInvocationResult:
    response: NormalizedResponse | None = None
    error: Exception | None = None
    interrupted: bool = False
    elapsed_ms: float = 0.0
    provider_name: str = ""
    api_mode: str = ""
    model: str | None = None
    transport: str | None = None
    transport_api_mode: str | None = None


class ProviderInvocationController:
    """Own the non-policy mechanics of one provider call."""

    def __init__(self, *, services: EngineServices, hooks: EngineHooks) -> None:
        self._services = services
        self._hooks = hooks

    def invoke(self, invocation: ProviderInvocationRequest) -> ProviderInvocationResult:
        provider = self._services.provider
        provider_name = str(getattr(provider, "provider_name", type(provider).__name__))
        api_mode = str(getattr(provider, "api_mode", "chat"))
        model = self._resolve_model(invocation.call_config, provider)
        transport = self._resolve_optional_string(provider, "transport_name")
        transport_api_mode = self._resolve_optional_string(provider, "transport_api_mode")
        api_call_count = self._next_api_request_attempt_count(invocation.context)
        api_hook_payload = self._build_api_hook_payload(
            request=invocation.request,
            messages=invocation.prepared_messages,
            tools=invocation.tools,
            call_config=invocation.call_config,
            context=invocation.context,
            provider=provider,
            api_call_count=api_call_count,
        )
        invocation.context.metadata["provider_invocation"] = {
            "provider": provider_name,
            "api_mode": api_mode,
            "transport": transport,
            "transport_api_mode": transport_api_mode,
            "model": model,
        }
        self._safe_call_hook("pre_api_request", **api_hook_payload)

        api_start = time.perf_counter()
        response: NormalizedResponse | None = None
        try:
            response = provider.generate(
                invocation.prepared_messages,
                invocation.tools,
                invocation.call_config,
                invocation.context,
                stream_callback=invocation.stream_callback,
                stream_silent_callback=invocation.stream_silent_callback,
            )
            ok, reasons = provider.validate_response(response)
            if not ok:
                raise ResponseInvalidError(
                    validation_errors=list(reasons),
                    body_summary="invalid response: " + "; ".join(reasons[:5]),
                    metadata={"phase": "validate_response"},
                )
            if invocation.on_valid_response is not None:
                invocation.on_valid_response(provider, invocation.context)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - api_start) * 1000
            self._safe_call_hook(
                "post_api_request",
                **self._build_post_api_hook_payload(
                    api_hook_payload,
                    elapsed_ms=elapsed_ms,
                    response=response,
                    error=exc,
                ),
            )
            return ProviderInvocationResult(
                error=exc,
                elapsed_ms=elapsed_ms,
                provider_name=provider_name,
                api_mode=api_mode,
                model=model,
                transport=transport,
                transport_api_mode=transport_api_mode,
            )

        elapsed_ms = (time.perf_counter() - api_start) * 1000
        self._safe_call_hook(
            "post_api_request",
            **self._build_post_api_hook_payload(
                api_hook_payload,
                elapsed_ms=elapsed_ms,
                response=response,
                error=None,
            ),
        )
        return ProviderInvocationResult(
            response=response,
            elapsed_ms=elapsed_ms,
            provider_name=provider_name,
            api_mode=api_mode,
            model=model,
            transport=transport,
            transport_api_mode=transport_api_mode,
        )

    def accumulate_usage(self, response: NormalizedResponse, context: TurnContext) -> None:
        """Add one successful LLM call's usage to the per-turn accumulator."""

        try:
            raw = (response.metadata or {}).get("usage") if response else None
            provider = self._services.provider
            provider_name = getattr(provider, "provider_name", "openai")
            api_mode = getattr(provider, "api_mode", "chat")
            this_call = normalize_usage(raw, provider=provider_name, api_mode=api_mode)
            acc = context.metadata.get("usage_accumulator")
            if not isinstance(acc, CanonicalUsage):
                acc = CanonicalUsage()
            context.metadata["usage_accumulator"] = acc.add(this_call)
            context.metadata["api_calls"] = int(context.metadata.get("api_calls", 0)) + 1
        except Exception:  # noqa: BLE001 - observability must never crash a turn
            self._services.logger.debug(
                "usage accumulation failed; leaving accumulator unchanged",
                exc_info=True,
            )

    @staticmethod
    def _next_api_request_attempt_count(context: TurnContext) -> int:
        next_count = int(context.metadata.get("_api_request_attempt_count", 0)) + 1
        context.metadata["_api_request_attempt_count"] = next_count
        return next_count

    @staticmethod
    def _resolve_model(call_config: ModelCallConfig, provider: ModelProvider) -> str:
        model = call_config.extra.get("model") if isinstance(call_config.extra, dict) else None
        if model is None:
            model = getattr(provider, "model", None)
        if model is None:
            model = "unknown"
        return str(model)

    @staticmethod
    def _resolve_optional_string(provider: ModelProvider, attr: str) -> str | None:
        value = getattr(provider, attr, None)
        if value is None:
            return None
        value = str(value)
        return value or None

    def _build_api_hook_payload(
        self,
        *,
        request: EngineRequest,
        messages: list[dict[str, Any]],
        tools: list[ToolDescriptor],
        call_config: ModelCallConfig,
        context: TurnContext,
        provider: ModelProvider,
        api_call_count: int,
    ) -> dict[str, Any]:
        try:
            approx_input_tokens = estimate_messages_tokens(messages)
        except Exception:
            approx_input_tokens = 0

        try:
            request_char_count = len(json.dumps(messages, ensure_ascii=False, default=str))
        except Exception:
            request_char_count = sum(len(str(message)) for message in messages)

        payload: dict[str, Any] = {
            "session_id": request.session_id,
            "iteration": context.iteration,
            "model": self._resolve_model(call_config, provider),
            "provider": str(getattr(provider, "provider_name", type(provider).__name__)),
            "api_mode": str(getattr(provider, "api_mode", "chat")),
            "api_call_count": api_call_count,
            "message_count": len(messages),
            "tool_count": len(tools or []),
            "approx_input_tokens": int(approx_input_tokens),
            "request_char_count": int(request_char_count),
            "max_tokens": call_config.max_tokens,
            "context_metadata": context.metadata,
        }
        transport = self._resolve_optional_string(provider, "transport_name")
        transport_api_mode = self._resolve_optional_string(provider, "transport_api_mode")
        if transport is not None:
            payload["transport"] = transport
        if transport_api_mode is not None:
            payload["transport_api_mode"] = transport_api_mode
        return payload

    @staticmethod
    def _build_post_api_hook_payload(
        pre_payload: dict[str, Any],
        *,
        elapsed_ms: float,
        response: NormalizedResponse | None,
        error: Exception | None,
    ) -> dict[str, Any]:
        payload = {
            "session_id": pre_payload["session_id"],
            "iteration": pre_payload["iteration"],
            "model": pre_payload["model"],
            "provider": pre_payload["provider"],
            "api_mode": pre_payload["api_mode"],
            "api_call_count": pre_payload["api_call_count"],
            "elapsed_ms": elapsed_ms,
            "response_finish_reason": response.finish_reason if response else None,
            "error": error,
            "context_metadata": pre_payload["context_metadata"],
        }
        if "transport" in pre_payload:
            payload["transport"] = pre_payload["transport"]
        if "transport_api_mode" in pre_payload:
            payload["transport_api_mode"] = pre_payload["transport_api_mode"]
        return payload

    def _safe_call_hook(self, name: str, **kwargs: Any) -> Any:
        hook = getattr(self._hooks, name, None)
        if hook is None:
            return None
        try:
            return hook(**kwargs)
        except Exception:
            self._services.logger.exception("Engine hook failed: %s", name)
            return None
