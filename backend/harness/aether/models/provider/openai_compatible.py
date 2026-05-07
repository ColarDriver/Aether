"""OpenAI-compatible provider with minimal compatibility handling."""

from __future__ import annotations

import json
import logging
from email.utils import parsedate_to_datetime
from time import time as _wallclock
from typing import Any, Iterable

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, StreamDeltaCallback, ToolCall, TurnContext
from aether.runtime.provider_errors import ProviderInvocationError
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)

# Maximum bytes we copy out of an error response body for classification /
# logging.  Keep this small — it ends up in logs and exception messages, and
# error bodies can be megabytes for some providers.
_MAX_BODY_SUMMARY_CHARS = 1024


class OpenAICompatibleModel(ModelProvider):
    """ModelProvider for standard OpenAI-compatible chat-completions APIs.

    Unlike the previous LangChain patch approach, this provider makes direct HTTP
    requests and intentionally preserves `thought_signature` on assistant tool calls.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        request_timeout_sec: int = 120,
    ) -> None:
        if not model:
            raise ValueError("OpenAICompatibleModel requires a model name")
        if not api_key:
            raise ValueError("OpenAICompatibleModel requires an api_key")
        if not base_url:
            raise ValueError("OpenAICompatibleModel requires a base_url")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_timeout_sec = max(5, int(request_timeout_sec))

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,  # noqa: ARG002
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        """Single-shot HTTP call to the chat-completions endpoint.

        Sprint 0 / PR 0.2: this provider no longer retries internally.  The
        engine's recovery layer (``runtime/recovery.py`` — added in PR 0.3)
        owns retry policy, backoff, fallback chains, and interrupt handling.
        We expose the provider's view of the error via
        ``ProviderInvocationError`` so the recovery layer can branch on
        ``status_code`` / ``retry_after_seconds`` / ``is_network_error``
        without poking at httpx internals.

        Any exception escaping this method MUST be a
        ``ProviderInvocationError`` — the engine relies on that contract.
        """
        payload = self._build_payload(messages, tools=tools, config=config)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.request_timeout_sec) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return self._parse_response(resp.json(), stream_callback=stream_callback)
        except httpx.HTTPStatusError as exc:
            # Server returned a response (4xx/5xx).  Capture status + Retry-After
            # + truncated body so the recovery layer has enough signal to classify.
            response = exc.response
            raise ProviderInvocationError(
                raw=exc,
                status_code=response.status_code,
                retry_after_seconds=_parse_retry_after(response.headers),
                body_summary=_summarize_body(response),
                is_network_error=False,
                metadata={"url": url, "method": "POST"},
            ) from exc
        except (httpx.TransportError, TimeoutError) as exc:
            # Couldn't even get a response back: DNS, TLS, connection reset,
            # or read timeout.  No status_code available.
            raise ProviderInvocationError(
                raw=exc,
                status_code=None,
                retry_after_seconds=None,
                body_summary=str(exc) or exc.__class__.__name__,
                is_network_error=True,
                metadata={"url": url, "method": "POST"},
            ) from exc
        except json.JSONDecodeError as exc:
            # Server replied with HTTP 200 but the body was not parseable JSON.
            # Treat this like a malformed-response transport error so the
            # recovery layer can apply its generic retry policy.
            raise ProviderInvocationError(
                raw=exc,
                status_code=None,
                retry_after_seconds=None,
                body_summary=f"non-JSON response body: {exc}",
                is_network_error=True,
                metadata={"url": url, "method": "POST"},
            ) from exc

    def _build_payload(
        self,
        messages: list[dict],
        *,
        tools: Iterable[ToolDescriptor],
        config: ModelCallConfig,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(config.extra.get("model", self.model)),
            "messages": self._convert_messages(messages),
            "stream": False,
        }
        if config.temperature is not None:
            payload["temperature"] = float(config.temperature)
        if config.max_tokens is not None:
            payload["max_tokens"] = int(config.max_tokens)

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        for key, value in config.extra.items():
            if key in {"model", "messages", "tools", "stream"}:
                continue
            payload.setdefault(key, value)

        return payload

    @classmethod
    def _convert_messages(cls, messages: list[dict]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"

            if role in {"system", "user"}:
                converted.append(
                    {
                        "role": role,
                        "content": cls._normalize_content(message.get("content")),
                    }
                )
                continue

            if role == "tool":
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "content": cls._normalize_content(message.get("content")),
                    "tool_call_id": str(message.get("tool_call_id") or message.get("id") or ""),
                }
                if message.get("name"):
                    tool_message["name"] = str(message.get("name"))
                converted.append(tool_message)
                continue

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": cls._normalize_content(message.get("content")),
            }
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                normalized_tool_calls: list[dict[str, Any]] = []
                for tool_call in raw_tool_calls:
                    normalized = cls._normalize_tool_call(tool_call)
                    if normalized is not None:
                        normalized_tool_calls.append(normalized)
                if normalized_tool_calls:
                    assistant_message["tool_calls"] = normalized_tool_calls
            converted.append(assistant_message)

        return converted

    @classmethod
    def _normalize_tool_call(cls, tool_call: Any) -> dict[str, Any] | None:
        if not isinstance(tool_call, dict):
            return None

        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "")
        if not name:
            return None

        raw_arguments = function.get("arguments", tool_call.get("arguments", "{}"))
        if isinstance(raw_arguments, dict):
            arguments = json.dumps(raw_arguments, ensure_ascii=False)
        else:
            arguments = str(raw_arguments or "{}")

        normalized_function: dict[str, Any] = {
            "name": name,
            "arguments": arguments,
        }

        # Compatibility point: preserve thought_signature for compatible gateways.
        if "thought_signature" in function:
            normalized_function["thought_signature"] = function.get("thought_signature")
        elif "thought_signature" in tool_call:
            normalized_function["thought_signature"] = tool_call.get("thought_signature")

        return {
            "id": str(tool_call.get("id") or tool_call.get("call_id") or ""),
            "type": "function",
            "function": normalized_function,
        }

    @staticmethod
    def _convert_tools(tools: Iterable[ToolDescriptor]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            parameters = dict(tool.parameters)
            if "type" not in parameters and "properties" not in parameters:
                parameters = {
                    "type": "object",
                    "properties": parameters,
                }
            if tool.required and "required" not in parameters:
                parameters["required"] = list(tool.required)

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": parameters,
                    },
                }
            )
        return converted

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        stream_callback: StreamDeltaCallback | None,
    ) -> NormalizedResponse:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return NormalizedResponse(content="", tool_calls=[], finish_reason="stop", metadata={"raw": data})

        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = self._normalize_content(message.get("content"))

        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            for raw in raw_tool_calls:
                parsed = self._parse_tool_call(raw)
                if parsed is not None:
                    tool_calls.append(parsed)

        if stream_callback and content and not tool_calls:
            try:
                stream_callback(content)
            except Exception:
                logger.exception("openai-compatible stream callback failed for final content fallback")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        metadata = {
            "model": data.get("model", self.model),
            "usage": usage,
            "token_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        finish_reason = str(choice.get("finish_reason") or "stop")
        if tool_calls:
            finish_reason = "tool_calls"

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            metadata=metadata,
        )

    @staticmethod
    def _parse_tool_call(raw: Any) -> ToolCall | None:
        if not isinstance(raw, dict):
            return None

        call_id = str(raw.get("id") or raw.get("call_id") or "")
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = str(function.get("name") or raw.get("name") or "")
        if not call_id or not name:
            return None

        arguments_raw = function.get("arguments", raw.get("arguments", "{}"))
        if isinstance(arguments_raw, dict):
            arguments = arguments_raw
        else:
            try:
                loaded = json.loads(arguments_raw)
                arguments = loaded if isinstance(loaded, dict) else {}
            except Exception:
                arguments = {}

        return ToolCall(id=call_id, name=name, arguments=arguments)

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)


# ---------------------------------------------------------------------------
# Module-level helpers (used by the single-shot generate path above)
# ---------------------------------------------------------------------------


def _parse_retry_after(headers: Any) -> float | None:
    """Best-effort parse of ``Retry-After`` / ``Retry-After-Ms`` headers.

    Accepts the three common encodings:
    1. Integer seconds: ``Retry-After: 30``
    2. Integer milliseconds: ``Retry-After-Ms: 5000`` (some providers)
    3. HTTP-date: ``Retry-After: Wed, 21 Oct 2026 07:28:00 GMT``

    Returns the wait time in **seconds** (clamped to non-negative), or
    ``None`` if no recognisable header is present.  Recovery strategies
    upstream are responsible for applying their own upper bound (we do not
    cap here so the structured value stays faithful to what the server
    asked for).
    """
    if headers is None or not hasattr(headers, "get"):
        return None

    # Try the millisecond variant first — only some providers send it but
    # when present it's authoritative.
    for key in ("retry-after-ms", "Retry-After-Ms"):
        raw = headers.get(key)
        if raw:
            try:
                return max(0.0, float(raw) / 1000.0)
            except (TypeError, ValueError):
                pass

    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None

    # Numeric seconds form
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass

    # HTTP-date form: convert to seconds-from-now
    try:
        target = parsedate_to_datetime(str(raw))
        return max(0.0, target.timestamp() - _wallclock())
    except (TypeError, ValueError, OverflowError):
        return None


def _summarize_body(response: Any) -> str | None:
    """Return a short, log-safe excerpt of an HTTP error body.

    We trim to ``_MAX_BODY_SUMMARY_CHARS`` characters and append an ellipsis
    marker when truncated, so readers can tell at a glance whether the
    summary is complete.
    """
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text:
        return None
    text = text.strip()
    if len(text) <= _MAX_BODY_SUMMARY_CHARS:
        return text
    return text[:_MAX_BODY_SUMMARY_CHARS] + "...(truncated)"
