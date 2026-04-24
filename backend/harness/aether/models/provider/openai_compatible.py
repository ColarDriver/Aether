"""OpenAI-compatible provider with minimal compatibility handling."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, StreamDeltaCallback, ToolCall, TurnContext
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


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
        retry_max_attempts: int = MAX_RETRIES,
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
        self.retry_max_attempts = max(1, int(retry_max_attempts))

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,  # noqa: ARG002
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        payload = self._build_payload(messages, tools=tools, config=config)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                with httpx.Client(timeout=self.request_timeout_sec) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    return self._parse_response(resp.json(), stream_callback=stream_callback)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if status_code in {429, 500, 502, 503, 504, 529} and attempt < self.retry_max_attempts:
                    wait_ms = self._calc_backoff_ms(attempt)
                    logger.warning(
                        "OpenAI-compatible API error %s, retrying %d/%d after %dms",
                        status_code,
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                    )
                    time.sleep(wait_ms / 1000)
                    continue
                raise
            except (httpx.TransportError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retry_max_attempts:
                    wait_ms = self._calc_backoff_ms(attempt)
                    logger.warning(
                        "OpenAI-compatible transport error, retrying %d/%d after %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        exc,
                    )
                    time.sleep(wait_ms / 1000)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI-compatible API call failed without an explicit exception")

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

    @staticmethod
    def _calc_backoff_ms(attempt: int) -> int:
        return 2000 * (1 << (attempt - 1))
