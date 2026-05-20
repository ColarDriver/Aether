"""OpenAI-compatible Chat Completions transport."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from aether.config.schema import ModelCallConfig
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    TurnContext,
)
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)


class OpenAIChatCompletionsTransport:
    """Pure conversion layer for OpenAI-style chat-completions APIs."""

    api_mode = "chat"
    name = "openai_chat_completions"

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Iterable[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del context, kwargs
        payload: dict[str, Any] = {
            "model": str(config.extra.get("model", model)),
            "messages": self.convert_messages(messages),
            "stream": False,
        }
        if config.temperature is not None:
            payload["temperature"] = float(config.temperature)
        if config.max_tokens is not None:
            payload["max_tokens"] = int(config.max_tokens)

        converted_tools = self.convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            # Some compatible gateways treat missing tool_choice as "none"
            # even though the OpenAI default is documented as "auto".
            payload.setdefault("tool_choice", "auto")

        for key, value in config.extra.items():
            if key in {"model", "messages", "tools", "stream"}:
                continue
            payload.setdefault(key, value)

        return payload

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"

            if role in {"system", "user"}:
                converted.append(
                    {
                        "role": role,
                        "content": self.normalize_content(message.get("content")),
                    }
                )
                continue

            if role == "tool":
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "content": self.normalize_content(message.get("content")),
                    "tool_call_id": str(message.get("tool_call_id") or message.get("id") or ""),
                }
                if message.get("name"):
                    tool_message["name"] = str(message.get("name"))
                converted.append(tool_message)
                continue

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": self.normalize_content(message.get("content")),
            }
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                normalized_tool_calls: list[dict[str, Any]] = []
                for tool_call in raw_tool_calls:
                    normalized = self.normalize_tool_call(tool_call)
                    if normalized is not None:
                        normalized_tool_calls.append(normalized)
                if normalized_tool_calls:
                    assistant_message["tool_calls"] = normalized_tool_calls
            converted.append(assistant_message)

        return converted

    def normalize_tool_call(self, tool_call: Any) -> dict[str, Any] | None:
        if not isinstance(tool_call, dict):
            return None

        raw_function = tool_call.get("function")
        function: dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
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

        if "thought_signature" in function:
            normalized_function["thought_signature"] = function.get("thought_signature")
        elif "thought_signature" in tool_call:
            normalized_function["thought_signature"] = tool_call.get("thought_signature")

        return {
            "id": str(tool_call.get("id") or tool_call.get("call_id") or ""),
            "type": "function",
            "function": normalized_function,
        }

    def convert_tools(self, tools: Iterable[ToolDescriptor], **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
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

    def normalize_response(
        self,
        response: Any,
        *,
        fallback_model: str,
        stream_callback: StreamDeltaCallback | None = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        del kwargs
        data = response if isinstance(response, dict) else {}
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return NormalizedResponse(
                content="",
                tool_calls=[],
                finish_reason="stop",
                metadata={"raw": data, "model": data.get("model", fallback_model)},
            )

        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = self.normalize_content(message.get("content"))

        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            for raw in raw_tool_calls:
                parsed = self.parse_tool_call(raw)
                if parsed is not None:
                    tool_calls.append(parsed)

        if stream_callback and content and not tool_calls:
            try:
                stream_callback(content)
            except Exception:
                logger.exception("openai-compatible stream callback failed for final content fallback")

        raw_usage = data.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        metadata = {
            "model": data.get("model", fallback_model),
            "usage": usage,
            "token_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "raw": data,
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

    def parse_tool_call(self, raw: Any) -> ToolCall | None:
        if not isinstance(raw, dict):
            return None

        call_id = str(raw.get("id") or raw.get("call_id") or "")
        raw_function = raw.get("function")
        function: dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
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

    def normalize_content(self, content: Any) -> str:
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

    def validate_raw_response(self, response: Any) -> tuple[bool, list[str]]:
        if not isinstance(response, dict):
            return False, ["raw response is not a dict"]

        reasons: list[str] = []
        err = response.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type") or "unknown"
            reasons.append(f"raw.error.{msg}")
        elif isinstance(err, str) and err:
            reasons.append(f"raw.error: {err[:100]}")

        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            reasons.append("raw.choices is empty or missing")

        return (len(reasons) == 0), reasons

    def validate_response(self, response: NormalizedResponse) -> tuple[bool, list[str]]:
        reasons: list[str] = []

        raw = response.metadata.get("raw") if isinstance(response.metadata, dict) else None
        if isinstance(raw, dict):
            err = raw.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("type") or "unknown"
                reasons.append(f"raw.error.{msg}")
            elif isinstance(err, str) and err:
                reasons.append(f"raw.error: {err[:100]}")

            choices = raw.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                if not response.content and not response.tool_calls:
                    reasons.append("raw.choices is empty or missing")

        return (len(reasons) == 0), reasons


__all__ = ["OpenAIChatCompletionsTransport"]
