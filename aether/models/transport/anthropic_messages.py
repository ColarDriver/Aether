"""Anthropic Messages API transport."""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterable

from aether.config.schema import ModelCallConfig
from aether.models.provider.hosted_web_search import (
    append_sources_section,
    dedupe_sources,
    is_web_search_tool,
)
from aether.runtime.core.contracts import NormalizedResponse, ToolCall, TurnContext
from aether.tools.base import ToolDescriptor


class AnthropicMessagesTransport:
    """Pure conversion layer for Anthropic Messages payloads and responses."""

    api_mode = "anthropic_messages"
    name = "anthropic_messages"

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Iterable[ToolDescriptor],
        config: ModelCallConfig,
        max_tokens: int,
        context: TurnContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        system, converted_messages = self.convert_messages(messages)

        payload: dict[str, Any] = {
            "model": str(config.extra.get("model", model)),
            "max_tokens": int(config.max_tokens if config.max_tokens is not None else max_tokens),
            "messages": converted_messages,
        }
        if system:
            payload["system"] = system
        if config.temperature is not None:
            payload["temperature"] = float(config.temperature)

        converted_tools = self.convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        for key in ("stop_sequences", "thinking", "metadata"):
            if key in config.extra:
                payload[key] = config.extra[key]

        payload.setdefault("metadata", {})
        if isinstance(payload["metadata"], dict) and context is not None:
            payload["metadata"].setdefault("session_id", context.session_id)
            payload["metadata"].setdefault("iteration", context.iteration)

        return payload

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        del kwargs
        system_blocks: list[dict[str, Any]] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = self.normalize_content(msg.get("content"))

            if role == "system":
                if content:
                    system_blocks.append({"type": "text", "text": content})
                continue

            if role == "user":
                converted.append({"role": "user", "content": content})
                continue

            if role == "assistant":
                assistant_content: list[dict[str, Any]] = []
                if content:
                    assistant_content.append({"type": "text", "text": content})

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for raw_call in tool_calls:
                        normalized = self.convert_assistant_tool_call(raw_call)
                        if normalized is not None:
                            assistant_content.append(normalized)

                if not assistant_content:
                    continue

                if len(assistant_content) == 1 and assistant_content[0].get("type") == "text":
                    converted.append({"role": "assistant", "content": assistant_content[0]["text"]})
                else:
                    converted.append({"role": "assistant", "content": assistant_content})
                continue

            if role == "tool":
                call_id = str(msg.get("tool_call_id") or msg.get("id") or "")
                if not call_id:
                    continue
                tool_result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                }
                if bool(msg.get("is_error")):
                    tool_result["is_error"] = True
                converted.append({"role": "user", "content": [tool_result]})
                continue

            if content:
                converted.append({"role": "user", "content": f"[{role}] {content}"})

        return system_blocks, converted

    def normalize_content(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [self.normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            for key in ("text", "output"):
                value = content.get(key)
                if isinstance(value, str):
                    return value
            nested = content.get("content")
            if nested is not None:
                return self.normalize_content(nested)
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)

        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    def convert_assistant_tool_call(self, raw_call: Any) -> dict[str, Any] | None:
        if not isinstance(raw_call, dict):
            return None

        function_raw = raw_call.get("function")
        fn: dict[str, Any] = function_raw if isinstance(function_raw, dict) else {}
        name = fn.get("name") or raw_call.get("name")
        if not name:
            return None

        raw_args = fn.get("arguments", raw_call.get("arguments", {}))
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                loaded = json.loads(raw_args)
                args = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = {}

        call_id = raw_call.get("id") or raw_call.get("call_id") or f"toolu_{uuid.uuid4().hex[:24]}"
        return {
            "type": "tool_use",
            "id": str(call_id),
            "name": str(name),
            "input": args,
        }

    def convert_tools(self, tools: Iterable[ToolDescriptor], **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if is_web_search_tool(tool):
                converted.append(
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 8,
                    }
                )
                continue

            schema = dict(tool.parameters)
            if "type" not in schema and "properties" not in schema:
                schema = {
                    "type": "object",
                    "properties": schema,
                }
            if tool.required and "required" not in schema:
                schema["required"] = list(tool.required)

            converted.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": schema,
                }
            )
        return converted

    def normalize_response(
        self,
        response: Any,
        *,
        fallback_model: str,
        **kwargs: Any,
    ) -> NormalizedResponse:
        del kwargs
        response_dict = as_dict(response)

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        hosted_search_calls: list[dict[str, Any]] = []
        hosted_search_results: list[dict[str, Any]] = []
        hosted_search_sources: list[dict[str, Any]] = []

        for block in response_dict.get("content", []):
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "text":
                content_parts.append(str(block.get("text", "")))
                hosted_search_sources.extend(
                    web_search_sources_from_citations(block.get("citations"))
                )
            elif block_type == "tool_use":
                input_raw = block.get("input")
                tool_arguments: dict[str, Any] = input_raw if isinstance(input_raw, dict) else {}
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=tool_arguments,
                    )
                )
            elif block_type == "server_tool_use" and block.get("name") == "web_search":
                input_raw = block.get("input")
                hosted_search_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "name": "web_search",
                        "input": input_raw if isinstance(input_raw, dict) else {},
                    }
                )
            elif block_type == "web_search_tool_result":
                result_entry, sources = parse_anthropic_web_search_result(block)
                hosted_search_results.append(result_entry)
                hosted_search_sources.extend(sources)

        raw_usage = response_dict.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        sources = dedupe_sources(hosted_search_sources)
        content = "".join(content_parts)
        if sources:
            content = append_sources_section(content, sources)
        metadata: dict[str, Any] = {
            "model": response_dict.get("model", fallback_model),
            "usage": usage,
        }
        if hosted_search_calls or hosted_search_results or sources:
            metadata["hosted_web_search"] = {
                "provider": "anthropic",
                "calls": hosted_search_calls,
                "results": hosted_search_results,
                "sources": sources,
                "source_count": len(sources),
            }

        stop_reason = str(response_dict.get("stop_reason") or "")
        if not stop_reason:
            stop_reason = "tool_calls" if tool_calls else "stop"

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=stop_reason,
            metadata=metadata,
        )

    def validate_raw_response(self, response: Any) -> tuple[bool, list[str]]:
        response_dict = as_dict(response)
        if not response_dict:
            return False, ["raw response is not a dict"]
        content = response_dict.get("content")
        if not isinstance(content, list) or not content:
            return False, ["raw.content is empty or missing"]
        return True, []


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dumped

    return {}


def web_search_sources_from_citations(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    sources: list[dict[str, str]] = []
    for citation in raw:
        data = as_dict(citation)
        if data.get("type") != "web_search_result_location":
            continue
        url = str(data.get("url") or "").strip()
        if not url:
            continue
        title = str(data.get("title") or "").strip() or url
        sources.append({"title": title, "url": url})
    return sources


def parse_anthropic_web_search_result(
    block: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    tool_use_id = str(block.get("tool_use_id") or "")
    content = block.get("content")
    if isinstance(content, dict):
        error = str(content.get("error_code") or content.get("message") or "unknown")
        return (
            {
                "tool_use_id": tool_use_id,
                "is_error": True,
                "error": error,
                "result_count": 0,
            },
            [],
        )
    if not isinstance(content, list):
        return (
            {
                "tool_use_id": tool_use_id,
                "is_error": False,
                "result_count": 0,
            },
            [],
        )

    sources: list[dict[str, str]] = []
    for item in content:
        data = as_dict(item)
        if data.get("type") != "web_search_result":
            continue
        url = str(data.get("url") or "").strip()
        if not url:
            continue
        title = str(data.get("title") or "").strip() or url
        sources.append({"title": title, "url": url})
    return (
        {
            "tool_use_id": tool_use_id,
            "is_error": False,
            "result_count": len(sources),
        },
        sources,
    )


__all__ = [
    "AnthropicMessagesTransport",
    "as_dict",
    "parse_anthropic_web_search_result",
    "web_search_sources_from_citations",
]
