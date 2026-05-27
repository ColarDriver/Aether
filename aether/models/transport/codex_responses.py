"""Codex Responses API transport."""

from __future__ import annotations

import json
from typing import Any, Iterable

from aether.config.schema import ModelCallConfig
from aether.models.provider.hosted_web_search import (
    append_sources_section,
    dedupe_sources,
    is_web_search_tool,
)
from aether.runtime.core.contracts import NormalizedResponse, ToolCall, TurnContext
from aether.tools.base import ToolDescriptor


class CodexResponsesTransport:
    """Pure conversion layer for the Codex/OpenAI Responses schema."""

    api_mode = "codex_responses"
    name = "codex_responses"

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Iterable[ToolDescriptor],
        config: ModelCallConfig,
        reasoning_effort: str,
        context: TurnContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del context, kwargs
        instructions, input_items = self.convert_messages(messages)

        effort = str(config.extra.get("reasoning_effort", reasoning_effort))
        payload: dict[str, Any] = {
            "model": str(config.extra.get("model", model)),
            "instructions": instructions,
            "input": input_items,
            "store": False,
            "stream": True,
            "reasoning": {"effort": effort, "summary": "detailed"}
            if effort != "none"
            else {"effort": "none"},
        }

        if config.max_tokens is not None:
            payload["max_output_tokens"] = int(config.max_tokens)
        if config.temperature is not None:
            payload["temperature"] = float(config.temperature)

        converted_tools = self.convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            payload.setdefault("tool_choice", "auto")

        for key, value in config.extra.items():
            if key in {"model", "reasoning_effort", "tools"}:
                continue
            payload.setdefault(key, value)

        return payload

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        del kwargs
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = self.normalize_content(msg.get("content"))

            if role == "system":
                if content:
                    instructions_parts.append(content)
                continue

            if role == "user":
                input_items.append({"role": "user", "content": self.convert_user_content(msg.get("content"))})
                continue

            if role == "assistant":
                if content:
                    input_items.append({"role": "assistant", "content": content})

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        normalized = self.convert_assistant_tool_call(tool_call)
                        if normalized is not None:
                            input_items.append(normalized)
                continue

            if role == "tool":
                call_id = str(msg.get("tool_call_id") or msg.get("id") or "")
                if not call_id:
                    continue
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": content,
                    }
                )
                continue

            if content:
                input_items.append({"role": "user", "content": f"[{role}] {content}"})

        instructions = "\n\n".join(part for part in instructions_parts if part) or "You are a helpful assistant."
        return instructions, input_items

    def convert_user_content(self, content: Any) -> str | list[dict[str, Any]]:
        if not isinstance(content, list):
            return self.normalize_content(content)
        parts: list[dict[str, Any]] = []
        has_image = False
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append({"type": "input_text", "text": item})
                continue
            if not isinstance(item, dict):
                text = self.normalize_content(item)
                if text:
                    parts.append({"type": "input_text", "text": text})
                continue
            image_url = self._image_url_from_part(item)
            if image_url:
                has_image = True
                parts.append({"type": "input_image", "image_url": image_url})
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "input_text", "text": text})
        if has_image:
            return parts or ""
        return "\n".join(str(part.get("text") or "") for part in parts if part.get("text"))

    @staticmethod
    def _image_url_from_part(part: dict[str, Any]) -> str:
        part_type = str(part.get("type") or "")
        if part_type not in {"image_url", "input_image"}:
            return ""
        image_url = part.get("image_url")
        if isinstance(image_url, str) and image_url.strip():
            return image_url.strip()
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        url = part.get("url")
        return url.strip() if isinstance(url, str) and url.strip() else ""

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
            nested_content = content.get("content")
            if nested_content is not None:
                return self.normalize_content(nested_content)
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)

        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    def convert_assistant_tool_call(self, tool_call: Any) -> dict[str, Any] | None:
        if not isinstance(tool_call, dict):
            return None

        raw_function = tool_call.get("function")
        fn: dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}
        name = fn.get("name") or tool_call.get("name")
        if not name:
            return None

        raw_arguments = fn.get("arguments", tool_call.get("arguments", {}))
        arguments = raw_arguments
        if isinstance(raw_arguments, dict):
            arguments = json.dumps(raw_arguments)

        call_id = tool_call.get("id") or tool_call.get("call_id") or ""
        return {
            "type": "function_call",
            "name": str(name),
            "arguments": arguments,
            "call_id": str(call_id),
        }

    def convert_tools(self, tools: Iterable[ToolDescriptor], **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        responses_tools: list[dict[str, Any]] = []
        for tool in tools:
            if is_web_search_tool(tool):
                responses_tools.append(
                    {
                        "type": "web_search",
                        "search_context_size": "medium",
                    }
                )
                continue

            parameters = dict(tool.parameters)
            if "type" not in parameters and "properties" not in parameters:
                parameters = {
                    "type": "object",
                    "properties": parameters,
                }
            if tool.required and "required" not in parameters:
                parameters["required"] = list(tool.required)

            responses_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": parameters,
                }
            )
        return responses_tools

    def normalize_response(
        self,
        response: Any,
        *,
        fallback_model: str,
        **kwargs: Any,
    ) -> NormalizedResponse:
        del kwargs
        data = response if isinstance(response, dict) else {}
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_content = ""
        hosted_search_calls: list[dict[str, Any]] = []
        hosted_search_sources: list[dict[str, Any]] = []

        for output_item in data.get("output", []):
            if not isinstance(output_item, dict):
                continue

            item_type = output_item.get("type")
            if item_type == "reasoning":
                for summary_item in output_item.get("summary", []):
                    if isinstance(summary_item, dict) and summary_item.get("type") == "summary_text":
                        reasoning_content += str(summary_item.get("text", ""))
                    elif isinstance(summary_item, str):
                        reasoning_content += summary_item
            elif item_type == "message":
                for part in output_item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        content_parts.append(str(part.get("text", "")))
                        hosted_search_sources.extend(
                            web_search_sources_from_annotations(part.get("annotations"))
                        )
            elif item_type == "function_call":
                parsed_arguments, invalid_reason = self.parse_tool_call_arguments(output_item)
                if invalid_reason:
                    content_parts.append(invalid_reason)
                    continue
                tool_calls.append(
                    ToolCall(
                        id=str(output_item.get("call_id", "")),
                        name=str(output_item.get("name", "")),
                        arguments=parsed_arguments,
                    )
                )
            elif item_type == "web_search_call":
                hosted_search_calls.append(codex_web_search_call_metadata(output_item))

        raw_usage = data.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        sources = dedupe_sources(hosted_search_sources)
        content = "".join(content_parts)
        if sources:
            content = append_sources_section(content, sources)
        metadata: dict[str, Any] = {
            "model": data.get("model", fallback_model),
            "usage": usage,
            "token_usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
        if hosted_search_calls or sources:
            metadata["hosted_web_search"] = {
                "provider": "codex",
                "calls": hosted_search_calls,
                "sources": sources,
                "source_count": len(sources),
            }

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            metadata=metadata,
        )

    def parse_tool_call_arguments(self, output_item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        raw_arguments = output_item.get("arguments", "{}")
        if isinstance(raw_arguments, dict):
            return raw_arguments, None

        normalized_arguments = raw_arguments or "{}"
        try:
            parsed_arguments = json.loads(normalized_arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            return {}, (
                f"Invalid tool call arguments for '{output_item.get('name')}': {exc}. "
                "Skipping tool call."
            )

        if not isinstance(parsed_arguments, dict):
            return {}, (
                f"Invalid tool call arguments for '{output_item.get('name')}': "
                "arguments must decode to a JSON object. Skipping tool call."
            )

        return parsed_arguments, None

    def validate_raw_response(self, response: Any) -> tuple[bool, list[str]]:
        if not isinstance(response, dict):
            return False, ["raw response is not a dict"]
        output = response.get("output")
        if not isinstance(output, list) or not output:
            return False, ["raw.output is empty or missing"]
        return True, []


def web_search_sources_from_annotations(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    sources: list[dict[str, str]] = []
    for annotation in raw:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("type") != "url_citation":
            continue
        url = str(annotation.get("url") or "").strip()
        if not url:
            continue
        title = str(annotation.get("title") or "").strip() or url
        sources.append({"title": title, "url": url})
    return sources


def codex_web_search_call_metadata(output_item: dict[str, Any]) -> dict[str, Any]:
    action = output_item.get("action")
    return {
        "id": str(output_item.get("id") or ""),
        "status": str(output_item.get("status") or ""),
        "action": action if isinstance(action, dict) else {},
    }


__all__ = [
    "CodexResponsesTransport",
    "codex_web_search_call_metadata",
    "web_search_sources_from_annotations",
]
