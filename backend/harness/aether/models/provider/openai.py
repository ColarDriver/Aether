"""OpenAI Codex provider using the ChatGPT Codex Responses API."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.credential_loader import CodexCliCredential, load_codex_cli_credential
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, StreamDeltaCallback, ToolCall, TurnContext
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
MAX_RETRIES = 3


class CodexChatModel(ModelProvider):
    """ModelProvider backed by ChatGPT Codex Responses API."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        reasoning_effort: str = "medium",
        retry_max_attempts: int = MAX_RETRIES,
        base_url: str = CODEX_BASE_URL,
        request_timeout_sec: int = 300,
        access_token: str | None = None,
        account_id: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.base_url = base_url.rstrip("/")
        self.request_timeout_sec = request_timeout_sec

        self._access_token = access_token or ""
        self._account_id = account_id or ""

        if not self._access_token:
            cred = self._load_codex_auth()
            if not cred:
                raise ValueError(
                    "Codex CLI credential not found. Expected ~/.codex/auth.json or CODEX_AUTH_PATH."
                )
            self._access_token = cred.access_token
            self._account_id = cred.account_id
            logger.info("Using Codex CLI credential (account: %s...)", (self._account_id or "unknown")[:8])

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,  # noqa: ARG002
        stream_callback: StreamDeltaCallback | None = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        response = self._call_codex_api(
            messages,
            tools=tools,
            config=config,
            stream_callback=stream_callback,
        )
        parsed = self._parse_response(response)
        if stream_callback and parsed.content and not parsed.tool_calls:
            try:
                stream_callback(parsed.content)
            except Exception:
                logger.exception("Codex stream callback failed for final content fallback")
        return parsed

    def _load_codex_auth(self) -> CodexCliCredential | None:
        return load_codex_cli_credential()

    def _call_codex_api(
        self,
        messages: list[dict],
        *,
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, tools=tools, config=config)
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "originator": "codex_cli_rs",
        }
        if self._account_id:
            headers["ChatGPT-Account-ID"] = self._account_id

        last_error: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return self._stream_response(headers, payload, stream_callback=stream_callback)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if status_code in (429, 500, 529) and attempt < self.retry_max_attempts:
                    wait_ms = self._calc_backoff_ms(attempt)
                    logger.warning(
                        "Codex API error %s, retrying %d/%d after %dms",
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
                        "Codex transport error, retrying %d/%d after %dms: %s",
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
        raise RuntimeError("Codex API call failed without an explicit exception")

    def _build_payload(
        self,
        messages: list[dict],
        *,
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
    ) -> dict[str, Any]:
        instructions, input_items = self._convert_messages(messages)

        effort = str(config.extra.get("reasoning_effort", self.reasoning_effort))
        payload: dict[str, Any] = {
            "model": str(config.extra.get("model", self.model)),
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

        if tools:
            payload["tools"] = self._convert_tools(tools)

        for key, value in config.extra.items():
            if key in {
                "model",
                "reasoning_effort",
                "tools",
            }:
                continue
            payload.setdefault(key, value)

        return payload

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [cls._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            for key in ("text", "output"):
                value = content.get(key)
                if isinstance(value, str):
                    return value
            nested_content = content.get("content")
            if nested_content is not None:
                return cls._normalize_content(nested_content)
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)

        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict[str, Any]]]:
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = self._normalize_content(msg.get("content"))

            if role == "system":
                if content:
                    instructions_parts.append(content)
                continue

            if role == "user":
                input_items.append({"role": "user", "content": content})
                continue

            if role == "assistant":
                if content:
                    input_items.append({"role": "assistant", "content": content})

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        normalized = self._convert_assistant_tool_call(tool_call)
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

            # Preserve unknown roles as user messages to avoid data loss.
            if content:
                input_items.append({"role": "user", "content": f"[{role}] {content}"})

        instructions = "\n\n".join(part for part in instructions_parts if part) or "You are a helpful assistant."
        return instructions, input_items

    def _convert_assistant_tool_call(self, tool_call: Any) -> dict[str, Any] | None:
        if not isinstance(tool_call, dict):
            return None

        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
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

    def _convert_tools(self, tools: Iterable[ToolDescriptor]) -> list[dict[str, Any]]:
        responses_tools: list[dict[str, Any]] = []
        for tool in tools:
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

    def _stream_response(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> dict[str, Any]:
        completed_response: dict[str, Any] | None = None
        streamed_output_items: dict[int, dict[str, Any]] = {}

        with httpx.Client(timeout=self.request_timeout_sec) as client:
            with client.stream("POST", f"{self.base_url}/responses", headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    data = self._parse_sse_data_line(line)
                    if not data:
                        continue

                    if stream_callback:
                        self._emit_stream_delta(stream_callback, data)

                    event_type = data.get("type")
                    if event_type == "response.output_item.done":
                        output_index = data.get("output_index")
                        output_item = data.get("item")
                        if isinstance(output_index, int) and isinstance(output_item, dict):
                            streamed_output_items[output_index] = output_item
                    elif event_type == "response.completed":
                        maybe_response = data.get("response")
                        if isinstance(maybe_response, dict):
                            completed_response = maybe_response

        if not completed_response:
            raise RuntimeError("Codex API stream ended without response.completed event")

        if streamed_output_items:
            merged_output: list[Any] = []
            response_output = completed_response.get("output")
            if isinstance(response_output, list):
                merged_output = list(response_output)

            max_index = max(max(streamed_output_items), len(merged_output) - 1)
            if max_index >= 0 and len(merged_output) <= max_index:
                merged_output.extend([None] * (max_index + 1 - len(merged_output)))

            for output_index, output_item in streamed_output_items.items():
                existing_item = merged_output[output_index]
                if not isinstance(existing_item, dict):
                    merged_output[output_index] = output_item

            completed_response = dict(completed_response)
            completed_response["output"] = [item for item in merged_output if isinstance(item, dict)]

        return completed_response


    @staticmethod
    def _emit_stream_delta(stream_callback: StreamDeltaCallback, event: dict[str, Any]) -> None:
        delta = CodexChatModel._extract_stream_delta(event)
        if not delta:
            return
        try:
            stream_callback(delta)
        except Exception:
            logger.exception("Codex stream callback failed while emitting delta")

    @staticmethod
    def _extract_stream_delta(event: dict[str, Any]) -> str:
        if not isinstance(event, dict):
            return ""

        event_type = str(event.get("type") or "")

        direct_delta = event.get("delta")
        if isinstance(direct_delta, str):
            return direct_delta
        if isinstance(direct_delta, dict):
            text = direct_delta.get("text")
            if isinstance(text, str):
                return text

        if "output_text" in event_type and isinstance(event.get("text"), str):
            return str(event.get("text"))

        item = event.get("item")
        if isinstance(item, dict):
            if isinstance(item.get("text"), str):
                return item.get("text", "")
            content = item.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        block_text = block.get("text")
                        if isinstance(block_text, str):
                            parts.append(block_text)
                if parts:
                    return "".join(parts)

        return ""

    @staticmethod
    def _parse_sse_data_line(line: str) -> dict[str, Any] | None:
        if not line.startswith("data:"):
            return None

        raw_data = line[5:].strip()
        if not raw_data or raw_data == "[DONE]":
            return None

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON Codex SSE frame: %s", raw_data)
            return None

        return data if isinstance(data, dict) else None

    def _parse_response(self, response: dict[str, Any]) -> NormalizedResponse:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_content = ""

        for output_item in response.get("output", []):
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
            elif item_type == "function_call":
                parsed_arguments, invalid_reason = self._parse_tool_call_arguments(output_item)
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

        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        metadata: dict[str, Any] = {
            "model": response.get("model", self.model),
            "usage": usage,
            "token_usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content

        return NormalizedResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            metadata=metadata,
        )

    def _parse_tool_call_arguments(self, output_item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
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

    @staticmethod
    def _calc_backoff_ms(attempt: int) -> int:
        return 2000 * (1 << (attempt - 1))

