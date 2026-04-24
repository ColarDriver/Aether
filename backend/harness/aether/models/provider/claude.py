"""Anthropic Claude provider with OAuth support and runtime message normalization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
import uuid
from typing import Any

import anthropic

from aether.config.schema import ModelCallConfig
from aether.models.credential_loader import (
    OAUTH_ANTHROPIC_BETAS,
    is_oauth_token,
    load_claude_code_credential,
)
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, StreamDeltaCallback, ToolCall, TurnContext
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
THINKING_BUDGET_RATIO = 0.8

_DEFAULT_BILLING_HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.85.351; "
    "cc_entrypoint=cli; cch=6c6d5;"
)
OAUTH_BILLING_HEADER = os.environ.get("ANTHROPIC_BILLING_HEADER", _DEFAULT_BILLING_HEADER)


class ClaudeChatModel(ModelProvider):
    """ModelProvider backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 16384,
        anthropic_api_key: str | None = None,
        enable_prompt_caching: bool = True,
        prompt_cache_size: int = 3,
        auto_thinking_budget: bool = True,
        retry_max_attempts: int = MAX_RETRIES,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.enable_prompt_caching = enable_prompt_caching
        self.prompt_cache_size = max(1, int(prompt_cache_size))
        self.auto_thinking_budget = auto_thinking_budget
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.default_headers = dict(default_headers or {})

        self._is_oauth = False
        self._oauth_access_token = ""
        self._api_key = self._resolve_api_key(anthropic_api_key)
        self._client = self._build_client()

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        payload = self._build_request_payload(messages, tools=tools, config=config, context=context)

        last_error: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                response = self._create(payload)
                parsed = self._parse_response(response)
                if stream_callback and parsed.content and not parsed.tool_calls:
                    try:
                        stream_callback(parsed.content)
                    except Exception:
                        logger.exception("Claude stream callback failed for final content fallback")
                return parsed
            except anthropic.RateLimitError as exc:
                last_error = exc
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, exc)
                logger.warning(
                    "Anthropic rate limited, retrying %d/%d after %dms",
                    attempt,
                    self.retry_max_attempts,
                    wait_ms,
                )
                time.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as exc:
                last_error = exc
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, exc)
                logger.warning(
                    "Anthropic server error, retrying %d/%d after %dms",
                    attempt,
                    self.retry_max_attempts,
                    wait_ms,
                )
                time.sleep(wait_ms / 1000)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Anthropic call failed without an explicit exception")

    def _resolve_api_key(self, configured_key: str | None) -> str:
        current_key = configured_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not current_key or current_key == "your-anthropic-api-key":
            cred = load_claude_code_credential()
            if cred:
                current_key = cred.access_token
                logger.info("Using Claude Code CLI credential (source: %s)", cred.source)

        if not current_key:
            raise ValueError(
                "No Anthropic API key or Claude Code OAuth credential found. "
                "Set ANTHROPIC_API_KEY or provide Claude Code credentials."
            )

        if is_oauth_token(current_key):
            self._is_oauth = True
            self._oauth_access_token = current_key
            self.default_headers = {
                **self.default_headers,
                "anthropic-beta": OAUTH_ANTHROPIC_BETAS,
            }
            # OAuth tokens have strict limits on cache_control blocks.
            self.enable_prompt_caching = False
            logger.info("OAuth token detected for Anthropic provider")

        return current_key

    def _build_client(self) -> anthropic.Anthropic:
        client = anthropic.Anthropic(
            api_key=self._api_key,
            default_headers=self.default_headers or None,
        )
        if self._is_oauth:
            self._patch_client_oauth(client)
        return client

    def _patch_client_oauth(self, client: Any) -> None:
        if hasattr(client, "api_key") and hasattr(client, "auth_token"):
            client.api_key = None
            client.auth_token = self._oauth_access_token

    def _build_request_payload(
        self,
        messages: list[dict],
        *,
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
    ) -> dict[str, Any]:
        system, converted_messages = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": str(config.extra.get("model", self.model)),
            "max_tokens": int(config.max_tokens if config.max_tokens is not None else self.max_tokens),
            "messages": converted_messages,
        }
        if system:
            payload["system"] = system
        if config.temperature is not None:
            payload["temperature"] = float(config.temperature)

        if tools:
            payload["tools"] = self._convert_tools(tools)

        for key in ("stop_sequences", "thinking", "metadata"):
            if key in config.extra:
                payload[key] = config.extra[key]

        payload.setdefault("metadata", {})
        if isinstance(payload["metadata"], dict):
            payload["metadata"].setdefault("session_id", context.session_id)
            payload["metadata"].setdefault("iteration", context.iteration)

        if self._is_oauth:
            self._apply_oauth_billing(payload)

        if self.enable_prompt_caching:
            self._apply_prompt_caching(payload)

        if self.auto_thinking_budget:
            self._apply_thinking_budget(payload)

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
            nested = content.get("content")
            if nested is not None:
                return cls._normalize_content(nested)
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)

        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system_blocks: list[dict[str, Any]] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = self._normalize_content(msg.get("content"))

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
                        normalized = self._convert_assistant_tool_call(raw_call)
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

    def _convert_assistant_tool_call(self, raw_call: Any) -> dict[str, Any] | None:
        if not isinstance(raw_call, dict):
            return None

        fn = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
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

    def _convert_tools(self, tools: list[ToolDescriptor]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
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

    def _apply_oauth_billing(self, payload: dict[str, Any]) -> None:
        billing_block = {"type": "text", "text": OAUTH_BILLING_HEADER}

        system = payload.get("system")
        if isinstance(system, list):
            filtered = [
                block
                for block in system
                if not (isinstance(block, dict) and OAUTH_BILLING_HEADER in str(block.get("text", "")))
            ]
            payload["system"] = [billing_block] + filtered
        elif isinstance(system, str):
            if OAUTH_BILLING_HEADER in system:
                payload["system"] = [billing_block]
            else:
                payload["system"] = [billing_block, {"type": "text", "text": system}]
        else:
            payload["system"] = [billing_block]

        if not isinstance(payload.get("metadata"), dict):
            payload["metadata"] = {}

        metadata = payload["metadata"]
        if "user_id" not in metadata:
            hostname = socket.gethostname()
            device_id = hashlib.sha256(f"aether-{hostname}".encode()).hexdigest()
            metadata["user_id"] = json.dumps(
                {
                    "device_id": device_id,
                    "account_uuid": "aether",
                    "session_id": str(uuid.uuid4()),
                }
            )

    def _apply_prompt_caching(self, payload: dict[str, Any]) -> None:
        system = payload.get("system")
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["cache_control"] = {"type": "ephemeral"}
        elif isinstance(system, str) and system:
            payload["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        messages = payload.get("messages", [])
        if isinstance(messages, list):
            cache_start = max(0, len(messages) - self.prompt_cache_size)
            for i in range(cache_start, len(messages)):
                msg = messages[i]
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block["cache_control"] = {"type": "ephemeral"}
                elif isinstance(content, str) and content:
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]

        tools = payload.get("tools")
        if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
            tools[-1]["cache_control"] = {"type": "ephemeral"}

    def _apply_thinking_budget(self, payload: dict[str, Any]) -> None:
        thinking = payload.get("thinking")
        if not isinstance(thinking, dict):
            return
        if thinking.get("type") != "enabled":
            return
        if thinking.get("budget_tokens"):
            return

        max_tokens = payload.get("max_tokens", self.max_tokens)
        thinking["budget_tokens"] = int(int(max_tokens) * THINKING_BUDGET_RATIO)

    @staticmethod
    def _strip_cache_control(payload: dict[str, Any]) -> None:
        for section in ("system", "messages"):
            items = payload.get(section)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item.pop("cache_control", None)
                content = item.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block.pop("cache_control", None)

        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool.pop("cache_control", None)

    def _create(self, payload: dict[str, Any]) -> Any:
        request_payload = dict(payload)
        if self._is_oauth:
            self._patch_client_oauth(self._client)
            self._strip_cache_control(request_payload)
        return self._client.messages.create(**request_payload)

    def _parse_response(self, response: Any) -> NormalizedResponse:
        response_dict = _as_dict(response)

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response_dict.get("content", []):
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "text":
                content_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
                    )
                )

        usage = response_dict.get("usage") if isinstance(response_dict.get("usage"), dict) else {}
        metadata: dict[str, Any] = {
            "model": response_dict.get("model", self.model),
            "usage": usage,
        }

        stop_reason = str(response_dict.get("stop_reason") or "")
        if not stop_reason:
            stop_reason = "tool_calls" if tool_calls else "stop"

        return NormalizedResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=stop_reason,
            metadata=metadata,
        )

    @staticmethod
    def _calc_backoff_ms(attempt: int, error: Exception) -> int:
        backoff_ms = 2000 * (1 << (attempt - 1))
        jitter_ms = int(backoff_ms * 0.2)
        total_ms = backoff_ms + jitter_ms

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            retry_after = headers.get("Retry-After")
            if retry_after:
                try:
                    return int(float(retry_after) * 1000)
                except (TypeError, ValueError):
                    pass

        return total_ms



def _as_dict(value: Any) -> dict[str, Any]:
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
