"""Codex provider using the ChatGPT Codex Responses API."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Iterable

import httpx

from aether.config.schema import ModelCallConfig
from aether.models.credential_loader import CodexCliCredential, load_codex_cli_credential
from aether.models.provider.base import ModelProvider
from aether.models.transport.codex_responses import (
    CodexResponsesTransport,
    codex_web_search_call_metadata as _transport_codex_web_search_call_metadata,
    web_search_sources_from_annotations as _transport_web_search_sources_from_annotations,
)
from aether.runtime.control.interrupt_signal import InterruptSignal
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)
_CODEX_RESPONSES_TRANSPORT = CodexResponsesTransport()

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
MAX_RETRIES = 3


class CodexChatModel(ModelProvider):
    """ModelProvider backed by ChatGPT Codex Responses API."""

    # Routes ``normalize_usage`` to the Codex / Responses
    # parser — input_tokens already excludes cached_tokens (unlike OpenAI
    # Chat Completions), and reasoning lives in output_tokens_details.
    provider_name: str = "codex"
    api_mode: str = "responses"
    transport_name: str | None = "codex_responses"
    transport_api_mode: str | None = "codex_responses"

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
        self._transport = CodexResponsesTransport()

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
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        interrupt_signal = context.interrupt_signal if context else None
        response = self._call_codex_api(
            messages,
            tools=tools,
            config=config,
            stream_callback=stream_callback,
            stream_silent_callback=stream_silent_callback,
            interrupt_signal=interrupt_signal,
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
        stream_silent_callback: StreamSilentCallback | None = None,
        interrupt_signal: InterruptSignal | None = None,
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
                return self._stream_response(
                    headers,
                    payload,
                    stream_callback=stream_callback,
                    stream_silent_callback=stream_silent_callback,
                    interrupt_signal=interrupt_signal,
                )
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
        return self._transport.build_payload(
            model=self.model,
            messages=messages,
            tools=tools,
            config=config,
            reasoning_effort=self.reasoning_effort,
        )

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        del cls
        return _CODEX_RESPONSES_TRANSPORT.normalize_content(content)

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict[str, Any]]]:
        return self._transport.convert_messages(messages)

    def _convert_assistant_tool_call(self, tool_call: Any) -> dict[str, Any] | None:
        return self._transport.convert_assistant_tool_call(tool_call)

    def _convert_tools(self, tools: Iterable[ToolDescriptor]) -> list[dict[str, Any]]:
        return self._transport.convert_tools(tools)

    def _stream_response(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
        interrupt_signal: InterruptSignal | None = None,
    ) -> dict[str, Any]:
        completed_response: dict[str, Any] | None = None
        streamed_output_items: dict[int, dict[str, Any]] = {}

        with httpx.Client(timeout=self.request_timeout_sec) as client:
            _unregister: Callable[[], None] | None = None
            if interrupt_signal is not None:
                _unregister = _register_interrupt_listener(interrupt_signal, client)
            try:
                with client.stream("POST", f"{self.base_url}/responses", headers=headers, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        data = self._parse_sse_data_line(line)
                        if not data:
                            continue

                        if stream_silent_callback:
                            self._emit_stream_silent_delta(stream_silent_callback, data)

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
            finally:
                if _unregister is not None:
                    _unregister()

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
    def _emit_stream_silent_delta(
        stream_silent_callback: StreamSilentCallback,
        event: dict[str, Any],
    ) -> None:
        delta = CodexChatModel._extract_stream_silent_delta(event)
        if not delta:
            return
        try:
            stream_silent_callback(delta)
        except Exception:
            logger.exception("Codex silent stream callback failed while emitting delta")

    @staticmethod
    def _extract_stream_delta(event: dict[str, Any]) -> str:
        if not isinstance(event, dict):
            return ""

        event_type = str(event.get("type") or "")
        if CodexChatModel._is_function_call_arguments_delta(event_type):
            return ""

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
    def _extract_stream_silent_delta(event: dict[str, Any]) -> str:
        if not isinstance(event, dict):
            return ""

        event_type = str(event.get("type") or "")
        if not CodexChatModel._is_function_call_arguments_delta(event_type):
            return ""

        delta = event.get("delta")
        return delta if isinstance(delta, str) else ""

    @staticmethod
    def _is_function_call_arguments_delta(event_type: str) -> bool:
        return "function_call_arguments" in event_type and event_type.endswith(".delta")

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
        return self._transport.normalize_response(response, fallback_model=self.model)

    def _parse_tool_call_arguments(self, output_item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        return self._transport.parse_tool_call_arguments(output_item)

    @staticmethod
    def _calc_backoff_ms(attempt: int) -> int:
        return 2000 * (1 << (attempt - 1))


def _register_interrupt_listener(
    signal: InterruptSignal,
    client: httpx.Client,
) -> Callable[[], None]:
    def _on_abort(_reason: str | None) -> None:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    signal.add_listener(_on_abort)

    def _unregister() -> None:
        signal.remove_listener(_on_abort)

    return _unregister


def _web_search_sources_from_annotations(raw: Any) -> list[dict[str, str]]:
    return _transport_web_search_sources_from_annotations(raw)


def _codex_web_search_call_metadata(output_item: dict[str, Any]) -> dict[str, Any]:
    return _transport_codex_web_search_call_metadata(output_item)
