"""Anthropic Claude provider with OAuth support and runtime message normalization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
import uuid
from typing import Any, Callable

import anthropic

from aether.runtime.control.interrupt_signal import InterruptSignal

from aether.config.schema import ModelCallConfig
from aether.models.credential_loader import (
    OAUTH_ANTHROPIC_BETAS,
    OAUTH_CONTEXT_1M_BETA,
    is_oauth_token,
    load_claude_code_credential,
)
from aether.models.provider.base import ModelProvider
from aether.models.transport.anthropic_messages import (
    AnthropicMessagesTransport,
    as_dict as _transport_as_dict,
    parse_anthropic_web_search_result as _transport_parse_anthropic_web_search_result,
    web_search_sources_from_citations as _transport_web_search_sources_from_citations,
)
from aether.runtime.core.contracts import (
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)
_ANTHROPIC_MESSAGES_TRANSPORT = AnthropicMessagesTransport()

MAX_RETRIES = 3
THINKING_BUDGET_RATIO = 0.8

_DEFAULT_BILLING_HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.85.351; "
    "cc_entrypoint=cli; cch=6c6d5;"
)
OAUTH_BILLING_HEADER = os.environ.get("ANTHROPIC_BILLING_HEADER", _DEFAULT_BILLING_HEADER)


class ClaudeChatModel(ModelProvider):
    """ModelProvider backed by the Anthropic Messages API."""

    # Routes ``normalize_usage`` to the Anthropic
    # parser so cache_read_input_tokens / cache_creation_input_tokens
    # are split correctly for billing math.
    provider_name: str = "anthropic"
    api_mode: str = "messages"

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
        self._transport = AnthropicMessagesTransport()

        self._is_oauth = False
        self._oauth_1m_beta_disabled = False
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
        # Accepted for signature parity with the base contract.
        # Anthropic's high-level ``messages.stream`` only exposes a
        # ``text_stream`` (text + thinking deltas already covered by
        # the visible ``stream_callback``); ``input_json_delta``
        # forwarding to a silent counter needs the lower-level event
        # stream and is left for a later implementation.
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        payload = self._build_request_payload(messages, tools=tools, config=config, context=context)

        last_error: Exception | None = None
        oauth_1m_beta_retry_used = False
        attempt = 1
        while attempt <= self.retry_max_attempts:
            try:
                # Route through the streaming API whenever the caller
                # supplied a ``stream_callback``.
                # Anthropic returns a single blob from ``messages.create``
                # and only signals usage at the end of the call — without
                # streaming, the CLI's ``↓ N tokens`` counter stays at 0
                # for the entire wait (response_chars is only fed by
                # ``stream_callback``).  Streaming forwards each text /
                # thinking chunk as it arrives, so the activity bar
                # ticks live, mirroring claude-code's behaviour.
                if stream_callback is not None:
                    interrupt_signal = context.interrupt_signal if context else None
                    response, streamed = self._create_streaming(payload, stream_callback, interrupt_signal=interrupt_signal)
                else:
                    response = self._create(payload)
                    streamed = False
                parsed = self._parse_response(response)
                # Only emit the fallback delta when we *didn't* already
                # stream — otherwise we'd duplicate every visible char
                # in the activity bar (response_chars would double).
                if (
                    stream_callback
                    and not streamed
                    and parsed.content
                    and not parsed.tool_calls
                ):
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
                attempt += 1
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
                attempt += 1
            except anthropic.APIStatusError as exc:
                last_error = exc
                if (
                    not oauth_1m_beta_retry_used
                    and self._maybe_disable_oauth_1m_beta(exc)
                ):
                    oauth_1m_beta_retry_used = True
                    context.metadata["oauth_1m_beta_disabled"] = True
                    continue
                raise

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
                "anthropic-beta": self._oauth_anthropic_betas(),
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

    def _oauth_anthropic_betas(self) -> str:
        betas = [
            beta.strip()
            for beta in OAUTH_ANTHROPIC_BETAS.split(",")
            if beta.strip()
        ]
        if self._oauth_1m_beta_disabled:
            betas = [beta for beta in betas if beta != OAUTH_CONTEXT_1M_BETA]
        return ",".join(betas)

    def _maybe_disable_oauth_1m_beta(self, error: anthropic.APIStatusError) -> bool:
        if not self._is_oauth or self._oauth_1m_beta_disabled:
            return False
        status_code = _anthropic_status_code(error)
        if status_code not in {400, 401, 403}:
            return False
        body = _anthropic_error_text(error)
        if "long context beta" not in body or "not yet available" not in body:
            return False
        current_beta_header = str(self.default_headers.get("anthropic-beta") or "")
        if OAUTH_CONTEXT_1M_BETA not in current_beta_header:
            return False

        self._oauth_1m_beta_disabled = True
        next_beta_header = self._oauth_anthropic_betas()
        if next_beta_header:
            self.default_headers["anthropic-beta"] = next_beta_header
        else:
            self.default_headers.pop("anthropic-beta", None)
        self._close_client_best_effort()
        self._client = self._build_client()
        logger.info("Disabled Anthropic OAuth 1M context beta after provider rejection")
        return True

    def _close_client_best_effort(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Ignoring Anthropic client close failure during rebuild", exc_info=True)

    def _build_request_payload(
        self,
        messages: list[dict],
        *,
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
    ) -> dict[str, Any]:
        payload = self._transport.build_payload(
            model=self.model,
            messages=messages,
            tools=tools,
            config=config,
            context=context,
            max_tokens=self.max_tokens,
        )

        if self._is_oauth:
            self._apply_oauth_billing(payload)

        if self.enable_prompt_caching:
            self._apply_prompt_caching(payload)

        if self.auto_thinking_budget:
            self._apply_thinking_budget(payload)

        return payload

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        del cls
        return _ANTHROPIC_MESSAGES_TRANSPORT.normalize_content(content)

    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return self._transport.convert_messages(messages)

    def _convert_assistant_tool_call(self, raw_call: Any) -> dict[str, Any] | None:
        return self._transport.convert_assistant_tool_call(raw_call)

    def _convert_tools(self, tools: list[ToolDescriptor]) -> list[dict[str, Any]]:
        return self._transport.convert_tools(tools)

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

    def _create_streaming(
        self,
        payload: dict[str, Any],
        stream_callback: StreamDeltaCallback,
        *,
        interrupt_signal: InterruptSignal | None = None,
    ) -> tuple[Any, bool]:
        """Stream the Anthropic Messages call and forward text deltas.

        Returns ``(final_message, streamed)``:

        * ``final_message`` is the same shape as ``messages.create``
          returns — fed into :meth:`_parse_response` as before.
        * ``streamed`` is ``True`` when at least one text/thinking
          chunk was forwarded to ``stream_callback``; the caller uses
          this to decide whether to emit the final full-content
          fallback (which would otherwise double-count).

        The Anthropic Python SDK exposes a high-level ``text_stream``
        iterator that already de-multiplexes ``content_block_delta``
        events for us — we don't need to re-implement the SSE state
        machine.  Tool-use, signature, and ping events are handled
        internally by the SDK; ``get_final_message`` returns the
        same fully-assembled message we'd get from the blocking call,
        including ``tool_use`` blocks and accurate ``usage`` totals.
        """
        request_payload = dict(payload)
        if self._is_oauth:
            self._patch_client_oauth(self._client)
            self._strip_cache_control(request_payload)

        streamed = False
        with self._client.messages.stream(**request_payload) as stream:
            _unregister: Callable[[], None] | None = None
            if interrupt_signal is not None:
                def _on_abort(_reason: str | None) -> None:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass

                interrupt_signal.add_listener(_on_abort)
                _unregister = lambda: interrupt_signal.remove_listener(_on_abort)  # noqa: E731
            try:
                for chunk in stream.text_stream:
                    if not chunk:
                        continue
                    streamed = True
                    try:
                        stream_callback(chunk)
                    except Exception:
                        logger.exception("Claude stream_callback raised; suppressing")
                final_message = stream.get_final_message()
            finally:
                if _unregister is not None:
                    _unregister()

        return final_message, streamed

    def _parse_response(self, response: Any) -> NormalizedResponse:
        return self._transport.normalize_response(response, fallback_model=self.model)

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
    return _transport_as_dict(value)


def _web_search_sources_from_citations(raw: Any) -> list[dict[str, str]]:
    return _transport_web_search_sources_from_citations(raw)


def _parse_anthropic_web_search_result(
    block: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    return _transport_parse_anthropic_web_search_result(block)


def _anthropic_status_code(error: anthropic.APIStatusError) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _anthropic_error_text(error: anthropic.APIStatusError) -> str:
    parts: list[str] = []
    body = getattr(error, "body", None)
    if isinstance(body, str):
        parts.append(body)
    elif isinstance(body, dict):
        try:
            parts.append(json.dumps(body, ensure_ascii=False))
        except TypeError:
            parts.append(str(body))
    elif body is not None:
        parts.append(str(body))

    response = getattr(error, "response", None)
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    if isinstance(text, str):
        parts.append(text)

    parts.append(str(error))
    return "\n".join(part for part in parts if part).casefold()
