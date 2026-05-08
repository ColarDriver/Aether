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
from aether.runtime.provider_errors import (
    ProviderInvocationError,
    ResponseInvalidError,
    StreamStallError,
)
from aether.tools.base import ToolDescriptor

logger = logging.getLogger(__name__)

# Maximum bytes we copy out of an error response body for classification /
# logging.  Keep this small — it ends up in logs and exception messages, and
# error bodies can be megabytes for some providers.
_MAX_BODY_SUMMARY_CHARS = 1024

# Sprint 1 / PR 1.1: SSE stale-stream detection budget.  If an in-flight
# streaming response produces zero bytes for this many seconds we abort
# with ``StreamStallError`` instead of waiting indefinitely.  Exposed as a
# class attribute on ``OpenAICompatibleModel`` so tests can override without
# the slow real-clock wait.
_DEFAULT_STREAM_READ_TIMEOUT_SEC: float = 90.0
# How long we are willing to wait for the *connect / write* phase, even
# while streaming is enabled.  The big timeout above is only for inter-event
# silence on an established connection.
_STREAM_CONNECT_TIMEOUT_SEC: float = 10.0
_STREAM_WRITE_TIMEOUT_SEC: float = 10.0


class OpenAICompatibleModel(ModelProvider):
    """ModelProvider for standard OpenAI-compatible chat-completions APIs.

    Unlike the previous LangChain patch approach, this provider makes direct HTTP
    requests and intentionally preserves `thought_signature` on assistant tool calls.
    """

    # Per-instance read timeout for streaming.  Exposed as a class attribute
    # so tests can override; production-side users can also bump it via
    # ``OpenAICompatibleModel.stream_read_timeout_sec = ...`` on a specific
    # instance.
    stream_read_timeout_sec: float = _DEFAULT_STREAM_READ_TIMEOUT_SEC

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

        # Sprint 1 / PR 1.1: per-instance flag flipped to True the first
        # time a streaming attempt fails with ``StreamStallError``.  Once
        # set, subsequent ``generate()`` calls skip streaming and go
        # straight to the (slower but more reliable) non-streaming path.
        # The flag deliberately lives on the instance, not the class —
        # different OpenAICompatibleModel instances pointing at different
        # gateways must not poison each other.
        self._disable_streaming: bool = False

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,  # noqa: ARG002
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        """Dispatcher that picks streaming vs non-streaming based on context.

        Decision rules:

        1. If the caller did not supply a ``stream_callback`` → non-streaming
           (no point streaming if no one is listening).
        2. If a previous call on this instance hit ``StreamStallError`` and
           we set ``_disable_streaming`` → non-streaming (one-strike-out
           per session, mirrors hermes 11369–11398).
        3. Otherwise → streaming.  If the streaming attempt itself fails
           with ``StreamStallError``, we flip ``_disable_streaming`` and
           re-attempt the same request once via the non-streaming path
           (so the user still gets *some* answer instead of a hard error).

        Any exception that escapes this method is — by contract — a
        ``ProviderInvocationError`` (or one of its structured subclasses).
        The engine's recovery layer (``runtime/recovery.py``) owns retry
        decisions for those.
        """
        payload = self._build_payload(messages, tools=tools, config=config)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        if stream_callback is not None and not self._disable_streaming:
            try:
                return self._streaming_generate(
                    url=url,
                    headers=headers,
                    payload=payload,
                    stream_callback=stream_callback,
                )
            except StreamStallError as stall:
                # Hot path for stuck SSE streams.  Disable streaming for the
                # rest of this provider instance's life and retry once
                # non-streaming so the user still gets an answer.
                logger.warning(
                    "SSE stream stalled after %.1fs; disabling streaming for this provider and falling back to non-streaming",
                    stall.stalled_after_seconds,
                )
                self._disable_streaming = True
                # Intentional fall-through: re-issue the same request via the
                # non-streaming path below.  We deliberately don't pass
                # stream_callback this time — there's no chunked output to
                # forward, and the engine will handle the final-response
                # one-shot fallback if needed.
                return self._non_streaming_generate(
                    url=url, headers=headers, payload=payload
                )

        return self._non_streaming_generate(
            url=url,
            headers=headers,
            payload=payload,
            stream_callback=stream_callback,
        )

    def _non_streaming_generate(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        """Original Sprint 0 single-shot path, factored out of generate().

        Behaviour and error contract are unchanged from PR 0.2:
        any HTTP / network / JSON failure is wrapped into a
        ``ProviderInvocationError``; success returns a parsed
        ``NormalizedResponse``.
        """
        # The non-streaming branch must turn off ``stream`` in the payload —
        # otherwise some gateways (notably Anthropic-via-OpenRouter) would
        # try to send back SSE and the synchronous client would block.
        non_streaming_payload = dict(payload)
        non_streaming_payload["stream"] = False

        try:
            with httpx.Client(timeout=self.request_timeout_sec) as client:
                resp = client.post(url, headers=headers, json=non_streaming_payload)
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

    def _streaming_generate(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        stream_callback: StreamDeltaCallback,
    ) -> NormalizedResponse:
        """SSE-based streaming path.

        Implements the four pieces called out in P0-1 of the run-loop
        roadmap:

        1. **Per-chunk callback fan-out** — every meaningful delta (content
           or tool-call argument fragment) triggers ``stream_callback``
           immediately, so consumers see characters appear in real time
           rather than as one giant blob.
        2. **Stale-stream watchdog** — httpx's read timeout is set to
           ``stream_read_timeout_sec`` (default 90 s).  If the server
           stops sending bytes for that long we surface a structured
           ``StreamStallError`` instead of letting it look like a generic
           network error.  The 90-second figure mirrors hermes' tested
           value; it's long enough to tolerate a slow first-token
           generation but short enough that hung gateways still get
           noticed.
        3. **First-delta detection** — we only need this internally so the
           non-streaming fallback path knows we already streamed
           *something* (so the engine's "fallback to one-shot
           stream_callback at the end" code doesn't double-emit).  We
           record this on the response metadata as ``streamed=True``.
        4. **Tool-call accumulation** — chat-completions sends tool-call
           arguments in fragments (one ``index`` per parallel call); we
           buffer them by index and reconstitute the final ``ToolCall``
           list when the stream completes.
        """
        streaming_payload = dict(payload)
        streaming_payload["stream"] = True
        streaming_payload.setdefault("stream_options", {})
        # Without ``include_usage`` the OpenAI streaming protocol does not
        # send a final usage block.  Set it here so token accounting still
        # works under streaming — Sprint 3 will rely on this for cost
        # reporting.
        if isinstance(streaming_payload["stream_options"], dict):
            streaming_payload["stream_options"].setdefault("include_usage", True)

        # Compose a timeout that gives the connect/write phases a small
        # budget but allows up to ``stream_read_timeout_sec`` of inter-byte
        # silence.  httpx's ReadTimeout fires when this elapses and we
        # convert it into ``StreamStallError`` below.
        timeout = httpx.Timeout(
            connect=_STREAM_CONNECT_TIMEOUT_SEC,
            read=self.stream_read_timeout_sec,
            write=_STREAM_WRITE_TIMEOUT_SEC,
            pool=5.0,
        )

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream(
                    "POST", url, headers=headers, json=streaming_payload
                ) as resp:
                    # Materialise the error body **inside** the stream
                    # context so ``response.text`` is available later.
                    # Without this, ``raise_for_status`` raises *after*
                    # the ``with client.stream(...)`` block has already
                    # closed the response, and any subsequent
                    # ``response.read()`` / ``.text`` access fails with
                    # ``httpx.ResponseNotRead`` — masking the real 4xx
                    # body that the recovery layer would otherwise log.
                    if resp.status_code >= 400:
                        try:
                            resp.read()
                        except Exception:        # noqa: BLE001 - best effort
                            pass
                    resp.raise_for_status()
                    return _parse_sse_stream(
                        resp.iter_lines(),
                        stream_callback=stream_callback,
                        fallback_model=self.model,
                    )
        except httpx.ReadTimeout as exc:
            raise StreamStallError(
                raw=exc,
                stalled_after_seconds=self.stream_read_timeout_sec,
                body_summary=f"no SSE event for {self.stream_read_timeout_sec:.0f}s",
                metadata={"url": url, "method": "POST", "phase": "stream-read"},
            ) from exc
        except httpx.HTTPStatusError as exc:
            # Stream attempt got an HTTP error response (e.g. 429 / 400
            # before streaming even started).  Body was already
            # materialised inside the ``with`` block above — we just
            # summarise it here for the recovery layer.
            response = exc.response
            raise ProviderInvocationError(
                raw=exc,
                status_code=response.status_code,
                retry_after_seconds=_parse_retry_after(response.headers),
                body_summary=_summarize_body(response),
                is_network_error=False,
                metadata={"url": url, "method": "POST", "phase": "stream-open"},
            ) from exc
        except (httpx.TransportError, TimeoutError) as exc:
            raise ProviderInvocationError(
                raw=exc,
                status_code=None,
                retry_after_seconds=None,
                body_summary=str(exc) or exc.__class__.__name__,
                is_network_error=True,
                metadata={"url": url, "method": "POST", "phase": "stream-open"},
            ) from exc

    def validate_response(
        self,
        response: NormalizedResponse,
    ) -> tuple[bool, list[str]]:
        """Detect malformed chat-completions responses post-parse.

        Sprint 1 / PR 1.1 — we deliberately keep this conservative.  A
        response is considered invalid only when it has neither content
        nor any tool calls *and* the upstream server signalled a non-stop
        finish reason that suggests an actual error rather than an empty
        completion.  False positives here are expensive: every invalid
        verdict goes back through the recovery layer and costs another
        round-trip.

        Examples that trigger invalid:
        - HTTP 200 with ``{"error": "..."}`` body that ``_parse_response``
          coerced into an empty NormalizedResponse + finish_reason="stop".
          We catch this via metadata sniffing.
        - Empty ``choices`` array in raw, again surfacing as
          ``finish_reason="stop"`` with no content/tools.

        Examples that DO NOT trigger invalid:
        - Legitimate empty assistant response (rare but allowed by the
          API spec) — engines downstream handle that as ExitReason
          EMPTY_RESPONSE, no need to retry.
        """
        reasons: list[str] = []

        raw = response.metadata.get("raw") if isinstance(response.metadata, dict) else None
        if isinstance(raw, dict):
            # OpenRouter-style 200 with embedded error object.
            err = raw.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("type") or "unknown"
                reasons.append(f"raw.error.{msg}")
            elif isinstance(err, str) and err:
                reasons.append(f"raw.error: {err[:100]}")

            # No choices at all — the server returned a structurally
            # impossible chat-completions response.
            choices = raw.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                # Only flag this when the response is also empty —
                # otherwise downstream parsing already coped.
                if not response.content and not response.tool_calls:
                    reasons.append("raw.choices is empty or missing")

        return (len(reasons) == 0), reasons

    def list_models(self) -> list[str]:
        """Fetch available model ids from ``GET {base_url}/models``.

        Returns ids sorted lexicographically.  On any error returns an
        empty list — the caller treats that as "endpoint unavailable" and
        falls back to a degraded picker.
        """
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=min(self.request_timeout_sec, 10)) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []

        ids: list[str] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
        return sorted(set(ids))

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
            # Spell ``tool_choice`` out explicitly.  The OpenAI spec says
            # "auto" is the default when ``tools`` is present, but in
            # practice some compatible gateways and Anthropic-mode
            # adapters fall back to "none" if the field is missing — the
            # symptom is a model that "narrates" its tool calls in prose
            # (e.g. a markdown ```bash``` block) without ever populating
            # the structured ``tool_calls`` field.  Setting "auto"
            # guarantees the model sees tool use as a live option.
            payload.setdefault("tool_choice", "auto")

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
            # Sprint 1 / PR 1.1: always carry the raw dict so
            # ``validate_response`` can introspect provider-specific error
            # shapes (e.g. OpenRouter's HTTP-200-with-error-body).
            return NormalizedResponse(
                content="",
                tool_calls=[],
                finish_reason="stop",
                metadata={"raw": data, "model": data.get("model", self.model)},
            )

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
            # Always retain the raw payload so the engine's post-LLM
            # validation step can inspect server-side error envelopes.
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

    Defensive against ``httpx.ResponseNotRead`` and similar — accessing
    ``response.text`` on a streaming response that hasn't been read can
    raise.  Returning ``None`` in that case is preferable to crashing
    the entire error path (the recovery layer needs the surrounding
    ``ProviderInvocationError`` to do its job, with or without a body).
    """
    try:
        text = getattr(response, "text", None)
    except Exception:        # noqa: BLE001 - defensive: never crash the error path
        return None
    if not isinstance(text, str) or not text:
        return None
    text = text.strip()
    if len(text) <= _MAX_BODY_SUMMARY_CHARS:
        return text
    return text[:_MAX_BODY_SUMMARY_CHARS] + "...(truncated)"


# ---------------------------------------------------------------------------
# SSE parser (Sprint 1 / PR 1.1)
# ---------------------------------------------------------------------------


def _parse_sse_stream(
    line_iter: Iterable[str],
    *,
    stream_callback: StreamDeltaCallback,
    fallback_model: str,
) -> NormalizedResponse:
    """Consume an SSE ``data:`` event stream and reconstruct a NormalizedResponse.

    Logic walks the ``data: {...}`` events emitted by an OpenAI-style
    chat-completions streaming endpoint:

    - Lines starting with ``:`` (SSE comments / keepalives) are ignored.
    - ``data: [DONE]`` terminates the stream.
    - Every other ``data: ...`` line is parsed as JSON; if parsing fails we
      log and skip the malformed event rather than aborting the whole
      generation (gateways occasionally emit garbage frames).

    The callback contract:

    - Every visible content delta is forwarded **immediately** via
      ``stream_callback(delta)``.  Empty strings are silently dropped so
      consumers don't see noise.
    - We do **not** forward tool-call argument fragments — they are
      buffered and will be exposed as a fully-parsed ``ToolCall`` list on
      the final response.  Forwarding them as text would just paint
      half-formed JSON onto the user's terminal.

    The function is a free function rather than a method so that tests can
    feed it a synthetic line iterator without spinning up an HTTP server.
    """
    accumulated_parts: list[str] = []
    tool_buffers: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    model_name: str | None = None
    usage: dict[str, Any] = {}
    last_event: dict[str, Any] | None = None

    for raw_line in line_iter:
        # ``iter_lines`` strips the trailing newline.  Server-Sent Events
        # delimit messages with blank lines (which appear as empty strings
        # here); we treat them as no-ops.  Keepalive comments (``:``) also
        # carry no payload.
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            # Some gateways prepend ``event: ...`` lines we don't care
            # about — skip silently.
            continue
        data_text = line[5:].lstrip()
        if data_text == "[DONE]":
            break
        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            logger.debug("openai SSE: skipping non-JSON data frame: %r", data_text[:120])
            continue
        last_event = event

        # OpenAI's streaming usage block (when stream_options.include_usage
        # is set) arrives in a final event whose ``choices`` list is empty.
        # Capture it before the choices-shape branches.
        ev_usage = event.get("usage")
        if isinstance(ev_usage, dict):
            usage = ev_usage

        if not model_name and event.get("model"):
            model_name = str(event["model"])

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else None
        if choice is None:
            continue

        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

        # Visible content delta — push to consumer immediately.
        delta_content = delta.get("content")
        if isinstance(delta_content, str) and delta_content:
            accumulated_parts.append(delta_content)
            try:
                stream_callback(delta_content)
            except Exception:
                logger.exception("openai SSE stream_callback raised; suppressing")

        # Tool-call argument fragments — buffer per index; the final
        # message reassembles them into ToolCall objects.
        delta_tool_calls = delta.get("tool_calls")
        if isinstance(delta_tool_calls, list):
            for raw in delta_tool_calls:
                if not isinstance(raw, dict):
                    continue
                try:
                    idx = int(raw.get("index", 0))
                except (TypeError, ValueError):
                    idx = 0
                buf = tool_buffers.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if raw.get("id"):
                    buf["id"] = str(raw["id"])
                fn = raw.get("function") if isinstance(raw.get("function"), dict) else None
                if fn is not None:
                    if fn.get("name"):
                        buf["name"] = str(fn["name"])
                    args_chunk = fn.get("arguments")
                    if isinstance(args_chunk, str) and args_chunk:
                        buf["arguments"] += args_chunk

        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])

    # Reconstitute the final assistant message.
    full_content = "".join(accumulated_parts)
    tool_calls: list[ToolCall] = []
    for idx in sorted(tool_buffers.keys()):
        buf = tool_buffers[idx]
        if not buf["id"] or not buf["name"]:
            # Drop fragments without identity — these are gateway artefacts
            # rather than legitimate calls.  Logged for diagnostics.
            logger.debug("openai SSE: dropping incomplete tool-call buffer at index %d", idx)
            continue
        args_text = buf["arguments"].strip()
        if args_text:
            try:
                parsed_args = json.loads(args_text)
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
            except json.JSONDecodeError:
                # Sprint 1 / PR 1.1 deliberately keeps this lenient.  Sprint
                # 1 / PR 1.3 (truncated tool-call detection, P0-4) is the
                # right place to flag this as truncated and bubble up — for
                # now we surface an empty args dict and let downstream
                # tool dispatch see the broken state.
                logger.warning(
                    "openai SSE: tool_call args at index %d failed to JSON-parse; treating as empty",
                    idx,
                )
                parsed_args = {}
        else:
            parsed_args = {}
        tool_calls.append(ToolCall(id=buf["id"], name=buf["name"], arguments=parsed_args))

    if not finish_reason:
        finish_reason = "tool_calls" if tool_calls else "stop"
    elif tool_calls:
        # OpenAI sends finish_reason="tool_calls" when applicable, but some
        # gateways send "stop" with a tool_calls payload — normalise here so
        # the engine's branch decision (text vs tool path) always works.
        finish_reason = "tool_calls"

    metadata: dict[str, Any] = {
        "model": model_name or fallback_model,
        "usage": usage,
        "token_usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0,
            "completion_tokens": usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0,
            "total_tokens": usage.get("total_tokens", 0) if isinstance(usage, dict) else 0,
        },
        # Validation-step context: keep the last raw event around so
        # ``validate_response`` can spot embedded error envelopes that some
        # gateways tack on at the very end of a stream.
        "raw": last_event or {},
        "streamed": True,
    }

    return NormalizedResponse(
        content=full_content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        metadata=metadata,
    )
