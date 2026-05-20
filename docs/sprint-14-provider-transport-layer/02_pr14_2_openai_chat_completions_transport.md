# PR 14.2 - OpenAI Chat Completions Transport

## Goal

Move OpenAI-compatible chat-completions message conversion, tool conversion,
payload construction, response normalization, finish reason mapping, and raw
validation into a transport class.

## Target Files

Add:

- `aether/models/transport/openai_chat.py`

Modify:

- `aether/models/provider/openai_compatible.py`

## Boundary

The provider still owns:

- HTTP client lifecycle.
- Request timeout and dead connection cleanup.
- Streaming vs non-streaming decision.
- Stream stall fallback.
- Raising `ProviderInvocationError` from HTTP failures.

The transport owns:

- `_build_payload` equivalent.
- Message sanitization required by chat completions.
- Tool schema conversion.
- Non-streaming response normalization.
- Streaming final chunk normalization where possible.
- Finish reason normalization.
- `validate_raw_response`.

## Migration Strategy

1. Copy current payload/normalization helpers into `OpenAIChatCompletionsTransport`.
2. Update `OpenAICompatibleModel` to instantiate the transport.
3. Replace internal helper calls with transport method calls.
4. Keep old private helper wrappers temporarily if tests import them.
5. Remove wrappers only when there are no imports.

## Hosted Web Search

Do not route local `web_search` through this transport. OpenAI-compatible
providers use Aether's local `web_search` tool unless a future provider-specific
native search capability is explicitly added.

## Tests

Add or update:

- `aether/tests/models/transport/test_openai_chat_transport.py`
- `aether/tests/models/test_openai_compatible_dead_connections.py`
- `aether/tests/models/test_provider_streaming.py`

Cover:

- Payload with no tools.
- Payload with tool descriptors.
- Tool calls round-trip into `NormalizedResponse.tool_calls`.
- Thought/signature metadata is preserved.
- Usage metadata shape is unchanged.
- Invalid response is rejected with the same reasons.

## Acceptance

- `OpenAICompatibleModel.generate()` behavior is unchanged.
- Provider tests pass.
- Targeted pyright passes for `aether/models/provider/openai_compatible.py` and
  `aether/models/transport/`.

## Detailed Implementation Notes

### Exact Methods to Move

Move these from `OpenAICompatibleModel` into
`OpenAIChatCompletionsTransport`:

- `_build_payload`
- `_convert_messages`
- `_normalize_tool_call`
- `_convert_tools`
- `_parse_response`
- `_parse_tool_call`
- `_normalize_content`

Keep these in `OpenAICompatibleModel`:

- `generate`
- `_non_streaming_generate`
- `_streaming_generate`
- `_get_client`
- `_build_http_client`
- `_rebuild_client`
- `_with_stale_connection_retry`
- `cleanup_dead_connections`
- `list_models`
- `_parse_sse_stream` unless stream event normalization can be moved without
  changing callback timing

### Payload Parity Requirements

The transport must preserve:

- `model`
- `messages`
- `temperature`
- `max_tokens`
- `stream`
- `tools`
- `tool_choice`
- any `config.extra` passthrough fields currently supported
- provider-specific preservation of `thought_signature` on assistant tool calls

Write fixture tests that compare the old payload and new payload during the
migration. If old private methods are temporarily kept as wrappers, the test can
call both until cleanup.

### Response Parity Requirements

The normalized response must preserve:

- `content`
- `tool_calls`
- `finish_reason`
- `metadata["usage"]`
- reasoning fields currently parsed from provider response, if present
- raw provider IDs only if they are already in metadata today

Do not add new public metadata unless it is necessary for transport
observability and documented in PR14.4.

### Streaming Detail

OpenAI-compatible streaming has two separate concerns:

- stream event parsing and incremental callbacks
- final `NormalizedResponse` construction

If the current `_parse_sse_stream` is tightly coupled to callbacks, keep it in
the provider for PR14.2 and only delegate the final response projection to the
transport. A forced move here would risk breaking the TUI token counter.

### Failure Modes

Provider HTTP failures must still be raised from the provider, not the
transport. The transport can raise `ValueError` or return validation failures
for malformed raw responses, but the provider/recovery layer owns HTTP status,
timeout, and stale connection semantics.

### Test Fixtures

Add fixtures for:

- plain assistant message
- assistant message with multiple tool calls
- tool call with dict arguments
- tool call with JSON-string arguments
- malformed `choices=[]`
- malformed `message=None`
- usage with prompt/completion/total tokens

### Review Checklist

- No `httpx` import in `openai_chat.py` unless it is only for typing raw
  response shapes, which should be avoided.
- The provider can be instantiated and used exactly as before.
- Existing `test_openai_compatible_dead_connections` remains meaningful because
  client lifecycle did not move.
