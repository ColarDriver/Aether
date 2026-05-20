# PR 14.3 - Anthropic and Codex Transports

## Goal

Extract Claude Messages and Codex Responses conversion into transports after the
OpenAI transport proves the boundary.

## Target Files

Add:

- `aether/models/transport/anthropic_messages.py`
- `aether/models/transport/codex_responses.py`

Modify:

- `aether/models/provider/claude.py`
- `aether/models/provider/codex.py`
- `aether/models/provider/hosted_web_search.py` only if helper placement needs
  cleanup.

## Boundary

Providers still own:

- API key and base URL.
- HTTP request execution.
- Streaming event reading.
- Interrupt checks.
- HTTP error conversion into structured provider errors.

Transports own:

- Anthropic system/user/tool message conversion.
- Anthropic tool schema conversion.
- Anthropic hosted `web_search_20250305` tool conversion.
- Anthropic response block normalization.
- Codex Responses input item conversion.
- Codex hosted `web_search` tool conversion.
- Codex output item normalization.
- Hosted web search source extraction.
- Provider-specific finish reason mapping.

## Hosted Search Rule

Keep one canonical helper for hosted search metadata:

- `hosted_web_search.is_web_search_tool`
- `hosted_web_search.append_sources_section`
- `hosted_web_search.dedupe_sources`

Do not mix hosted search with the local `WebSearchTool` backend selection.

## Tests

Add:

- `aether/tests/models/transport/test_anthropic_messages_transport.py`
- `aether/tests/models/transport/test_codex_responses_transport.py`

Cover:

- Tool schema conversion.
- Hosted web search tool conversion.
- Text response normalization.
- Tool call normalization.
- Reasoning/signature metadata preservation.
- Usage metadata preservation.
- Citation/source section preservation.
- Invalid raw response validation.

## Acceptance

- Claude/Codex provider behavior remains unchanged.
- Existing hosted web search tests still pass.
- `aether/models/provider/claude.py` and `aether/models/provider/codex.py` are
  materially smaller and mostly IO-oriented.

## Detailed Implementation Notes

### Anthropic Transport Scope

Move these behaviors into `AnthropicMessagesTransport`:

- system/user/assistant/tool message conversion
- assistant tool-use block conversion
- `ToolDescriptor` to Anthropic tool schema conversion
- hosted `web_search` descriptor to Anthropic `web_search_20250305`
- text/content block normalization
- server tool use and web search result metadata extraction
- citation/source extraction
- finish reason mapping
- raw response shape validation

Keep these in `ClaudeChatModel`:

- `anthropic.Anthropic` client construction
- OAuth/key resolution
- beta header selection
- prompt caching policy application if it requires provider runtime state
- `_create` and `_create_streaming`
- retry/backoff around API calls
- interrupt listener cleanup

### Codex Transport Scope

Move these behaviors into `CodexResponsesTransport`:

- canonical messages to Responses input conversion
- assistant tool call conversion
- `ToolDescriptor` to Responses tool schema conversion
- hosted `web_search` descriptor to Codex `web_search`
- output item normalization
- response item IDs and call IDs required for future provider turns
- hosted web search call/source metadata extraction
- finish reason mapping
- raw response shape validation

Keep these in `CodexChatModel`:

- credential loading from Codex auth
- HTTP request execution
- response streaming loop if callback timing is provider-owned
- request retry/timeout
- auth error classification

### Hosted Web Search Invariants

Hosted search is provider-native. Local `WebSearchTool` is Aether-native. They
share the tool name `web_search`, so tests must prove the conversion layer
chooses correctly:

- When provider is Claude/Codex, `ToolDescriptor(name="web_search")` is
  converted into native hosted search schema.
- When provider is OpenAI-compatible, `ToolDescriptor(name="web_search")` stays
  an ordinary function tool for the local backend.
- Sources from hosted provider responses are appended or surfaced exactly as
  today.

### Migration Step Detail

For each provider:

1. Add transport unit tests from raw fixture objects.
2. Instantiate transport in provider `__init__`.
3. Replace payload construction call.
4. Replace final response parsing call.
5. Replace validation call or delegate `validate_response`.
6. Keep compatibility wrappers for one PR if external tests import private
   methods.
7. Remove wrappers after test update.

Do Claude and Codex in the same PR only because both hosted-search conversions
should share helper cleanup. If either becomes too large, split Codex into
PR14.3b rather than mixing partial migrations.

### Test Detail

Anthropic fixtures should include:

- text-only response block
- tool_use block
- server_tool_use web_search block
- web_search_tool_result block
- citations in text blocks
- empty content list invalid response

Codex fixtures should include:

- output_text response
- function/tool call output item
- web_search_call output item
- annotations with URLs
- missing output invalid response

### Review Checklist

- Provider files still own IO.
- Transport files are pure and network-free.
- Hosted web search metadata remains backward compatible.
