# Sprint 14 - Provider Transport Layer

## Decision

This must be its own sprint. It should not be bundled with turn runner cleanup,
credential runtime, context compression, or parallel tool scheduling.

Provider transport changes touch the provider contract surface, streaming payload
shape, hosted web search conversion, usage parsing, and recovery classification.
Those are high-blast-radius paths. Keeping this sprint focused lets us preserve
`ModelProvider.generate()` externally while moving provider-specific data
conversion behind smaller contracts.

## Motivation

Aether currently has provider-specific conversion and IO mixed together in:

- `aether/models/provider/openai_compatible.py`
- `aether/models/provider/claude.py`
- `aether/models/provider/codex.py`

Each provider owns all of these concerns:

- Convert canonical messages to provider request messages.
- Convert Aether `ToolDescriptor` objects to provider tool schemas.
- Build HTTP request payloads.
- Stream provider-specific chunks.
- Normalize provider responses into `NormalizedResponse`.
- Validate response shape.
- Extract usage, reasoning, hosted web search metadata, and finish reasons.

Hermes separates the pure data path with a `ProviderTransport` style boundary:

```text
convert_messages -> convert_tools -> build_kwargs -> normalize_response
```

Aether should adopt the same boundary, but preserve Aether's existing
`ModelProvider.generate()` contract during this sprint.

## Goals

- Add a transport package under `aether/models/transport/`.
- Move provider payload and response conversion out of provider IO classes.
- Preserve existing provider behavior, tool names, hosted web search behavior,
  stream callbacks, usage metadata, and recovery errors.
- Make adding a future provider mostly a transport + thin client task.
- Keep `AgentEngine` and `ProviderInvocationController` mostly unchanged.

## Non-Goals

- Do not add credential pools or OAuth.
- Do not rewrite provider selection or `/model`.
- Do not change gateway/TUI schemas.
- Do not change `ModelProvider.generate()` public signature.
- Do not add a Codex app-server runtime.
- Do not change web search product semantics beyond moving hosted conversion.
- Do not fix unrelated pyright baseline errors.

## PR Roadmap

| PR | File | Boundary |
|---|---|---|
| 14.1 | `01_pr14_1_transport_contract_and_registry.md` | Transport types, registry, tests, no provider behavior change |
| 14.2 | `02_pr14_2_openai_chat_completions_transport.md` | Extract OpenAI-compatible chat-completions conversion |
| 14.3 | `03_pr14_3_anthropic_and_codex_transports.md` | Extract Claude Messages and Codex Responses conversion |
| 14.4 | `04_pr14_4_provider_runtime_integration.md` | Wire providers through transports while preserving generate contract |
| 14.5 | `05_pr14_5_tests_observability_and_acceptance.md` | Cross-provider regression, metadata, recovery, docs |

## Dependency

```text
14.1 contract
  -> 14.2 openai-compatible transport
  -> 14.3 anthropic/codex transports
  -> 14.4 runtime integration cleanup
  -> 14.5 acceptance
```

## Completion Criteria

- `openai_compatible.py`, `claude.py`, and `codex.py` no longer contain large
  provider payload/normalization blocks that belong to transports.
- `ModelProvider.generate()` still works the same for engine callers.
- Existing provider tests pass.
- Hosted web search through Claude/Codex still maps to provider-native tools.
- Local `web_search` tool remains independent from hosted provider search.

## Detailed Split Rationale

Do not treat this sprint as a simple file move. The provider files currently
mix at least four different layers:

- **Provider IO**: HTTP client lifecycle, request retry, auth headers, timeout,
  stream iteration, interrupt listener cleanup.
- **Transport conversion**: canonical Aether messages/tools to provider-native
  payloads.
- **Response projection**: provider-native response/events to
  `NormalizedResponse`, `ToolCall`, usage metadata, reasoning metadata, and
  hosted web-search metadata.
- **Engine recovery boundary**: provider-specific HTTP/status/body errors to
  `ProviderInvocationError` subclasses.

Sprint 14 should extract only the conversion/projection layer. Provider IO and
recovery error construction stay in provider classes until a later provider
runtime sprint. This matches the useful part of Hermes' transport design without
copying Hermes' full credential/runtime stack.

## Current Aether Anchors

Use these as the migration map:

- `OpenAICompatibleModel._build_payload`
- `OpenAICompatibleModel._convert_messages`
- `OpenAICompatibleModel._convert_tools`
- `OpenAICompatibleModel._parse_response`
- `OpenAICompatibleModel._parse_tool_call`
- `OpenAICompatibleModel.validate_response`
- `ClaudeChatModel._build_request_payload`
- `ClaudeChatModel._convert_messages`
- `ClaudeChatModel._convert_tools`
- `ClaudeChatModel._parse_response`
- `CodexChatModel._build_payload`
- `CodexChatModel._convert_messages`
- `CodexChatModel._convert_tools`
- `CodexChatModel._stream_response` event normalization helpers
- `aether/models/provider/hosted_web_search.py`

Do not start by rewriting these methods. First wrap the existing behavior in
transport tests, then move one provider at a time.

## Hermes Reference Points

Read these files before implementation:

- `/workspace/hermes-agent/agent/transports/base.py`
- `/workspace/hermes-agent/agent/transports/chat_completions.py`
- `/workspace/hermes-agent/agent/transports/anthropic.py`
- `/workspace/hermes-agent/agent/transports/codex.py`
- `/workspace/hermes-agent/agent/transports/types.py`

The target is the boundary shape, not a line-for-line port. Aether has its own
`NormalizedResponse`, `ToolCall`, `ToolDescriptor`, interrupt model, and hosted
search helpers.

## Hard Compatibility Rules

- `ModelProvider.generate()` must remain the only method called by
  `ProviderInvocationController`.
- `NormalizedResponse.metadata["usage"]` shape must remain compatible with
  `aether/runtime/observability/usage.py`.
- `stream_callback` and `stream_silent_callback` semantics must not change.
- Hosted web search metadata must remain under `metadata["hosted_web_search"]`.
- Provider validation failures must still become `ResponseInvalidError` through
  `ProviderInvocationController`.
- No raw provider response object should leak into public metadata.

## Implementation Order Guardrail

Each provider migration should follow this order:

1. Add pure transport tests using fixture dictionaries or lightweight fake raw
   objects.
2. Move conversion helpers behind the transport while preserving old private
   method wrappers when tests still import them.
3. Re-run existing provider tests.
4. Only then delete stale wrappers.

Skipping step 1 makes it too easy to accidentally change provider payloads in a
way that only appears during live model calls.
