# PR 14.4 - Provider Runtime Integration

## Goal

Make transport selection explicit in providers and provider observability while
preserving `ModelProvider.generate()`.

## Scope

Modify:

- `aether/models/provider/base.py`
- `aether/models/provider/openai_compatible.py`
- `aether/models/provider/claude.py`
- `aether/models/provider/codex.py`
- `aether/agents/runtime/provider_invocation.py`

## Required Changes

- Add an optional `transport` or `transport_api_mode` attribute to provider
  implementations.
- Ensure `provider.api_mode` remains the stable public metadata value.
- Ensure `ProviderInvocationController` metadata can include:
  - `provider_name`
  - `api_mode`
  - `transport`
  - `model`
- Keep hook payload fields backward-compatible.
- Keep usage normalization keyed by the same provider/api mode values unless
  tests prove a better split is required.

## Do Not Do

- Do not add credential runtime.
- Do not add provider auto-selection.
- Do not change `/model`.
- Do not change gateway model methods.

## Tests

Add/update:

- `aether/tests/agents/runtime/test_provider_invocation_controller.py`
- `aether/tests/models/transport/test_provider_transport_integration.py`

Cover:

- Provider invocation metadata includes transport when available.
- Existing hook payload fields are still present.
- Usage accumulation still works.
- Provider validation errors still route through recovery.

## Acceptance

- Transport-aware metadata is observable without breaking old tests.
- Existing provider invocation hooks keep their old field names.
- `python -m pytest aether/tests/models aether/tests/agents/runtime/test_provider_invocation_controller.py` passes.

## Detailed Implementation Notes

### Provider Attributes

Add optional provider attributes without changing the abstract base requirement:

- `transport_name: str | None`
- `transport_api_mode: str | None`

For existing providers:

- OpenAI-compatible: `transport_name="openai_chat_completions"`,
  `transport_api_mode="chat"`
- Claude: `transport_name="anthropic_messages"`,
  `transport_api_mode="anthropic_messages"`
- Codex: `transport_name="codex_responses"`,
  `transport_api_mode="codex_responses"`

Keep `provider_name` and `api_mode` stable unless a test explicitly documents a
needed change.

### Hook Payload Compatibility

`ProviderInvocationController._build_api_hook_payload` currently exposes:

- `session_id`
- `iteration`
- `model`
- `provider`
- `api_mode`
- `api_call_count`
- `message_count`
- `tool_count`
- `approx_input_tokens`
- `request_char_count`
- `max_tokens`
- `context_metadata`

Do not rename these. Add optional fields:

- `transport`
- `transport_api_mode`

Consumers that ignore them should remain unaffected.

### Metadata Placement

If result metadata is extended, use a nested stable shape:

```python
context.metadata["provider_invocation"] = {
    "provider": "...",
    "api_mode": "...",
    "transport": "...",
    "model": "...",
}
```

Do not place raw payloads, headers, API keys, or raw response objects here.

### Runtime Contract

The engine still calls:

```python
provider.generate(messages, tools, config, context, ...)
```

The provider chooses its transport internally. Do not let
`ProviderInvocationController` construct transports. That would couple engine
runtime to provider internals.

### Tests

Use `ScriptedProvider` for controller tests and a fake provider with transport
attributes for metadata tests. Do not make live provider calls.

Add assertions that old hook payload keys remain present. This prevents a
silent breaking change in observability plugins.

### Review Checklist

- No provider raw payload in hook metadata.
- No new required abstract method in `ModelProvider`.
- Existing scripted provider tests still pass without adding transport fields.
