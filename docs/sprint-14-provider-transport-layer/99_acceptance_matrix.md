# Sprint 14 - Acceptance Matrix

| Scenario | 14.1 Contract | 14.2 OpenAI | 14.3 Claude/Codex | 14.4 Runtime | 14.5 Acceptance |
|---|---|---|---|---|---|
| OpenAI text | registry available | payload + normalize | - | metadata stable | regression pass |
| OpenAI tool call | registry available | tool schema + tool call | - | hooks stable | regression pass |
| OpenAI streaming | no behavior change | stream final normalization | - | stream metadata stable | regression pass |
| Claude text | registry available | - | normalize blocks | metadata stable | regression pass |
| Claude hosted search | hosted helper stable | - | native search schema | sources stable | regression pass |
| Codex text | registry available | - | output item normalize | metadata stable | regression pass |
| Codex hosted search | hosted helper stable | - | native search schema | sources stable | regression pass |
| Invalid response | validation contract | validation parity | validation parity | recovery routes | regression pass |
| Usage | no behavior change | usage preserved | usage preserved | accumulation stable | regression pass |

## Unit Test Map

| File | Purpose |
|---|---|
| `aether/tests/models/transport/test_registry.py` | transport registration and lookup |
| `aether/tests/models/transport/test_openai_chat_transport.py` | OpenAI-compatible payload and response projection |
| `aether/tests/models/transport/test_anthropic_messages_transport.py` | Claude message/tool/hosted-search conversion |
| `aether/tests/models/transport/test_codex_responses_transport.py` | Codex Responses conversion and hosted-search metadata |
| `aether/tests/models/transport/test_provider_transport_integration.py` | provider metadata and transport wiring |
| existing `aether/tests/models/**` | provider regression |
| `aether/tests/agents/runtime/test_provider_invocation_controller.py` | hook/usage/recovery boundary regression |

## Manual Checklist

- Run one OpenAI-compatible text turn.
- Run one OpenAI-compatible tool turn.
- Run one Claude hosted web-search turn if credentials are available.
- Run one Codex hosted web-search turn if credentials are available.
- Confirm local `web_search` still works on openai-compatible provider.
- Confirm provider hook logs do not include raw payloads or keys.

## Non-Regression Rules

- Do not change `ModelProvider.generate()` signature.
- Do not move HTTP client lifecycle into transports.
- Do not move credential lookup into transports.
- Do not change gateway/TUI schema.
- Do not merge local web-search backend selection with hosted provider search.

## Verification Result

Completed on `provider-transport-layer`:

- `python -m pytest aether/tests/models aether/tests/agents/runtime/test_provider_invocation_controller.py aether/tests/engine/test_streaming_generate.py aether/tests/tools/test_web_search_tool.py -q`
  - Result: `107 passed`.
- `python -m pytest aether/tests -q`
  - Result: `1642 passed`.
- `uv run pyright aether/models/provider aether/models/transport aether/agents/runtime/provider_invocation.py`
  - Result: `0 errors, 0 warnings, 0 informations`.
- Static import review:
  - `aether/models/transport/` has no `httpx`, `anthropic`, `AgentEngine`, gateway, or TUI imports.

Implementation acceptance:

- OpenAI-compatible, Claude Messages, and Codex Responses providers are backed by pure transport classes.
- Provider classes still own credentials, HTTP clients, retries, streaming IO, and provider error wrapping.
- Runtime hooks keep existing field names and add optional `transport` / `transport_api_mode`.
- `context.metadata["provider_invocation"]` exposes provider, API mode, transport, transport API mode, and model without raw payloads or credentials.
