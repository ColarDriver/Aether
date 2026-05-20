# PR 19.3 - Provider, Auth, and Model Services

## Goal

Move provider discovery, model discovery, runtime provider selection, credential
readiness, and auxiliary-slot reporting into shared services.

## Current Problem

`providers_methods.py` currently owns multiple business paths:

- provider list display metadata
- static model catalog
- OpenAI-compatible live model discovery via `/models`, `/v1/models`, and
  `/api/models`
- fallback discovery metadata for `/model`
- provider runtime status via `resolve_main_provider_runtime`
- credential status via `default_credential_lookup`
- auxiliary model slots via `resolve_auxiliary_slot`

That behavior is useful outside stdio JSON-RPC, but it is currently embedded in
gateway handler code.

## Changes

Add `aether/services/providers/contracts.py`:

- `ProviderSummary`
- `ModelSummary`
- `ModelDiscoveryStatus`
- `ProviderModelList`
- `ProviderRuntimeStatus`
- `CredentialStatus`
- `CredentialSetStatus`
- `AuxiliarySlotStatus`
- `ProviderSelectionRequest`
- `ProviderSelectionResult`

Add `ProviderService`:

- `list_providers() -> list[ProviderSummary]`
- `resolve_provider_name(name: str) -> str`
- `get_provider_defaults(name: str) -> dict[str, object]`
- `list_models(provider: str, *, base_url: str | None = None, api_key: str | None = None) -> ProviderModelList`
- `runtime_current(...) -> ProviderRuntimeStatus`
- `auxiliary_slots(slots: list[str] | None = None) -> list[AuxiliarySlotStatus]`

ProviderService must reuse:

- `aether.cli.providers.resolve_provider_name`
- `aether.cli.providers.list_providers`
- `aether.cli.providers.get_provider_defaults`
- `aether.config.provider_runtime.resolve_main_provider_runtime`
- `aether.config.auxiliary_slots.resolve_auxiliary_slot`

Move the static model catalog and provider display metadata out of
`providers_methods.py` into service-owned constants or catalog helpers.

Add `AuthService`:

- `credentials_status(provider_family: str | None = None, ...) -> CredentialSetStatus`
- `runtime_credential_status(runtime: ProviderRuntimeConfig) -> CredentialStatus`

AuthService must reuse `aether.runtime.credentials.default_credential_lookup`
and only return redacted public metadata. It must never expose raw secret values.

Add `ModelDiscoveryService` or keep discovery internal to `ProviderService`:

- Preserve current OpenAI-compatible live discovery behavior.
- Preserve probe order: `/models`, `/v1/models`, `/api/models`.
- Preserve fallback to static catalog when discovery fails.
- Preserve discovery metadata fields currently surfaced to TUI.
- Keep network timeouts bounded.

Add `ModelSelectionService`:

- resolve final provider/model/base_url inputs
- read and write `last_model_by_provider` through `PrefsService`
- return readiness and missing credential metadata
- do not instantiate an `AgentEngine`

Gateway migration is deferred to PR 19.7. This PR adds services and tests only.

## Tests

Add:

- `aether/tests/services/test_provider_service.py`
- `aether/tests/services/test_auth_service.py`
- `aether/tests/services/test_model_selection_service.py`

Cover:

- provider list matches current gateway provider display fields
- alias resolution matches CLI provider helper behavior
- static catalog includes current Claude/OpenAI/Codex model ids
- OpenAI-compatible live discovery parses common payload shapes
- discovery fallback preserves current `discovery` metadata semantics
- base URL probe candidates match current gateway behavior
- runtime current returns provider/model/base URL/API key env names without
  leaking secret values
- credentials status reports configured/missing keys with redacted values
- auxiliary slots return stable public metadata and reject unknown slots
- model selection persists last model preference through `PrefsService`

## Migration Notes

- Keep gateway response shapes unchanged until PR 19.7.
- Do not duplicate provider runtime or credential resolution. Wrap the existing
  modules.
- Avoid adding new provider constructors. Provider construction for actual runs
  belongs to AgentRunService in PR 19.6.
- Network discovery tests should use `httpx.MockTransport` or equivalent fake
  transports; no real network calls in tests.

## Risks

- `/model` relies on subtle discovery diagnostics. Dropping a field can regress
  TUI troubleshooting.
- Provider names use both `openai` and `openai-compatible` concepts in nearby
  code. Contracts should distinguish provider display name, runtime family, and
  transport family explicitly.
- Auth readiness must be public-safe. Never include raw API keys, OAuth tokens,
  or unredacted headers in service results.

## Non-Goals

- Do not introduce a new credential pool.
- Do not implement OAuth.
- Do not implement login/logout.
- Do not change provider constructors.
- Do not change gateway provider RPC schemas.
- Do not implement a Web settings UI.
- Do not migrate gateway handlers in this PR.

## Acceptance

- Provider/model/auth/auxiliary-slot business behavior is available through
  services.
- Service tests prove parity with existing gateway behavior.
- Secret values are never exposed in service read models.
- TUI `/model`, future CLI provider commands, and future Web settings can share
  the same service path after adapter migration.
