# PR 17.1 - Provider Runtime Config

## Goal

Centralize provider/model runtime config resolution without changing provider
behavior.

## New Module

Add:

- `aether/config/provider_runtime.py`

Recommended types:

- `ProviderRuntimeConfig`
- `ProviderRuntimeChoice`
- `resolve_main_provider_runtime()`
- `resolve_provider_runtime_from_env()`

## Environment Rules

Keep current user-facing names stable:

- `AETHER_PROVIDER`
- provider-specific model/base URL/API key variables already present in `.env`
- `WEB_SEARCH_PROVIDER`
- `WEB_SEARCH_API_KEY`

Do not reintroduce `AETHER_SUBAGENT_PROVIDER` as the global control. Subagent is
an auxiliary slot and should be handled in PR17.4.

## Provider Choices

Supported global provider choices:

- `codex`
- `claude`
- `openai-compatible`

Map defaults:

- `codex` -> provider `codex`, default model `gpt-5.4`
- `claude` -> provider `claude`, default model `sonnet`
- `openai-compatible` -> provider `openai`, default model `gpt-5.4`

Use exact Aether model IDs already established in config/tests.

## Tests

Add:

- `aether/tests/config/test_provider_runtime.py`

Cover:

- Env parsing.
- Unknown provider error.
- Default model mapping.
- Existing `.env` style remains valid.

## Acceptance

- Provider construction can use `ProviderRuntimeConfig`.
- No credential pool yet.
- No behavior change for active provider.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/config/provider_runtime.py`
- `aether/tests/config/test_provider_runtime.py`

### Data Model

Recommended dataclasses:

```python
@dataclass(frozen=True, slots=True)
class ProviderRuntimeChoice:
    family: Literal["codex", "claude", "openai-compatible"]
    provider_name: str
    default_model: str

@dataclass(frozen=True, slots=True)
class ProviderRuntimeConfig:
    family: str
    provider_name: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_value: str | None = None
```

The first PR may omit `api_key_value` if credential sources are strictly PR17.2.
If included, it must not appear in `repr`.

### Alias Handling

Normalize these values:

- `openai-compatible`
- `openai_compatible`
- `openai`

to family `openai-compatible`.

Normalize common Claude typo aliases only if current code already accepts them.
Do not silently accept `sonnect` as a provider family; that is a model alias
problem and should be handled separately if needed.

### Config Resolution Order

For main provider:

1. explicit `EngineConfig` provider value if one exists
2. `AETHER_PROVIDER`
3. current default provider behavior

For model:

1. explicit model config
2. provider-specific env model if it already exists
3. family default from this module

Document any existing env variable names found during implementation. Do not
rename them without compatibility.

### Provider Construction Integration

Do not move provider construction in PR17.1 unless the call site is already
simple. The minimum useful change is that call sites can ask:

```python
runtime = resolve_main_provider_runtime(config, environ=os.environ)
```

and receive a stable object.

### Tests

Cover:

- `AETHER_PROVIDER=codex`
- `AETHER_PROVIDER=claude`
- `AETHER_PROVIDER=openai-compatible`
- missing `AETHER_PROVIDER`
- invalid provider value includes clear error text
- explicit config beats env
- env/default model mapping
- `repr(runtime)` does not include secrets if value field exists

### Review Checklist

- No tool imports `provider_runtime.py` yet except tests unless needed.
- No `.env` file mutation.
- No secret redaction implementation yet; that is PR17.2.
