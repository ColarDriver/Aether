# PR 17.3 - Credential Pool and Rate-Limit Rotation

## Goal

Add an optional local credential pool and integrate it with rate-limit recovery.

## New Modules

Add:

- `aether/runtime/credentials/pool.py`

## Pool Shape

Keep the first version simple:

```json
{
  "providers": {
    "openai": [
      {"name": "primary", "api_key_env": "OPENAI_API_KEY"},
      {"name": "backup", "api_key_env": "OPENAI_API_KEY_2"}
    ]
  }
}
```

Support strategies:

- `fill_first`
- `round_robin`

Defer `least_used` and weighted routing unless needed.

## Recovery Integration

When provider error classification indicates rate limit or quota exhaustion:

- mark current credential as temporarily unhealthy
- try next credential if available
- record metadata without raw keys

## Non-Goals

- No persistent encrypted vault.
- No OAuth refresh.
- No billing dashboard.

## Tests

Add:

- `aether/tests/runtime/credentials/test_pool.py`
- recovery integration tests for rate-limit rotation

Cover:

- Pool load.
- Round-robin selection.
- Mark unhealthy.
- No raw secret in metadata.
- Fallback to env credential when no pool exists.

## Acceptance

- Existing single-key behavior remains default.
- Rate-limit recovery can rotate credentials if configured.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/runtime/credentials/pool.py`
- `aether/tests/runtime/credentials/test_pool.py`

### Pool Data Model

Recommended dataclasses:

```python
@dataclass(slots=True)
class PooledCredential:
    provider: str
    name: str
    credential: CredentialValue
    healthy: bool = True
    last_error: str | None = None

@dataclass(slots=True)
class CredentialPoolSelection:
    credential: PooledCredential
    strategy: str
```

Do not store raw secrets in JSON state. Pool config can point to env var names.

### Config Source

First version can read from `EngineConfig` or a small JSON/TOML file under
`AETHER_HOME`. If a file is used, document path and schema. Keep `.env`
fallback as default.

Do not require users to create a pool file for normal operation.

### Rotation Semantics

On rate limit/quota-like errors:

1. classify error through existing recovery classifier
2. mark selected credential unhealthy for the current process/session
3. choose next healthy credential for the same provider
4. retry through existing recovery path
5. record metadata with provider, credential name, and reason

Do not retry indefinitely. Respect existing retry budgets.

### Recovery Integration Boundary

Integrate with:

- `aether/runtime/recovery/rate_guard.py`
- `aether/agents/runtime/recovery_controller.py`
- provider construction or provider credential update path

If providers cannot swap credentials in-place safely, build a new provider
instance through the existing provider factory. Do not mutate random provider
attributes from recovery code without a typed helper.

### Metadata

Allowed:

- credential pool enabled
- credential name
- provider
- rotation count
- failure class

Forbidden:

- raw key
- env var value
- authorization header

### Tests

Cover:

- single env credential path unaffected
- pool load with two credentials
- `fill_first` always chooses first healthy
- `round_robin` cycles healthy credentials
- unhealthy credential is skipped
- all unhealthy returns clear error
- rate-limit recovery requests rotation once
- metadata redacted

### Review Checklist

- Rotation respects existing recovery max attempts.
- Provider state is not mutated unsafely.
- Pool is optional and backward compatible.
