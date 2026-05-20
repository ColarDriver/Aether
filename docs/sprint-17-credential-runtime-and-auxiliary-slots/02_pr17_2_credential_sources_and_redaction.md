# PR 17.2 - Credential Sources and Redaction

## Goal

Add a small credential lookup abstraction and secret redaction helper.

## New Modules

Add:

- `aether/runtime/credentials/__init__.py`
- `aether/runtime/credentials/sources.py`
- `aether/runtime/credentials/redaction.py`

## Contract

Recommended types:

- `CredentialSource`
- `EnvCredentialSource`
- `CredentialLookup`
- `CredentialValue`

Credential lookup should return metadata about where a credential came from
without exposing the raw secret in public metadata.

## Redaction

Add:

- `redact_secret(value: str) -> str`
- `redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]`
- `contains_secret_like_text(text: str) -> bool`

Use this for logs and metadata snapshots where provider config may appear.

## Non-Goals

- No credential files.
- No OAuth.
- No key rotation.

## Tests

Add:

- `aether/tests/runtime/credentials/test_sources.py`
- `aether/tests/runtime/credentials/test_redaction.py`

Cover:

- Env lookup.
- Missing credential.
- Redacted display.
- Metadata never includes raw API key.

## Acceptance

- Existing `.env` API keys still work.
- Provider construction can consume lookup results.
- Logs and public metadata do not expose raw keys.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/runtime/credentials/__init__.py`
- `aether/runtime/credentials/sources.py`
- `aether/runtime/credentials/redaction.py`
- `aether/tests/runtime/credentials/test_sources.py`
- `aether/tests/runtime/credentials/test_redaction.py`

### Credential Source Contract

Recommended contract:

```python
@dataclass(frozen=True, slots=True)
class CredentialValue:
    value: str
    source: str
    key_name: str

    def redacted(self) -> str: ...

class CredentialSource(Protocol):
    name: str
    def get(self, key_name: str) -> CredentialValue | None: ...
```

`CredentialValue.__repr__` must not show the raw value.

### Env Source

`EnvCredentialSource` should accept an explicit env mapping for tests:

```python
EnvCredentialSource(environ=os.environ)
```

Do not read `.env` directly here. Existing env loading should continue to happen
at process startup.

### Provider Key Mapping

Centralize mapping from provider family to expected env names, for example:

- Codex: existing Codex key/env/auth path
- Claude: `ANTHROPIC_API_KEY` or current accepted names
- OpenAI-compatible: existing openai-compatible key env names
- Web search: `WEB_SEARCH_API_KEY`

During implementation, read current config code first and preserve existing
names exactly.

### Redaction Rules

Redaction should:

- keep empty values empty
- hide full key body
- optionally show last 4 characters only
- redact common bearer tokens in strings
- redact dict values for keys like `api_key`, `authorization`, `token`,
  `refresh_token`, `access_token`

Do not over-redact ordinary model names or base URLs.

### Integration Points

Use redaction in:

- provider runtime status metadata
- gateway status methods if added later
- failed request dump metadata if provider config is included
- debug logs where credential config is printed

Do not thread redaction into every logger call in this PR. Start with the
provider runtime surfaces.

### Tests

Cover:

- env source returns value and source metadata
- missing key returns `None`
- `repr(CredentialValue)` is redacted
- mapping redacts nested auth headers
- strings with `Bearer <token>` are redacted
- public metadata helper never includes raw key

### Review Checklist

- No raw secret in snapshots.
- No credential files yet.
- Existing provider setup still accepts env values.
