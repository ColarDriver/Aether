# PR 14.1 - Transport Contract and Registry

## Goal

Introduce the transport boundary without changing any provider behavior.

## New Package

Create `aether/models/transport/`.

Recommended files:

- `aether/models/transport/__init__.py`
- `aether/models/transport/base.py`
- `aether/models/transport/types.py`
- `aether/models/transport/registry.py`

## Contract

Define a small `ProviderTransport` protocol or abstract base class:

```python
class ProviderTransport(Protocol):
    api_mode: str

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...
    def convert_tools(self, tools: list[ToolDescriptor], **kwargs: Any) -> Any: ...
    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse: ...
    def validate_raw_response(self, response: Any) -> tuple[bool, list[str]]: ...
```

Use `build_payload`, not Hermes' `build_kwargs`, because Aether providers use
direct HTTP payloads today. If a future SDK provider needs kwargs, it can expose
that through the same shape or a subclass.

## Registry

Add:

- `register_transport(api_mode: str, factory: type[ProviderTransport]) -> None`
- `get_transport(api_mode: str) -> ProviderTransport | None`
- `available_transports() -> tuple[str, ...]`

Discovery should be explicit at first. Avoid import-time magic that makes test
order unstable.

## Compatibility Rules

- Do not edit `openai_compatible.py`, `claude.py`, or `codex.py` behavior yet.
- Do not change provider classes to depend on the new registry in this PR.
- Do not change `NormalizedResponse`.
- Do not change `ToolDescriptor`.

## Tests

Add `aether/tests/models/transport/test_registry.py`.

Cover:

- Register/get/available behavior.
- Duplicate registration is deterministic.
- Unknown api mode returns `None`.
- Contract stub can build a payload and normalize a response.

## Acceptance

- The new transport package imports without provider side effects.
- Existing provider tests pass unchanged.
- No engine code imports transport directly yet.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/models/transport/__init__.py`
- `aether/models/transport/base.py`
- `aether/models/transport/types.py`
- `aether/models/transport/registry.py`
- `aether/tests/models/transport/test_registry.py`

`__init__.py` should only export the stable public names. Do not import concrete
transports here in PR14.1, because that creates unnecessary import-order and
optional-dependency risk.

### Transport Type Details

Prefer `Protocol` plus dataclasses over a large inheritance tree. Concrete
transports should be easy to instantiate in tests without provider clients.

Recommended dataclasses:

```python
@dataclass(slots=True)
class TransportPayload:
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class TransportValidation:
    ok: bool
    reasons: list[str] = field(default_factory=list)
```

Keep this small. Do not introduce a full request object unless concrete provider
work proves it is needed.

### Registry Semantics

The registry should be deterministic:

- Registering the same api mode with the same class is allowed and idempotent.
- Registering the same api mode with a different class should raise
  `ValueError` unless tests prove replacement is necessary.
- `get_transport("missing")` returns `None`.
- `available_transports()` returns sorted names for stable tests.

### Dependency Direction

Allowed:

- `transport/*` imports `aether.runtime.core.contracts`
- `transport/*` imports `aether.tools.base`
- provider modules import `transport/*` in later PRs

Forbidden:

- `transport/*` imports `AgentEngine`
- `transport/*` imports gateway/TUI modules
- `transport/*` constructs HTTP clients
- `transport/*` reads `.env`

### Test Detail

The registry test should assert no concrete provider dependency is imported as a
side effect. A simple way is to register a local fake transport class inside the
test file and never import `openai_compatible`, `claude`, or `codex`.

### Review Checklist

- No network libraries are required by the new package.
- No provider package import errors can occur just by importing
  `aether.models.transport`.
- Type names are generic enough for chat-completions, Anthropic Messages,
  Codex Responses, and future Bedrock-style providers.
