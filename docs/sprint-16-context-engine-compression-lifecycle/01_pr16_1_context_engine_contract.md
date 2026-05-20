# PR 16.1 - Context Engine Contract

## Goal

Introduce a pluggable context engine contract and adapt current compaction
behavior to it without changing runtime behavior.

## New Package

Add:

- `aether/runtime/context/__init__.py`
- `aether/runtime/context/engine.py`
- `aether/runtime/context/default_engine.py`

## Contract

Recommended interface:

```python
class ContextEngine(Protocol):
    name: str

    def should_compress_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> bool: ...

    def compact_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> ContextEngineResult: ...

    def apply_provider_projection(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]: ...
```

## Boundary

The context engine owns:

- Whether preflight compaction is needed.
- Calling the current compaction pipeline.
- Collapse/provider projection.
- Context-engine metadata namespace.

It does not own:

- Memory retrieval.
- Skill nudge injection.
- Plan mode reminder injection.
- Tool result storage.

## Tests

Add:

- `aether/tests/runtime/context/test_context_engine_contract.py`
- `aether/tests/agents/runtime/test_context_assembly_pipeline.py` updates

Cover:

- Default engine preserves current compaction behavior.
- Provider projection does not mutate canonical messages.
- Context metadata does not leak live objects.

## Acceptance

- Existing compaction tests pass.
- No behavior change for ordinary turns.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/runtime/context/__init__.py`
- `aether/runtime/context/engine.py`
- `aether/runtime/context/default_engine.py`
- `aether/tests/runtime/context/test_context_engine_contract.py`

### Result Type

Use a result object that can carry both messages and observability:

```python
@dataclass(slots=True)
class ContextEngineResult:
    messages: list[dict[str, Any]]
    changed: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
```

Do not return raw compaction pipeline internals to `ContextAssemblyPipeline`.

### Default Engine Adapter

`DefaultContextEngine` should be a thin adapter over current behavior:

- preflight compaction delegates to existing compaction pipeline
- collapse projection delegates to existing collapse view logic
- metadata mirrors current `compaction` metadata

In PR16.1, it is acceptable for the adapter to call legacy engine methods
through a narrow protocol. The goal is the contract, not full decoupling.

### Dependency Direction

Allowed:

- `context/default_engine.py` imports compaction service modules
- `ContextAssemblyPipeline` imports `ContextEngine`

Forbidden:

- context engine imports TUI/gateway
- context engine constructs providers
- context engine reads `.env`
- context engine directly mutates session store outside documented lifecycle

### Registration

Do not add plugin discovery in this PR. Use explicit injection through
`EngineServices` or `AgentEngine` construction. If no engine is provided, use
`DefaultContextEngine`.

### Tests

Direct tests should assert:

- default engine returns unchanged messages below threshold
- default engine returns changed messages when a fake compaction adapter changes
  messages
- provider projection returns a new list
- canonical input list is not mutated
- metadata is JSON-safe

### Review Checklist

- The contract is small enough for future alternative engines.
- Existing compaction behavior remains available.
- No manual `/compress` control surface appears yet.
