# PR 16.2 - Compression Lifecycle Service

## Goal

Move compression lifecycle orchestration into a service that can be reused by
auto-compaction and manual compression.

## New Module

Add:

- `aether/runtime/context/compression_lifecycle.py`

Recommended types:

- `CompressionLifecycleService`
- `CompressionRequest`
- `CompressionResult`
- `CompressionStatus`

## Scope

The service owns:

- Compression preflight.
- Calling the selected `ContextEngine`.
- Recording compression start/end metadata.
- Rebuilding or preserving system prompt according to existing semantics.
- Surfacing failure reasons without corrupting the transcript.

## Not In Scope

- Credential selection for auxiliary models. That belongs to Sprint 17.
- New summarization algorithms.
- Plugin context engines.

## Tests

Add:

- `aether/tests/runtime/context/test_compression_lifecycle.py`

Cover:

- No-op when below threshold.
- Successful compaction records metadata.
- Failed compaction leaves messages usable.
- System prompt behavior is preserved.

## Acceptance

- Context assembly calls lifecycle service instead of directly invoking
  compaction helpers.
- Existing compaction and collapse tests pass.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/runtime/context/compression_lifecycle.py`
- `aether/tests/runtime/context/test_compression_lifecycle.py`

### Service Contract

Recommended shape:

```python
@dataclass(slots=True)
class CompressionRequest:
    messages: list[dict[str, Any]]
    context: TurnContext
    trigger_reason: str
    force: bool = False
    focus: str | None = None

@dataclass(slots=True)
class CompressionResult:
    messages: list[dict[str, Any]]
    status: Literal["skipped", "compressed", "failed"]
    metadata: dict[str, Any]
    error: str | None = None
```

The lifecycle service should be called from context assembly preflight and later
from manual compression if PR16.4 implements it.

### Lifecycle Steps

The service should execute in this order:

1. Snapshot input message count and token estimate if available.
2. Ask context engine whether compression should run unless `force=True`.
3. Emit/record `compression_started` metadata.
4. Call context engine compression.
5. Validate returned messages are a list of dicts with roles.
6. Record success/failure metadata.
7. Return messages without mutating input unless current behavior mutates.

### Prompt Rebuild Boundary

If existing compaction rebuilds or persists system prompt, keep that behavior
behind the lifecycle adapter. Do not silently rebuild system prompts on every
compression unless existing tests expect it.

### Metadata Shape

Use nested metadata:

```python
context.metadata["context_engine"] = {
    "name": "...",
    "compression": {
        "status": "...",
        "trigger_reason": "...",
        "source_message_count": 10,
        "result_message_count": 4,
    },
}
```

Keep old top-level `compaction` metadata for compatibility.

### Tests

Use fake context engines:

- `NoopContextEngine`
- `CompressingContextEngine`
- `FailingContextEngine`

Cover:

- skipped when not needed
- forced compression
- successful compression
- invalid message output becomes failure
- exception becomes failure without losing input messages
- metadata is stable and JSON-safe

### Review Checklist

- No provider calls in lifecycle service unless delegated through existing
  compaction pipeline.
- Failure path leaves messages usable.
- Auto-compaction and future manual compression share the same request/result.
