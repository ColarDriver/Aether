# PR 16.3 - Memory, Diagnostics, and Session Split

## Goal

Make compression boundaries safe for memory, diagnostics, and session metadata.

## Scope

Add lifecycle hooks around compression:

- `before_context_compression`
- `after_context_compression`
- `context_compression_failed`

Integrate:

- memory runtime pre-compression capture
- diagnostics attachment preservation or invalidation
- task/session metadata continuity
- trajectory metadata showing compression lineage

## Session Split

Do not rotate session IDs in this PR unless Aether already has a compatible
session persistence primitive. Instead, record compression lineage metadata:

- `compression_count`
- `compressed_at`
- `trigger_reason`
- `source_message_count`
- `result_message_count`

Session ID rotation can be a later sprint if needed.

## Tests

Add/update:

- `aether/tests/memory/test_memory_injection.py`
- `aether/tests/runtime/test_compaction_pipeline.py`
- `aether/tests/engine/test_compaction_integration.py`

Cover:

- Memory is notified before old context is removed.
- Diagnostics do not reattach stale edits incorrectly.
- Compression metadata is public and JSON-safe.

## Acceptance

- Compression no longer silently drops memory/diagnostic context.
- Existing memory and diagnostics tests pass.

## Detailed Implementation Notes

### Hooks to Add

Add hook names only if the existing `EngineHooks` model can carry them without
breaking consumers:

- `before_context_compression`
- `after_context_compression`
- `context_compression_failed`

Payload should include:

- `session_id`
- `trigger_reason`
- `source_message_count`
- `result_message_count` when available
- `context_metadata`

Do not include raw full messages in hook payload by default. They may contain
secrets or large tool outputs.

### Memory Integration

Before compression removes or summarizes older context:

- call a memory pre-compression hook/provider method if available
- give memory code a bounded textual summary or metadata view, not raw live
  objects
- record whether memory contributed compression hints

If no memory provider is enabled, the hook should be a no-op.

### Diagnostics Integration

Compression can make diagnostic attachments stale. Required behavior:

- Preserve diagnostic tracker state in runtime metadata.
- Do not reattach diagnostics for edits that were compressed away unless they
  still refer to active files.
- Record compression generation/counter so attachment code can tell whether it
  is using stale context.

### Session Split Decision

Do not rotate session IDs in this PR. Aether's session/gateway resume semantics
are not the same as Hermes' SQLite session split. Instead record lineage:

```python
context.metadata["compression_lineage"] = {
    "generation": 2,
    "trigger_reason": "preflight",
    "source_message_count": 120,
    "result_message_count": 18,
}
```

If a later sprint adds session split, it should build on this metadata.

### Tests

Add focused tests with fake memory and fake diagnostics:

- memory hook called before compression
- memory hook failure does not abort compression unless current memory behavior
  already aborts
- diagnostic generation increments after compression
- missing diagnostic file/state does not crash compression
- compression metadata appears in result metadata

Regression tests:

- `aether/tests/memory`
- `aether/tests/agents/test_diagnostic_attachment_pipeline.py`
- `aether/tests/tools/test_file_edit_diagnostic_wire.py`

### Review Checklist

- No raw messages or secrets in public compression metadata.
- No session ID rotation.
- Memory and diagnostics failures are isolated.
