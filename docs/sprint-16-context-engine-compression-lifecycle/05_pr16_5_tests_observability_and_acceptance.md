# PR 16.5 - Tests, Observability, and Acceptance

## Goal

Close Sprint 16 with explicit regression coverage.

## Verification

Run:

- `python -m pytest aether/tests/runtime/test_compaction_pipeline.py`
- `python -m pytest aether/tests/engine/test_compaction_integration.py`
- `python -m pytest aether/tests/engine/test_collapse_integration.py`
- `python -m pytest aether/tests/memory`
- `python -m pytest aether/tests/agents/runtime/test_context_assembly_pipeline.py`
- targeted `uv run pyright` for context/runtime files

## Observability

Expose:

- `context_engine.name`
- `context_engine.compression_count`
- `context_engine.last_trigger_reason`
- `compaction` metadata unchanged for backward compatibility

## Acceptance

- Existing auto-compaction behavior remains stable.
- Manual compression is either complete or absent.
- Context engine boundary is tested without provider network calls.

## Detailed Acceptance Procedure

### Static Review

Confirm:

- `ContextAssemblyPipeline` depends on a context engine/lifecycle boundary.
- Existing compaction pipeline is still reachable through the default engine.
- Manual compression, if present, does not duplicate compression logic.
- Context engine files do not import gateway/TUI/provider clients.

### Automated Tests

Focused:

```bash
python -m pytest aether/tests/runtime/context
python -m pytest aether/tests/agents/runtime/test_context_assembly_pipeline.py
python -m pytest aether/tests/runtime/test_compaction_pipeline.py
python -m pytest aether/tests/engine/test_compaction_integration.py
python -m pytest aether/tests/engine/test_collapse_integration.py
python -m pytest aether/tests/memory
```

If manual RPC/slash exists:

```bash
python -m pytest aether/tests/gateway/test_context_methods.py
```

Targeted type check:

```bash
uv run pyright aether/runtime/context aether/agents/runtime/context_assembly.py
```

### Observability Checklist

The result metadata should let us answer:

- Did compression run?
- Which context engine ran?
- Why did it run?
- How many messages went in/out?
- Did it fail?
- Did memory/diagnostics hooks run?

But metadata must not include:

- raw full transcript
- raw tool output bodies
- raw credentials
- live context engine objects

### Deferred Work Log

At the end of the sprint, document whether these were deferred:

- plugin context engines
- session ID rotation/split
- manual slash command
- auxiliary compression model slot integration

Each deferred item must include the reason and the sprint that should own it.
