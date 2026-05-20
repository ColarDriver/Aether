# Sprint 16 - Acceptance Matrix

| Scenario | 16.1 Contract | 16.2 Lifecycle | 16.3 Memory/Diagnostics | 16.4 Manual | 16.5 Acceptance |
|---|---|---|---|---|---|
| Normal turn | default engine no-op | no compression | no hook noise | status stable | regression pass |
| Preflight compaction | engine decides | service executes | metadata preserved | optional status | regression pass |
| Collapse projection | projection isolated | lifecycle records | canonical safe | - | regression pass |
| Compression failure | engine reports | service preserves messages | hooks see failure | user-readable | regression pass |
| Memory before compression | contract allows hook | lifecycle calls hook | memory preserved | - | regression pass |
| Manual compress | engine reusable | service executes | metadata stable | RPC/slash works | regression pass |

## Unit Test Map

| File | Purpose |
|---|---|
| `aether/tests/runtime/context/test_context_engine_contract.py` | context engine contract and default adapter |
| `aether/tests/runtime/context/test_compression_lifecycle.py` | compression lifecycle service |
| `aether/tests/agents/runtime/test_context_assembly_pipeline.py` | assembly integration |
| `aether/tests/runtime/test_compaction_pipeline.py` | compaction parity |
| `aether/tests/engine/test_compaction_integration.py` | engine compaction parity |
| `aether/tests/engine/test_collapse_integration.py` | provider projection/collapse parity |
| `aether/tests/memory/**` | memory pre-compression safety |
| `aether/tests/gateway/test_context_methods.py` | manual RPC if implemented |

## Implementation Status

| Area | Status | Evidence |
|---|---|---|
| Context engine contract | Implemented | `aether/runtime/context/engine.py` defines `ContextEngine` and `ContextEngineResult`. |
| Default engine adapter | Implemented | `aether/runtime/context/default_engine.py` wraps existing compaction and collapse helpers through a narrow adapter. |
| Compression lifecycle service | Implemented | `aether/runtime/context/compression_lifecycle.py` owns preflight/forced compression, metadata, hook calls, validation, and failure preservation. |
| Context assembly integration | Implemented | `ContextAssemblyPipeline` calls `CompressionLifecycleService` for preflight compression and `ContextEngine.apply_provider_projection` for collapse view. |
| Memory boundary | Implemented | Existing memory `before_compaction` remains on the default engine path; regression tests assert preflight compaction sees canonical messages without injected memory. |
| Diagnostics continuity | Implemented | Successful compression records `diagnostics.compression_generation` and `compression_lineage`. |
| Session split | Deferred | Sprint 16 records lineage metadata and does not rotate session IDs. |
| Manual backend control | Implemented | `context.status` and `context.compress` RPC methods are registered and covered by gateway tests. |
| Manual slash command | Deferred | No `/compress` command is registered; backend RPC is available first, avoiding a half-wired TUI path. |

## Observability Shape

`EngineResult.metadata` now includes:

- `context_engine.name`
- `context_engine.compression_count`
- `context_engine.last_trigger_reason`
- `context_engine.compression`
- `compression_lineage`

The existing top-level `compaction` counters remain unchanged for backward
compatibility.

## Manual Checklist

- Long context still auto-compacts.
- Collapse view still affects provider payload only.
- Failed compression leaves the session usable.
- Memory retrieval still appears after compression.
- Manual `/compress` only exists if backend RPC is complete.

## Non-Regression Rules

- Do not rotate session IDs in Sprint 16.
- Do not add plugin context engine discovery yet.
- Do not duplicate compression logic in gateway/TUI.
- Do not put full transcripts into public metadata.
- Do not change plan-mode reminder semantics.

## Verification

Run on `context-engine-compression-lifecycle`:

```text
python -m pytest aether/tests/runtime/context
# 12 passed

python -m pytest aether/tests/agents/runtime/test_context_assembly_pipeline.py
# 5 passed

python -m pytest aether/tests/runtime/test_compaction_pipeline.py
# 16 passed

python -m pytest aether/tests/engine/test_compaction_integration.py
# 7 passed

python -m pytest aether/tests/engine/test_collapse_integration.py
# 8 passed

python -m pytest aether/tests/memory
# 52 passed

python -m pytest aether/tests/gateway/test_context_methods.py
# 7 passed

uv run pyright aether/runtime/context aether/agents/runtime/context_assembly.py aether/gateway/handlers/context_methods.py
# 0 errors, 0 warnings, 0 informations
```

## Deferred Work

| Item | Reason | Owning Sprint |
|---|---|---|
| Plugin context engines | Sprint 16 only adds explicit injection and a stable contract; discovery/marketplace policy belongs outside compression lifecycle. | Future plugin/context-engine sprint |
| Session ID rotation/split | Aether's current resume/session persistence model does not have Hermes-style split primitives. This sprint records lineage instead. | Future session persistence sprint |
| TUI `/compress` slash command | Backend RPC is complete, but no TUI command is registered yet to avoid UI behavior churn in this runtime sprint. | Future TUI control-surface sprint |
| Auxiliary compression model slot | Provider/model slot routing is part of Sprint 17 credential/runtime config work, not this lifecycle boundary. | Sprint 17 |
