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
