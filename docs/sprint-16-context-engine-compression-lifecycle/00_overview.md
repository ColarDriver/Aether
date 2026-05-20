# Sprint 16 - Context Engine and Compression Lifecycle

## Decision

This must be its own sprint.

Context compression touches canonical transcript shape, session persistence,
memory, diagnostics, prompt rebuilding, and token-budget behavior. It should not
be mixed with provider transport or credential work.

## Motivation

Aether already has compaction and collapse behavior, but it is still primarily
an engine-internal pipeline. Hermes has a more explicit `ContextEngine`
boundary, making compression and future long-context strategies replaceable.

Aether needs a similar boundary before adding more sophisticated summarization,
manual compression, or plugin context engines.

## Goals

- Define a `ContextEngine` contract.
- Wrap current compaction pipeline as the default engine.
- Add explicit compression lifecycle events.
- Preserve canonical/prepared message separation from Sprint 13.
- Support manual compression without changing normal auto-compaction behavior.
- Keep memory and diagnostics hooks coherent around compression.

## Non-Goals

- Do not add credential pools.
- Do not rewrite provider transports.
- Do not introduce a plugin marketplace.
- Do not change TUI rendering beyond minimal slash/RPC output if manual
  compression is included.
- Do not change plan mode semantics.

## PR Roadmap

| PR | File | Boundary |
|---|---|---|
| 16.1 | `01_pr16_1_context_engine_contract.md` | Contract and default adapter for existing compaction |
| 16.2 | `02_pr16_2_compression_lifecycle_service.md` | Compression service, preflight, prompt rebuild boundary |
| 16.3 | `03_pr16_3_memory_diagnostics_and_session_split.md` | Memory/diagnostic hooks and session metadata |
| 16.4 | `04_pr16_4_manual_compression_control_surface.md` | Optional `/compress` RPC/slash path |
| 16.5 | `05_pr16_5_tests_observability_and_acceptance.md` | Full regression and acceptance |

## Completion Criteria

- `ContextAssemblyPipeline` calls a context engine boundary, not raw compaction
  helpers directly.
- Compression lifecycle is observable.
- Manual compression path is either implemented or explicitly deferred with no
  half-wired UI.

## Current Aether Anchors

Review these before implementing:

- `aether/agents/runtime/context_assembly.py`
- `aether/agents/core/agent.py` methods around `_get_compaction_pipeline`,
  `_maybe_compact_messages`, `_apply_collapse_view`, and memory helpers.
- `aether/services/compact.py`
- `aether/runtime/recovery/strategies.py`
- `aether/runtime/session/session_runtime.py`
- `aether/memory/*`
- `aether/runtime/diagnostics/*`

The current system already has multiple compaction tiers and a collapse view.
Sprint 16 should wrap and clarify those behaviors, not replace them with a new
summarizer.

## Hermes Reference Points

Read:

- `/workspace/hermes-agent/agent/context_engine.py`
- `/workspace/hermes-agent/agent/context_compressor.py`
- `/workspace/hermes-agent/agent/conversation_compression.py`
- `/workspace/hermes-agent/agent/memory_provider.py`

Important idea to borrow: context compression is a lifecycle with hooks,
preflight, metadata, and recovery behavior. Do not borrow Hermes' entire session
DB split model unless Aether has equivalent persistence.

## Message Model Rules

- Canonical transcript remains the source of truth.
- Provider projection/collapse view must not mutate canonical messages.
- Compression output must be valid engine messages.
- Tool result spill references must remain usable after compression.
- Plan mode reminders and diagnostics attachments remain per-turn injections,
  not permanent compressed transcript content unless current behavior already
  makes them permanent.

## Compression Failure Rules

If compression fails:

- keep current messages usable
- record a public-safe failure reason
- do not clear memory/diagnostic state
- do not mark the turn as successful compression
- let provider recovery decide whether to continue, retry, or fail

## Manual Compression Product Rule

Manual `/compress` is optional in this sprint. If implemented, it must use the
same `CompressionLifecycleService` as auto-compression. If not implemented, do
not register a slash command or gateway method stub that returns fake success.
