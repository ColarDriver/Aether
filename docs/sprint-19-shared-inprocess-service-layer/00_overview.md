# Sprint 19 - Shared In-Process Service Layer

## Background

Aether now has a stdio gateway, a TypeScript TUI, a CLI, an agent runtime, a
provider transport layer, credential runtime helpers, auxiliary model slots,
context services, and tool scheduling. Reusable business behavior is still
spread across `aether/gateway/handlers/*`, `aether/cli/*`, runtime helpers, and
persistence helpers. The gateway has become the accidental business API for the
TUI.

Sprint 19 creates a shared in-process service layer so gateway, CLI, and future
HTTP/WebSocket/SSE adapters can call the same business operations without
copying gateway handler logic or importing UI/transport code.

## Goals

- Establish `aether/services/*` as the reusable application service boundary.
- Keep `aether/gateway/handlers/*` as JSON-RPC adapters only.
- Preserve all existing gateway RPC method names, request fields, response
  fields, error codes, and TUI-consumed event shapes.
- Move low-risk read/state behavior first: sessions, prefs, config, providers,
  auth readiness, model selection, tools, skills, diagnostics, and health.
- Define agent-run event contracts before moving run lifecycle code.
- Migrate `agent_methods.py` last, after event-contract golden tests exist.
- Keep services transport-neutral so future CLI/Web adapters can use them
  directly.
- Leave `aether/services/compact` in place as the existing service reference.

## Non-Goals

- Do not implement a Web UI.
- Do not add a production HTTP server, WebSocket server, or SSE server.
- Do not change gateway wire schemas or TUI event names.
- Do not rewrite `AgentEngine`.
- Do not migrate or rewrite `aether/services/compact`.
- Do not add OAuth, login/logout, or a new credential pool.
- Do not implement full Hermes-style `config.yaml` migration.
- Do not make the TUI call Python services directly in this sprint.

## Layer Boundaries

`aether/services/*` owns transport-neutral business operations:

- session lifecycle and transcript read models
- effective config, environment path, and preference resolution
- provider discovery, auth readiness, model selection, and auxiliary slots
- tool, skill, diagnostic, and health read models
- agent run lifecycle, cancellation, result persistence, and service events

`aether/gateway/handlers/*` owns JSON-RPC adaptation:

- validate RPC params
- map params into service request contracts
- call services
- map service results into existing JSON-RPC responses/events
- map service exceptions into existing `GatewayError` codes

Future `aether/cli`, `aether/http`, and `aether/web` adapters follow the same
rule. They call services and map service contracts to their own protocol or
display surface. They must not import gateway handler internals.

## Target Directory Shape

```text
aether/services/
  compact/
  common/
    contracts.py
    errors.py
    import_guard.py
  sessions/
    contracts.py
    service.py
    __init__.py
  config/
    contracts.py
    service.py
    prefs.py
    __init__.py
  providers/
    contracts.py
    service.py
    auth.py
    model_selection.py
    __init__.py
  tools/
    contracts.py
    service.py
    __init__.py
  skills/
    contracts.py
    service.py
    __init__.py
  diagnostics/
    contracts.py
    service.py
    __init__.py
  health/
    contracts.py
    service.py
    __init__.py
  runs/
    contracts.py
    events.py
    builder.py
    service.py
    __init__.py
```

Each service package exports stable public contracts from `__init__.py`. Service
contracts are dataclasses or typed models that do not embed JSON-RPC envelopes,
Pydantic gateway schemas, Ink/React state, or HTTP/WebSocket framing.

## Roadmap

| PR | File | Boundary |
|---|---|---|
| 19.1 | `01_pr19_1_service_layer_contract_and_boundaries.md` | Service contract rules, errors, import guardrails |
| 19.2 | `02_pr19_2_session_config_and_prefs_services.md` | Session, config, prefs services |
| 19.3 | `03_pr19_3_provider_auth_and_model_services.md` | Provider, auth, model selection services |
| 19.4 | `04_pr19_4_tools_skills_diagnostics_health_services.md` | Tools, skills, diagnostics, health read services |
| 19.5 | `05_pr19_5_agent_run_event_contract.md` | Agent run request/result/event contracts and golden event tests |
| 19.6 | `06_pr19_6_agent_run_service_core.md` | AgentRunService core lifecycle without gateway wire changes |
| 19.7 | `07_pr19_7_low_risk_gateway_adapter_migration.md` | Migrate session/prefs/provider/tools/status handlers |
| 19.8 | `08_pr19_8_agent_gateway_adapter_migration.md` | Migrate `agent_methods.py` last |
| 19.9 | `09_pr19_9_cli_web_readiness.md` | CLI/Web adapter readiness checks |
| 19.10 | `10_pr19_10_acceptance_and_hardening.md` | Final acceptance, import guards, regression matrix |

## Dependency Graph

```text
19.1 service contract rules
  -> 19.2 sessions/config/prefs
  -> 19.3 providers/auth/model selection
  -> 19.4 tools/skills/diagnostics/health
  -> 19.5 run request/result/event contracts
  -> 19.6 AgentRunService core
  -> 19.7 low-risk gateway adapters
  -> 19.8 agent gateway adapter
  -> 19.9 CLI/Web readiness
  -> 19.10 acceptance and hardening
```

`19.8` must not start until `19.5` has golden tests for current gateway event
shapes and `19.6` can run against a fake provider. Low-risk handler migrations
in `19.7` should be merged before touching `agent_methods.py`.

## Current-Code Anchors

- `aether/gateway/handlers/session_methods.py` owns current session wire
  conversion and directly calls `aether.cli.sessions`.
- `aether/gateway/handlers/prefs_methods.py` owns current prefs RPC adaptation
  and directly calls `aether.cli.prefs`.
- `aether/gateway/handlers/providers_methods.py` mixes provider catalog, live
  OpenAI-compatible model discovery, provider runtime status, credential status,
  and auxiliary-slot reporting.
- `aether/gateway/handlers/tools_methods.py` owns current tool catalog RPC.
- `aether/gateway/handlers/agent_methods.py` owns run handle registration,
  provider/engine construction, streaming event translation, permission bridge
  wiring, cancellation, result persistence, and final response mapping.
- `aether/config/provider_runtime.py`, `aether/config/auxiliary_slots.py`, and
  `aether/runtime/credentials/*` are existing provider/auth foundations and
  should be reused, not reimplemented.

## Acceptance Summary

- TUI behavior remains unchanged.
- Gateway RPC method names, request fields, response fields, and event shapes
  remain unchanged.
- Services expose transport-neutral contracts and stable `__init__.py` exports.
- Service tests own business behavior.
- Gateway tests own wire compatibility and adapter error mapping.
- Services do not import gateway handlers, gateway protocol envelopes, CLI
  parsers, Ink/React code, or Web transport code.
- Gateway handlers become thin adapters, with `agent_methods.py` migrated last.
- Future CLI/Web adapters can call services directly without depending on
  private gateway implementation details.

## Verification Plan

Docs verification:

- Every file listed in the roadmap exists.
- `README.md` links every PR document and the acceptance matrix.
- This overview includes Goals, Non-Goals, Roadmap, Dependency Graph, Current
  Code Anchors, and Acceptance Summary.
- Every PR document includes Goal, Current Problem, Changes, Tests, Non-Goals,
  Acceptance, Migration Notes, and Risks.
- `99_acceptance_matrix.md` maps scenarios to PRs.

Implementation verification:

- `python -m pytest aether/tests/services`
- `python -m pytest aether/tests/gateway`
- `python -m pytest aether/tests/cli`
- `python -m pytest aether/tests/agents`
- `uv run pyright aether/services aether/gateway/handlers aether/cli`
- A static import guard proves `aether/services/**` does not import
  `aether.gateway.handlers`, `aether.gateway.protocol`, TUI code, React/Ink
  code, or CLI parser modules.
