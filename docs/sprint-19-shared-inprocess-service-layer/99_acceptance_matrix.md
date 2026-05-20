# Sprint 19 - Acceptance Matrix

| # | Scenario | 19.1 | 19.2 | 19.3 | 19.4 | 19.5 | 19.6 | 19.7 | 19.8 | 19.9 | 19.10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Service packages import safely | boundaries and errors | sessions/config/prefs export | provider exports | read-service exports | run contracts export | run service export | adapters import services | agent adapter imports service | CLI import guard | final guard |
| E2 | TUI command catalog starts | no UI imports in services | unchanged | unchanged | unchanged | unchanged | unchanged | low-risk RPC unchanged | agent RPC unchanged | smoke | regression pass |
| E3 | session create/list/current/resume | contract rules | service owns behavior | - | - | run request depends on sessions | run uses SessionService | RPC unchanged | agent run uses service session | CLI ready | regression pass |
| E4 | transcript rendering | no gateway schemas in services | transcript service model | - | - | event fixtures | persistence parity | session RPC unchanged | run persistence unchanged | smoke | regression pass |
| E5 | prefs and last model | no adapter state | PrefsService | model selection consumes prefs | - | - | provider/run uses prefs as needed | prefs RPC unchanged | agent unaffected | CLI ready | regression pass |
| E6 | `/model` list/switch | boundary | prefs available | provider/auth/model service | - | - | run builder consumes provider selection | provider RPC unchanged | agent provider build unchanged | Web ready | regression pass |
| E7 | provider credentials/status | no secret contracts | config paths | AuthService redacts | health aggregates | - | run builder uses readiness | provider RPC unchanged | agent errors compatible | doctor ready | regression pass |
| E8 | tools list | service package rules | - | - | ToolService | - | run builder tool registry hook | tools RPC unchanged | tool events unchanged | CLI ready | regression pass |
| E9 | skills list | service package rules | config paths | - | SkillService | - | run builder skill catalog | status/tools RPC unchanged | agent skill catalog unchanged | Web ready | regression pass |
| E10 | diagnostics/health | common status contracts | config included | auth included | diagnostics/health services | event status contract | run status available | low-risk adapter stable | agent status stable | doctor ready | regression pass |
| E11 | agent run streaming | event boundary rule | session dependency | provider dependency | diagnostic dependency | golden event contract | AgentRunService core | not migrated | agent adapter maps events | WS/SSE ready | regression pass |
| E12 | permission prompts | no UI in services | - | - | - | permission event contract | prompter protocol injection | unchanged | gateway bridge compatible | smoke | regression pass |
| E13 | cancel run | service error rules | - | - | health status optional | cancel contract | cancel service | unchanged | agent.cancel adapter | Web ready | regression pass |
| E14 | future HTTP adapter | transport-neutral contracts | reusable | reusable | reusable | service events reusable | run lifecycle reusable | gateway no longer source of truth | agent adapter proves mapping | documented | accepted |

## Required Files

| File | Purpose |
|---|---|
| `00_overview.md` | sprint goals, boundaries, roadmap, dependency graph, acceptance summary |
| `01_pr19_1_service_layer_contract_and_boundaries.md` | service skeleton, common errors, import guards |
| `02_pr19_2_session_config_and_prefs_services.md` | session/config/prefs services |
| `03_pr19_3_provider_auth_and_model_services.md` | provider/auth/model services |
| `04_pr19_4_tools_skills_diagnostics_health_services.md` | tools/skills/diagnostics/health read services |
| `05_pr19_5_agent_run_event_contract.md` | run contracts and event golden tests |
| `06_pr19_6_agent_run_service_core.md` | AgentRunService core lifecycle |
| `07_pr19_7_low_risk_gateway_adapter_migration.md` | non-agent gateway adapter migration |
| `08_pr19_8_agent_gateway_adapter_migration.md` | agent gateway adapter migration |
| `09_pr19_9_cli_web_readiness.md` | CLI/Web readiness |
| `10_pr19_10_acceptance_and_hardening.md` | final hardening and regression evidence |

## Future Implementation Test Map

| File | Purpose |
|---|---|
| `aether/tests/services/test_service_import_boundaries.py` | services do not import gateway/UI/transport modules |
| `aether/tests/services/test_service_exports.py` | stable service package public exports |
| `aether/tests/services/test_session_service.py` | session lifecycle and transcript service behavior |
| `aether/tests/services/test_config_service.py` | effective config/default/path read behavior |
| `aether/tests/services/test_prefs_service.py` | scoped prefs and last-model persistence |
| `aether/tests/services/test_provider_service.py` | provider catalog, model listing, discovery fallback |
| `aether/tests/services/test_auth_service.py` | credential readiness without secret leakage |
| `aether/tests/services/test_model_selection_service.py` | deterministic provider/model/base URL selection |
| `aether/tests/services/test_tools_service.py` | tool list/grouping read model |
| `aether/tests/services/test_skills_service.py` | skill discovery read model |
| `aether/tests/services/test_diagnostics_service.py` | diagnostics readiness and missing tracker behavior |
| `aether/tests/services/test_health_service.py` | aggregated public health status |
| `aether/tests/services/test_agent_run_contracts.py` | run request/result/event contracts |
| `aether/tests/services/test_agent_run_service.py` | run lifecycle with fake dependencies |
| `aether/tests/services/test_agent_run_service_events.py` | event ordering and payload semantics |
| `aether/tests/services/test_agent_run_service_cancel.py` | cancellation behavior |
| `aether/tests/gateway/test_service_adapter_compat.py` | RPC response compatibility after adapter migration |
| `aether/tests/gateway/test_agent_run_event_compat.py` | service-event to gateway-event golden compatibility |
| `aether/tests/cli/test_cli_service_boundaries.py` | CLI does not import gateway handlers for serviceized behavior |

## Required Automated Checks

- `python -m pytest aether/tests/services`
- `python -m pytest aether/tests/gateway`
- `python -m pytest aether/tests/cli`
- `python -m pytest aether/tests/agents`
- `python -m pytest aether/tests/tools`
- `uv run pyright aether/services aether/gateway/handlers aether/cli`

## Manual Acceptance

- TUI starts normally and command catalog loads.
- Session create/list/resume/current works through the existing TUI path.
- Transcript rendering still shows user, assistant, tool, metadata, and tool
  calls correctly.
- `/model` lists models, shows discovery diagnostics, and persists selection.
- Agent run streams text/reasoning and emits tool/status/token events.
- Permission and approval flows still use the gateway prompter bridges.
- `agent.cancel` cancels an active run and leaves a coherent final state.
- Tools, skills, diagnostics, and health/status surfaces still render.

## Non-Regression Rules

- Gateway RPC method names stay stable.
- Gateway request and response fields stay stable.
- TUI event shapes stay stable.
- Services do not import gateway handlers, gateway protocol envelopes, CLI
  parser/UI modules, TUI code, React/Ink code, or Web transport code.
- Adapters do not duplicate business logic owned by services.
- Gateway prompter and protocol serialization stay in gateway.
- `commands_methods.py` may remain a catalog-only handler.
- Future HTTP/WebSocket/SSE adapters call services rather than gateway handlers.

## Final Default Position

Sprint 19 should not end with a new public Web server or a second service layer.
It should end with a clean in-process service boundary, gateway adapters over
that boundary, and compatibility evidence proving the TUI did not regress.

## Implementation Evidence

Automated checks run on `shared-inprocess-service-layer`:

- `python -m pytest aether/tests/services` - 39 passed.
- `python -m pytest aether/tests/gateway` - 241 passed.
- `python -m pytest aether/tests/cli` - 33 passed.
- `python -m pytest aether/tests/agents` - 138 passed.
- `python -m pytest aether/tests/tools` - 341 passed.
- `uv run pyright aether/services aether/gateway/handlers aether/cli` -
  0 errors, 0 warnings.

Implemented service packages:

- `aether/services/common`
- `aether/services/sessions`
- `aether/services/config`
- `aether/services/providers`
- `aether/services/tools`
- `aether/services/skills`
- `aether/services/diagnostics`
- `aether/services/health`
- `aether/services/runs`

Gateway adapter status:

- `prefs_methods.py` uses `PrefsService`.
- `session_methods.py` uses `SessionService`.
- `providers_methods.py` uses `ProviderService` and `AuthService`.
- `tools_methods.py` uses `ToolService`.
- `agent_methods.py` uses `AgentRunService`.
- Gateway protocol serialization and prompter bridges remain in gateway.
- `commands_methods.py` remains catalog-only.

Intentional follow-ups:

- Production HTTP/WebSocket/SSE adapters are still future work.
- Pure session persistence still lives under `aether.cli.sessions`; services
  wrap it as the current source of truth.
- Gateway compatibility builder hooks remain in `agent_methods.py` for existing
  tests, but run lifecycle ownership is in `AgentRunService`.
