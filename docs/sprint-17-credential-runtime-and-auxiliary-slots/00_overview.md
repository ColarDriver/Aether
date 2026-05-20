# Sprint 17 - Credential Runtime and Auxiliary Slots

## Decision

This must be its own sprint.

Credential runtime touches secrets, provider construction, rate-limit recovery,
subagent model choice, compression model choice, and observability. It should
not be mixed with provider transport extraction.

## Motivation

Aether currently uses environment variables and `EngineConfig` directly for
provider setup. Recent work added global provider choices and subagent defaults,
but there is no unified runtime for:

- credential lookup
- credential redaction
- per-provider credential availability
- multiple keys
- key rotation after rate limits
- auxiliary model slots
- per-slot provider/model overrides

Hermes has a much larger credential pool/runtime-provider system. Aether should
add the useful core without copying every OAuth or platform integration.

## Goals

- Normalize provider runtime configuration.
- Add credential source abstraction.
- Add optional local credential pool with rotation.
- Define auxiliary model slots for subagent, compression, verifier, curator-like
  future tasks, and web search if needed.
- Keep current `.env` values working.

## Non-Goals

- Do not implement browser OAuth flows.
- Do not add cloud-specific setup wizards.
- Do not rewrite provider transports.
- Do not force users into credential files if `.env` works.
- Do not add a skill curator in this sprint.

## PR Roadmap

| PR | File | Boundary |
|---|---|---|
| 17.1 | `01_pr17_1_provider_runtime_config.md` | Normalize provider/model config and env names |
| 17.2 | `02_pr17_2_credential_sources_and_redaction.md` | CredentialSource, env source, redaction |
| 17.3 | `03_pr17_3_credential_pool_and_rate_limit_rotation.md` | Optional pool and recovery integration |
| 17.4 | `04_pr17_4_auxiliary_model_slots.md` | Slot config for subagent/compression/verifier |
| 17.5 | `05_pr17_5_gateway_cli_tests_acceptance.md` | Observability, commands, tests |

## Completion Criteria

- `.env` compatibility remains intact.
- Provider construction reads through a single runtime config path.
- Subagent default provider/model no longer has bespoke parsing inside the tool.
- Secrets are redacted from logs and metadata.

## Current Aether Anchors

Review:

- `.env` in the repo root, but never delete or reorder user values casually.
- `aether/config/schema.py`
- provider construction paths in CLI/gateway startup
- `aether/tools/builtins/agent_tool.py` subagent provider/model default logic
- `aether/models/provider/openai_compatible.py`
- `aether/models/provider/claude.py`
- `aether/models/provider/codex.py`
- `aether/runtime/recovery/rate_guard.py`
- `aether/runtime/recovery/provider_errors.py`
- `aether/tools/builtins/web_search.py`

Sprint 17 is the place to make provider choice and credential lookup explicit.
Do not keep adding one-off env parsing in tools.

## Environment Naming Principles

Use stable, plain names:

- `AETHER_PROVIDER` controls the main/default provider family.
- `WEB_SEARCH_PROVIDER` controls local search backend.
- `WEB_SEARCH_API_KEY` supplies local search backend credential.

Do not use `AETHER_WEB_SEARCH_PROVIDER` unless a future migration deliberately
renames the variable with backward compatibility. The user-facing decision has
already been to keep `WEB_SEARCH_PROVIDER`.

Auxiliary slots should use explicit names, for example:

- `AETHER_AUX_SUBAGENT_PROVIDER`
- `AETHER_AUX_SUBAGENT_MODEL`
- `AETHER_AUX_COMPRESSION_PROVIDER`
- `AETHER_AUX_COMPRESSION_MODEL`

Do not overload `AETHER_PROVIDER` to mean both main and every auxiliary task.

## Provider Family Mapping

Canonical provider family values:

- `codex`
- `claude`
- `openai-compatible`

Internal provider names may remain:

- `codex`
- `claude`
- `openai`

Keep this mapping centralized. Do not make individual tools parse aliases.

## Security Rules

- Never write raw API keys into `EngineResult.metadata`.
- Never include raw API keys in hook payloads.
- Never include raw Authorization headers in failed request dumps.
- Logs may include credential source name, but not secret value.
- Redaction must run before any provider runtime config is exposed through
  gateway or CLI status.

## Hermes Reference Points

Read for design ideas only:

- `/workspace/hermes-agent/agent/credential_sources.py`
- `/workspace/hermes-agent/agent/credential_pool.py`
- `/workspace/hermes-agent/hermes_cli/runtime_provider.py`
- `/workspace/hermes-agent/agent/redact.py`

Do not copy Hermes OAuth flows or full setup wizard into this sprint.
