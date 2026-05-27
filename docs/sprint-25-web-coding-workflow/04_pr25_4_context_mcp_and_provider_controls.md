# PR25.4 - Context MCP And Provider Controls

## Objective

Make runtime readiness visible before and during web runs. Users should see
model context pressure, prompt composition, compression state, provider/model
status, MCP server/resource readiness, and credential state without leaking
secrets.

## Current State

Already present or partially implemented:

- composer token ring and activity bar token usage;
- usage reducer normalizes several provider token shapes;
- context routes and service exist;
- context compression can be triggered through backend routes;
- backend routes expose MCP status/config/server mutation/runtime
  refresh/resources/resource reads;
- web `/mcp` inspector and settings views can list configured/imported tools,
  save/delete managed servers, refresh runtime discovery, and read text
  resources;
- provider settings and preflight diagnostics exist.

Main gaps:

- token display is still mostly active-run/session metadata;
- next-run prompt budget is not reconstructed with enough fidelity;
- compaction state and model window pressure are not visible enough;
- provider-specific usage fields are not fully normalized;
- MCP credential UX is shallow;
- remote MCP validation errors need to be more actionable;
- resource actions are basic/read-only;
- provider error/readiness state is not consistently surfaced in the same
  runtime control surface.

## Backend Scope

Primary files:

- `aether/services/context/contracts.py`
- `aether/services/context/service.py`
- `aether/web/routes/context.py`
- `aether/services/runs/builder.py`
- `aether/services/runs/service.py`
- `aether/services/providers/contracts.py`
- `aether/services/providers/service.py`
- `aether/services/tools/contracts.py`
- `aether/services/tools/service.py`
- `aether/web/routes/tools.py`
- `aether/web/routes/providers.py`
- `aether/runtime/mcp/*`
- `aether/agents/runtime/context_assembly.py`
- `aether/services/compact/*`
- `aether/services/environment/contracts.py`
- `aether/services/environment/service.py`

Context contracts:

- `GET /api/context/{session_id}/status`
  - active model and provider;
  - context window and unknown-window state;
  - estimated prompt tokens;
  - recent transcript, system prompt, memory/context, attachments, and
    tool-result token breakdowns;
  - compression lineage/state;
  - pressure level: `low`, `medium`, `high`, `critical`;
  - next action: `none`, `compress`, `split`, `blocked`.
- `POST /api/context/{session_id}/estimate`
  - accepts draft user text and attachments;
  - returns next-run estimate without mutating session state.
- `POST /api/context/{session_id}/compress`
  - returns compression result, lineage, and updated accounting.

Provider rules:

- model catalog exposes context window when known;
- unknown windows render as unknown, not zero;
- usage normalization preserves prompt/input, completion/output, reasoning,
  cache read/write, and total tokens for OpenAI-compatible, Anthropic, Codex,
  and hosted-tool variants;
- provider preflight errors include endpoint, status, body summary, retry hint,
  and suggested fix where safe.

MCP contracts:

- credential status per server: configured, missing, invalid, unknown;
- expose env key names and redacted display values only;
- never return raw secret values in normal API responses;
- validate stdio command existence, remote URL scheme, connect timeout,
  header/env key shape, and tool/resource discovery;
- resource read returns inline text for text resources;
- binary/unsupported resources return metadata and safe open/download status
  when supported;
- credential reveal or mutation is audited where current environment service
  supports it.

## Frontend Scope

Primary files:

- `web/src/api/types.ts`
- `web/src/api/client.ts`
- `web/src/stores/chatStore.ts`
- `web/src/stores/providerStore.ts`
- `web/src/stores/toastStore.ts`
- `web/src/components/chat/Composer.tsx`
- `web/src/components/chat/ComposerInspectorPanel.tsx`
- `web/src/components/chat/ActivityBar.tsx`
- `web/src/components/chat/ChatWorkbenchHeader.tsx`
- `web/src/components/settings/ProviderSettings.tsx`
- `web/src/components/settings/ToolsView.tsx`
- `web/src/styles.css`

Required UI:

- context ring shows model-window pressure, not just latest usage;
- context inspector shows source breakdown and compression lineage;
- draft estimates update when user text, attachments, workspace references,
  model, or session changes;
- estimates are debounced and stale responses are ignored;
- compression action appears only when backend supports it;
- after compression, timeline receives a summary notice;
- critical pressure is visible before send;
- provider status shows selected provider/model, endpoint/preflight state, and
  actionable failure details;
- MCP server editor supports name, enabled, transport, command/args, URL,
  env/header keys, timeout, and connect timeout;
- MCP validation panel shows status, discovered tools/resources, missing
  credentials, last error, and suggestion;
- resource browser supports server filter, text preview, copy URI/content, and
  open/download when safe;
- no raw secret values appear in forms, logs, screenshots, or normal views.

Accessibility:

- keyboard navigation through MCP server/resource lists;
- validation errors associated with form inputs;
- dialogs use existing modal focus behavior.

## Tests

Python:

- context status returns model window and source breakdown;
- draft estimate includes text, attachments, and workspace references;
- unknown provider window handled safely;
- compression result updates lineage/status;
- provider usage normalization covers OpenAI-compatible, Anthropic, Codex, cache,
  and reasoning fields;
- provider preflight errors are structured and redacted;
- MCP config upsert redacts credentials;
- invalid remote URL returns validation error;
- missing env key appears as missing without value;
- resource read returns text content and safe metadata.

TypeScript:

- context ring renders low/medium/high/critical states;
- inspector renders source breakdown;
- draft estimate debounces and cancels stale responses;
- compression action updates visible state;
- unknown context window is readable;
- provider settings show preflight success/error details;
- MCP server editor renders stdio and remote fields conditionally;
- missing credential state is visible without leaking secrets;
- resource browser filters by server and reads text resource;
- delete server asks for confirmation.

Browser/manual:

- attach a large workspace file and verify pressure changes;
- switch model and verify context window changes;
- trigger compression and verify updated context notice;
- configure a fixture stdio MCP server and refresh tools;
- configure a bad remote URL and verify validation output;
- read a text resource and copy content;
- verify screenshots/logs do not include secret values.

## Acceptance

- Users can tell whether the next run is safe, near limit, or likely to
  compress.
- Token usage during/after runs is consistent with provider metadata.
- Provider readiness and errors are actionable.
- MCP servers can be configured, validated, refreshed, and deleted from web.
- Credential state is clear and secrets are not exposed.
- MCP resources can be browsed and read safely.

## Explicit Exclusions

- Exact tokenizer parity for every third-party OpenAI-compatible provider.
- Billing/cost accounting.
- Automatic session splitting UI unless already backed by backend contracts.
- OAuth device-flow credential setup.
- Hosted MCP marketplace.
- Arbitrary MCP resource mutation unless the runtime exposes a safe operation.
