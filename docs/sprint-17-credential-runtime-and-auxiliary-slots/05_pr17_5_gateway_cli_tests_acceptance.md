# PR 17.5 - Gateway, CLI, Tests, and Acceptance

## Goal

Expose provider runtime state safely and close Sprint 17.

## Gateway/CLI Surface

Optional RPCs:

- `provider.runtime_current`
- `provider.credentials_status`
- `provider.auxiliary_slots`

These must never return raw secrets.

## Observability

Expose:

- provider choice
- model
- credential source name
- credential pool entry name
- redacted key suffix only if useful
- auxiliary slot used

Do not expose:

- raw API keys
- full Authorization headers
- refresh tokens

## Verification

Run:

- `python -m pytest aether/tests/config`
- `python -m pytest aether/tests/runtime/credentials`
- `python -m pytest aether/tests/subagents`
- `python -m pytest aether/tests/engine/test_tool_permission_gate.py`
- targeted `uv run pyright` for config/credential files

## Acceptance

- Runtime config and credentials are observable without leaking secrets.
- `.env` compatibility remains intact.
- Subagent slot behavior is covered.
- Rate-limit rotation is covered if PR17.3 enabled it.

## Detailed Acceptance Procedure

### Gateway Method Shape

If adding RPCs, prefer read-only status methods:

- `provider.runtime_current`
- `provider.credentials_status`
- `provider.auxiliary_slots`

Return examples:

```json
{
  "provider": "openai",
  "family": "openai-compatible",
  "model": "gpt-5.4",
  "credential": {
    "source": "env",
    "name": "OPENAI_API_KEY",
    "configured": true,
    "redacted": "sk-...abcd"
  }
}
```

Do not include raw key fields.

### CLI/TUI Rules

If exposing status in CLI/TUI:

- display provider family and model
- display whether credential is configured
- display redacted credential identifier only
- display auxiliary slot provider/model
- do not display base URL credentials embedded in URLs

### Full Compatibility Checks

Manually verify `.env` scenarios:

- only openai-compatible variables configured
- Claude variables configured
- Codex variables configured
- `WEB_SEARCH_PROVIDER=brave` and `WEB_SEARCH_API_KEY=...`
- no web search key configured

Do not modify the user's `.env` values as part of tests.

### Automated Tests

Run:

```bash
python -m pytest aether/tests/config
python -m pytest aether/tests/runtime/credentials
python -m pytest aether/tests/subagents
python -m pytest aether/tests/tools/test_web_search_tool.py
python -m pytest aether/tests/gateway
```

Targeted type check:

```bash
uv run pyright aether/config aether/runtime/credentials aether/tools/builtins/agent_tool.py
```

### Security Review Checklist

Search for accidental leaks:

```bash
rg -n "api_key|Authorization|Bearer|WEB_SEARCH_API_KEY|ANTHROPIC_API_KEY|CODEX" aether
```

This search is not itself a failure. Review any new logging/metadata code and
confirm it uses redaction.

### Rollback Plan

If credential pool rotation proves risky, keep PR17.1, PR17.2, and PR17.4. Gate
pool rotation behind config disabled by default until recovery tests are strong.
