# PR 14.5 - Tests, Observability, and Acceptance

## Goal

Close Sprint 14 with cross-provider verification and explicit migration notes.

## Test Matrix

Run:

- `python -m pytest aether/tests/models`
- `python -m pytest aether/tests/agents/runtime/test_provider_invocation_controller.py`
- `python -m pytest aether/tests/engine/test_streaming_generate.py`
- `python -m pytest aether/tests/tools/test_web_search_tool.py`
- targeted `uv run pyright` for provider and transport files

## Regression Areas

Verify:

- OpenAI-compatible non-streaming response.
- OpenAI-compatible streaming response.
- Claude text response.
- Claude tool response.
- Claude hosted web search response.
- Codex text response.
- Codex tool response.
- Codex hosted web search response.
- Provider validation failures.
- HTTP/provider error recovery classification.
- Usage normalization.
- Silent streaming token counter.

## Docs

Update:

- `docs/sprint-14-provider-transport-layer/99_acceptance_matrix.md`
- Any provider architecture doc if one exists.

## Acceptance

- Providers are transport-backed.
- Provider classes are primarily IO/client lifecycle wrappers.
- Transport tests cover payload and normalization without network.
- No gateway/TUI schema changes.

## Detailed Acceptance Procedure

### Static Review

Confirm these facts by reading code:

- `aether/models/transport/` has pure conversion modules.
- `openai_compatible.py` has no large standalone message/tool conversion block
  except compatibility wrappers.
- `claude.py` has no large content-block normalization block except wrappers.
- `codex.py` has no large output-item normalization block except wrappers.
- Provider classes still own HTTP clients and auth.
- No transport imports `httpx`, `anthropic`, gateway, TUI, or `AgentEngine`.

### Automated Tests

Run a focused suite first:

```bash
python -m pytest aether/tests/models/transport
python -m pytest aether/tests/models
python -m pytest aether/tests/agents/runtime/test_provider_invocation_controller.py
python -m pytest aether/tests/engine/test_streaming_generate.py
python -m pytest aether/tests/tools/test_web_search_tool.py
```

Then run the broader Python suite if the focused suite is green:

```bash
python -m pytest aether/tests
```

Targeted type check:

```bash
uv run pyright aether/models/provider aether/models/transport aether/agents/runtime/provider_invocation.py
```

If repository-wide `uv run pyright` still fails from known baseline issues,
record that separately and do not hide Sprint 14 regressions.

### Fixture Coverage Checklist

Transport tests must include:

- text-only response
- response with one tool call
- response with multiple tool calls
- malformed response validation
- usage metadata
- reasoning metadata when supported
- hosted web search for Claude/Codex
- streaming final response projection for providers where this moved

### Observability Checklist

Ensure these remain true:

- `api_calls` increments once per successful provider response.
- `usage.total_tokens` matches previous normalization.
- `pre_api_request` and `post_api_request` hook payloads are backward compatible.
- `provider_error_retries` and recovery trails are not affected by pure transport
  validation.

### Rollback Plan

If a provider migration causes live-call uncertainty, keep the transport class
and switch the provider back to the old helper path behind a local flag or
temporary wrapper. Do not revert unrelated transports that already passed.
