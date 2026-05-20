# Sprint 17 - Acceptance Matrix

| Scenario | 17.1 Runtime Config | 17.2 Sources | 17.3 Pool | 17.4 Slots | 17.5 Acceptance |
|---|---|---|---|---|---|
| Main provider from env | parsed | key resolved | optional | main slot | regression pass |
| Missing key | config valid | clear error | fallback none | slot reports missing | regression pass |
| Subagent default | provider mapped | key redacted | optional | subagent slot | regression pass |
| Caller model override | parsed | no leak | no rotation | caller wins | regression pass |
| Rate limit | provider classified | source known | rotate key | slot unchanged | regression pass |
| Metadata/logging | provider visible | secrets redacted | pool name only | slot visible | regression pass |

## Unit Test Map

| File | Purpose |
|---|---|
| `aether/tests/config/test_provider_runtime.py` | provider family/model resolution |
| `aether/tests/runtime/credentials/test_sources.py` | env credential source |
| `aether/tests/runtime/credentials/test_redaction.py` | secret redaction |
| `aether/tests/runtime/credentials/test_pool.py` | optional pool and rotation |
| `aether/tests/config/test_auxiliary_slots.py` | auxiliary slot resolution |
| existing `aether/tests/subagents/**` | subagent provider/model behavior |
| existing `aether/tests/tools/test_web_search_tool.py` | web-search env compatibility |

## Manual Checklist

- Existing `.env` openai-compatible setup still starts.
- Existing `.env` Claude setup still starts.
- Existing `.env` Codex setup still starts.
- `WEB_SEARCH_PROVIDER` and `WEB_SEARCH_API_KEY` still control local search.
- Subagent default provider/model can be inspected without exposing keys.

## Non-Regression Rules

- Do not delete or rewrite user `.env` values.
- Do not expose raw API keys in gateway/CLI/status metadata.
- Do not overload `AETHER_PROVIDER` for every auxiliary slot.
- Do not implement OAuth flows in this sprint.
- Do not make credential pool mandatory.
