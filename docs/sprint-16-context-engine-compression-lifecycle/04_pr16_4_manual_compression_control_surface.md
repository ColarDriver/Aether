# PR 16.4 - Manual Compression Control Surface

## Goal

Expose manual compression only after the lifecycle service is stable.

## Control Surface

Recommended gateway RPC:

- `context.compress`
- `context.status`

Optional slash command:

- `/compress`
- `/compress <focus>`

## Behavior

- If there is not enough content to compress, return a readable no-op result.
- If compression succeeds, show before/after message and token estimates.
- If compression fails, keep the current session usable and report the reason.
- Do not auto-enter plan mode.
- Do not change provider/model selection.

## Tests

Add:

- `aether/tests/gateway/test_context_methods.py`
- TUI slash tests only if `/compress` is implemented.

Cover:

- No-op status.
- Successful compression response.
- Failed compression response.
- Missing session error.

## Acceptance

- Manual compression uses the same lifecycle as auto-compression.
- If slash UI is not implemented, no partial command is registered.

## Detailed Implementation Notes

### Gateway RPC Shape

If implemented, add methods under gateway handlers:

- `context.status`
- `context.compress`

Suggested `context.status` response:

```json
{
  "session_id": "s",
  "context_engine": "default",
  "compression_count": 1,
  "last_compression": {
    "status": "compressed",
    "trigger_reason": "manual",
    "source_message_count": 42,
    "result_message_count": 12
  }
}
```

Suggested `context.compress` params:

```json
{
  "session_id": "s",
  "focus": "optional text",
  "force": true
}
```

### Session Lookup

The RPC must operate on the active session transcript. If the current session
store cannot provide canonical messages safely, defer the RPC rather than
inventing an alternate transcript path.

### TUI Slash Behavior

If adding slash command:

- `/compress` calls `context.compress` with `force=true`
- `/compress <focus>` passes focus text
- result prints a short status line with before/after counts
- failure prints reason and leaves session usable

Do not stream compression progress in the first version unless gateway events
already support it cleanly.

### No-Op Behavior

If not enough content exists:

- return `status="skipped"`
- include `reason="not_enough_context"` or equivalent
- do not mark compression count as incremented

### Tests

Gateway tests:

- missing session returns RPC error
- no-op returns skipped
- success returns compressed metadata
- failure returns failed without transcript loss

TUI tests only if slash is implemented:

- `/compress` calls RPC
- `/compress focus` passes focus
- skipped/success/failure render readable output

### Review Checklist

- Manual path uses `CompressionLifecycleService`.
- No duplicate compression implementation in gateway/TUI.
- No fake slash command if backend is not ready.
