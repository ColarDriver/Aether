# PR 15.4 - Stream Controller

## Goal

Extract stream callback construction and stream-related metadata from
`AgentEngine`.

## New Module

Add:

- `aether/agents/runtime/stream_controller.py`

Recommended types:

- `StreamController`
- `StreamCallbacks`
- `StreamEventAdapter`

## Scope

Move:

- Visible stream callback wrapper.
- Silent stream callback wrapper.
- Token counter metadata.
- Interrupt checks during stream callback.
- Final one-shot stream fallback if current behavior has one.
- Stream hook/event calls that are currently engine-private.

## Boundary

The stream controller does not read provider chunks directly. Providers still
call callbacks; the controller only creates those callbacks and updates runtime
metadata/events.

## Tests

Add:

- `aether/tests/agents/runtime/test_stream_controller.py`

Cover:

- Visible delta forwarded once.
- Silent delta increments token counter without visible text.
- Interrupt during callback raises/marks the current expected signal.
- Metadata remains JSON-safe.

## Acceptance

- Provider streaming tests pass.
- TUI streaming event shape is unchanged.
- Runner receives callbacks from `StreamController`, not `AgentEngine` methods.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/agents/runtime/stream_controller.py`
- `aether/tests/agents/runtime/test_stream_controller.py`

### Current Engine Methods

Move:

- `_build_stream_callback`
- `_build_stream_silent_callback`

Also inspect any related metadata writes in provider invocation and TUI gateway
streaming tests before moving.

### Controller Contract

Recommended shape:

```python
@dataclass(slots=True)
class StreamCallbacks:
    visible: StreamDeltaCallback | None
    silent: StreamSilentCallback | None

class StreamController:
    def build_callbacks(
        self,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> StreamCallbacks: ...
```

If the current behavior depends on `EngineHooks`, inject hooks explicitly. Do
not reach back into `AgentEngine`.

### Visible vs Silent Semantics

Visible callback:

- forwards user-visible text deltas
- updates any existing stream metadata
- respects interrupt signal

Silent callback:

- counts hidden/tool-argument/reasoning chunks
- does not emit user-visible text
- must keep the TUI token indicator alive during tool-only generation

Do not collapse these two callbacks into one. The TUI behavior depends on this
distinction.

### Interrupt Detail

If callback interruption currently raises a special `BaseException` subclass or
sets context metadata, preserve that exact mechanism. Providers may rely on the
exception crossing out of the streaming loop.

### Tests

Direct tests should cover:

- visible delta forwarded to request callback
- visible callback handles missing request callback
- silent delta increments metadata/count callback
- interrupt before visible callback
- interrupt during silent callback
- callback exceptions are not swallowed when old behavior propagated them

Regression tests:

- `aether/tests/engine/test_streaming_generate.py`
- `aether/tests/engine/test_streaming_engine_gate.py`
- `aether/tests/models/test_provider_streaming.py`
- gateway streaming tests

### Review Checklist

- No TUI component imports from stream controller.
- No provider-specific parsing in stream controller.
- Token counter metadata names remain stable.
