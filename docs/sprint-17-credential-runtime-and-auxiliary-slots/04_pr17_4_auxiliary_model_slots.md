# PR 17.4 - Auxiliary Model Slots

## Goal

Replace ad-hoc subagent provider/model parsing with explicit auxiliary slots.

## Slots

Recommended first slots:

- `main`
- `subagent`
- `compression`
- `verifier`
- `title`

Future slots:

- `curator`
- `background_review`

## Config Shape

Add config support similar to:

```toml
[auxiliary.subagent]
provider = "openai-compatible"
model = "gpt-5.4"

[auxiliary.compression]
provider = "openai-compatible"
model = "gpt-5.4"
```

Environment fallback:

- `AETHER_PROVIDER` controls main provider.
- `AETHER_AUX_SUBAGENT_PROVIDER`
- `AETHER_AUX_SUBAGENT_MODEL`
- `AETHER_AUX_COMPRESSION_PROVIDER`
- `AETHER_AUX_COMPRESSION_MODEL`

If env names are considered too verbose, document the final chosen names before
implementation. Do not silently overload `AETHER_PROVIDER` for every slot.

## Subagent Migration

Move current subagent default profile logic out of:

- `aether/tools/builtins/agent_tool.py`

Into:

- provider runtime / auxiliary slot resolver

The tool should ask for the `subagent` slot default, then write
`provider_override` and `model_override` metadata.

## Tests

Add:

- `aether/tests/config/test_auxiliary_slots.py`
- update subagent model parameter tests

Cover:

- Slot defaults.
- Env overrides.
- Caller `model=` argument still wins.
- `model="inherit"` still inherits.
- Async direct test contexts do not trigger real provider calls.

## Acceptance

- Subagent provider/model default logic is no longer inside the tool.
- Compression can later use the same slot resolver.
- Existing subagent tests pass.

## Detailed Implementation Notes

### Files to Add

Create or extend:

- `aether/config/auxiliary_slots.py`
- `aether/tests/config/test_auxiliary_slots.py`

If provider runtime config already owns slots after PR17.1, keep slot resolver
there instead of adding another parallel module. The important part is one
resolver, not the exact file name.

### Slot Data Model

Recommended:

```python
@dataclass(frozen=True, slots=True)
class AuxiliarySlotConfig:
    slot: Literal["subagent", "compression", "verifier", "title"]
    provider_family: str
    provider_name: str
    model: str
    inherited: bool = False
```

`inherited=True` means the slot should use the parent/main provider unless a
caller overrides it.

### Resolution Order

For each slot:

1. explicit caller override, if the tool/API has one
2. explicit `EngineConfig`/config file slot value
3. slot-specific env vars
4. `AETHER_PROVIDER` family default
5. hardcoded safe default

For subagent `task(..., model="...")`, caller override wins. For
`model="inherit"`, do not apply slot defaults.

### Subagent Migration

Current logic in `aether/tools/builtins/agent_tool.py` should be replaced:

- remove provider/model env parsing from the tool
- call slot resolver for `subagent`
- write `provider_override` and `model_override` into `SubagentTask.metadata`
- keep direct test contexts from accidentally making real provider calls

The last point is important: direct async tests may not have full engine config.
The resolver should be able to return "inherit" when there is no engine runtime
context and no explicit env/config.

### Compression Slot

Do not wire compression provider switching unless Sprint 16 has a stable
compression lifecycle. It is enough for PR17.4 to define the slot and test
resolution. Actual compression usage can be a follow-up.

### Env Names

Use explicit names:

- `AETHER_AUX_SUBAGENT_PROVIDER`
- `AETHER_AUX_SUBAGENT_MODEL`
- `AETHER_AUX_COMPRESSION_PROVIDER`
- `AETHER_AUX_COMPRESSION_MODEL`
- `AETHER_AUX_VERIFIER_PROVIDER`
- `AETHER_AUX_VERIFIER_MODEL`

If older env names exist from prior experiments, support them as deprecated
aliases with tests and a migration note.

### Tests

Cover:

- default subagent slot
- env subagent slot provider/model
- caller `model="gpt"` wins
- caller `model="inherit"` disables default override
- invalid slot provider returns clear error
- direct async subagent test context does not use env default unless engine
  config says to
- compression slot resolves but is not used by engine yet

### Review Checklist

- No provider/model parsing remains in `AgentTool` except caller argument
  validation.
- Slot resolver has no network calls.
- Secrets are not part of slot config.
