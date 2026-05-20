# Sprint 18 - Acceptance Matrix

| Scenario | 18.1 Capabilities | 18.2 Scheduler | 18.3 Interrupt/Permission | 18.4 Dispatch | 18.5 Acceptance |
|---|---|---|---|---|---|
| Multiple reads | safe | parallel | cancellable | integrated behind flag | passed |
| Read + write | mixed unsafe | sequential | permission stable | integrated fallback | passed |
| Two writes same path | unsafe/path overlap | sequential | permission stable | integrated fallback | passed |
| Unknown external tool | unsafe default | sequential | unchanged | integrated fallback | passed |
| Plan mode write | blocked | not scheduled | no prompt | integrated fallback | passed |
| Interrupt running tools | classified | pending workers cancel | deterministic errors | transcript stable | passed |
| Post-tool hooks | unchanged | ordered results | no double fire | ordered hooks | passed |

## Unit Test Map

| File | Purpose |
|---|---|
| `aether/tests/runtime/tools/test_capabilities.py` | tool safety classification |
| `aether/tests/runtime/tools/test_parallel_scheduler.py` | planning, path overlap, ordered results |
| `aether/tests/runtime/tools/test_parallel_scheduler_interrupt.py` | interrupt and cleanup behavior |
| `aether/tests/agents/runtime/test_tool_dispatch_controller.py` | dispatch integration |
| existing `aether/tests/engine/test_tool_permission_gate.py` | permission parity |
| existing `aether/tests/engine/test_interrupt_message_injection.py` | interrupt parity |
| existing `aether/tests/tools/**` | built-in tool regression |

## Manual Checklist

- Multiple independent reads run as one parallel batch.
- Read plus write falls back to sequential.
- Permission prompt still appears once and blocks execution until answered.
- Plan mode write blocker fires before scheduling.
- Interrupt during slow tools returns a coherent transcript.

## Automated Verification

Completed on this branch:

- `python -m pytest aether/tests/runtime/tools` - 27 passed.
- `python -m pytest aether/tests/agents/runtime/test_tool_dispatch_controller.py` - 12 passed.
- `python -m pytest aether/tests/engine/test_tool_permission_gate.py` - 12 passed.
- `python -m pytest aether/tests/engine/test_interrupt_message_injection.py` - 2 passed.
- `python -m pytest aether/tests/tools` - 341 passed.
- `python -m pytest aether/tests/engine` - 169 passed.
- `python -m pytest aether/tests/agents` - 138 passed.
- `uv run pyright aether/runtime/tools aether/agents/runtime/tool_dispatch.py aether/tools` - 0 errors.

## Default-On Decision

`EngineConfig.parallel_tool_execution_enabled` remains `False` by default for
this sprint. The implementation is fully wired and regression-covered, but
keeping the first integration opt-in preserves existing operator behavior while
read-only tool thread-safety is validated in real TUI sessions.

## Non-Regression Rules

- Unknown tools are sequential.
- Shell is sequential.
- Browser/LSP tools are sequential until thread safety is proven.
- Tool result messages preserve original tool-call order.
- ToolDispatchController remains the owner of post hooks and transcript append.
