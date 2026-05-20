# Sprint 18 - Acceptance Matrix

| Scenario | 18.1 Capabilities | 18.2 Scheduler | 18.3 Interrupt/Permission | 18.4 Dispatch | 18.5 Acceptance |
|---|---|---|---|---|---|
| Multiple reads | safe | parallel | cancellable | integrated | regression pass |
| Read + write | mixed unsafe | sequential | permission stable | integrated | regression pass |
| Two writes same path | unsafe/path overlap | sequential | permission stable | integrated | regression pass |
| Unknown external tool | unsafe default | sequential | unchanged | integrated | regression pass |
| Plan mode write | blocked | not scheduled | no prompt | integrated | regression pass |
| Interrupt running tools | classified | workers stop | deterministic errors | transcript stable | regression pass |
| Post-tool hooks | unchanged | ordered results | no double fire | ordered hooks | regression pass |

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

## Non-Regression Rules

- Unknown tools are sequential.
- Shell is sequential.
- Browser/LSP tools are sequential until thread safety is proven.
- Tool result messages preserve original tool-call order.
- ToolDispatchController remains the owner of post hooks and transcript append.
