# Sprint 13 — Acceptance Matrix

场景 × PR 二维矩阵。每行是一个端到端行为，每列记录该 PR 完成后应保持或新增的验证点。

## E2E Matrix

| # | 场景 | 13.1 Metadata | 13.2 Provider | 13.3 Context | 13.4 Tool | 13.5 Recovery | 13.6 Lifecycle | 13.7 Acceptance |
|---|---|---|---|---|---|---|---|---|
| E1 | 普通文本回复 | `turn` snapshot 无 internal refs | provider controller 返回 response | context assembly 不改 messages | — | — | result building 不变 | full run 通过 |
| E2 | read-only tool call | tool metadata snapshot 正常 | provider 返回 tool_calls | prepared messages 顺序不变 | registry dispatch 成功 | — | turn finalize 不变 | tool integration 通过 |
| E3 | dangerous shell permission approve | permission metadata 不泄露 live object | — | — | permission approve 后 dispatch | — | session rule 保持 | TUI permission 手测通过 |
| E4 | dangerous shell permission reject | reject metadata 稳定 | — | — | reject 生成 error ToolResult | — | loop 继续 | reject 回归通过 |
| E5 | plan mode blocker | `plan_mode_active` 可观察 | — | plan reminder 仍注入 | blocker 在 permission 前触发 | — | mode lifecycle 不变 | `/plan` 手测通过 |
| E6 | `exit_plan_mode` | approval refs 不泄露 | — | plan attachment 不回归 | 不被 write blocker 拦截 | — | approve/reject mode 不变 | plan approval 通过 |
| E7 | provider invalid response | retry counters 初始化 | validation 进入 error | — | — | response invalid recovery | result exit reason 不变 | recovery test 通过 |
| E8 | provider 429 rate limit | metadata trail 保留 | provider error 包装 | — | — | fallback / RATE_LIMITED 不变 | finalize metadata 不变 | rate-limit test 通过 |
| E9 | empty response | counters 不泄露 | empty response 标准化 | — | — | empty recovery 不变 | exit reason 不变 | empty tests 通过 |
| E10 | length continuation | continuation keys internal | provider retry 可调用 | override messages 保持 | — | continuation 拼接不变 | result 不变 | continuation tests 通过 |
| E11 | truncated tool call | retry counters internal | provider retry 可调用 | broken msg 不进 canonical | — | truncated retry 不变 | finalize 不变 | retry tests 通过 |
| E12 | invalid JSON tool args | invalid_json counter internal | — | — | tool error injection shape 不变 | retry 超限行为不变 | result 不变 | JSON recovery 通过 |
| E13 | context overflow compaction | compaction metadata stable | provider error 包装 | preflight path 保持 | — | reactive compaction 不变 | result exit reason 不变 | compaction tests 通过 |
| E14 | diagnostics after edit | diagnostic refs internal | — | `<diagnostics>` 注入顺序不变 | edited_paths metadata 保持 | — | next turn lifecycle 不变 | diagnostics tests 通过 |
| E15 | verifier reminder | verifier flags stable | — | reminder 注入顺序不变 | task dispatch 可用 | — | session flags 不变 | verifier tests 通过 |
| E16 | memory injection | memory metadata stable | — | memory 不污染 canonical messages | — | — | session memory state 不变 | memory tests 通过 |
| E17 | skill nudge | skill flags stable | — | skill nudge 顺序不变 | `skill` tool 可用 | — | nudge counters per-session | skill tests 通过 |
| E18 | subagent sync task | task refs internal | child provider 可调用 | parent context 不污染 child | task tool dispatch 正常 | child recovery 不变 | child session lifecycle 不变 | subagent tests 通过 |
| E19 | subagent async notification | task metadata stable | child provider 可调用 | notification drain 顺序不变 | task_output / send_message 正常 | — | root engine queue 不变 | async tests 通过 |
| E20 | interrupt during LLM call | interrupt refs internal | controller reports interrupted | — | — | recovery 不吞 interrupt | finalize INTERRUPTED | interrupt tests 通过 |
| E21 | interrupt during tool call | cleanup metadata stable | — | — | dispatch observes interrupt path | — | resource cleanup executes | interrupt tests 通过 |
| E22 | middleware before_llm throws | error metadata stable | — | assembly returns MIDDLEWARE_ERROR path | — | — | result failed 不变 | middleware tests 通过 |
| E23 | after_tool middleware throws | metadata stable | — | — | dispatch failure path 不变 | — | result failed 不变 | middleware tests 通过 |
| E24 | session resume | session metadata stable | — | restored messages preserved | — | — | stored prompt / mode 不变 | resume tests 通过 |
| E25 | task resource cleanup | cleanup metadata stable | — | — | — | — | cleanup hook runs | cleanup tests 通过 |

## Unit Test Map

| 文件 | 覆盖 PR | 目的 |
|---|---|---|
| `aether/tests/runtime/test_turn_metadata.py` | 13.1 | metadata key grouping、snapshot、internal refs filtering |
| `aether/tests/agents/test_engine_metadata_contract.py` | 13.1 | EngineResult metadata backward compatibility |
| `aether/tests/agents/runtime/test_provider_invocation_controller.py` | 13.2 | provider call、hooks、usage、validation |
| `aether/tests/agents/runtime/test_context_assembly_pipeline.py` | 13.3 | PRE_LLM 注入顺序和 canonical/prepared 分离 |
| `aether/tests/agents/runtime/test_tool_dispatch_controller.py` | 13.4 | permission、plan blocker、dedup、hooks、registry dispatch |
| `aether/tests/agents/runtime/test_recovery_controller.py` | 13.5 | provider error、empty、length、context overflow、fallback |
| `aether/tests/agents/runtime/test_session_lifecycle.py` | 13.6 | session start/end、system prompt、cwd、cleanup |
| `aether/tests/agents/test_agent_engine_facade_compat.py` | 13.6 | public API compatibility |
| `aether/tests/agents/runtime/test_sprint13_acceptance.py` | 13.7 | controller stack smoke、stable metadata、tool message shape |
| existing `aether/tests/agents/**` | 13.7 | full engine regression |
| existing `aether/tests/tools/**` | 13.7 | tool behavior regression |
| existing `aether/tests/runtime/**` | 13.7 | runtime behavior regression |
| existing `aether/tests/subagents/**` | 13.7 | subagent behavior regression |

## Automated Verification

- `python -m pytest aether/tests`
- `uv run pyright aether/agents/runtime aether/agents/core/agent.py aether/tools/builtins/agent_tool.py aether/tests/agents/runtime/test_provider_invocation_controller.py aether/tests/agents/runtime/test_recovery_controller.py aether/tests/agents/runtime/test_sprint13_acceptance.py aether/tests/runtime/test_turn_metadata.py aether/tests/agents/test_engine_metadata_contract.py aether/tests/agents/test_agent_engine_facade_compat.py`
- Repository-wide `uv run pyright` is still a tracked follow-up baseline task: as of PR13.7 it reports existing non-Sprint-13 issues such as unresolved optional imports, sandbox prototype imports, provider optional typing, and test stub typing.

## Manual Checklist

- `uv run aether` 能启动并普通对话。
- read-only tool 正常。
- dangerous tool permission approve / reject 正常。
- `/plan` 进入 plan mode，写工具 blocker 正常。
- `exit_plan_mode` approval 正常。
- edit 后 diagnostics 下一轮出现。
- subagent sync / async 正常。
- interrupt 不导致 hanging child。
- 长上下文或 mock overflow 下 compaction/recovery 不回归。

## Non-Regression Rules

- 不改 `ModelProvider.generate()`。
- 不改 `ToolRegistry.dispatch()`。
- 不改 gateway JSON-RPC schema。
- 不改 TUI event schema。
- 不改 plan mode 文案和行为，除非单独测试确认必须修 bug。
- 不将 live object 写入 public metadata snapshot。
- 不把 provider-bound projection 写回 canonical transcript。
