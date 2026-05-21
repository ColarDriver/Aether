# Sprint 21 - Acceptance Matrix

| # | Scenario | 21.1 | 21.2 | 21.3 | 21.4 | 21.5 | 21.6 | 21.7 |
|---|---|---|---|---|---|---|---|---|
| E1 | Aether-owned TS renderer contract | block union | reducer types | component props | tool types | prompt types | integrated | tests |
| E2 | No Ink/web terminal dependency | boundary note | event model | DOM renderer | DOM diff | DOM prompts | DOM timeline | regression |
| E3 | Persisted transcript parity | helper types | transcript normalize | message render | tool/result render | prompt history | timeline | tests |
| E4 | Live streaming parity | block states | frame reducer | assistant/thinking | tool events | prompt events | timeline status | tests |
| E5 | User messages | type | normalize | component | order with tools | prompt context | timeline | visual |
| E6 | Assistant messages | type | deltas | markdown/caret | order with tools | prompt context | timeline | visual |
| E7 | Thinking/reasoning | type | reasoning.delta | component | grouped order | unaffected | status placement | tests |
| E8 | Tool call/result | type | tool events | placeholder | full render model | question escape | no duplicate stack | tests |
| E9 | Diff/code rendering | diff type | metadata extraction | code primitive | full diff viewer | permission preview | timeline alignment | visual |
| E10 | Permission prompts | prompt type | request reducer | base shell | diff preview | actions/state | modal+timeline | tests |
| E11 | Approval prompts | approval type | request reducer | markdown | unaffected | actions/state | modal+timeline | tests |
| E12 | Ask user question | question type | parse/update | base shell | not tool group | full component | timeline | tests |
| E13 | Streaming status | status type | run status reducer | status block | tool-use status | prompt pause | scroll-safe | tests |
| E14 | Scroll stability | n/a | n/a | no forced thinking scroll | expandable blocks | modal focus | auto-scroll rule | manual |
| E15 | Build/test health | type tests | store tests | component tests | component tests | prompt tests | integration | full suite |

## Required Files

| File | Purpose |
|---|---|
| `00_overview.md` | sprint goal, current gaps, cc-haha reference, Ink decision, target coverage |
| `01_pr21_1_chat_render_contract_and_package_boundary.md` | typed TypeScript block contract and React-free package boundary |
| `02_pr21_2_event_normalization_and_streaming_state.md` | transcript and WebSocket frame normalization into blocks |
| `03_pr21_3_message_primitives_user_assistant_thinking_markdown.md` | user/assistant/thinking/system/error/markdown DOM components |
| `04_pr21_4_tool_result_diff_code_blocks.md` | tool grouping, results, diffs, code viewer |
| `05_pr21_5_permission_approval_question_components.md` | permission, approval, and ask_user_question components |
| `06_pr21_6_chat_timeline_integration_and_scroll.md` | ChatView integration and scroll behavior |
| `07_pr21_7_tests_visual_acceptance_and_regression.md` | automated/manual acceptance and regression matrix |
| `99_acceptance_matrix.md` | scenario-to-PR verification map |
| `README.md` | sprint index |

## Implementation Evidence Expected

Before Sprint 21 is considered complete, expect evidence in:

- `web/src/chat-rendering/*`
- `web/src/components/chat/ChatTimeline.tsx`
- `web/src/components/chat/blocks/*`
- `web/src/stores/chatStore.ts`
- `web/src/api/runSocket.ts`
- `web/src/api/types.ts`
- `web/src/styles.css`
- frontend tests under `web/src/**/*.test.ts(x)`

## Final Acceptance

- Aether web renders a full agent turn through a single typed timeline.
- User, assistant, thinking, tool call, tool result, diff, permission, approval,
  ask_user_question, and streaming states are all represented as first-class
  blocks.
- Persisted transcript and live events produce compatible output.
- Prompt decisions use correct backend payloads and leave historical context in
  the chat.
- Streaming and scroll behavior are browser-native and do not emulate terminal
  rendering.
