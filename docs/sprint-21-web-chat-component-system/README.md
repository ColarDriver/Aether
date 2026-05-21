# Sprint 21 - Web Chat Component System

Sprint 21 defines and implements Aether's browser-native chat rendering
foundation.

The key decision is that Aether web should not build an Ink clone. It should own
a TypeScript `ChatBlock` contract, normalization layer, render model, and React
DOM component set for agent chat.

## PRs

1. `01_pr21_1_chat_render_contract_and_package_boundary.md`
2. `02_pr21_2_event_normalization_and_streaming_state.md`
3. `03_pr21_3_message_primitives_user_assistant_thinking_markdown.md`
4. `04_pr21_4_tool_result_diff_code_blocks.md`
5. `05_pr21_5_permission_approval_question_components.md`
6. `06_pr21_6_chat_timeline_integration_and_scroll.md`
7. `07_pr21_7_tests_visual_acceptance_and_regression.md`

## Required Reading

- `00_overview.md`
- `99_acceptance_matrix.md`

## Reference Implementations

- cc-haha:
  - `desktop/src/types/chat.ts`
  - `desktop/src/components/chat/MessageList.tsx`
  - `desktop/src/components/chat/AssistantMessage.tsx`
  - `desktop/src/components/chat/ThinkingBlock.tsx`
  - `desktop/src/components/chat/ToolCallBlock.tsx`
  - `desktop/src/components/chat/ToolResultBlock.tsx`
  - `desktop/src/components/chat/DiffViewer.tsx`
  - `desktop/src/components/chat/AskUserQuestion.tsx`
  - `desktop/src/components/chat/StreamingIndicator.tsx`
- Aether current web:
  - `web/src/components/chat/*`
  - `web/src/stores/chatStore.ts`
  - `web/src/api/runSocket.ts`
  - `aether/services/runs/events.py`
  - `aether/web/ws/events.py`

## Completion Rule

The sprint is complete only when the browser chat surface is driven by one typed
timeline model and every target block kind has normalization plus render/test
coverage.
