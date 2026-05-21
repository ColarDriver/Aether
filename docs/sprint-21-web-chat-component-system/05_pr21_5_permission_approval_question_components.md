# PR21.5 - Permission, Approval, and Question Components

## Goal

Make interactive prompts first-class web chat blocks while preserving modal
affordances where they are useful.

Coverage:

- tool permission,
- plan approval,
- generic approval/questions,
- `ask_user_question`,
- prompt resolution state in the timeline.

## Required Files

Add:

```text
web/src/components/chat/blocks/PermissionRequestBlock.tsx
web/src/components/chat/blocks/ApprovalRequestBlock.tsx
web/src/components/chat/blocks/AskUserQuestionBlock.tsx
web/src/components/chat/promptState.ts
```

Modify:

```text
web/src/components/chat/PermissionDialog.tsx
web/src/components/chat/ApprovalDialog.tsx
web/src/stores/chatStore.ts
web/src/api/runSocket.ts
web/src/styles.css
```

## Design Rule

Prompts have two surfaces:

- a timeline block that remains visible after resolution,
- an active modal or focused panel while a decision is pending.

The modal may disappear after response. The timeline block must remain and show
what happened.

## PermissionRequestBlock

Render:

- tool name,
- risk/category,
- reason,
- command/body/diff preview,
- arguments details,
- pending/allowed/denied/expired state,
- allow once,
- allow session when backend says `allow_session`,
- deny.

Decision payloads:

- allow once -> `{ type: 'allow_once' }`
- allow session -> `{ type: 'allow_session' }`
- deny -> `{ type: 'deny' }`

If future UI supports argument edits, use:

```ts
{ type: 'allow_once', updated_arguments: { ... } }
```

Do not add argument editing until the visual and security model is clear.

## ApprovalRequestBlock

Render:

- `approvalKind`,
- plan markdown when present,
- plan path when present,
- pending/approved/rejected/answered state,
- approve/reject buttons for plan approvals,
- question answer controls for question approvals.

Plan approval behavior:

- approve sends `{ confirmed: true }`,
- reject sends `{ confirmed: false }`,
- reject keeps timeline block with rejected state and allows a later revised
  plan block to appear as a new approval request.

Question approval behavior:

- answers are collected into `Record<string, string>`,
- response sends `{ confirmed: true, answers }` unless backend requires
  question-only payload; keep `runSocket` aligned with `aether/web/ws/runs.py`.

## AskUserQuestionBlock

`ask_user_question` is a tool, but it should not feel like a generic tool call.
It is an interaction surface.

Support:

- one or more questions,
- optional headers,
- mutually exclusive options,
- optional free-text answer,
- submitted answer display,
- disabled state after answer,
- keyboard Enter submit while respecting IME composition.

Parsing must support:

- `{ questions: [{ question, header, options }] }`
- `{ question, header, options }`
- existing Aether tool schema fields from `aether.tools.builtins.ask_user_question`.

## Prompt Resolution

Add timeline resolution updates when possible:

- `prompt.resolved` with matching prompt ID updates block state,
- `permission.respond` can optimistically update block state before server echo,
- `approval.respond` can optimistically update approval state before server echo,
- failed send rolls back to pending and shows an error notice.

If the backend does not emit enough resolution data, add a small web event or
client-side optimistic map in this PR. Prefer backend event if it already has a
natural place in `aether/web/ws/runs.py`.

## Modal Compatibility

Keep `PermissionDialog` and `ApprovalDialog` as active overlays if the current
web UX needs them, but refactor them to consume the same typed prompt blocks.

No duplicated rendering logic:

- shared `PermissionPreview` subcomponent,
- shared `PlanApprovalContent` subcomponent,
- shared `QuestionForm` subcomponent.

## Tests

Add tests for:

- permission block renders diff preview and action buttons,
- allow once/session/deny send correct payload,
- resolved permission stays visible with final state,
- approval block renders markdown plan,
- approve/reject send nested `approval.respond.payload.result`,
- question approval sends answers,
- ask_user_question parses both supported input shapes,
- IME composition does not submit early,
- modal and timeline block share preview rendering.

## Acceptance

- Tool permissions, plan approvals, and questions are understandable and
  actionable in the browser.
- Closing/resolving prompts does not erase the historical prompt/diff content.
- Prompt action payloads match the backend protocol exactly.
