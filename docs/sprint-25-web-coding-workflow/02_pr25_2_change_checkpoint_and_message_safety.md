# PR25.2 - Change Checkpoint And Message Safety

## Objective

Turn web coding changes into a safe workflow. Users should be able to inspect
changed files, accept or reject supported changes, detect conflicts before
restore, verify changes, and use retry/rewind/fork/undo actions without silently
leaving the workspace in an inconsistent state.

## Current State

Already present or partially implemented:

- `DiffViewer.tsx`, `DiffBlock.tsx`, and `CurrentTurnChangeCard.tsx` render
  diffs and changed-file summaries;
- diagnostics and same-turn shell/LSP verification bundles can be shown;
- per-file restore can use pre-run workspace checkpoints and git fallback;
- session fork and rewind routes exist;
- workspace checkpoint creation/list/restore routes exist;
- the run WebSocket can create an opt-in pre-run workspace checkpoint;
- web message actions include copy, quote, user edit-to-composer, retry,
  assistant retry, fork, and rewind.

Main gaps:

- restore is not a complete accept/reject workflow;
- conflict detection is shallow;
- verification follow-up actions are not first-class;
- message actions are transcript-aware but not fully workspace-state-aware;
- undo/replay state feedback is too limited for coding sessions.

## Backend Scope

Primary files:

- `aether/services/workspace/contracts.py`
- `aether/services/workspace/service.py`
- `aether/web/routes/workspace.py`
- `aether/services/sessions/contracts.py`
- `aether/services/sessions/service.py`
- `aether/web/routes/sessions.py`
- `aether/services/runs/contracts.py`
- `aether/services/runs/service.py`

Change-management contracts:

- `GET /api/workspace/changes`
  - returns git/checkpoint-derived changes with status, paths, hunks,
    staged/untracked state, checkpoint availability, and conflict risk.
- `POST /api/workspace/changes/accept`
  - accepts file paths or change IDs;
  - records that the current content is user accepted;
  - does not mutate content unless later staging support is explicitly added.
- `POST /api/workspace/changes/reject`
  - rejects paths by restoring from a pre-run checkpoint when available;
  - falls back to git restore only for safe tracked modifications;
  - refuses untracked/delete/rename/binary cases without explicit support.
- `POST /api/workspace/changes/verify`
  - executes a known verification command or enqueues a verifier task when the
    backend supports it.

Message-action contracts:

- `GET /api/sessions/{session_id}/message-actions/{index}`
  - returns supported actions and reasons for unsupported actions.
- `POST /api/sessions/{session_id}/actions/retry`
  - retries from a message index;
  - optionally restores the relevant pre-run checkpoint first;
  - returns new run/session metadata.
- `POST /api/sessions/{session_id}/actions/rewind`
  - rewinds transcript and optionally restores workspace state.
- `POST /api/sessions/{session_id}/actions/fork`
  - forks transcript and preserves checkpoint metadata without mutating the
    source session.
- `POST /api/sessions/{session_id}/actions/undo-run`
  - restores workspace to the checkpoint captured before a selected run and
    rewinds transcript after that run.

Safety rules:

- compare current file hash against the hash captured when the diff/change card
  was rendered; return `409 conflict` before overwriting changed content;
- return structured `checkpoint_missing` when restore data is unavailable;
- reject paths outside the active workspace;
- show transcript-only actions when no checkpoint exists;
- never restore workspace content as a side effect of quote/edit;
- never replay a message into the wrong workspace root.

Change data model:

- `change_id`, `path`, `old_path`, `status`, `source`, `accepted`, `rejected`,
  `conflict`, `verification_status`, `last_verified_at`, `commands`,
  `checkpoint_id`, `before_hash`, `current_hash`.

## Frontend Scope

Primary files:

- `web/src/api/types.ts`
- `web/src/api/client.ts`
- `web/src/stores/chatStore.ts`
- `web/src/components/chat/MessageActionBar.tsx`
- `web/src/components/chat/ChatTimeline.tsx`
- `web/src/components/chat/ChatView.tsx`
- `web/src/components/chat/DiffViewer.tsx`
- `web/src/components/chat/blocks/DiffBlock.tsx`
- `web/src/components/chat/blocks/CurrentTurnChangeCard.tsx`
- `web/src/components/chat/blocks/DiagnosticsBlock.tsx`
- `web/src/components/shared/ConfirmDialog.tsx`
- `web/src/styles.css`

Required UI:

- per-file action row: copy path, open preview, accept, reject, verify;
- accepted/rejected/conflict states remain visible in timeline history;
- conflict dialog explains why restore is blocked and offers copy path, open
  current file, and view current diff;
- rejected files update the card without removing the historical context;
- verification action appends a visible verification result or task link;
- message action availability comes from backend when possible;
- distinct labels for retry only, retry from checkpoint, rewind transcript only,
  rewind and restore workspace, fork transcript only, fork with checkpoint
  metadata, and undo run;
- destructive workspace restore always asks for confirmation;
- result notices include session ID, checkpoint ID, and run ID when relevant;
- disabled actions show reasons instead of disappearing.

Visual rules:

- diff rows stay aligned with adjacent timeline cards;
- red/green backgrounds extend across full changed rows;
- code text, line numbers, plus/minus markers, and actions do not reflow code on
  narrow screens;
- conflict and confirmation dialogs use the existing modal style, not browser
  default dialogs.

## Tests

Python:

- changes endpoint classifies modified/created/deleted/untracked/binary files;
- accept records state without content mutation;
- reject restores a modified file from checkpoint;
- reject refuses when current hash changed after the diff snapshot;
- git fallback works only for safe tracked modifications;
- message-action capability endpoint returns supported actions and reasons;
- undo-run restores checkpoint and rewinds transcript;
- retry from checkpoint starts a new run with restored CWD/workspace metadata;
- fork preserves checkpoint metadata and leaves the source session unchanged.

TypeScript:

- change card renders accept/reject/verify actions for supported files;
- accepted/rejected/conflict state persists in the rendered card;
- conflict responses render actionable UI;
- rejected file action reloads workspace preview if the file is open;
- message action menu renders backend-supported actions;
- destructive restore asks for confirmation;
- transcript-only actions are clearly labeled.

Browser/manual:

- generate a run with two file changes, accept one and reject one;
- verify the accepted file remains and the rejected file is restored;
- edit a file before reject and verify conflict prevents overwrite;
- undo a checkpointed run and verify file content plus transcript state;
- retry from checkpoint and verify the new run starts from restored state;
- fork a previous message and verify the original session remains unchanged.

## Acceptance

- Users can reject a specific changed file without losing unrelated changes.
- Users can accept a file and see that acceptance recorded in UI.
- Conflicts are detected before content restore.
- Verification output is attached to the change workflow.
- Retry, rewind, fork, and undo never imply checkpoint-backed behavior when no
  checkpoint exists.

## Explicit Exclusions

- Hunk-level accept/reject.
- Git commit creation.
- Multi-branch merge conflict resolution.
- Distributed checkpoint storage.
- Cross-repository checkpoint restore.
