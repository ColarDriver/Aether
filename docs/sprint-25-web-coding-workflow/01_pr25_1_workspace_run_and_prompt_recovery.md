# PR25.1 - Workspace Run And Prompt Recovery

## Objective

Make the web console run against the selected workspace root and recover live
prompt state predictably. A user should be able to switch repositories, start or
resume a session, send a run, reload the browser, and understand whether a
pending permission/approval/question is active, replayable, or stale.

## Current State

Already present or partially implemented:

- workspace browsing, search, preview, editing, git status, and checkpoints
  exist under a server-side workspace root;
- web sessions have partial CWD/session-state plumbing;
- pending permission and approval prompts can be replayed after same-process
  browser WebSocket reconnect;
- frontend prompt state queues permission/approval prompts by `prompt_id` and
  deduplicates replayed prompt frames.

Main gaps:

- the active workspace root must be visible, switchable, remembered, and used by
  `EngineRequest.cwd`;
- composer workspace references, file preview/editing, git/checkpoint routes,
  and run CWD must all point at the same root;
- browser reload should restore active prompts when the backend process still
  owns them;
- backend restart must mark old prompts as stale instead of leaving active
  approve/reject buttons that cannot resolve anything.

## Backend Scope

Primary files:

- `aether/services/workspace/contracts.py`
- `aether/services/workspace/service.py`
- `aether/web/routes/workspace.py`
- `aether/web/app.py`
- `aether/services/runs/contracts.py`
- `aether/services/runs/builder.py`
- `aether/services/runs/service.py`
- `aether/services/sessions/contracts.py`
- `aether/services/sessions/service.py`
- `aether/web/routes/sessions.py`
- `aether/web/ws/runs.py`
- `aether/web/ws/prompts.py`
- `aether/runtime/session/session_state.py`

Workspace/root contracts:

- `GET /api/workspace/root`
  - returns `root`, `name`, `exists`, `readable`, `git_root`, `is_git`, and
    `recent_roots`;
  - includes enough metadata for the web header/rail to explain non-git roots.
- `PUT /api/workspace/root`
  - accepts `path`, optional `session_id`, and optional `remember`;
  - validates that the path exists, is a directory, and is readable;
  - normalizes symlinks before storing;
  - persists `workspace.active_root` and deduped capped `workspace.recent_roots`
    when `remember` is true;
  - updates the current session CWD when `session_id` is supplied;
  - never lets workspace file routes escape the active root.
- app startup
  - constructs `WorkspaceService(root=remembered_active_root)` when the pref is
    valid;
  - falls back to the project/default root with a structured warning when the
    remembered root is invalid.
- run creation
  - resolves CWD in this order: explicit request CWD, tracked session CWD,
    active workspace root, process CWD fallback;
  - sets `EngineRequest.cwd`;
  - returns normalized `cwd` and `workspace_root` in `run.accepted`.

Prompt/recovery contracts:

- durable prompt record shape:
  - `prompt_id`, `run_id`, `session_id`, `kind`, `frame`, `created_at`,
    `expires_at`, `status`, `resolution`, and `process_instance_id`;
  - statuses: `pending`, `resolved`, `stale`, `expired`, `disconnected`.
- prompt broker writes a pending record before sending the browser frame.
- resolve, timeout, disconnect, and `reject_run` update the durable record.
- startup cleanup marks prompts owned by a dead process as `stale` or `expired`
  unless a durable run backend can actually resume the provider call.
- WebSocket replay sends only prompts resolvable by the current broker.
- resolving an unknown prompt returns `prompt.missing`; resolving a stale prompt
  returns `prompt.stale` with a reason.

## Frontend Scope

Primary files:

- `web/src/api/types.ts`
- `web/src/api/client.ts`
- `web/src/api/runSocket.ts`
- `web/src/stores/chatStore.ts`
- `web/src/components/chat/WorkspaceRail.tsx`
- `web/src/components/chat/WorkspaceFilePanel.tsx`
- `web/src/components/chat/ChatWorkbenchHeader.tsx`
- `web/src/components/chat/Composer.tsx`
- `web/src/components/chat/WorkspaceReferencePopover.tsx`
- `web/src/components/chat/PermissionDialog.tsx`
- `web/src/components/chat/ApprovalDialog.tsx`
- `web/src/components/chat/AskUserQuestionBlock.tsx`
- `web/src/components/chat/PromptContent.tsx`
- `web/src/components/settings/WorkspaceView.tsx`

Required UI:

- visible workspace root control in the header or workspace rail;
- current root, recent roots, validation errors, and manual server-local path
  entry;
- refresh workspace tree/search/git/checkpoints after root switch;
- clear or mark stale selected workspace references that are invalid in the new
  root;
- new sessions inherit the active root unless explicitly overridden;
- resumed sessions show their tracked CWD when it differs from the global active
  root;
- active prompts show normal action buttons;
- stale prompts render historically with disabled actions and clear stale text;
- duplicate prompt frames update existing prompt blocks instead of adding
  duplicates;
- reload with an active backend run restores the modal; reload after backend
  restart shows stale state and instructs the user to rerun or inspect the
  transcript.

## Tests

Python:

- root info returns valid metadata;
- root switch normalizes paths, persists prefs, and updates session CWD;
- invalid paths, files, and unreadable dirs return structured errors;
- app startup uses remembered root and falls back safely;
- run WebSocket acceptance frame includes selected root CWD;
- `AgentRunService` receives `EngineRequest.cwd` from request/session/root;
- prompt broker persists before send and updates on resolve/reject/timeout;
- restart cleanup marks orphaned records stale;
- stale prompt resolution returns an explicit frame/error.

TypeScript:

- API client methods for root read/switch and prompt-state frames;
- workspace root menu renders active/recent roots and errors;
- root switch refreshes rail/search/file state;
- composer invalidates stale workspace references after root switch;
- run socket deduplicates replayed prompts;
- active prompt modal survives reconnect;
- stale prompt blocks render disabled actions.

Browser/manual:

- switch to a fixture repo and verify rail files, git status, and run CWD;
- attach a workspace file, switch root, and verify stale attachment protection;
- start a permission prompt, reload browser, approve successfully;
- start a permission prompt, restart backend, verify stale prompt UI;
- open two tabs and verify only one prompt resolution wins.

## Acceptance

- A selected server-local repository drives browsing, file preview/editing, git
  status, checkpoints, composer references, and model run CWD.
- Refreshing the browser preserves the remembered root.
- Same-process prompt reconnect works.
- Backend restart never leaves an unresolvable active prompt.
- Historical prompt states distinguish approved, rejected, missing, stale, and
  expired.

## Current Implementation Notes

Implemented on the active branch:

- `GET /api/workspace/root` and `PUT /api/workspace/root` expose and switch the
  active workspace root.
- `create_app` reads `workspace.active_root` from prefs and falls back safely.
- `AgentRunRequest.cwd` flows through `RunDependencyBuilder` into
  `EngineRequest.cwd`.
- web `run.start` uses explicit CWD, tracked session CWD, or active workspace
  root and returns `cwd` / `workspace_root` in `run.accepted`.
- new/resumed web sessions seed session CWD from the active workspace root when
  needed.
- `WorkspaceRail` shows active root, recent roots, manual root switching, git
  status, and reloads tree/git/checkpoint state after switching.
- `Composer` reloads project context and clears stale workspace references after
  workspace root changes.

Remaining work in this PR is durable prompt status beyond same-process replay.

## Explicit Exclusions

- Full durable provider-stream resumption unless the run runtime already
  supports it.
- Remote clone/auth flows.
- Git commit/push/branch mutation UI.
- Multi-root workspace support.
- Cloud workspace provisioning.
