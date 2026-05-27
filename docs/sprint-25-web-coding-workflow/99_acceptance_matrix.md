# Sprint 25 Acceptance Matrix

## Required Verification By PR

| PR | Backend checks | Frontend checks | Browser/manual checks |
| --- | --- | --- | --- |
| PR25.1 Workspace Run And Prompt Recovery | root switch, session CWD, run CWD, prompt persistence, stale cleanup | root UI, composer invalidation, prompt queues, stale prompts | switch repo, run against repo, reload active prompt, restart stale prompt |
| PR25.2 Change Checkpoint And Message Safety | changes API, accept/reject, conflict, checkpoint message actions | change card actions, conflict modal, message action labels | accept/reject fixture, undo run, retry from checkpoint |
| PR25.3 Task Artifact And A2A Console | task detail, hierarchy, artifact safety, stop/follow-up | task tree, child streams, artifact panel, action feedback | task hierarchy, stop, follow-up, text/image/binary artifacts |
| PR25.4 Context MCP And Provider Controls | context estimate/status/compress, provider usage/preflight, MCP validation | context ring/inspector, provider status, MCP editor/resource browser | large attachment pressure, compression, bad MCP URL, resource read |
| PR25.5 Tool Preview And Live Acceptance | provider/tool fixtures where backend metadata exists | renderer fixtures, unsafe URL/path handling, streaming stability | screenshots, live streaming, provider error, permission/approval, task run |

## Sprint-Level Acceptance

- The selected workspace root is the single source for file browsing, previews,
  attachments, git/checkpoints, and run CWD.
- Permission, approval, and ask-user-question prompts are never left as active
  controls when the backend can no longer resolve them.
- Changed files can be inspected and safely accepted/rejected where the backend
  can guarantee correctness.
- Checkpoint-backed message actions are clearly distinguished from
  transcript-only actions.
- Subagent/task history, actions, and artifacts are inspectable from web without
  reading runtime files directly.
- Context pressure, compression state, provider readiness, and MCP readiness are
  visible before users rely on a run.
- Known tool/provider outputs render as structured previews; unknown outputs
  remain readable through fallback views.
- Browser acceptance covers narrow layouts, long outputs, streaming updates,
  permissions, approvals, prompts, workspace operations, task artifacts,
  provider errors, and screenshot baselines.

## Commands

Use focused tests for each PR while developing, then run this before declaring
the sprint complete:

```bash
cd web && npm run typecheck
cd web && npm test
cd web && npm run test:e2e
python -m pytest aether/tests/web aether/tests/services -q
git diff --check
```
