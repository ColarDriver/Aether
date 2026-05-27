# Sprint 25 - Web Coding Workflow

This sprint is one implementation folder for the remaining web coding workflow
work. It is not a broad "migration" bucket and it is not split into unrelated
mini-sprints. The PR files below group the required work by the user-visible
workflow they complete.

The target is an Aether web console that can run coding sessions against the
right workspace, survive prompt/reconnect edge cases, manage changed files and
checkpoints safely, inspect task/subagent artifacts, expose context/MCP/provider
state, and render live tool output with enough visual acceptance to trust it.

## PR Files

1. `01_pr25_1_workspace_run_and_prompt_recovery.md`
2. `02_pr25_2_change_checkpoint_and_message_safety.md`
3. `03_pr25_3_task_artifact_and_a2a_console.md`
4. `04_pr25_4_context_mcp_and_provider_controls.md`
5. `05_pr25_5_tool_preview_and_live_acceptance.md`
6. `99_acceptance_matrix.md`

## Former Plan Mapping

The earlier nine-part outline is intentionally collapsed here:

- workspace root and repository launch controls, plus durable prompt/run
  recovery, are handled by PR25.1 because both define whether a live run can be
  resumed and controlled correctly.
- change management and checkpoint-backed message actions are handled by
  PR25.2 because both are workspace-state safety problems.
- task, artifact, and A2A controls stay together in PR25.3 because they are one
  subagent inspection workflow.
- context budget, compaction visibility, MCP credentials/resources, and provider
  controls are handled by PR25.4 because they are runtime configuration and
  readiness surfaces.
- tool preview edge cases and visual/live-provider acceptance are handled by
  PR25.5 because renderer breadth only matters if it is verified in real web
  flows.

## Execution Order

1. Land workspace/run context and prompt recovery first. Every later workflow
   depends on the active root, session CWD, and resolvable live prompts being
   correct.
2. Land change/checkpoint/message safety next. This protects user files before
   adding richer coding actions.
3. Land task/artifact/A2A depth. Subagent work becomes inspectable and
   interruptible from web.
4. Land context/MCP/provider controls. The web console should show whether the
   runtime is ready before a run fails.
5. Land tool preview and live acceptance. This locks down rendering, browser
   behavior, and provider/manual workflows.

## Non-Goals

- No UI-only buttons for behavior that the backend cannot execute safely.
- No claim of completion based only on static rendering tests.
- No broad "migration complete" label. The completion bar is concrete coding
  workflow behavior that is connected, recoverable, and tested.
- No cloud workspace provisioning, hosted MCP marketplace, or remote
  collaborative prompt queue in this sprint.
