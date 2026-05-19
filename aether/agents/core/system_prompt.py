"""System-prompt augmentation utilities used by the engine.

This module owns three prepended sections, assembled in order before
the caller-supplied system prompt:

1. ``<tool_use_contract>`` — registry-derived enumeration of the
   available tools and a hard ban on prose-style tool emission.  The
   single strongest lever against Kimi-class models that *want* to
   call a tool but write the call as a markdown fence.
2. ``<verification_directive>`` — forces the model to verify its
   work (re-read, type-check, grep callers) before reporting a task
   complete.  Parity with
   ``open-claude-code/src/constants/prompts.ts:211``.
3. ``<faithful_reporting>`` — bans defensive hedging and dishonest
   summaries when a verification step fails.  Parity with
   ``open-claude-code/src/constants/prompts.ts:240``.

Each section is independently switchable via :class:`SystemPromptOptions`
so a caller can A/B individual blocks without rewriting strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aether.tools.base import ToolDescriptor


_TOOL_CONTRACT_TEMPLATE = (
    "<tool_use_contract>\n"
    "You have these tools available: {names}.\n"
    "You MUST invoke them via the structured ``tool_calls`` field of your "
    "response. Do NOT write tool calls in markdown code blocks "
    "(```bash...```), ``<function=NAME>{{...}}``, ``<functions.shell:N>{{...}}``, "
    "``<invoke name=...>``, ``<tool_call>``, or any other prose form — "
    "such text will be discarded and the run loop will exit without "
    "executing anything.\n"
    "Common mappings: to run a shell command, call the ``shell`` tool; "
    "to read a file, call ``read_file``; to list a directory, call "
    "``list_dir``; to search file contents, call ``grep``; to find files "
    "by name, call ``glob``; to write a file, call ``write_file``.\n"
    "</tool_use_contract>"
)

# Parity with open-claude-code/src/constants/prompts.ts:211.
_VERIFICATION_DIRECTIVE = (
    "<verification_directive>\n"
    "Before reporting a task complete, verify it actually works: run the "
    "test, execute the script, type-check the module, read back the "
    "edited file. Minimum complexity means no gold-plating — it does NOT "
    "mean skipping the finish line. If you cannot verify (no test exists, "
    "cannot run the code, no language server), say so explicitly rather "
    "than claiming success.\n"
    "Specifically after editing source files, you SHOULD:\n"
    "  1. Re-read the changed file or run a syntax/type check "
    "(``pyright``, ``tsc --noEmit``, ``python -c 'import module'``, "
    "etc.).\n"
    "  2. Search the rest of the codebase for callers of any renamed / "
    "removed symbol via ``grep``; missed call-sites are the #1 source "
    "of regressions.\n"
    "  3. If the editor surfaces a ``<diagnostics>`` block in a "
    "subsequent user turn, treat its contents as authoritative and fix "
    "them before moving on.\n"
    "</verification_directive>"
)

# Parity with open-claude-code/src/constants/prompts.ts:394 (the
# "non-trivial work needs an adversarial verifier" paragraph).  Named
# explicitly so the soft engine-side gate in agent.py and the system
# prompt share one source of truth for the rule.
_VERIFIER_GATE = (
    "<verifier_gate>\n"
    "When non-trivial implementation has happened on your turn — "
    "defined as 3+ files edited, backend/API changes, or "
    "infrastructure changes — you MUST request an independent "
    "verification BEFORE reporting completion.  Spawn the ``task`` "
    "tool with ``subagent_type=\"Verifier\"``.  Pass:\n"
    "  - the original user request,\n"
    "  - the list of files changed (by anyone — you, a fork, or a "
    "subagent),\n"
    "  - the approach you took,\n"
    "  - any plan file path if you authored one.\n"
    "Your own checks, caveats, and a fork's self-checks do NOT "
    "substitute for the verifier's verdict.  On FAIL: fix, resume the "
    "verifier with the fix, repeat until PASS.  On PARTIAL: report "
    "exactly what passed and what could not be verified.  On PASS: "
    "spot-check it — re-run 2–3 commands from its report; if any PASS "
    "lacks a matching command block or diverges from your re-run, "
    "resume the verifier.\n"
    "</verifier_gate>"
)

# Plan-mode reminder.  Modeled on
# ``open-claude-code/src/utils/messages.ts:3227-3292`` (the 5-phase
# workflow attachment) but adapted to Aether's tool surface.
#
# Two structural choices that matter:
#
# 1. The reminder is injected as a *user-role* ``<system-reminder>``
#    message just before the model call, not appended to the system
#    prompt.  Salience next to the user turn is the lever that gets
#    the model to actually obey — see
#    ``_maybe_inject_plan_mode_attachment`` in ``agent.py``.
# 2. The body lays out *what to do*, not just what's forbidden.  The
#    old reminder was a list of "don'ts" and left the model to
#    improvise the workflow; it often improvised "run shell to
#    explore."  This version names the right tool/subagent for each
#    of five phases.
#
# Only ``{plan_path}`` and ``{plan_file_clause}`` vary per session;
# the rest is a stable string so prompt-cache hits keep paying off
# across turns.
_PLAN_MODE_REMINDER_TEMPLATE = (
    "<system-reminder>\n"
    "Plan mode is active. The user indicated that they do not want "
    "you to execute yet -- you MUST NOT make any edits (with the "
    "exception of the plan file mentioned below), run any "
    "non-readonly tools (including changing configs or making "
    "commits), or otherwise make any changes to the system. This "
    "supercedes any other instructions you have received.\n"
    "\n"
    "## Plan File Info:\n"
    "{plan_file_clause}\n"
    "You should build your plan incrementally by writing to or "
    "editing this file. NOTE that this is the only file you are "
    "allowed to edit - other than this you are only allowed to take "
    "READ-ONLY actions.\n"
    "\n"
    "## Plan Workflow\n"
    "\n"
    "### Phase 1: Initial Understanding\n"
    "Goal: Gain a comprehensive understanding of the user's request "
    "by reading through code and asking them questions. In this "
    "phase, dispatch ``Explore`` subagents via the ``task`` tool, or "
    "use the read-only tools (``read_file``, ``list_dir``, ``grep``, "
    "``glob``, ``web_fetch``, ``web_search``) directly. Actively "
    "search for existing functions, utilities, and patterns that can "
    "be reused -- avoid proposing new code when a suitable "
    "implementation already exists.\n"
    "\n"
    "Launch **up to 3 ``Explore`` subagents IN PARALLEL** (single "
    "response, multiple ``task`` calls) when the scope is uncertain "
    "or multiple areas of the codebase are involved. Use 1 agent for "
    "isolated/targeted changes. Quality over quantity -- 3 is the "
    "cap, fewer is usually better.\n"
    "\n"
    "### Phase 2: Design\n"
    "Goal: Design an implementation approach.\n"
    "\n"
    "Launch a ``Plan`` subagent via ``task(subagent_type=\"Plan\", "
    "...)`` to design the implementation given the user's intent and "
    "your Phase 1 findings. Pass it the file paths and code traces "
    "you collected, the requirements/constraints, and a request for "
    "a detailed implementation plan. For truly trivial work (typo, "
    "single-line fix, simple rename) you may skip the Plan subagent.\n"
    "\n"
    "### Phase 3: Review\n"
    "Goal: Review the design from Phase 2 and confirm it matches the "
    "user's intent. Read the critical files the subagent identified, "
    "and use ``ask_user_question`` to clarify any remaining "
    "ambiguity. Do NOT use ``ask_user_question`` to request plan "
    "approval -- that's what ``exit_plan_mode`` is for.\n"
    "\n"
    "### Phase 4: Final Plan\n"
    "Write the final plan to the plan file above using ``write_file`` "
    "(if the file doesn't yet exist) or ``file_edit`` (to revise). "
    "Structure it as:\n"
    "  - **Context** -- why this change is being made\n"
    "  - **Approach** -- the recommended implementation (one approach, "
    "not all alternatives)\n"
    "  - **Files** -- absolute paths of files to be modified, with the "
    "specific changes\n"
    "  - **Reused utilities** -- existing functions to reuse, with "
    "paths\n"
    "  - **Verification** -- how to test end-to-end\n"
    "Keep the file concise enough to scan, detailed enough to "
    "execute. The only file you may write or edit in plan mode is the "
    "plan file above.\n"
    "\n"
    "### Phase 5: Call ``exit_plan_mode``\n"
    "At the very end of your turn, once the plan file is ready, "
    "always call ``exit_plan_mode`` to request user approval. Your "
    "turn must end with either ``ask_user_question`` (to clarify) or "
    "``exit_plan_mode`` (to request approval) -- not with prose. "
    "Phrases like \"Is this plan okay?\", \"Should I proceed?\", "
    "\"How does this plan look?\" MUST be expressed by calling "
    "``exit_plan_mode``, not by asking in text.\n"
    "\n"
    "Reminder: ``read_file``, ``list_dir``, ``grep``, ``glob``, "
    "``write_file``, ``file_edit``, ``shell``, ``task`` are tool "
    "names. Invoke them via structured ``tool_calls``, not by typing "
    "them as shell commands -- the ``shell`` tool is BLOCKED in plan "
    "mode regardless. Calling ``read_file <path>`` through ``shell`` "
    "will fail.\n"
    "</system-reminder>"
)

_PLAN_FILE_CLAUSE_EXISTS = (
    "A plan file already exists at {plan_path}. Read it and make "
    "incremental edits using ``file_edit``."
)
_PLAN_FILE_CLAUSE_MISSING = (
    "No plan file exists yet. Create your plan at {plan_path} using "
    "``write_file``."
)
_PLAN_FILE_CLAUSE_UNKNOWN = (
    "Plan file path is unavailable for this session. Resolve it via "
    "``exit_plan_mode``-side persistence; in the meantime, treat all "
    "writes as forbidden."
)

_PLAN_MODE_REMINDER = _PLAN_MODE_REMINDER_TEMPLATE.format(
    plan_path="(unavailable)",
    plan_file_clause=_PLAN_FILE_CLAUSE_UNKNOWN,
)


def build_plan_mode_reminder(session_id: str) -> str:
    """Return the plan-mode reminder body for *session_id*.

    Body only — does **not** include role wrapping. Callers that need
    a user-role message should use :func:`build_plan_mode_attachment`.
    """
    plan_path = "(unavailable)"
    plan_exists: bool | None = None
    try:
        from aether.runtime.session.plan_artifact import get_plan_path, read_plan
    except Exception:  # pragma: no cover - defensive
        pass
    else:
        try:
            path = get_plan_path(session_id)
        except ValueError:
            pass
        else:
            plan_path = str(path)
            plan_exists = read_plan(session_id) is not None
    if plan_exists is True:
        clause = _PLAN_FILE_CLAUSE_EXISTS.format(plan_path=plan_path)
    elif plan_exists is False:
        clause = _PLAN_FILE_CLAUSE_MISSING.format(plan_path=plan_path)
    else:
        clause = _PLAN_FILE_CLAUSE_UNKNOWN
    return _PLAN_MODE_REMINDER_TEMPLATE.format(
        plan_path=plan_path,
        plan_file_clause=clause,
    )


def build_plan_mode_attachment(session_id: str | None) -> dict[str, object]:
    """Return a user-role message dict carrying the plan-mode reminder.

    The body is wrapped in ``<system-reminder>`` (already part of the
    template) and tagged with ``metadata.source = "plan_mode"`` so
    middleware / observers can identify it. Returned as a plain dict
    so the engine can append it to the outbound message list without
    a model-side type change.
    """
    body = build_plan_mode_reminder(session_id) if session_id else _PLAN_MODE_REMINDER
    return {
        "role": "user",
        "content": body,
        "metadata": {"source": "plan_mode"},
    }


def append_plan_mode_reminder(
    system: str | None,
    *,
    session_id: str | None = None,
) -> str | None:
    """Deprecated: append the plan-mode reminder to *system*.

    Kept for backward compatibility with callers that splice into the
    system prompt. New call sites should use
    :func:`build_plan_mode_attachment` instead — injecting a user-role
    ``<system-reminder>`` message just before the model call gives the
    constraint far higher salience than burying it in the system
    prompt prefix.
    """
    reminder = build_plan_mode_reminder(session_id) if session_id else _PLAN_MODE_REMINDER
    if system and system.strip():
        return f"{system}\n\n{reminder}"
    return reminder


# Parity with open-claude-code/src/constants/prompts.ts:240.
_FAITHFUL_REPORTING = (
    "<faithful_reporting>\n"
    "Report outcomes faithfully. If tests fail, say so with the relevant "
    "output; if you did not run a verification step, say that rather "
    "than implying it succeeded. Never claim ``all tests pass`` when the "
    "output shows failures, never suppress or simplify failing checks "
    "(tests, lints, type errors) to manufacture a green result, and "
    "never characterize incomplete or broken work as done. Equally, "
    "when a check did pass or a task is complete, state it plainly — do "
    "not hedge confirmed results with unnecessary disclaimers, downgrade "
    "finished work to ``partial``, or re-verify things you already "
    "checked. The goal is an accurate report, not a defensive one.\n"
    "</faithful_reporting>"
)


# Subprocess-statelessness contract for the ``shell`` tool.  The
# round-trip in ``ShellTool`` makes CWD persist across calls (so
# ``cd /workspace/foo`` carries forward) but env vars and activated
# venvs do NOT survive a Popen boundary — there's no persistent shell
# process.  The model needs to know this explicitly or it will keep
# trying ``source .venv/bin/activate && python …`` chains that
# silently lose the activation on the next call.
_SHELL_TOOL_CONTRACT = (
    "<shell_tool_contract>\n"
    "The ``shell`` tool launches each command in a fresh subprocess. "
    "Two consequences you must respect:\n"
    "\n"
    "1. **CWD persists.** Aether captures ``pwd -P`` after every "
    "shell call and uses it as the default CWD for the next call. "
    "Running ``cd /workspace/foo`` once is enough — subsequent "
    "``shell`` / ``read_file`` / ``grep`` / ``glob`` / ``write_file`` "
    "/ ``file_edit`` calls will treat ``/workspace/foo`` as the "
    "base. You do NOT need to chain ``cd /workspace/foo && ...`` on "
    "every command.\n"
    "\n"
    "2. **Env vars and venv activations do NOT persist.** "
    "``source .venv/bin/activate`` in one call has zero effect on the "
    "next — the activated shell exited. To use a venv across calls, "
    "invoke its interpreter directly: ``./.venv/bin/python …`` or "
    "``./.venv/bin/pip install …``. Same for ``export FOO=bar`` — "
    "either inline (``FOO=bar cmd``) or use the venv path pattern. "
    "If a project needs a venv, create it inside the project "
    "(``python -m venv .venv``) and use ``./.venv/bin/python`` "
    "thereafter — do not rely on a pre-existing activation.\n"
    "\n"
    "Chain dependent commands with ``&&`` within a single call when "
    "they must share env (``cd foo && python -m venv .venv && "
    "./.venv/bin/pip install -e .``). Across calls, only CWD "
    "survives.\n"
    "</shell_tool_contract>"
)


@dataclass(slots=True, frozen=True)
class SystemPromptOptions:
    """Per-call switches for which prepended sections to emit."""

    include_tool_contract: bool = True
    include_verification_directive: bool = True
    include_faithful_reporting: bool = True
    include_verifier_gate: bool = True
    include_shell_tool_contract: bool = True


def augment_system_prompt(
    system: str | None,
    descriptors: Iterable[ToolDescriptor],
    options: SystemPromptOptions = SystemPromptOptions(),
) -> str | None:
    """Return *system* with Aether's standard sections prepended.

    Sections are joined with a blank line between them and the
    caller-supplied prompt is appended at the bottom (preserving the
    user's text verbatim).  When every requested section drops out
    (no tools to advertise and both directive switches off), the
    original *system* is returned unchanged.
    """
    sections: list[str] = []

    if options.include_tool_contract:
        names = sorted({d.name for d in descriptors if d.name})
        if names:
            sections.append(
                _TOOL_CONTRACT_TEMPLATE.format(
                    names=", ".join(f"``{n}``" for n in names)
                )
            )

    # Only emit the shell-subprocess contract when ``shell`` is
    # actually in the tool kit — saves prompt tokens for read-only
    # subagent loadouts that don't carry the shell tool.
    if options.include_shell_tool_contract:
        shell_present = any(d.name == "shell" for d in descriptors)
        if shell_present:
            sections.append(_SHELL_TOOL_CONTRACT)

    if options.include_verification_directive:
        sections.append(_VERIFICATION_DIRECTIVE)

    if options.include_verifier_gate:
        sections.append(_VERIFIER_GATE)

    if options.include_faithful_reporting:
        sections.append(_FAITHFUL_REPORTING)

    if not sections:
        return system

    header = "\n\n".join(sections)
    if system and system.strip():
        return f"{header}\n\n{system}"
    return header


def augment_system_with_tool_contract(
    system: str | None,
    descriptors: Iterable[ToolDescriptor],
) -> str | None:
    """Backwards-compatible alias.

    Existing callers expected a single ``<tool_use_contract>`` block.
    This shim now also emits the verification directive and
    faithful-reporting sections by default — callers that need to
    suppress them should switch to :func:`augment_system_prompt` with
    an explicit :class:`SystemPromptOptions`.
    """
    return augment_system_prompt(system, descriptors)


__all__ = [
    "SystemPromptOptions",
    "append_plan_mode_reminder",
    "augment_system_prompt",
    "augment_system_with_tool_contract",
    "build_plan_mode_attachment",
    "build_plan_mode_reminder",
]
