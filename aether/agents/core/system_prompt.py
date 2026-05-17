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

# Parity with open-claude-code/src/constants/prompts.ts:240.
# Sprint 12 PR 12.3 — per-turn reminder that the session is in plan
# mode.  Short, stable text so the prompt cache prefix doesn't churn
# between turns.  Appended LAST (after directive sections, after the
# caller-supplied system prompt) so its instructions are the closest
# to the user turn the model sees, overriding any general "verify and
# report" guidance.
_PLAN_MODE_REMINDER_TEMPLATE = (
    "<plan_mode_reminder>\n"
    "Plan mode is active. Do not execute implementation work yet.\n"
    "Plan file: {plan_path}\n"
    "Plan file status: {plan_status}.\n"
    "You may read, search, list files, fetch context, and ask clarifying "
    "questions. The only file you may write or edit is the plan file "
    "above; use write_file to create it or file_edit to revise it. Do "
    "not write other files, run shell commands, commit, dispatch "
    "subagents, update todos, or create/update/delete memory. When the "
    "plan is ready, call exit_plan_mode to request approval. Do not ask "
    "for plan approval in ordinary text or ask_user_question.\n"
    "</plan_mode_reminder>"
)

_PLAN_MODE_REMINDER = _PLAN_MODE_REMINDER_TEMPLATE.format(
    plan_path="(unavailable)",
    plan_status="unknown",
)


def build_plan_mode_reminder(session_id: str) -> str:
    """Return the plan-mode reminder for *session_id*."""
    plan_path = "(unavailable)"
    plan_status = "unknown"
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
            content = read_plan(session_id)
            plan_status = "plan file exists" if content is not None else "no plan file exists yet"
    return _PLAN_MODE_REMINDER_TEMPLATE.format(
        plan_path=plan_path,
        plan_status=plan_status,
    )


def append_plan_mode_reminder(
    system: str | None,
    *,
    session_id: str | None = None,
) -> str | None:
    """Append the plan-mode reminder to *system*.

    Used by the engine's per-turn prompt assembly when
    ``session_state.get_mode(session_id) == "plan"``.  When *system* is
    empty/None we return the reminder by itself so the model still sees
    the constraint.
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


@dataclass(slots=True, frozen=True)
class SystemPromptOptions:
    """Per-call switches for which prepended sections to emit."""

    include_tool_contract: bool = True
    include_verification_directive: bool = True
    include_faithful_reporting: bool = True
    include_verifier_gate: bool = True


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
    "build_plan_mode_reminder",
]
