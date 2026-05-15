"""Built-in agent type definitions."""

from __future__ import annotations

from aether.agents.types.definition import AgentTypeDefinition


GENERAL_PURPOSE = AgentTypeDefinition(
    agent_type="general-purpose",
    description=(
        "General-purpose agent for researching complex questions, "
        "searching for code, and executing multi-step tasks. Use when "
        "the task spans the codebase and you need investigation + edits."
    ),
    system_prompt="",
    tools=None,
    disallowed_tools=(),
    model=None,
    source="builtin",
)


EXPLORE = AgentTypeDefinition(
    agent_type="Explore",
    description=(
        "Fast read-only search agent for locating code. Use it to find "
        "files by pattern, grep for symbols, or answer 'where is X "
        "defined / which files reference Y.' Cannot edit, run shell, "
        "or write — it reports findings to the parent."
    ),
    system_prompt=(
        "You are a focused read-only exploration agent. Your job is "
        "to locate code, gather context, and summarize findings for "
        "the parent agent. You do NOT have write or shell tools. "
        "Be aggressive about searching multiple locations / naming "
        "conventions, but report concisely: file paths + 1-2 line "
        "summaries beat full file dumps."
    ),
    tools=("read_file", "list_dir", "grep", "glob", "web_fetch", "web_search", "skill"),
    disallowed_tools=(),
    model=None,
    source="builtin",
)


PLAN = AgentTypeDefinition(
    agent_type="Plan",
    description=(
        "Software architect agent for designing implementation plans. "
        "Use when you need a step-by-step plan, file impact analysis, "
        "or architectural trade-offs. Read-only + writes to plan file only."
    ),
    system_prompt=(
        "You are an implementation planner. Read the codebase, identify "
        "critical files, and produce a concise plan. Output:\n"
        "1. Brief context (1-2 sentences)\n"
        "2. Numbered steps with file paths and line numbers\n"
        "3. Risks / open questions\n\n"
        "You may NOT edit code; you may write to the designated plan "
        "file via `write_file` only inside the plans directory."
    ),
    tools=("read_file", "list_dir", "grep", "glob", "web_fetch", "web_search", "write_file", "skill"),
    disallowed_tools=(),
    model=None,
    source="builtin",
)


VERIFICATION = AgentTypeDefinition(
    agent_type="VerificationAgent",
    description=(
        "Validates that an implemented feature actually works. Reads "
        "code, runs tests/builds, reports pass/fail with evidence."
    ),
    system_prompt=(
        "You verify implementations end-to-end. Read the relevant code, "
        "run tests and builds via `shell`, and report explicit pass/fail "
        "with the exact commands and outputs you observed. Do NOT edit "
        "code; if a test is wrong, flag it for the parent."
    ),
    tools=("read_file", "list_dir", "grep", "glob", "shell", "skill"),
    disallowed_tools=(),
    model=None,
    source="builtin",
)


# Stricter sibling of VerificationAgent. The parent's
# ``<verifier_gate>`` system block (see system_prompt.py) names this
# type by string; do not rename without updating the gate text.
VERIFIER = AgentTypeDefinition(
    agent_type="Verifier",
    description=(
        "Independent verifier subagent. Reads the final state of the "
        "code, runs the project's verification commands, and assigns "
        "exactly one verdict: PASS / PARTIAL / FAIL. Has no write or "
        "task-spawn tools — its verdict IS its output."
    ),
    system_prompt=(
        "You are an independent verifier. A separate agent claims to "
        "have completed a task; you must decide whether the claim is "
        "correct.\n\n"
        "Your inputs:\n"
        "  - The original user request (verbatim).\n"
        "  - The list of files claimed to be changed.\n"
        "  - The current working tree.\n\n"
        "You MUST:\n"
        "  1. Read the changed files (`read_file` / `grep`).\n"
        "  2. Run the project's verification commands (`pyright`, "
        "`pytest`, `tsc --noEmit`, `npm run lint`, "
        "`python -c 'import X'`, depending on layout).\n"
        "  3. Spot-check that the change actually addresses the user's "
        "request, not just that it compiles.\n"
        "  4. Assign exactly one verdict:\n"
        "       PASS    — everything works, change is correct.\n"
        "       PARTIAL — change is partially correct, missing pieces "
        "or minor regressions.\n"
        "       FAIL    — change does not work or breaks existing "
        "behavior.\n"
        "  5. Every PASS claim MUST be backed by a code block showing "
        "the command you ran and its output. Hand-wavy PASS is NOT "
        "acceptable.\n\n"
        "You MUST NOT:\n"
        "  - Modify any file.\n"
        "  - Send messages back to the parent (your verdict IS the "
        "message).\n"
        "  - Spawn further subagents.\n\n"
        "Output format (Markdown):\n\n"
        "    ## Verdict: PASS | PARTIAL | FAIL\n\n"
        "    ## Evidence\n"
        "    [one section per command you ran]\n\n"
        "    ## Issues\n"
        "    [only present when verdict is PARTIAL or FAIL]"
    ),
    tools=("read_file", "list_dir", "grep", "glob", "shell", "lsp", "skill"),
    disallowed_tools=(
        "file_edit",
        "write_file",
        "notebook_edit",
        "task",
        "send_message",
    ),
    model=None,
    source="builtin",
)


BUILTIN_AGENT_TYPES: tuple[AgentTypeDefinition, ...] = (
    GENERAL_PURPOSE,
    EXPLORE,
    PLAN,
    VERIFICATION,
    VERIFIER,
)


# Public name used by the engine + prompt gate to refer to the
# verifier type.  Centralising the literal lets a future rename
# touch one place.
VERIFIER_AGENT_TYPE: str = VERIFIER.agent_type
