"""Engine configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class ModelCallConfig:
    """Provider call configuration for a single turn."""

    temperature: float | None = None
    max_tokens: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineConfig:
    """Runtime configuration for the loop engine."""

    # Hard cap on LLM iterations per turn. ``IterationBudget`` layers
    # cheap-tool refunds and the one-shot summary call on top of this
    # ceiling. Default ``32`` matches ``aether chat --max-iterations``
    # so CLI, SDK callers, subagents, and unattended tests share the
    # same headroom.
    max_iterations: int = 32
    fail_on_tool_error: bool = False
    raise_on_middleware_error: bool = False
    fail_on_unknown_tool: bool = True
    enable_todo_hydration: bool = False
    memory_nudge_interval: int = 0
    skill_nudge_interval: int = 0
    skill_listing_enabled: bool = True
    skill_listing_token_budget: int = 4096
    # Memory subsystem defaults. The default mode targets task/project
    # context rather than user profiling. Memory blocks are retrieved as
    # transient provider-bound context only.
    memory_enabled: bool = True
    memory_mode: str = "project"
    memory_token_budget_pct: float = 0.08
    memory_token_budget_max: int = 2_500
    memory_block_token_max: int = 500
    memory_retrieval_timeout_ms: int = 200
    memory_project_store_enabled: bool = True
    memory_user_profile_enabled: bool = False
    memory_auto_write_enabled: bool = False
    memory_llm_rerank_enabled: bool = False
    memory_debug_log_content: bool = False
    # Emergency rollback switch for SSE streaming. When False, the
    # engine refuses to forward ``request.stream_callback`` to the
    # provider and always uses the non-streaming path, even if the user
    # passes a callback. Useful when a provider gateway has a broken SSE
    # implementation.
    streaming_enabled: bool = True
    # Enable or disable finish_reason="length" continuation logic. When
    # False, the engine surfaces the truncated answer as-is.
    length_continuation_enabled: bool = True
    # Maximum number of continuation attempts after a response ends with
    # finish_reason="length".  ``0`` effectively disables retries while still
    # allowing the thinking-budget detector and partial-return path.
    max_length_continue_retries: int = 3
    # Emergency rollback switch for truncated tool-call detection. When
    # False, the engine skips the "args don't end with } or ]"
    # heuristic and the finish_reason="length" + tool_calls retry path;
    # malformed JSON falls through to the dispatcher unchanged. Useful
    # if the heuristic ever produces false positives on a specific
    # model.
    truncated_tool_call_detection_enabled: bool = True
    # Number of times the engine re-issues the same provider call when
    # the model returns ``finish_reason="length"`` together with
    # tool_calls. The broken assistant message is not appended to
    # history for these attempts, so the model gets a clean retry at
    # producing complete arguments.
    max_truncated_tool_call_retries: int = 1
    # When tool_call arguments fail to parse as JSON (and truncation
    # has been ruled out), how many times to silently re-issue the API
    # call before injecting a tool-error message back into history so
    # the model can self-correct.
    max_invalid_json_retries: int = 3
    # Disable the tool-error self-correction injection entirely if it
    # ever causes infinite recovery loops in practice.  Defaults to True
    # because the injection is significantly safer than letting broken
    # JSON poison a tool runtime.
    invalid_json_recovery_enabled: bool = True
    # Phantom-tool recovery: when the model returns a
    # response with no structured ``tool_calls`` *but* the visible body
    # carries clear evidence of attempted tool invocation (\u0060\u0060\u0060bash
    # blocks, ``<function=NAME>`` inline tags, ``<invoke>`` XML), inject
    # a corrective ``role=user`` message and retry instead of silently
    # finalising as TEXT_RESPONSE. The retry budget bounds the loop so
    # a model that consistently refuses structured tool_calls
    # eventually exits with PHANTOM_TOOL_INTENT instead of looping
    # forever. Disabling falls back to surfacing the diagnostic and
    # finalising the turn without a corrective retry.
    phantom_tool_recovery_enabled: bool = True
    # Two retries cover the common case where one or two corrective
    # turns are enough. Set to 0 to disable retries while keeping the
    # diagnostic.
    max_phantom_tool_retries: int = 2
    # When ``True`` and the model emits prose-style
    # tool intents (\u0060\u0060\u0060bash\u0060\u0060\u0060 fences, ``<function=NAME>``,
    # ``<functions.shell:N>``, ``<invoke>``) instead of structured
    # ``tool_calls``, the engine attempts to synthesize ``ToolCall``s
    # from the parsed prose and dispatch them through the registry as
    # if the model had emitted them properly.  This keeps the loop
    # alive for Kimi-class models that habitually narrate tool calls.
    # Synthesis only runs when the prose maps cleanly to a registered
    # tool name (after fuzzy normalisation); otherwise the corrective-
    # message retry path runs as before.  Disable to keep the strict
    # claude-code-style "no synthesis, prose is just text" semantics.
    phantom_tool_synthesis_enabled: bool = True
    # When ``True`` and ``tool_registry`` is not
    # explicitly passed to ``AgentEngine``, the engine populates it with
    # the bundled tool kit (``shell``, ``read_file``, ``write_file``,
    # ``list_dir``, ``grep``, ``glob``).  Set ``False`` to get the
    # empty-registry behavior — useful for tests that install
    # only mocks, or for callers shipping their own toolset and that
    # want no surprises.
    use_builtin_tools: bool = True
    # When ``True`` the engine prepends a small
    # ``<tool_use_contract>`` system block enumerating the registered
    # tools and forbidding prose-style tool emission (markdown ``bash``
    # fences, ``<function=NAME>``, ``<functions.shell:N>``, ``<invoke>``,
    # ``<tool_call>``).  Single strongest lever against Kimi-class
    # models that hallucinate tool calls in ``content``.  Suppressed
    # automatically when the registry is empty (no tools to advertise).
    tool_use_contract_enabled: bool = True
    # When ``True`` the engine prepends a ``<verification_directive>``
    # system block instructing the model to verify its work (re-read,
    # type-check, grep callers) before reporting a task complete.
    # Parity with open-claude-code/src/constants/prompts.ts:211 — see
    # ``aether/agents/core/system_prompt.py``.
    verification_directive_enabled: bool = True
    # When ``True`` the engine prepends a ``<faithful_reporting>``
    # system block banning defensive hedging and dishonest summaries
    # when a verification step fails.  Parity with
    # open-claude-code/src/constants/prompts.ts:240.
    faithful_reporting_enabled: bool = True
    # When ``True`` the engine prepends a ``<verifier_gate>`` system
    # block AND fires the in-loop soft gate that nudges the model to
    # spawn an independent ``Verifier`` subagent once the session has
    # edited ``verifier_gate_file_threshold`` distinct files.  Parity
    # with open-claude-code/src/constants/prompts.ts:394.
    verifier_gate_enabled: bool = True
    # Number of distinct files edited in a single session before the
    # soft gate fires a reminder.  Three is the OCC default; raise it
    # for chatty refactors, drop to 1 in CI guardrail mode.
    verifier_gate_file_threshold: int = 3
    # Master switch for the classifier-aware recovery composite
    # (rate-limit / context-overflow / payload / thinking-signature /
    # response-invalid strategies). When ``False`` the engine falls
    # back to ``GenericBackoffStrategy``. Default ``True`` because the
    # specific strategies delegate unknown reasons to the generic path.
    classified_recovery_enabled: bool = True
    # When ``True`` and ``EngineServices.fallback_chain`` carries more
    # than one provider, recovery decisions tagged
    # ``activate_fallback`` rotate the active provider. Keep ``False``
    # to disable provider failover even when a chain is configured.
    fallback_chain_enabled: bool = False
    # maximum number of provider rotations the
    # fallback chain may perform inside a single turn.  Bounds the
    # worst-case "every provider returns 429 in sequence" scenario so
    # an interactive turn never spins through 10 providers before
    # giving up.  ``0`` disables fallback entirely (equivalent to
    # ``fallback_chain_enabled=False``).
    max_fallback_activations_per_turn: int = 4
    # ``Retry-After`` waits longer than this many seconds force an
    # immediate fallback (or give-up if the chain is exhausted)
    # instead of blocking the turn for the full duration.
    rate_limit_fallback_threshold_seconds: float = 30.0
    # cross-session provider/base_url lockout.  When a
    # session observes a rate-limit response with a concrete reset hint, it
    # writes a short-lived file lock so other sessions skip that same upstream
    # and use fallback instead of repeatedly burning failed requests.
    rate_guard_enabled: bool = True
    rate_guard_dir: Path | None = None
    # per-turn retry budget for unrepairable tool
    # names.  Each iteration that surfaces at least one unknown +
    # unrepairable name bumps a counter; once it reaches this cap the
    # turn finalises with ``ExitReason.INVALID_TOOL_REPEATED`` so
    # observers see "the model never figured out the right name"
    # instead of an opaque tool failure.
    invalid_tool_max_retries: int = 3
    # maximum delegate-class tool calls that may
    # actually dispatch in a single turn.  Excess calls become
    # synthetic error ``ToolResult``s telling the model to consolidate
    # work into fewer delegations.  ``0`` disables delegate dispatch
    # entirely (useful for debugging).
    max_delegate_calls_per_turn: int = 4
    # tool names treated as "delegate-class" by
    # the per-turn cap above.  Compared via the same name normalisation
    # the repair path uses (case-fold + dash↔underscore + namespace
    # strip) so ``DelegateTask`` / ``delegate-task`` /
    # ``mcp__router__delegate_task`` all hit the cap. Defaults cover
    # the built-in delegation entry points; operators can extend the
    # tuple to bring custom delegators under the cap.
    delegate_tool_names: tuple[str, ...] = (
        "delegate_task",
        "delegate",
        "subagent",
        "subagent_dispatch",
        "spawn_subagent",
    )
    # when ``True`` the engine deduplicates identical
    # tool calls (same name, same canonicalised args) within a single
    # iteration — the first call dispatches, the rest become stub
    # ``ToolResult``s that point at the original call id.  Default
    # ``True`` because duplicate dispatch is almost always model
    # confusion (re-reading the same file 5x); set to ``False`` to
    # restore direct-dispatch behavior for diagnostics.
    tool_dedup_enabled: bool = True
    # cheap-tool refund list.  Tools whose call
    # doesn't consume real iteration budget — typically bookkeeping
    # operations (``update_todo``, ``memory_*``) and read-only state
    # queries (``session_search``).  When an iteration's tool_calls
    # are *entirely* drawn from this set, ``IterationBudget.refund()``
    # cancels the slot the loop just consumed so the model gets the
    # same effective headroom it would have had without the
    # bookkeeping call.  Names are compared with the same normalisation
    # the tool-repair path uses (case-fold + dash↔underscore + namespace
    # strip) so ``UpdateTodo``, ``update-todo`` and
    # ``mcp__router__update_todo`` all match ``update_todo``.  Set to
    # ``()`` to make every tool call consume a slot — useful for
    # debugging or for workloads where every
    # tool has real cost (e.g. paid-API-only tool kits).
    cheap_tool_names: tuple[str, ...] = (
        "update_todo",
        "todo_write",
        "memory",
        "memory_write",
        "memory_read",
        "skill_manage",
        "session_search",
    )
    # When ``True``, exhausting ``max_iterations`` triggers a single
    # one-shot LLM call (no tools, no streaming) to generate a summary
    # of what the turn accomplished. The summary becomes
    # ``EngineResult.final_response`` while ``exit_reason`` stays
    # ``MAX_ITERATIONS``. Set to ``False`` to leave
    # ``final_response`` empty on budget exhaustion.
    summary_on_budget_exhausted: bool = True
    # Master switch for the five-tier compaction pipeline. Default
    # ``False`` keeps context-overflow recovery on the clean-exit path
    # instead of attempting an LLM-generated summary. Operators can
    # enable it after validating the summarizer path with their
    # provider credentials.
    compression_enabled: bool = False
    # Pre-LLM threshold as a fraction of the resolved model context window.
    # When estimated history size is below this threshold, preflight
    # compaction is a no-op.  Context-overflow recovery still forces the
    # pipeline even if the estimator says the prompt is smaller.
    compression_pre_llm_pct: float = 0.85
    # Tier 5 autocompact threshold.  Kept separate so cheaper tiers can
    # eventually run at one threshold while the expensive LLM-fork tier
    # fires later.
    compression_autocompact_pct: float = 0.85
    # Consecutive Tier 5 failures before the circuit breaker skips further
    # summariser calls for this turn.
    compression_max_failures: int = 3
    # Tier 5-specific switch.  ``compression_enabled=True`` with this set
    # ``False`` keeps the pipeline skeleton active for cheaper future tiers.
    autocompact_enabled: bool = True
    # Messages protected verbatim during Tier 5 summarisation.
    compression_protect_first_n: int = 2
    compression_protect_last_n: int = 6
    # Target output budget for the summary text generated by the LLM fork.
    compression_target_summary_tokens: int = 4_000
    # Tier 3 (TimeBasedMicrocompactor) tuning.
    #
    # ``microcompact_gap_threshold_minutes``: minutes since the last
    # assistant message that must elapse before Tier 3 considers the
    # cache "cold" enough to clear old tool results.  Default 5 mirrors
    # claude-code's ``tengu_time_based_microcompact_gap_threshold_minutes``
    # GrowthBook default.  Set to ``math.inf`` (or any very large value)
    # to effectively disable Tier 3 without unwiring the tier.
    microcompact_gap_threshold_minutes: float = 5.0
    # ``microcompact_keep_recent``: how many of the most recent
    # compactable tool results to PRESERVE in original form when Tier 3
    # fires.  Default 5 matches claude-code.  The Tier 3 implementation
    # floors this to 1 — leaving zero would strand the model with no
    # working tool context at all on the next call.
    microcompact_keep_recent: int = 5
    # ``microcompact_compactable_tools``: tool names whose
    # ``tool_result`` payloads Tier 3 will replace with the
    # ``[Old tool result content cleared]`` placeholder.  Compared via
    # the same normalisation phantom-tool repair uses (case-fold +
    # dash↔underscore + namespace strip), so ``ReadFile``, ``read-file``
    # and ``mcp__router__read_file`` all match the ``read_file`` entry.
    # Default covers the read-class builtins; ``write_file`` is omitted
    # because its result is already a tiny "wrote N bytes" string.
    microcompact_compactable_tools: tuple[str, ...] = (
        "read_file",
        "shell",
        "grep",
        "glob",
        "list_dir",
    )
    # Tier 2 (Snipper) tuning.
    #
    # ``snip_enabled``: master switch for the Tier 2 snipper.  Can
    # be disabled even when the rest of the pipeline is active —
    # e.g. if a specific snip rule causes unexpected behaviour and
    # we want to ship a hotfix without reverting the whole
    # pipeline.  Default ON because the three rules are conservative
    # by design (only delete provably-redundant content).
    snip_enabled: bool = True
    # Per-rule switches.  All on by default; flip individual rules
    # off if one misbehaves on real workloads.
    #
    # ``snip_dupe_enabled`` (Rule 1): keep only the LAST occurrence
    # of each ``(name, input)`` tuple for a small whitelist of
    # read-only tools (read_file, list_dir, glob, grep).  Earlier
    # duplicate calls and their results are deleted.  shell /
    # write_file are intentionally excluded — same input there can
    # have side-effects.
    snip_dupe_enabled: bool = True
    # ``snip_fail_enabled`` (Rule 2): drop a failed (``is_error=True``)
    # tool_result + its tool_use when a later same-key call
    # succeeds.  Applies to any tool name (failure superseded by
    # success is universally redundant).  Pure failures with no
    # follow-up success are preserved (failure itself is signal).
    snip_fail_enabled: bool = True
    # ``snip_empty_enabled`` (Rule 3): drop assistant messages whose
    # content is empty / whitespace / only-empty thinking-blocks /
    # only-empty text-blocks (and no tool_use).  These appear when
    # a streaming response was interrupted between blocks or when
    # the model emitted a blank "let me think..." pause.
    snip_empty_enabled: bool = True
    # Tier 4 (ContextCollapseTier) tuning.
    #
    # ``context_collapse_enabled``: master switch for the projection-
    # based collapse tier.  Default ``False`` because Tier 4 mutates
    # the *view* sent to the model and competes with Tier 5; we want
    # production observation first before flipping the default.
    # Operators enable explicitly when Tier 5 fires too aggressively
    # on long sessions (e.g. by inflating Tier 5 fork cost).
    context_collapse_enabled: bool = False
    # ``context_collapse_commit_pct``: token utilisation (as fraction
    # of model window) at which Tier 4 starts proposing collapse
    # segments.  0.90 mirrors claude-code's commit-start threshold.
    # Below this the tier is read-only — it just applies whatever
    # committed segments already exist on the store.
    context_collapse_commit_pct: float = 0.90
    # ``context_collapse_blocking_pct``: token utilisation at which a
    # newly-proposed segment becomes *committed* (and therefore takes
    # effect in subsequent projections).  0.95 mirrors claude-code's
    # blocking threshold.  Between commit_pct and blocking_pct,
    # segments are proposed but not committed — the pipeline can
    # still re-evaluate before they bind.
    context_collapse_blocking_pct: float = 0.95
    # ``context_collapse_segment_max_messages``: maximum messages the
    # tier will fold into a single collapse segment.  Larger values
    # mean fewer summary calls but each one is harder for the LLM
    # to summarise well.  Default 20 is a sweet spot in profiling.
    context_collapse_segment_max_messages: int = 20
    # master switch for per-tool result spill.
    # When ``True`` (default), tools that produce more than their own
    # ``MAX_RESULT_CHARS`` write the full output to disk under
    # ``tool_result_spill_dir`` and keep only an inline preview plus a
    # standardised ``... [output truncated: ... saved to ...] ...``
    # notice in ``ToolResult.content``.  When ``False``, tools fall back
    # to plain truncation behaviour with a generic ``[output truncated]``
    # marker — emergency rollback if the spill directory becomes
    # problematic (read-only NFS, full disk, untrusted on shared host).
    tool_result_spill_enabled: bool = True
    # override directory for spilled results.
    # When ``None`` (default) the helpers in
    # :mod:`aether.runtime.tools.tool_result_storage` use
    # ``~/.aether/tool_results/<session_id>/<call_id>.<ext>``.  Set to a
    # faster local disk on systems where ``$HOME`` is on slow networked
    # storage, or to a tmpfs path on ephemeral CI workers.
    tool_result_spill_dir: Path | None = None
    # web tool master switches and limits.
    # ``web_search_provider`` selects the search backend; only ``brave``
    # is implemented today, but having an explicit knob keeps the door
    # open for ddgs / tavily / serpapi without touching tool code.
    web_fetch_enabled: bool = True
    web_search_enabled: bool = True
    web_search_api_key: str | None = None
    web_search_provider: str = "brave"
    web_fetch_max_download_bytes: int = 5 * 1024 * 1024
    web_fetch_timeout_seconds: int = 30
    # subagent dispatch master switch.  When
    # ``False`` the ``task`` / ``task_stop`` tools refuse to dispatch
    # and return a structured error.  Cheap rollback for environments
    # where letting the model fan out is undesirable (cost, isolation).
    allow_subagent_dispatch: bool = True
    # interaction tool master switches.
    # ``plan_mode_enabled`` gates ``EnterPlanMode`` / ``ExitPlanMode``;
    # ``ask_user_question_enabled`` gates ``AskUserQuestion``.  When
    # disabled the tools refuse with a clear "disabled by configuration"
    # error rather than failing silently.
    plan_mode_enabled: bool = True
    ask_user_question_enabled: bool = True
    ask_user_question_timeout_seconds: int = 600
    # tool permission confirmation.  Dangerous tools are gated
    # before dispatch so confirmations render in the CLI control plane
    # instead of scrollback/transcript.  Non-interactive callers deny
    # dangerous tools by default unless they explicitly opt into allow.
    tool_permissions_enabled: bool = False
    tool_permission_default: str = "ask"
    tool_permission_auto_allow_readonly: bool = True
    tool_permission_non_interactive_default: str = "deny"
    tool_permission_session_allow_enabled: bool = True
    # skill catalog configuration.  Empty
    # ``skill_search_paths`` means "use sensible defaults" inside
    # ``SkillCatalog`` callers (project-root ``skills/`` then
    # ``~/.aether/skills``).  ``skill_list_in_system_prompt`` is OFF
    # by default to avoid silently inflating the system prompt with
    # the discovered ~70 skills; enable explicitly when desired.
    skill_tool_enabled: bool = True
    skill_search_paths: tuple[Path, ...] = ()
    skill_list_in_system_prompt: bool = False
    agent_type_search_paths: tuple[Path, ...] = ()
    agent_type_registry_enabled: bool = True
    # On-disk task store for subagent lifecycle persistence (PR 10.4).
    # When enabled, subagent task records, message streams, and final
    # results land under ``task_store_path`` (defaults to
    # ``~/.aether/tasks``).  Disable to opt out entirely — the engine
    # falls back to in-memory state and ``task_output`` becomes
    # equivalent to "no such task" for any historical lookup.
    task_store_enabled: bool = True
    task_store_path: Path | None = None
    # Heartbeat staleness threshold (seconds).  RUNNING records older
    # than this are marked KILLED at root engine startup so a crashed
    # gateway never leaves "still running" zombies in the store.
    task_store_stale_seconds: float = 60.0
    # LSP tool gate.  ``True`` by default
    # because the tool degrades gracefully (it returns a friendly
    # "language server not installed" message when no LSP binary is
    # on PATH), so leaving it on never crashes a turn that doesn't
    # need it.  ``lsp_server_overrides`` lets operators steer specific
    # languages at a custom binary (e.g. ``"python": [["pylsp"]]``)
    # without re-publishing the harness.
    lsp_tool_enabled: bool = True
    lsp_server_overrides: Dict[str, Any] = field(default_factory=dict)
    lsp_request_timeout_seconds: int = 8
    # empty-response recovery and structured
    # tool-error rollback knobs.
    legitimate_empty_passthrough_enabled: bool = True
    empty_response_recovery_enabled: bool = True
    empty_response_max_retries: int = 3
    empty_response_partial_stream_recovery_enabled: bool = True
    housekeeping_fallback_enabled: bool = True
    housekeeping_tool_names: tuple[str, ...] = (
        "memory",
        "update_todo",
        "todo_write",
        "skill_manage",
        "session_search",
    )
    post_tool_empty_nudge_enabled: bool = True
    thinking_prefill_enabled: bool = True
    thinking_prefill_max_retries: int = 2
    codex_intermediate_ack_enabled: bool = True
    codex_intermediate_ack_max_retries: int = 2
    error_withholding_enabled: bool = True
    max_provider_recovery_attempts: int = 8
    tool_error_structured_format_enabled: bool = True
    tool_schema_precheck_enabled: bool = True
    # failed provider request debug dumps.
    # Disabled by default because request bodies can contain sensitive user
    # data even after credential/header redaction.
    dump_failed_requests: bool = False
    request_dump_dir: Path = Path("./request_dumps")
    # optional trajectory JSONL persistence.
    # Disabled by default because transcripts can contain sensitive user/tool data.
    save_trajectories: bool = False
    trajectory_dir: Path = Path("./trajectories")
    # Headless Chromium browser tool.
    # **Disabled by default** because Playwright is a heavyweight
    # optional dependency (~150 MB Chromium download) and most
    # turns don't need a real browser.  Flip on after running
    # ``pip install playwright && playwright install chromium``.
    # ``web_browser_idle_timeout_seconds`` controls how long an idle
    # browser stays warm before the manager shuts it down to free
    # memory.  ``web_browser_navigation_timeout_seconds`` bounds any
    # single page load.
    web_browser_enabled: bool = False
    web_browser_idle_timeout_seconds: int = 30
    web_browser_navigation_timeout_seconds: int = 30
