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

    # Hard cap on LLM iterations per turn — the structured ceiling
    # ``IterationBudget`` (Sprint 3 / PR 3.2) is built on top of.  Default
    # ``32`` matches the ``aether chat --max-iterations`` CLI default so
    # SDK callers, subagents, and unattended tests see the same headroom
    # interactive users do.  Cheap-tool refund + max-iterations summary
    # provide soft guardrails before this number becomes painful, so the
    # tradeoff favours "long task completes" over "cost is bounded";
    # benchmarks / cost-sensitive deployments should override explicitly.
    max_iterations: int = 32
    fail_on_tool_error: bool = False
    raise_on_middleware_error: bool = False
    fail_on_unknown_tool: bool = True
    enable_todo_hydration: bool = False
    memory_nudge_interval: int = 0
    skill_nudge_interval: int = 0
    # Sprint 1 / PR 1.1: emergency rollback switch for the new SSE streaming
    # path.  When False, the engine refuses to forward ``request.stream_callback``
    # to the provider, forcing the (older, well-tested) non-streaming path
    # even if the user passes a callback.  Useful for shipping deployments
    # where a buggy provider gateway breaks SSE.
    streaming_enabled: bool = True
    # Sprint 1 / PR 1.2: enable/disable finish_reason="length" continuation
    # logic.  When False, the engine will not try to stitch continuations and
    # will simply surface the truncated answer as-is.
    length_continuation_enabled: bool = True
    # Maximum number of continuation attempts after a response ends with
    # finish_reason="length".  ``0`` effectively disables retries while still
    # allowing the thinking-budget detector and partial-return path.
    max_length_continue_retries: int = 3
    # Sprint 1 / PR 1.3: emergency rollback switch for the truncated
    # tool-call detector.  When False, the engine skips the
    # "args don't end with } or ]" heuristic and the
    # finish_reason="length" + tool_calls retry path; broken JSON falls
    # through to the dispatcher (current pre-PR-1.3 behaviour).  Useful if
    # the heuristic ever produces false positives on a specific model.
    truncated_tool_call_detection_enabled: bool = True
    # Number of times we re-issue the same provider call when the model
    # returns ``finish_reason="length"`` together with tool_calls.  We
    # deliberately do NOT append the broken assistant message to history
    # for these attempts — the goal is to give the model a second chance
    # at producing complete arguments without poisoning the conversation.
    # Hermes ships ``1`` for this knob; we mirror it.
    max_truncated_tool_call_retries: int = 1
    # Sprint 1 / PR 1.3: when tool_call arguments fail to parse as JSON
    # (and we have ruled out truncation), how many times we silently
    # re-issue the API call before injecting a tool-error message back
    # into history so the model can self-correct.  Hermes ships ``3``.
    max_invalid_json_retries: int = 3
    # Disable the tool-error self-correction injection entirely if it
    # ever causes infinite recovery loops in practice.  Defaults to True
    # because the injection is significantly safer than letting broken
    # JSON poison a tool runtime.
    invalid_json_recovery_enabled: bool = True
    # Sprint 1.5 / phantom-tool recovery: when the model returns a
    # response with no structured ``tool_calls`` *but* the visible body
    # carries clear evidence of attempted tool invocation (\u0060\u0060\u0060bash
    # blocks, ``<function=NAME>`` inline tags, ``<invoke>`` XML), inject
    # a corrective ``role=user`` message and retry instead of silently
    # finalising as TEXT_RESPONSE.  The retry budget bounds the loop
    # so a model that *consistently* refuses structured tool_calls
    # eventually exits with PHANTOM_TOOL_INTENT instead of looping
    # forever.  Disabling falls back to today's "show diagnostic +
    # finalise" behaviour, which is fine for non-Kimi-class models
    # that always populate ``tool_calls`` correctly.
    phantom_tool_recovery_enabled: bool = True
    # Default chosen empirically: typical Kimi-class failures self-
    # correct within 1–2 corrective turns; 2 retries lets us nudge
    # twice before giving up.  Set to 0 to disable retries entirely
    # while keeping the diagnostic.
    max_phantom_tool_retries: int = 2
    # Sprint 1.5 / P0-9: when ``True`` and the model emits prose-style
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
    # Sprint 1.5 / P0-9: when ``True`` and ``tool_registry`` is not
    # explicitly passed to ``AgentEngine``, the engine populates it with
    # the bundled tool kit (``shell``, ``read_file``, ``write_file``,
    # ``list_dir``, ``grep``, ``glob``).  Set ``False`` to get the
    # legacy empty-registry behaviour — useful for tests that install
    # only mocks, or for callers shipping their own toolset and that
    # want no surprises.
    use_builtin_tools: bool = True
    # Sprint 1.5 / P0-9: when ``True`` the engine prepends a small
    # ``<tool_use_contract>`` system block enumerating the registered
    # tools and forbidding prose-style tool emission (markdown ``bash``
    # fences, ``<function=NAME>``, ``<functions.shell:N>``, ``<invoke>``,
    # ``<tool_call>``).  Single strongest lever against Kimi-class
    # models that hallucinate tool calls in ``content``.  Suppressed
    # automatically when the registry is empty (no tools to advertise).
    tool_use_contract_enabled: bool = True
    # Sprint 2 / PR 2.2: master switch for the classifier-aware
    # recovery composite (rate-limit / context-overflow / payload /
    # thinking-signature / response-invalid strategies).  When ``False``
    # the engine falls back to the Sprint 0 ``GenericBackoffStrategy``
    # — useful as an emergency rollback if a future strategy
    # mis-classifies a hot upstream failure mode.  Default ``True``
    # because every shipped strategy is a strict superset of the
    # generic backoff path (they delegate to it for unknown reasons).
    classified_recovery_enabled: bool = True
    # Sprint 2 / PR 2.2: when ``True`` and ``EngineServices.fallback_chain``
    # carries more than one provider, recovery decisions tagged
    # ``activate_fallback`` actually rotate the active provider.
    # Default ``False`` per the sprint plan: failover is only useful
    # once the chain has been validated end-to-end with real provider
    # credentials; flipping this on without that prep just hides
    # configuration mistakes.  Single ``True`` flip after
    # ``FallbackChain`` is wired in production.
    fallback_chain_enabled: bool = False
    # Sprint 2 / PR 2.2: maximum number of provider rotations the
    # fallback chain may perform inside a single turn.  Bounds the
    # worst-case "every provider returns 429 in sequence" scenario so
    # an interactive turn never spins through 10 providers before
    # giving up.  ``0`` disables fallback entirely (equivalent to
    # ``fallback_chain_enabled=False``).
    max_fallback_activations_per_turn: int = 4
    # Sprint 2 / PR 2.2: ``Retry-After`` waits longer than this many
    # seconds force an immediate fallback (or give-up if the chain is
    # exhausted) instead of blocking the turn for the full duration.
    # Mirrors hermes' "if rate-limit wait > 30s, rotate" heuristic.
    rate_limit_fallback_threshold_seconds: float = 30.0
    # Sprint 2 / PR 2.3: per-turn retry budget for unrepairable tool
    # names.  Each iteration that surfaces at least one unknown +
    # unrepairable name bumps a counter; once it reaches this cap the
    # turn finalises with ``ExitReason.INVALID_TOOL_REPEATED`` so
    # observers see "the model never figured out the right name"
    # instead of an opaque tool failure.  Default ``3`` matches the
    # P0-5 acceptance criteria in ``02_p0_critical_gaps.md``.
    invalid_tool_max_retries: int = 3
    # Sprint 2 / PR 2.3: maximum delegate-class tool calls that may
    # actually dispatch in a single turn.  Excess calls become
    # synthetic error ``ToolResult``s telling the model to consolidate
    # work into fewer delegations.  ``0`` disables delegate dispatch
    # entirely (useful for debugging).
    max_delegate_calls_per_turn: int = 4
    # Sprint 2 / PR 2.3: tool names treated as "delegate-class" by
    # the per-turn cap above.  Compared via the same name normalisation
    # the repair path uses (case-fold + dash↔underscore + namespace
    # strip) so ``DelegateTask`` / ``delegate-task`` /
    # ``mcp__router__delegate_task`` all hit the cap.  Defaults cover
    # the names hermes-agent and Aether's subagent layer use; operators
    # can extend the tuple to bring custom delegators under the cap.
    delegate_tool_names: tuple[str, ...] = (
        "delegate_task",
        "delegate",
        "subagent",
        "subagent_dispatch",
        "spawn_subagent",
    )
    # Sprint 2 / PR 2.3: when ``True`` the engine deduplicates identical
    # tool calls (same name, same canonicalised args) within a single
    # iteration — the first call dispatches, the rest become stub
    # ``ToolResult``s that point at the original call id.  Default
    # ``True`` because duplicate dispatch is almost always model
    # confusion (re-reading the same file 5x); set to ``False`` to
    # restore Sprint 1 behaviour for diagnostics.
    tool_dedup_enabled: bool = True
    # Sprint 3 / PR 3.2: cheap-tool refund list.  Tools whose call
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
    # ``()`` to restore the legacy "every tool call costs a slot"
    # behaviour — useful for debugging or for workloads where every
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
    # Sprint 3 / PR 3.2: when ``True``, exhausting ``max_iterations``
    # triggers a single one-shot LLM call (no tools, no streaming) to
    # generate a summary of what the turn accomplished.  The summary
    # text becomes ``EngineResult.final_response`` while
    # ``exit_reason`` stays ``MAX_ITERATIONS`` so observability sees
    # the same termination signal as before — only the human-facing
    # response payload changes.  Bounded cost (one extra round per
    # max-iterations event) and large UX win: previously users saw
    # ``done · 0.0s`` with an empty body after a 60-second turn.
    # Set to ``False`` to restore the pre-PR-3.2 silent-break
    # behaviour (e.g. for benchmarks where the empty-string response
    # is part of the test contract).
    summary_on_budget_exhausted: bool = True
    # Sprint 3 / PR 3.4: master switch for the five-tier compaction
    # pipeline.  Default ``False`` preserves Sprint 2 behaviour: context
    # overflow recovery exits cleanly instead of attempting an LLM-fork
    # summary.  Operators can enable after validating the summariser path
    # with their provider credentials.
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
    # Sprint 3.5 / PR 3.5.1: master switch for per-tool result spill.
    # When ``True`` (default), tools that produce more than their own
    # ``MAX_RESULT_CHARS`` write the full output to disk under
    # ``tool_result_spill_dir`` and keep only an inline preview plus a
    # standardised ``... [output truncated: ... saved to ...] ...``
    # notice in ``ToolResult.content``.  When ``False``, tools fall back
    # to plain truncation behaviour with a generic ``[output truncated]``
    # marker — emergency rollback if the spill directory becomes
    # problematic (read-only NFS, full disk, untrusted on shared host).
    tool_result_spill_enabled: bool = True
    # Sprint 3.5 / PR 3.5.1: override directory for spilled results.
    # When ``None`` (default) the helpers in
    # :mod:`aether.runtime.tool_result_storage` use
    # ``~/.aether/tool_results/<session_id>/<call_id>.<ext>``.  Set to a
    # faster local disk on systems where ``$HOME`` is on slow networked
    # storage, or to a tmpfs path on ephemeral CI workers.
    tool_result_spill_dir: Path | None = None
    # Sprint 3.5 / PR 3.5.5 — web tool master switches and limits.
    # ``web_search_provider`` selects the search backend; only ``brave``
    # is implemented today, but having an explicit knob keeps the door
    # open for ddgs / tavily / serpapi without touching tool code.
    web_fetch_enabled: bool = True
    web_search_enabled: bool = True
    web_search_api_key: str | None = None
    web_search_provider: str = "brave"
    web_fetch_max_download_bytes: int = 5 * 1024 * 1024
    web_fetch_timeout_seconds: int = 30
    # Sprint 3.5 / PR 3.5.6 — subagent dispatch master switch.  When
    # ``False`` the ``task`` / ``task_stop`` tools refuse to dispatch
    # and return a structured error.  Cheap rollback for environments
    # where letting the model fan out is undesirable (cost, isolation).
    allow_subagent_dispatch: bool = True
    # Sprint 3.5 / PR 3.5.7 — interaction tool master switches.
    # ``plan_mode_enabled`` gates ``EnterPlanMode`` / ``ExitPlanMode``;
    # ``ask_user_question_enabled`` gates ``AskUserQuestion``.  When
    # disabled the tools refuse with a clear "disabled by configuration"
    # error rather than failing silently.
    plan_mode_enabled: bool = True
    ask_user_question_enabled: bool = True
    ask_user_question_timeout_seconds: int = 600
    # Sprint 3.5 / PR 3.5.8 — skill catalog configuration.  Empty
    # ``skill_search_paths`` means "use sensible defaults" inside
    # ``SkillCatalog`` callers (project-root ``skills/`` then
    # ``~/.aether/skills``).  ``skill_list_in_system_prompt`` is OFF
    # by default to avoid silently inflating the system prompt with
    # the discovered ~70 skills; enable explicitly when desired.
    skill_tool_enabled: bool = True
    skill_search_paths: tuple[Path, ...] = ()
    skill_list_in_system_prompt: bool = False
    # Sprint 3.5 / PR 3.5.9 — LSP tool gate.  ``True`` by default
    # because the tool degrades gracefully (it returns a friendly
    # "language server not installed" message when no LSP binary is
    # on PATH), so leaving it on never crashes a turn that doesn't
    # need it.  ``lsp_server_overrides`` lets operators steer specific
    # languages at a custom binary (e.g. ``"python": [["pylsp"]]``)
    # without re-publishing the harness.
    lsp_tool_enabled: bool = True
    lsp_server_overrides: Dict[str, Any] = field(default_factory=dict)
    lsp_request_timeout_seconds: int = 8
    # Sprint 3.5 / PR 3.5.10 — Headless Chromium browser tool.
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
