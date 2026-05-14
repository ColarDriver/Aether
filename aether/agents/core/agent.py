"""Primary agent implementation for Aether harness."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List

from aether.agents.core.phantom_tool import (
    PhantomToolIntent,
    build_corrective_user_message,
    detect_phantom_tool_intent,
    synthesize_tool_calls_from_phantom,
)
from aether.agents.core.tool_hardening import ToolDispatchPlan, prepare_tool_calls
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.memory import (
    MemoryBundle,
    MemoryMode,
    MemoryProvider,
    MemoryQuery,
    ProjectMemoryStore,
    RetrievalMemoryProvider,
    TaskMemoryProvider,
    append_memory_context,
    default_memory_metadata,
    estimate_text_tokens,
    metadata_from_bundle,
    normalize_memory_mode,
    pack_memory_blocks,
    render_memory_bundle,
    resolve_memory_token_budget,
    scopes_for_mode,
    strip_memory_context_from_messages,
)
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    LoopState,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.recovery.error_classifier import FailoverReason
from aether.runtime.core.exceptions import EngineInterrupted
from aether.runtime.control.interrupt_messages import FETCH_INTERRUPTED_MESSAGE, select_interrupt_marker
from aether.runtime.control.interrupt_signal import InterruptSignal
from aether.runtime.recovery.fallback_chain import FallbackChain
from aether.runtime.core.hooks import EngineHooks, HookOutcome
from aether.runtime.recovery.image_shrink import shrink_image_parts_in_messages
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.iteration_budget import IterationBudget
from aether.runtime.recovery.provider_errors import ProviderInvocationError, ResponseInvalidError
from aether.runtime.recovery.rate_guard import RateGuard, RateGuardCheck
from aether.runtime.recovery.strategies import (
    AttemptState,
    ClassifiedRecoveryStrategy,
    GenericBackoffStrategy,
    RecoveryDecision,
    RecoveryStrategy,
    wait_interruptible,
)
from aether.runtime.recovery.response_classification import (
    EmptyKind,
    ResponseClassification,
    is_legitimate_empty,
    strip_thinking_tags,
)
from aether.runtime.observability.reasoning import extract_last_reasoning
from aether.runtime.observability.request_dump import dump_api_request_debug
from aether.runtime.core.services import EngineServices
from aether.runtime.session.session_runtime import (
    TURN_KEY_CODEX_ACK_RETRIES,
    TURN_KEY_EMPTY_RESPONSE_RETRIES,
    TURN_KEY_EMPTY_RECOVERY_LAST_STEP,
    TURN_KEY_INVALID_JSON_RETRIES,
    TURN_KEY_PHANTOM_TOOL_RETRIES,
    TURN_KEY_PHANTOM_TOOL_SYNTHESIZED,
    TURN_KEY_POST_TOOL_EMPTY_RETRIED,
    TURN_KEY_PROVIDER_ERROR_RETRIES,
    TURN_KEY_STREAMED_ASSISTANT_TEXT,
    TURN_KEY_THINKING_PREFILL_RETRIES,
    TURN_KEY_TRUNCATED_RESPONSE_PREFIX,
    TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES,
    SessionRuntimeRegistry,
    SessionRuntimeState,
)
from aether.runtime.session.session_store import InMemorySessionStore, SessionStore
from aether.runtime.recovery.schema_sanitizer import sanitize_tool_descriptors
from aether.runtime.core.state_machine import EngineStateMachine
from aether.runtime.control.steer import SteerInbox
from aether.runtime.tools.task_cleanup import normalize_task_cleanup_metadata, release_task_resources
from aether.runtime.observability.trajectory import build_trajectory_record, save_trajectory_record
from aether.runtime.observability.unicode_sanitizer import (
    sanitize_provider_credentials_non_ascii,
    sanitize_structure_non_ascii,
    sanitize_structure_surrogates,
    strip_surrogates,
)
from aether.runtime.observability.usage import CanonicalUsage, normalize_usage
from aether.runtime.tools.tool_error_format import (
    FormattedToolError,
    format_invalid_tool_args_error,
    format_schema_error,
    format_unknown_tool_error,
)
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionMode,
    ToolPermissionPreview,
    ToolPermissionRequest,
    build_fallback_preview,
    build_permission_denied_result,
    build_session_rule_for_request,
    default_permission_stats,
    find_matching_rule,
    is_dangerous_tool,
    make_permission_request,
    normalize_permission_mode,
)
from aether.services.compact import estimate_messages_tokens
from aether.services.compact.autocompact import AutoCompactor
from aether.services.compact.compactor import CompactionPipeline, CompactionResult
from aether.services.compact.collapse import (
    CollapseStore,
    ContextCollapseTier,
)
from aether.services.compact.llm_fork import LLMForkSummarizer
from aether.services.compact.microcompact import TimeBasedMicrocompactor
from aether.services.compact.snip import Snipper
from aether.tools.base import UnknownToolError
from aether.tools.registry import ToolRegistry, check_plan_mode_block

if TYPE_CHECKING:
    from aether.subagents.contracts import SubagentResult, SubagentTask
    from aether.subagents.manager import SubagentManager


# keys that live on context.metadata as runtime helpers
# but should NOT leak into the JSON-serialisable ``EngineResult.metadata['turn']``
# snapshot.  Their normalised, public-facing form lives at the metadata top
# level (e.g. ``usage_accumulator`` → ``metadata['usage']``).  Add new
# internal-only keys here whenever a future feature introduces one.
_METADATA_INTERNAL_KEYS: frozenset[str] = frozenset({
    "usage_accumulator",
    # live ``IterationBudget`` instance — kept on
    # ``context.metadata`` so ``_handle_max_iterations`` can find it
    # without threading another argument through every recovery path.
    # The JSON-friendly snapshot lives at ``metadata['iteration_budget']``
    # (also on ``context.metadata``) and is mirrored into
    # ``EngineResult.metadata['iteration_budget']`` by ``_build_result``.
    "_iteration_budget_obj",
    # live ``EngineConfig`` reference — tools
    # consult it via ``maybe_spill_for_tool`` to read the per-tool
    # spill master switch and override directory.  Storing the live
    # object (not a snapshot) keeps tools picking up config changes
    # mid-session if a future feature ever rebuilds config; the price
    # is that we MUST exclude it from the JSON-serialised
    # ``EngineResult.metadata['turn']`` snapshot.
    "_engine_config",
    # Live references for the subagent and interaction tool family.
    # Stored as objects, never as JSON.
    "_parent_agent",
    "_subagent_manager",
    # interactive prompter forwarded from
    # ``EngineRequest.approval_prompter`` to ``ExitPlanModeTool`` and
    # ``AskUserQuestionTool``.  Excluded for the same reason as the
    # other live-object keys above.
    "_approval_prompter",
    # live SkillCatalog reference used by
    # ``SkillTool`` when no catalog was injected via constructor.
    "_skill_catalog",
    # live LSPManager reference for
    # :class:`LSPTool`.  Same rationale as the other live-object
    # keys above: must never leak into the JSON-serialised
    # ``EngineResult.metadata['turn']`` snapshot.
    "_lsp_manager",
    # live BrowserManager reference for
    # :class:`WebBrowserTool`.
    "_browser_manager",
    "_tool_permission_prompter",
    "_tool_permission_preview_plans",
    "_compaction_pipeline",
    "_compaction_in_progress",
    "_api_request_attempt_count",
    "_failed_request_dump_written",
    "turn_start_idx",
    "_task_resource_handles",
    "_task_resource_keys",
    "_task_cleanup_done",
    "_schema_sanitized_tool_descriptors",
    "_schema_sanitizer_retry_attempted",
    "_image_shrink_retry_attempted",
    "_interrupt_signal",
    "_loop_state_callback",
    "_project_memory_store",
})


_CONTINUE_LOOP_SENTINEL = object()


class _EmptyRecoveryStep(str, Enum):
    TRUNCATED_PREFIX_CONCAT = "truncated_prefix_concat"
    PARTIAL_STREAM_RECOVERY = "partial_stream_recovery"
    HOUSEKEEPING_FALLBACK = "housekeeping_fallback"
    POST_TOOL_EMPTY_NUDGE = "post_tool_empty_nudge"
    THINKING_PREFILL = "thinking_prefill"
    RETRY_OR_FALLBACK = "retry_or_fallback"
    TERMINAL_EMPTY = "terminal_empty"
    CODEX_INTERMEDIATE_ACK = "codex_intermediate_ack"


@dataclass(slots=True, frozen=True)
class _EmptyRecoveryOutcome:
    step: _EmptyRecoveryStep
    final_response: str | None = None
    exit_reason: ExitReason | None = None
    continue_loop: bool = False


@dataclass(slots=True, frozen=True)
class _FinalizeResponseOutcome:
    final_response: str | object | None
    exit_reason: ExitReason | None
    error_text: str | None = None


@dataclass(slots=True)
class _WithholdingState:
    pending_errors: list[ProviderInvocationError] = field(default_factory=list)
    cascade_log: list[str] = field(default_factory=list)
    suppressed_callback_notifications: int = 0
    compression_attempted_for: set[str] = field(default_factory=set)


class AgentEngine:
    """Main agent orchestrator with an explicit run loop state machine."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        tool_registry: ToolRegistry | None = None,
        middleware_pipeline: MiddlewarePipeline | None = None,
        config: EngineConfig | None = None,
        interrupt_controller: InterruptController | None = None,
        logger: logging.Logger | None = None,
        delegate_depth: int = 0,
        subagent_id: str | None = None,
        parent_subagent_id: str | None = None,
        subagent_manager: "SubagentManager | None" = None,
        session_store: SessionStore | None = None,
        hooks: EngineHooks | None = None,
        recovery_strategy: RecoveryStrategy | None = None,
        fallback_chain: FallbackChain | None = None,
        skill_catalog: "object | None" = None,
        lsp_manager: "object | None" = None,
        browser_manager: "object | None" = None,
        steer_inbox: SteerInbox | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        if tool_registry is None:
            if self.config.use_builtin_tools:
                # Local import keeps subprocess/pathlib import cost
                # off the critical path for callers that explicitly
                # disable builtins (use_builtin_tools=False) or pass
                # their own registry.
                from aether.tools.builtins import build_default_tool_registry

                tool_registry = build_default_tool_registry()
            else:
                tool_registry = ToolRegistry()
        # pick the default recovery strategy:
        # * If the caller injected one, honour it untouched (test
        #   determinism / scripted-provider compat).
        # * If ``classified_recovery_enabled`` is True (default),
        #   install the new ``ClassifiedRecoveryStrategy`` so 429 /
        #   413 / response-invalid / context-overflow / thinking-
        #   signature failures get the right recovery shape with no
        #   call-site changes.
        # * Otherwise fall back to the generic backoff path —
        #   single emergency rollback knob if the classifier ever
        #   misbehaves.
        if recovery_strategy is None:
            if getattr(self.config, "classified_recovery_enabled", True):
                recovery_strategy = ClassifiedRecoveryStrategy()
            else:
                recovery_strategy = GenericBackoffStrategy()

        # Per-session state registry (cross-turn nudge counters, task memory,
        # future cached system prompt for prefix-cache stability).  Storing
        # these on a session-keyed registry instead of plain ``self._*``
        # attributes is what makes a single ``AgentEngine`` instance safe to
        # share across many concurrent sessions — see
        # ``runtime/session_runtime.py`` for the rationale.
        #
        # Note: per-turn retry counters (empty_response_retries,
        # provider_error_retries) live on ``TurnContext.metadata`` instead
        # of here, because their lifetime is exactly one turn.
        self._session_runtime = SessionRuntimeRegistry()
        if memory_provider is not None:
            effective_memory_provider = memory_provider
            self._project_memory_store: ProjectMemoryStore | None = None
        else:
            project_store: ProjectMemoryStore | None = None
            try:
                project_store = ProjectMemoryStore(Path.cwd())
            except Exception:
                pass
            self._project_memory_store = project_store
            effective_memory_provider = RetrievalMemoryProvider(
                session_runtime=self._session_runtime,
                project_store=project_store,
            )

        # When a fallback chain is supplied, defer to it for the
        # active provider — see ``EngineServices.provider``.  The
        # constructor-time ``provider`` argument is then only used as
        # the fallback if no chain is configured.  We deliberately
        # accept *both* arguments so callers that only have one
        # provider can keep the old call shape, and callers that have
        # a chain can still pass ``provider=chain.current_provider``
        # for type-checker friendliness even though it's redundant.
        self.services = EngineServices(
            provider=provider,
            tool_registry=tool_registry,
            middleware_pipeline=middleware_pipeline or MiddlewarePipeline(),
            interrupt_controller=interrupt_controller or InterruptController(),
            logger=logger or logging.getLogger(__name__),
            recovery_strategy=recovery_strategy,
            fallback_chain=fallback_chain,
            steer_inbox=steer_inbox,
            memory_provider=effective_memory_provider,
        )
        if self.services.middleware_pipeline.logger is None:
            self.services.middleware_pipeline.logger = self.services.logger

        self._delegate_depth = max(0, int(delegate_depth))
        self._subagent_id = subagent_id
        self._parent_subagent_id = parent_subagent_id
        self._subagent_manager = subagent_manager
        # Optional shared skill catalog.  When
        # set the SkillTool will look here if it was registered without
        # an explicit catalog.  Stored on the engine so subagents can
        # inherit (see DefaultSubagentBuilder).
        self._skill_catalog = skill_catalog
        # Optional shared LSP and browser managers. Both default to
        # ``None`` so callers that
        # never use the corresponding tool pay zero cost (no LSP
        # binary lookup, no Playwright import).  When set, they're
        # forwarded into ``TurnContext.metadata`` for tools to pick up.
        self._lsp_manager = lsp_manager
        self._browser_manager = browser_manager
        self._compaction_pipeline: CompactionPipeline | None = None
        self._compaction_pipeline_provider: ModelProvider | None = None

        self._current_session_id: str | None = None
        self._active_children: dict[str, AgentEngine] = {}
        self._active_children_lock = RLock()
        self._session_store = session_store or InMemorySessionStore()
        self._hooks = hooks or EngineHooks()

    @property
    def delegate_depth(self) -> int:
        return self._delegate_depth

    @property
    def subagent_id(self) -> str | None:
        return self._subagent_id

    @property
    def parent_subagent_id(self) -> str | None:
        return self._parent_subagent_id

    @property
    def subagent_manager(self) -> "SubagentManager | None":
        return self._subagent_manager

    def set_subagent_manager(self, manager: "SubagentManager") -> None:
        self._subagent_manager = manager

    def _register_child(self, child: "AgentEngine") -> None:
        key = child.subagent_id or f"child-{id(child)}"
        with self._active_children_lock:
            self._active_children[key] = child

    def _unregister_child(self, child: "AgentEngine") -> None:
        with self._active_children_lock:
            to_remove = [k for k, v in self._active_children.items() if v is child]
            for key in to_remove:
                self._active_children.pop(key, None)

    def _interrupt_active_children(self, reason: str | None = None) -> None:
        with self._active_children_lock:
            children = list(self._active_children.values())
        for child in children:
            child.interrupt(reason=reason)

    def interrupt(self, session_id: str | None = None, reason: str | None = None) -> None:
        effective_session = session_id or self._current_session_id
        if effective_session:
            self.services.interrupt_controller.request(effective_session, reason)
            self.services.steer_inbox.clear(effective_session)
        self._interrupt_active_children(reason=reason)

    def clear_interrupt(self, session_id: str | None = None) -> None:
        effective_session = session_id or self._current_session_id
        if effective_session:
            self.services.interrupt_controller.clear(effective_session)

    def send_steer(self, session_id: str, text: str) -> bool:
        return self.services.steer_inbox.append(session_id, text)

    def run_subagents(
        self,
        tasks: List["SubagentTask"],
        *,
        max_concurrent_children: int | None = None,
    ) -> List["SubagentResult"]:
        if self._subagent_manager is None:
            raise RuntimeError("Subagent manager is not configured")
        return self._subagent_manager.run_tasks(
            parent=self,
            tasks=tasks,
            max_concurrent_children=max_concurrent_children,
        )

    def resume(self, request: EngineRequest) -> EngineResult:
        return self.run_loop(request)

    def run_turn(self, request: EngineRequest) -> EngineResult:
        return self.run_loop(request)

    def run_loop(self, request: EngineRequest) -> EngineResult:
        """Execute one full agent loop turn until completion/failure/interruption."""
        self._current_session_id = request.session_id
        context: TurnContext | None = None
        active_system_prompt: str | None = None
        stream_callback_wrapped = None

        final_response: str | None = None
        error_text: str | None = None
        exit_reason = ExitReason.EMPTY_RESPONSE
        iterations = 0

        try:
            state_machine, messages, context = self._prepare_turn_entry(request)
            messages, active_system_prompt = self._prepare_session_and_system_prompt(
                request,
                messages,
                context,
            )
            self._apply_turn_nudges(context)
            stream_callback_wrapped = self._build_stream_callback(request, context)
            stream_silent_callback_wrapped = self._build_stream_silent_callback(
                request, context
            )

            state_machine.transition(LoopState.PREPARE)
            if self._is_interrupted(request.session_id, context):
                state_machine.transition(LoopState.INTERRUPTED)
                exit_reason = ExitReason.INTERRUPTED
            else:
                state_machine.transition(LoopState.PRE_LLM)

                # Track the live iteration budget on ``context.metadata``
                # so ``_handle_max_iterations`` and other observability
                # hooks can inspect it without extra argument plumbing.
                #
                # ``iterations`` stays a **monotonic** loop-body counter
                # and ``context.iteration`` stays a 1-indexed
                # observability value for middleware, hooks, and CLI
                # output. Cheap-tool refunds only affect
                # ``budget.used``; they do not rewrite the exposed
                # iteration count.
                budget = IterationBudget(max_total=self.config.max_iterations)
                context.metadata["_iteration_budget_obj"] = budget
                context.metadata["iteration_budget"] = budget.to_dict()

                # Main iterative loop: each iteration can produce tool calls or a terminal text response.
                # ``iterations`` is bumped AFTER a successful LLM round
                # (see the ``iterations += 1`` further down) so failed
                # turns surface the same iteration count to
                # ``_build_result``.  ``context.iteration`` is the
                # 1-indexed "we are about to issue round N" value used
                # by middleware and phantom-tool ID prefixes. Cheap-tool
                # refunds affect ``budget.used``, not this observability
                # counter.
                while budget.consume():
                    context.iteration = iterations + 1

                    if self._is_interrupted(request.session_id, context):
                        state_machine.transition(LoopState.INTERRUPTED)
                        exit_reason = ExitReason.INTERRUPTED
                        break

                    if not context.metadata.get("_preflight_compaction_done"):
                        context.metadata["_preflight_compaction_done"] = True
                        preflight = self._maybe_compact_messages(
                            messages,
                            context=context,
                            trigger_reason="preflight",
                        )
                        if preflight is not None:
                            messages = preflight.compressed_messages

                    hook_outcome = self._collect_pre_llm_hook_outcome(
                        "pre_llm_call",
                        session_id=request.session_id,
                        iteration=context.iteration,
                        messages=copy.deepcopy(messages),
                        context_metadata=context.metadata,
                    )

                    # PRE_LLM middleware stage: rewrite/enrich outbound message list.
                    #  continuation path can override the message list
                    # for exactly one next iteration (partial assistant +
                    # continuation user instruction).  Pop it here so retries
                    # do not accidentally keep reusing a stale override.
                    loop_messages = context.metadata.pop("_messages_override", None)
                    if isinstance(loop_messages, list):
                        messages = loop_messages

                    hook_outcome = self._merge_memory_context_into_hook_outcome(
                        messages,
                        hook_outcome,
                        context=context,
                    )

                    outbound_messages = self._apply_hook_outcome_to_messages(
                        messages,
                        hook_outcome,
                        context=context,
                    )

                    try:
                        prepared_messages = self.services.middleware_pipeline.run_before_llm(outbound_messages, context)
                    except Exception as exc:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break

                    # projection-view application.
                    # Tier 4 (``ContextCollapseTier``) writes a
                    # ``CollapseStore`` onto ``context.metadata`` but
                    # never mutates the local ``messages`` list (so
                    # session_record / replay sees the un-collapsed
                    # past).  The view is applied here, on the
                    # post-middleware payload, so every provider call
                    # within this turn sees the same projection — and
                    # only the wire payload changes, not the stored
                    # transcript.
                    prepared_messages = self._apply_collapse_view(
                        prepared_messages, context
                    )

                    state_machine.transition(LoopState.LLM_CALL)
                    # Allow middleware to short-circuit the provider call (e.g. circuit breaker).
                    response = hook_outcome.short_circuit_response
                    if response is None:
                        response = self._pop_context_response(context, "llm_pre_response")
                    if response is None:
                        # ProviderInvocationError goes through the engine
                        # recovery strategy. Any other Exception bypasses
                        # that strategy and goes straight to middleware
                        # ``on_error`` so programming errors and scripted
                        # test provider failures stay visible.
                        invoke_outcome = self._invoke_provider_with_recovery(
                            request=request,
                            canonical_messages=messages,
                            prepared_messages=prepared_messages,
                            stream_callback=stream_callback_wrapped,
                            stream_silent_callback=stream_silent_callback_wrapped,
                            context=context,
                        )
                        if invoke_outcome.interrupted:
                            state_machine.transition(LoopState.INTERRUPTED)
                            exit_reason = ExitReason.INTERRUPTED
                            break
                        if invoke_outcome.response is not None:
                            response = invoke_outcome.response
                        else:
                            # Recovery exhausted (or never applicable): fall
                            # through to the existing middleware on_error
                            # pipeline so it can build a user-facing message.
                            exc = invoke_outcome.error
                            assert exc is not None  # type-checker: mutually exclusive with response
                            self._handle_pipeline_error(exc, state_machine.state, context)
                            recovered_response = self._pop_context_response(context, "llm_error_response")
                            if recovered_response is None:
                                error_text = str(exc)
                                # the recovery strategy
                                # may have pinned a more specific terminal
                                # (RATE_LIMITED / CONTEXT_EXHAUSTED /
                                # PAYLOAD_TOO_LARGE / FALLBACK_EXHAUSTED)
                                # via context.metadata.  Honour that hint
                                # before falling back to the default
                                # PROVIDER_ERROR / RESPONSE_INVALID picks
                                # so observability surfaces the precise
                                # cause of the give-up.
                                terminal_hint = context.metadata.pop(
                                    "recovery_terminal_exit_reason", None
                                )
                                exit_reason = self._resolve_terminal_exit_reason(
                                    terminal_hint, exc
                                )
                                state_machine.transition(LoopState.FAILED)
                                break
                            response = recovered_response

                    state_machine.transition(LoopState.POST_LLM)
                    # POST_LLM middleware stage: normalize/annotate response before control-flow split.
                    try:
                        response = self.services.middleware_pipeline.run_after_llm(response, context)
                    except Exception as exc:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        error_text = str(exc)
                        exit_reason = ExitReason.MIDDLEWARE_ERROR
                        state_machine.transition(LoopState.FAILED)
                        break

                    # accumulate token usage per LLM call.
                    # We do this AFTER the after_llm middleware pipeline so any
                    # response-level normalisation it performed is reflected in
                    # the raw ``metadata["usage"]`` dict we read here.  The
                    # accumulator lives on context.metadata so it is per-turn
                    # and per-session safe (no cross-session leak).
                    self._accumulate_usage(response, context)
                    iterations += 1

                    self._safe_call_hook(
                        "post_llm_call",
                        session_id=request.session_id,
                        iteration=context.iteration,
                        response_text=response.content or "",
                        context_metadata=context.metadata,
                    )

                    length_text = (response.content or "").strip()
                    if response.finish_reason == "length" and (
                        response.tool_calls or length_text
                    ):
                        # ``finish_reason="length"`` +
                        # tool_calls is a structurally different failure
                        # from prose-truncation.  The continuation prompt
                        # used by ordinary length continuation cannot
                        # finish a half-emitted JSON
                        # tool argument, so we route this case through a
                        # dedicated retry-once-then-refuse handler instead
                        # of the regular length-continuation path.
                        if (
                            response.tool_calls
                            and getattr(self.config, "truncated_tool_call_detection_enabled", True)
                        ):
                            handled = self._handle_length_with_tool_calls(
                                response=response,
                                messages=messages,
                                context=context,
                            )
                        else:
                            handled = self._handle_length_finish_reason(
                                response=response,
                                messages=messages,
                                request=request,
                                context=context,
                            )
                        if handled.action == "continue":
                            messages = handled.messages
                            state_machine.transition(LoopState.CHECK_EXIT)
                            if budget.exhausted:
                                state_machine.transition(LoopState.FINALIZE)
                                exit_reason = ExitReason.MAX_ITERATIONS
                                break
                            state_machine.transition(LoopState.PRE_LLM)
                            continue
                        if handled.action == "finalize":
                            messages = handled.messages
                            final_response = handled.final_response
                            exit_reason = handled.exit_reason
                            state_machine.transition(LoopState.FINALIZE)
                            break

                    if response.tool_calls:
                        # gate the dispatcher on
                        # tool-call argument validation.  The validator
                        # normalises argument types in place, decides
                        # whether the response is truncated (and we
                        # should retry without poisoning history), and
                        # falls back to a tool-error injection so the
                        # model can self-correct on persistent JSON
                        # mistakes.  See ``_validate_tool_call_arguments``
                        # for the full state machine.
                        if getattr(self.config, "truncated_tool_call_detection_enabled", True):
                            validation = self._validate_tool_call_arguments(
                                response=response,
                                messages=messages,
                                context=context,
                            )
                            if validation.action == "retry":
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if budget.exhausted:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue
                            if validation.action == "truncated":
                                rollback = self._get_messages_up_to_last_assistant(messages)
                                visible_text = self._extract_visible_text(response.content or "")
                                if visible_text:
                                    prefix_parts = context.metadata.setdefault(
                                        "truncated_response_prefix_parts", []
                                    )
                                    if isinstance(prefix_parts, list):
                                        prefix_parts.append(visible_text)
                                context.metadata["partial"] = True
                                context.metadata.setdefault(
                                    "length_exit_reason", "tool_call_truncated"
                                )
                                messages = rollback
                                final_response = visible_text or None
                                exit_reason = ExitReason.TOOL_CALL_TRUNCATED
                                state_machine.transition(LoopState.FINALIZE)
                                break
                            if validation.action == "inject_error":
                                # Append assistant tool_calls + per-call
                                # tool error stubs.  The model sees the
                                # JSON it produced and the parse error,
                                # then has the next iteration to retry.
                                tool_result_start_idx = len(messages)
                                messages.extend(validation.injection_messages)
                                self._apply_pending_steer_to_tool_results(
                                    messages,
                                    session_id=request.session_id,
                                    start_idx=tool_result_start_idx,
                                    context=context,
                                )
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if budget.exhausted:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue
                            # action == "ok" → fall through to dispatch.
                        self._register_skill_nudge(context)

                        # Tool path: append assistant tool-call message first, then execute each call.
                        state_machine.transition(LoopState.TOOL_DISPATCH)
                        self._append_assistant_tool_message(messages, response)
                        tool_result_start_idx = len(messages)

                        # sanitise the call batch
                        # before dispatch:
                        #   * fuzzy-repair tool names
                        #   * cap delegate-class fan-out per turn
                        #   * dedup identical calls
                        # Returns a plan whose entries either dispatch
                        # normally or carry a synthetic ToolResult that
                        # the engine appends in lieu of dispatching.
                        # When the per-turn unrepairable-name budget is
                        # exhausted the plan also pins a terminal
                        # exit reason (INVALID_TOOL_REPEATED).
                        dispatch_plan = prepare_tool_calls(
                            response.tool_calls,
                            registry=self.services.tool_registry,
                            config=self.config,
                            context=context,
                        )
                        # Surface counters in turn metadata for
                        # observability — the CLI footer reads these
                        # to render a "↻ repaired 1, deduped 2" hint.
                        if dispatch_plan.repaired_count:
                            context.metadata["tool_names_repaired"] = (
                                int(context.metadata.get("tool_names_repaired", 0))
                                + dispatch_plan.repaired_count
                            )
                        if dispatch_plan.deduped_count:
                            context.metadata["tool_calls_deduped"] = (
                                int(context.metadata.get("tool_calls_deduped", 0))
                                + dispatch_plan.deduped_count
                            )
                        if dispatch_plan.capped_count:
                            context.metadata["tool_calls_capped"] = (
                                int(context.metadata.get("tool_calls_capped", 0))
                                + dispatch_plan.capped_count
                            )

                        if dispatch_plan.exit_reason is None:
                            schema_injection = self._maybe_inject_schema_errors(
                                dispatch_plan=dispatch_plan,
                                response=response,
                                context=context,
                            )
                            if schema_injection is not None:
                                state_machine.transition(LoopState.TOOL_EXECUTE)
                                tool_result_start_idx = len(messages)
                                messages.extend(schema_injection)
                                self._apply_pending_steer_to_tool_results(
                                    messages,
                                    session_id=request.session_id,
                                    start_idx=tool_result_start_idx,
                                    context=context,
                                )
                                state_machine.transition(LoopState.CHECK_EXIT)
                                if budget.exhausted:
                                    state_machine.transition(LoopState.FINALIZE)
                                    exit_reason = ExitReason.MAX_ITERATIONS
                                    break
                                state_machine.transition(LoopState.PRE_LLM)
                                continue

                        state_machine.transition(LoopState.TOOL_EXECUTE)
                        tool_failed = False
                        for prepared in dispatch_plan.prepared:
                            call = prepared.call
                            if self._is_interrupted(request.session_id, context):
                                self._record_interrupt_metadata(
                                    context,
                                    was_in_tool_call=True,
                                )
                                state_machine.transition(LoopState.INTERRUPTED)
                                exit_reason = ExitReason.INTERRUPTED
                                break

                            # Skip dispatch when the sanitiser already
                            # produced a synthetic result (cap / dedup /
                            # unrepairable name).  We still go through
                            # the after_tool middleware path so the
                            # synthetic result gets the same redaction
                            # / observability treatment as a real one.
                            if prepared.synthetic_result is not None:
                                result = prepared.synthetic_result
                                try:
                                    result = self.services.middleware_pipeline.run_after_tool(result, context)
                                except Exception as exc:
                                    self._handle_pipeline_error(exc, state_machine.state, context)
                                    error_text = str(exc)
                                    exit_reason = ExitReason.MIDDLEWARE_ERROR
                                    state_machine.transition(LoopState.FAILED)
                                    tool_failed = True
                                    break
                                self._record_tool_result_error(context, result)
                                self._append_tool_result_message(messages, result)
                                continue

                            permission_checked = self._apply_tool_permission_gate(
                                call,
                                request=request,
                                context=context,
                            )

                            if isinstance(permission_checked, ToolResult):
                                result = permission_checked
                            else:
                                try:
                                    # before_tool can rewrite a ToolCall or short-circuit with ToolResult
                                    # (for guardrails/policy blocks).
                                    pre_tool = self.services.middleware_pipeline.run_before_tool(
                                        permission_checked,
                                        context,
                                    )
                                except Exception as exc:
                                    self._handle_pipeline_error(exc, state_machine.state, context)
                                    error_text = str(exc)
                                    exit_reason = ExitReason.MIDDLEWARE_ERROR
                                    state_machine.transition(LoopState.FAILED)
                                    tool_failed = True
                                    break

                                if isinstance(pre_tool, ToolResult):
                                    result = pre_tool
                                    tool_call = None
                                else:
                                    tool_call = pre_tool

                            if not isinstance(permission_checked, ToolResult) and tool_call is not None:
                                # Track active call so middleware on_error handlers can build
                                # a deterministic fallback ToolResult for this exact invocation.
                                context.metadata.pop("tool_error_result", None)
                                context.metadata["_active_tool_call"] = tool_call
                                context.metadata["_tool_interrupt_behavior"] = getattr(
                                    self.services.tool_registry.get(tool_call.name),
                                    "interrupt_behavior",
                                    "block",
                                )
                                try:
                                    result = self.services.tool_registry.dispatch(tool_call, context)
                                except UnknownToolError:
                                    # repair already
                                    # ran and produced no match; this
                                    # branch is now hit only if the
                                    # registry mutated mid-turn (rare).
                                    # Honour ``fail_on_unknown_tool`` for
                                    # backward compat with callers that
                                    # explicitly want hard failure.
                                    if self.config.fail_on_unknown_tool:
                                        error_text = f"Unknown tool: {tool_call.name}"
                                        exit_reason = ExitReason.UNKNOWN_TOOL
                                        state_machine.transition(LoopState.FAILED)
                                        tool_failed = True
                                        break
                                    result = ToolResult(
                                        tool_call_id=tool_call.id,
                                        name=tool_call.name,
                                        content=self._format_unknown_tool_content(
                                            tool_call.name,
                                            context=context,
                                        ),
                                        is_error=True,
                                        metadata={
                                            "_unknown_tool_recovery": True,
                                            "_tool_error_category": "unknown_tool",
                                        }
                                        if getattr(
                                            self.config,
                                            "tool_error_structured_format_enabled",
                                            True,
                                        )
                                        else {},
                                    )
                                except Exception as exc:
                                    if self.config.fail_on_tool_error:
                                        # If strict mode is enabled, middleware may still recover
                                        # by providing a synthetic ToolResult in metadata.
                                        self._handle_pipeline_error(exc, state_machine.state, context)
                                        recovered_tool_result = context.metadata.pop("tool_error_result", None)
                                        if not isinstance(recovered_tool_result, ToolResult):
                                            error_text = str(exc)
                                            exit_reason = ExitReason.TOOL_ERROR
                                            state_machine.transition(LoopState.FAILED)
                                            tool_failed = True
                                            break
                                        result = recovered_tool_result
                                    else:
                                        result = ToolResult(
                                            tool_call_id=tool_call.id,
                                            name=tool_call.name,
                                            content=f"Tool execution error: {exc}",
                                            is_error=True,
                                        )
                                finally:
                                    context.metadata.pop("_active_tool_call", None)
                                    context.metadata.pop("_tool_interrupt_behavior", None)

                            try:
                                # after_tool middleware stage for redaction, auditing, or shaping.
                                result = self.services.middleware_pipeline.run_after_tool(result, context)
                            except Exception as exc:
                                self._handle_pipeline_error(exc, state_machine.state, context)
                                error_text = str(exc)
                                exit_reason = ExitReason.MIDDLEWARE_ERROR
                                state_machine.transition(LoopState.FAILED)
                                tool_failed = True
                                break

                            self._record_tool_result_error(context, result)
                            self._append_tool_result_message(messages, result)
                            if bool(result.metadata.get("interrupted")):
                                self._record_interrupt_metadata(
                                    context,
                                    was_in_tool_call=True,
                                )
                            if self._is_permission_abort_result(result):
                                self._record_interrupt_metadata(
                                    context,
                                    was_in_tool_call=True,
                                )
                                exit_reason = ExitReason.INTERRUPTED
                                state_machine.transition(LoopState.INTERRUPTED)
                                break

                        if state_machine.state in {LoopState.FAILED, LoopState.INTERRUPTED}:
                            break
                        if tool_failed:
                            break
                        self._apply_pending_steer_to_tool_results(
                            messages,
                            session_id=request.session_id,
                            start_idx=tool_result_start_idx,
                            context=context,
                        )

                        # when the per-turn
                        # invalid-tool retry budget is exhausted the
                        # sanitiser pinned ``dispatch_plan.exit_reason``.
                        # Synthetic ToolResults for each unrepairable
                        # name have already been appended to the
                        # message stream above, so the model has the
                        # full diagnostic context — we just need to
                        # finalise this turn instead of looping again.
                        # Walk through CHECK_EXIT first so the state
                        # machine's TOOL_EXECUTE → CHECK_EXIT → FINALIZE
                        # contract holds (TOOL_EXECUTE → FINALIZE is
                        # explicitly disallowed in state_machine.py).
                        if dispatch_plan.exit_reason is not None:
                            try:
                                exit_reason = ExitReason(dispatch_plan.exit_reason)
                            except ValueError:
                                exit_reason = ExitReason.UNKNOWN_TOOL
                            context.metadata["partial"] = True
                            state_machine.transition(LoopState.CHECK_EXIT)
                            state_machine.transition(LoopState.FINALIZE)
                            break

                        # cheap-tool refund.  When
                        # the model's tool_calls for this round are
                        # *entirely* drawn from the cheap-tool
                        # whitelist (todo bookkeeping, memory writes,
                        # session search...), refund the budget slot
                        # this iteration consumed so the model gets
                        # the same effective headroom it would have
                        # had without the bookkeeping call.  Mixed
                        # iterations (one cheap call + one real call)
                        # do NOT refund — the real call still
                        # warranted the slot.  Refund happens BEFORE
                        # the exhaustion check so a cheap-only
                        # iteration that would otherwise have been
                        # the budget-exhausting one keeps the loop
                        # alive for substantive follow-up work.
                        if response.tool_calls and all(
                            self._is_cheap_tool(call.name)
                            for call in response.tool_calls
                        ):
                            budget.refund()
                            context.metadata["iteration_budget"] = budget.to_dict()

                        # Continue iterative tool-use loop unless max iteration budget is exhausted.
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break

                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    # No tool calls -> finalize response (or recover
                    # from phantom-tool intent first).
                    prefix_parts = context.metadata.pop("truncated_response_prefix_parts", None)
                    prefix = " ".join(part.strip() for part in prefix_parts if isinstance(part, str) and part.strip()) if isinstance(prefix_parts, list) else ""
                    suffix = (response.content or "").strip()
                    combined_content = ((prefix + " " + suffix).strip() if prefix and suffix else (prefix or suffix))
                    response_to_store = response
                    if combined_content != (response.content or ""):
                        response_to_store = NormalizedResponse(
                            content=combined_content,
                            tool_calls=list(response.tool_calls),
                            finish_reason=response.finish_reason,
                            metadata=dict(response.metadata),
                        )

                    # Phantom-tool recovery — when the assistant body
                    # carries clear evidence of attempted tool use
                    # (\u0060\u0060\u0060bash blocks, ``<function=NAME>`` inline
                    # tags, ``<invoke>`` XML) but ``tool_calls`` is
                    # empty, the model is "describing" rather than
                    # invoking.  Without this branch the loop would
                    # silently finalise as TEXT_RESPONSE, leaving the
                    # user with a green checkmark and no work done.
                    #
                    # Three outcomes:
                    #
                    # * ``"retry"`` — append the assistant's prose to
                    #   history, append a corrective ``role=user``
                    #   nudge, bump the retry counter, continue the
                    #   loop and let the model self-correct.
                    # * ``"exhausted"`` — phantom intent was present
                    #   but ``max_phantom_tool_retries`` already
                    #   burned; fall through to finalise but tag the
                    #   turn ``PHANTOM_TOOL_INTENT`` so the UI shows
                    #   "model never invoked anything" instead of a
                    #   misleading green checkmark.
                    # * ``"none"`` — body was prose, no recovery
                    #   needed; fall through unchanged.
                    phantom_outcome = self._maybe_recover_phantom_tool_intent(
                        response_to_store=response_to_store,
                        messages=messages,
                        context=context,
                    )
                    if phantom_outcome == "synthesized":
                        # The recovery method has populated
                        # ``response_to_store.tool_calls`` from the
                        # parsed prose.  Run the synthesized calls
                        # through the dispatch path inline so the
                        # next LLM iteration sees a clean tool/result
                        # pair, just as if the model had emitted
                        # structured tool_calls in the first place.
                        synth_outcome, synth_exit, synth_error = self._dispatch_synthesized_tool_calls(
                            response=response_to_store,
                            messages=messages,
                            context=context,
                            state_machine=state_machine,
                            request=request,
                        )
                        if synth_outcome == "interrupted":
                            exit_reason = ExitReason.INTERRUPTED
                            break
                        if synth_outcome == "failed":
                            assert synth_exit is not None
                            exit_reason = synth_exit
                            error_text = synth_error
                            break
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    if phantom_outcome == "retry":
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue

                    finalized = self._finalize_empty_response(
                        response=response,
                        response_to_store=response_to_store,
                        messages=messages,
                        context=context,
                        request=request,
                        phantom_outcome=phantom_outcome,
                        prefix=prefix,
                    )
                    if finalized.final_response is _CONTINUE_LOOP_SENTINEL:
                        state_machine.transition(LoopState.CHECK_EXIT)
                        if budget.exhausted:
                            state_machine.transition(LoopState.FINALIZE)
                            exit_reason = ExitReason.MAX_ITERATIONS
                            break
                        state_machine.transition(LoopState.PRE_LLM)
                        continue
                    final_response = (
                        finalized.final_response
                        if isinstance(finalized.final_response, str)
                        else None
                    )
                    exit_reason = finalized.exit_reason or ExitReason.EMPTY_RESPONSE
                    error_text = finalized.error_text
                    state_machine.transition(LoopState.FINALIZE)
                    break

            # Force a deterministic terminal state if loop exits by condition rather than break.
            if state_machine.state not in {LoopState.FAILED, LoopState.INTERRUPTED, LoopState.FINALIZE}:
                state_machine.transition(LoopState.FINALIZE)
                if budget.exhausted:
                    exit_reason = ExitReason.MAX_ITERATIONS

            # max-iterations summary fallback.
            # Triggered ONCE per turn (guarded by ``IterationBudget.grace_call``)
            # when the loop terminated because the iteration budget
            # was exhausted.  Generates a one-shot non-tool LLM
            # response so the user sees what got done instead of an
            # empty ``final_response``.  No-op when:
            #   * exit_reason isn't MAX_ITERATIONS (other terminations
            #     either succeeded or have their own diagnostic text);
            #   * ``final_response`` already carries text (defensive —
            #     we never want to clobber a real model answer);
            #   * ``summary_on_budget_exhausted`` is False (rollback);
            #   * the grace round was already consumed (defensive).
            if (
                exit_reason == ExitReason.MAX_ITERATIONS
                and not final_response
                and "budget" in locals()
            ):
                summary_text = self._handle_max_iterations(
                    request, messages, context
                )
                if summary_text:
                    final_response = summary_text

            # FINALIZE -> DONE transition for successful/terminal completion paths.
            if state_machine.state == LoopState.FINALIZE:
                state_machine.transition(LoopState.DONE)

            pending_steer = self.services.steer_inbox.drain(request.session_id)
            if pending_steer:
                context.metadata["pending_steer"] = pending_steer

            self._observe_memory_turn(messages, context)

            result = self._build_result(
                request,
                messages,
                iterations,
                final_response,
                error_text,
                exit_reason,
                context=context,
                active_system_prompt=active_system_prompt,
            )
            self._save_trajectory_if_enabled(
                result=result,
                messages=messages,
                context=context,
            )
            self._cleanup_task_resources_if_needed(
                result=result,
                context=context,
            )
            self.services.interrupt_controller.clear(request.session_id)

            self._safe_call_hook(
                "on_session_end",
                session_id=request.session_id,
                completed=result.status in {EngineStatus.COMPLETED, EngineStatus.MAX_ITERATIONS},
                interrupted=result.status == EngineStatus.INTERRUPTED,
                context_metadata=context.metadata,
            )
            return result
        finally:
            if context is not None and not context.metadata.get("_task_cleanup_done"):
                self._cleanup_task_resources(
                    context=context,
                    completed=False,
                    interrupted=self._is_interrupted(context.session_id, context),
                )
            if context is not None:
                self.clear_interrupt(session_id=context.session_id)
            self._current_session_id = None

    def _get_compaction_pipeline(self) -> CompactionPipeline:
        """Return a compaction pipeline bound to the currently-active provider."""
        provider = self.services.provider
        if (
            self._compaction_pipeline is None
            or self._compaction_pipeline_provider is not provider
        ):
            # both Tier 4 (collapse) and
            # Tier 5 (autocompact) need an LLM-fork summariser.  We
            # construct a single shared instance and let both tiers
            # share it: same provider, same usage-bridge, identical
            # accounting semantics.  Sharing avoids double-binding
            # the provider object and keeps the usage_sink wiring
            # in one place.
            shared_summarizer = LLMForkSummarizer(
                provider=provider,
                config=self.config,
                logger=self.services.logger,
                # bridge fork-call usage back
                # into the parent turn's accumulator so the cost of
                # compaction is *not* silently excluded from
                # ``metadata['usage']`` / ``api_calls``.
                usage_sink=self._accumulate_usage,
            )
            self._compaction_pipeline = CompactionPipeline(
                tiers=[
                    # Tier 2 redundancy snipper.
                    # Cheap, local, no LLM round-trip; deletes
                    # repeated read_file/list_dir/glob/grep calls,
                    # superseded failed pairs, and empty assistant
                    # frames.  Runs first so downstream tiers see
                    # the lean view (their token estimates and
                    # commit/blocking thresholds get the benefit).
                    Snipper(
                        config=self.config,
                        logger=self.services.logger,
                    ),
                    # Tier 3 lives between snip
                    # (Tier 2) and collapse (Tier 4).  Cheap and local;
                    # consults ``_aether_meta.timestamp`` stamped on
                    # every assistant message we emit.
                    TimeBasedMicrocompactor(
                        config=self.config,
                        logger=self.services.logger,
                    ),
                    # Tier 4 projection-based
                    # collapse.  Default disabled
                    # (``context_collapse_enabled=False``); when on,
                    # owns the headroom flag that Tier 5 respects so
                    # the two tiers don't both fire on the same
                    # pass.
                    ContextCollapseTier(
                        config=self.config,
                        summarizer=shared_summarizer,
                        logger=self.services.logger,
                    ),
                    AutoCompactor(
                        config=self.config,
                        summarizer=shared_summarizer,
                        logger=self.services.logger,
                    ),
                ],
                token_estimator=estimate_messages_tokens,
                config=self.config,
                logger=self.services.logger,
            )
            self._compaction_pipeline_provider = provider
        return self._compaction_pipeline

    def _apply_collapse_view(
        self,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> List[Dict[str, Any]]:
        """Return ``CollapseStore.as_view(messages)`` if a store is live.

        applied right before each
        ``provider.generate`` call so every provider invocation in
        this turn sees the same projection.  Returns ``messages``
        unchanged when:

        * compaction is disabled at master level (no store can exist),
        * Tier 4 itself is disabled (no store will exist),
        * a store exists but has no committed segments yet (``as_view``
          short-circuits to an identity return).

        Crucially we do NOT touch the local ``messages`` list — the
        view is a *projection* that only affects the wire payload, so
        ``session_record`` / replay / steer / transcript export still
        see the full original conversation.
        """
        store = context.metadata.get("_collapse_store")
        if not isinstance(store, CollapseStore):
            return messages
        return store.as_view(messages)

    def _maybe_compact_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> CompactionResult | None:
        """Run the compaction pipeline when compression is enabled."""
        if not getattr(self.config, "compression_enabled", False):
            return None

        self._memory_before_compaction(messages, context)

        provider = self.services.provider
        model = str(getattr(provider, "model", "") or "unknown")
        result = self._get_compaction_pipeline().maybe_compress(
            messages,
            turn_context=context,
            model=model,
            model_window=self._resolve_model_window(),
            trigger_reason=trigger_reason,
        )
        if result.tiers_run:
            context.metadata["compaction_last_result"] = {
                "trigger_reason": trigger_reason,
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "tiers_run": list(result.tiers_run),
                "exhausted": result.exhausted,
            }
        return result

    def _resolve_model_window(self) -> int:
        """Best-effort resolve the active model's context window."""
        provider = self.services.provider
        for attr in ("context_window", "max_context_tokens", "max_context_length"):
            window = getattr(provider, attr, None)
            if isinstance(window, int) and window > 0:
                return window

        model = str(getattr(provider, "model", "") or "").lower()
        if "claude" in model:
            return 200_000
        if "kimi" in model or "moonshot" in model:
            return 200_000
        if "gpt-4o" in model or "gpt-4-turbo" in model:
            return 128_000
        if "gpt-4.1" in model or "gpt-5" in model:
            return 128_000
        return 32_000

    def _prepare_turn_entry(
        self,
        request: EngineRequest,
    ) -> tuple[EngineStateMachine, List[Dict[str, Any]], TurnContext]:
        messages = self._sanitize_messages(copy.deepcopy(request.messages))

        if request.user_message is not None:
            messages.append({"role": "user", "content": self._sanitize_text(request.user_message)})

        task_id = self._sanitize_text(str(request.metadata.get("task_id", "")).strip()) if request.metadata else ""
        if not task_id:
            task_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())

        metadata = dict(request.metadata)
        loop_state_callback = metadata.get("_loop_state_callback")
        state_machine = EngineStateMachine(
            on_transition=loop_state_callback if callable(loop_state_callback) else None
        )
        turn_start_idx = len(messages)
        # stamp the active provider's identity onto the
        # turn so middleware (TokenUsageMiddleware) and downstream tools can
        # pick the right ``normalize_usage`` parser without dragging
        # ``EngineServices`` into the middleware contract.  Re-read from
        # ``self.services.provider`` each turn so the fallback chain
        # (which swaps the active provider mid-session) gets a fresh value.
        active_provider = self.services.provider
        metadata.update(
            {
                "entry_prepared": True,
                "task_id": task_id,
                "turn_id": turn_id,
                "turn_start_idx": turn_start_idx,
                "request_has_stream_callback": bool(request.stream_callback),
                "active_provider_name": getattr(active_provider, "provider_name", "openai"),
                "active_provider_api_mode": getattr(active_provider, "api_mode", "chat"),
                "memory": default_memory_metadata(
                    enabled=bool(getattr(self.config, "memory_enabled", False)),
                    mode=str(getattr(self.config, "memory_mode", "off") or "off"),
                ),
                # Per-turn retry counters live on TurnContext.metadata so they
                # cannot leak across concurrent sessions sharing this engine.
                # Initialised to 0 here; mutated by the run-loop's empty-
                # response / provider-error / truncated-tool-call branches.
                TURN_KEY_EMPTY_RESPONSE_RETRIES: 0,
                TURN_KEY_PROVIDER_ERROR_RETRIES: 0,
                TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES: 0,
                TURN_KEY_INVALID_JSON_RETRIES: 0,
                TURN_KEY_PHANTOM_TOOL_RETRIES: 0,
                TURN_KEY_PHANTOM_TOOL_SYNTHESIZED: 0,
                TURN_KEY_THINKING_PREFILL_RETRIES: 0,
                TURN_KEY_CODEX_ACK_RETRIES: 0,
                TURN_KEY_STREAMED_ASSISTANT_TEXT: "",
                TURN_KEY_POST_TOOL_EMPTY_RETRIED: False,
                TURN_KEY_EMPTY_RECOVERY_LAST_STEP: "",
            }
        )

        interrupt_signal = self._signal_for_request(request)
        metadata["_interrupt_signal"] = interrupt_signal
        context = TurnContext(
            session_id=request.session_id,
            iteration=0,
            metadata=metadata,
            task_id=task_id,
            turn_id=turn_id,
            interrupt_signal=interrupt_signal,
        )
        # inject the live ``EngineConfig`` so
        # tools can consult per-tool feature switches (currently
        # ``tool_result_spill_enabled`` / ``tool_result_spill_dir``)
        # without us threading a config argument through every
        # ToolExecutor.execute signature.  The key is in
        # ``_METADATA_INTERNAL_KEYS`` so the live dataclass never
        # leaks into the JSON-serialised EngineResult.metadata['turn'].
        context.metadata["_engine_config"] = self.config
        # expose the parent agent, the
        # subagent manager, the optional approval prompter and the
        # optional skill catalog to tools.  Same rationale as
        # ``_engine_config`` above — every key is in
        # ``_METADATA_INTERNAL_KEYS`` so live objects never leak into
        # the JSON-serialised result snapshot.
        context.metadata["_parent_agent"] = self
        context.metadata["_subagent_manager"] = self._subagent_manager
        context.metadata["_approval_prompter"] = getattr(
            request, "approval_prompter", None
        )
        context.metadata["_tool_permission_prompter"] = getattr(
            request, "tool_permission_prompter", None
        )
        context.metadata["tool_permissions"] = default_permission_stats(
            enabled=bool(getattr(self.config, "tool_permissions_enabled", True))
        )
        context.metadata["_skill_catalog"] = getattr(self, "_skill_catalog", None)
        # LSP and browser manager references. Tools fetch them lazily;
        # passing the live
        # objects (not config snapshots) keeps subagents and parent
        # sharing one warm subprocess pool.
        context.metadata["_lsp_manager"] = getattr(self, "_lsp_manager", None)
        context.metadata["_browser_manager"] = getattr(self, "_browser_manager", None)
        context.metadata["_project_memory_store"] = getattr(self, "_project_memory_store", None)
        context.metadata["empty_recovery"] = {}
        return state_machine, messages, context

    def _prepare_session_and_system_prompt(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> tuple[List[Dict[str, Any]], str | None]:
        stored_prompt = None
        is_new_session = not bool(request.messages)
        if self._session_store:
            session_row = self._session_store.get_session(request.session_id)
            if session_row:
                maybe_prompt = session_row.get("system_prompt")
                if isinstance(maybe_prompt, str) and maybe_prompt.strip():
                    stored_prompt = self._sanitize_text(maybe_prompt)
                is_new_session = False
            else:
                is_new_session = True

        requested_prompt = self._sanitize_text(request.system_message) if request.system_message else None
        selected_prompt = requested_prompt or stored_prompt

        # Augment the prompt with a registry-derived
        # <tool_use_contract> block before injecting it into messages.
        # We deliberately do NOT mutate ``selected_prompt`` itself —
        # storage / metadata / EngineResult should reflect what the
        # caller passed in, not the engine's boilerplate.  The contract
        # is regenerated each turn from the live registry, so resuming
        # a session with a different toolset still produces an accurate
        # block.
        prompt_for_messages = selected_prompt
        if self.config.tool_use_contract_enabled:
            from aether.agents.core.system_prompt import (
                augment_system_with_tool_contract,
            )

            prompt_for_messages = augment_system_with_tool_contract(
                selected_prompt,
                self.services.tool_registry.list_descriptors(),
            )

        if prompt_for_messages:
            messages = self._inject_system_prompt(messages, prompt_for_messages)

        context.metadata["system_prompt_applied"] = bool(selected_prompt)
        context.metadata["system_prompt_source"] = (
            "request_field" if requested_prompt else ("session_store" if stored_prompt else "none")
        )
        context.metadata["tool_use_contract_applied"] = bool(
            self.config.tool_use_contract_enabled
            and prompt_for_messages is not None
            and prompt_for_messages != selected_prompt
        )

        if self.config.enable_todo_hydration:
            self._hydrate_todo_snapshot(messages, context)

        if selected_prompt and self._session_store and (is_new_session or requested_prompt is not None):
            self._session_store.update_system_prompt(request.session_id, selected_prompt)

        context.metadata["system_prompt_preview"] = selected_prompt[:120] if selected_prompt else ""

        if is_new_session:
            self._safe_call_hook(
                "on_session_start",
                session_id=request.session_id,
                context_metadata=context.metadata,
            )

        return messages, selected_prompt

    def _apply_turn_nudges(self, context: TurnContext) -> None:
        """Bump the memory-review counter for this session and surface the flag.

        Cadence is per-session (not per-engine), so the counter must come from
        ``SessionRuntimeRegistry`` rather than an instance attribute.  This is
        what guarantees that two concurrent sessions sharing the same
        ``AgentEngine`` cannot interfere with each other's nudge timing.
        """
        context.metadata.setdefault("should_review_memory", False)
        context.metadata.setdefault("should_review_skills", False)

        interval = max(0, int(getattr(self.config, "memory_nudge_interval", 0)))
        if interval <= 0:
            return

        state = self._session_runtime.get(context.session_id)
        state.memory_nudge_counter += 1
        if state.memory_nudge_counter >= interval:
            context.metadata["should_review_memory"] = True
            state.memory_nudge_counter = 0

    def _register_skill_nudge(self, context: TurnContext) -> None:
        """Bump the skill-review counter on every tool-call iteration.

        Like ``_apply_turn_nudges`` this lives on ``SessionRuntimeState`` so
        the cadence cannot bleed across concurrent sessions.
        """
        interval = max(0, int(getattr(self.config, "skill_nudge_interval", 0)))
        if interval <= 0:
            return

        state = self._session_runtime.get(context.session_id)
        state.skill_nudge_counter += 1
        if state.skill_nudge_counter >= interval:
            context.metadata["should_review_skills"] = True
            state.skill_nudge_counter = 0

    def _maybe_recover_phantom_tool_intent(
        self,
        *,
        response_to_store: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> str:
        """Detect phantom-tool intent and decide what to do about it.

        Returns one of four string values:

        * ``"synthesized"`` — phantom intent was detected **and** the
          parsed prose mapped cleanly to registered tools, so the
          engine turned them into structured ``ToolCall``s.  The
          caller's ``response_to_store`` was mutated in place
          (``response_to_store.tool_calls`` now carries the synth
          calls) so the caller can fall straight through the normal
          tool-dispatch branch as if the model had emitted them.  Also
          bumps ``TURN_KEY_PHANTOM_TOOL_SYNTHESIZED`` and stashes the
          per-call notes in ``context.metadata['phantom_synth_notes']``
          so the UI can render a soft "synthesized" hint instead of
          the loud ``inline tool tags`` warning.
        * ``"retry"``  — phantom intent detected, retry budget had
          room, ``messages`` was extended with the assistant's prose
          + a corrective ``role=user`` nudge, and the per-turn
          counter was bumped.  The caller should ``continue`` the
          loop (after honouring max_iterations).
        * ``"exhausted"`` — phantom intent was present but
          ``EngineConfig.max_phantom_tool_retries`` had already been
          burned this turn.  ``messages`` is left untouched; the
          caller falls through to the normal text-response finalise
          path but should set ``exit_reason = PHANTOM_TOOL_INTENT``.
          Also stashes ``"phantom_tool_attempts"`` in
          ``context.metadata`` so observers / diagnostics can show
          *what* the model tried to invoke without re-parsing.
        * ``"none"`` — body was prose; no recovery needed.  Caller
          falls through unchanged.

        Designed as a pure decision routine: callers control the loop
        transitions and ``exit_reason`` assignment.  This keeps the
        hot path (no phantom intent at all — the common case) at one
        ``re.search`` against the response body and zero list
        mutations.
        """
        if not getattr(self.config, "phantom_tool_recovery_enabled", True):
            return "none"

        intent = detect_phantom_tool_intent(response_to_store.content or "")
        if intent.is_empty():
            return "none"

        # Cache the parsed intent for downstream observers (footer,
        # diagnostic).  Stored as plain dicts so middleware can
        # serialise it without importing the dataclass.
        context.metadata["phantom_tool_attempts"] = {
            "shell_commands": list(intent.shell_commands),
            "invoke_calls": [
                {"name": name, "arguments": dict(args)}
                for name, args in intent.invoke_calls
            ],
            "raw_intents_count": intent.raw_intents_count,
        }

        # Synthesis path — try to turn the prose intents into real
        # ``ToolCall``s the registry can dispatch *this iteration*.
        # When successful the caller skips the corrective-message
        # retry entirely and the loop continues forward with one
        # fewer wasted round-trip.  Synthesis is conservative: it
        # only emits calls for tools the registry already knows about
        # (after case-fold + underscore-normalise), so unknown tool
        # names still fall through to the corrective-message path.
        if getattr(self.config, "phantom_tool_synthesis_enabled", True):
            synth = synthesize_tool_calls_from_phantom(
                intent,
                self.services.tool_registry,
                id_prefix=f"phantom_{context.iteration}",
            )
            if synth is not None and synth.tool_calls:
                response_to_store.tool_calls = list(synth.tool_calls)
                # Strip phantom-tool prose out of the visible content
                # — leaving raw ``<function=...>`` tags in the
                # assistant message would re-trigger detection on the
                # next turn (when the message is re-sent in history).
                # We keep a brief acknowledgement in the content slot
                # so the UI's "● Listing 1 directory…" headline still
                # has *something* to render.
                response_to_store.content = ""
                context.metadata.setdefault("phantom_synth_notes", []).extend(synth.notes)
                if synth.unresolved:
                    context.metadata.setdefault(
                        "phantom_synth_unresolved", []
                    ).extend(
                        [{"name": name, "reason": reason} for name, reason in synth.unresolved]
                    )
                context.metadata[TURN_KEY_PHANTOM_TOOL_SYNTHESIZED] = (
                    int(context.metadata.get(TURN_KEY_PHANTOM_TOOL_SYNTHESIZED, 0))
                    + len(synth.tool_calls)
                )
                try:
                    self.services.logger.debug(
                        "phantom_tool_synthesis: emitted=%d unresolved=%d session=%s",
                        len(synth.tool_calls),
                        len(synth.unresolved),
                        context.session_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return "synthesized"

        attempts = int(context.metadata.get(TURN_KEY_PHANTOM_TOOL_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "max_phantom_tool_retries", 2)))
        if attempts >= max_attempts:
            return "exhausted"

        # Append the assistant's prose so the next iteration's
        # context window includes the model's broken response, then
        # the corrective nudge.  Order matters: removing the
        # assistant message would let the model "forget" what it
        # just wrote and re-emit the same broken pattern.
        self._append_assistant_text_message(messages, response_to_store)
        messages.append(
            build_corrective_user_message(intent, attempt_index=attempts + 1)
        )
        context.metadata[TURN_KEY_PHANTOM_TOOL_RETRIES] = attempts + 1
        # Reset the empty-response counter — this iteration produced
        # *content*, just in the wrong shape.  Carrying the
        # empty-response budget over would cause spurious 9-step
        # degradation kicks in when phantom recovery is doing
        # the right thing.
        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
        try:
            self.services.logger.debug(
                "phantom_tool_recovery: attempt=%d / max=%d shell=%d invoke=%d session=%s",
                attempts + 1,
                max_attempts,
                len(intent.shell_commands),
                len(intent.invoke_calls),
                context.session_id,
            )
        except Exception:  # noqa: BLE001
            pass
        return "retry"

    def _dispatch_synthesized_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        state_machine: EngineStateMachine,
        request: EngineRequest,
    ) -> tuple[str, ExitReason | None, str | None]:
        """Dispatch ``response.tool_calls`` synthesized from prose intent.

        Returns ``(outcome, exit_reason, error_text)`` where ``outcome``
        is one of:

        * ``"continue"`` — every call dispatched, results appended,
          loop should iterate to the next LLM call.
        * ``"interrupted"`` — user pressed Ctrl-C mid-dispatch; caller
          should ``break`` and surface ``ExitReason.INTERRUPTED``.
        * ``"failed"`` — middleware / strict-mode tool error; caller
          should ``break`` with the returned ``exit_reason`` and
          ``error_text``.

        This helper is intentionally a tighter version of the regular
        tool-dispatch branch: we skip the truncation /
        invalid-JSON validators because the synthesized arguments
        were built by us (not parsed from a model JSON blob), so the
        failure modes those gates protect against do not apply.
        """
        state_machine.transition(LoopState.TOOL_DISPATCH)
        self._append_assistant_tool_message(messages, response)
        tool_result_start_idx = len(messages)

        state_machine.transition(LoopState.TOOL_EXECUTE)
        for call in response.tool_calls:
            if self._is_interrupted(request.session_id, context):
                self._record_interrupt_metadata(
                    context,
                    was_in_tool_call=True,
                )
                state_machine.transition(LoopState.INTERRUPTED)
                return "interrupted", ExitReason.INTERRUPTED, None

            permission_checked = self._apply_tool_permission_gate(
                call,
                request=request,
                context=context,
            )

            if isinstance(permission_checked, ToolResult):
                result = permission_checked
            else:
                try:
                    pre_tool = self.services.middleware_pipeline.run_before_tool(
                        permission_checked,
                        context,
                    )
                except Exception as exc:
                    self._handle_pipeline_error(exc, state_machine.state, context)
                    state_machine.transition(LoopState.FAILED)
                    return "failed", ExitReason.MIDDLEWARE_ERROR, str(exc)

                if isinstance(pre_tool, ToolResult):
                    result = pre_tool
                    tool_call = None
                else:
                    tool_call = pre_tool

            if not isinstance(permission_checked, ToolResult) and tool_call is not None:
                context.metadata.pop("tool_error_result", None)
                context.metadata["_active_tool_call"] = tool_call
                context.metadata["_tool_interrupt_behavior"] = getattr(
                    self.services.tool_registry.get(tool_call.name),
                    "interrupt_behavior",
                    "block",
                )
                try:
                    result = self.services.tool_registry.dispatch(tool_call, context)
                except UnknownToolError:
                    if self.config.fail_on_unknown_tool:
                        state_machine.transition(LoopState.FAILED)
                        return (
                            "failed",
                            ExitReason.UNKNOWN_TOOL,
                            f"Unknown tool: {tool_call.name}",
                        )
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
                except Exception as exc:
                    if self.config.fail_on_tool_error:
                        self._handle_pipeline_error(exc, state_machine.state, context)
                        recovered_tool_result = context.metadata.pop("tool_error_result", None)
                        if not isinstance(recovered_tool_result, ToolResult):
                            state_machine.transition(LoopState.FAILED)
                            return "failed", ExitReason.TOOL_ERROR, str(exc)
                        result = recovered_tool_result
                    else:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=f"Tool execution error: {exc}",
                            is_error=True,
                        )
                finally:
                    context.metadata.pop("_active_tool_call", None)
                    context.metadata.pop("_tool_interrupt_behavior", None)

            try:
                result = self.services.middleware_pipeline.run_after_tool(result, context)
            except Exception as exc:
                self._handle_pipeline_error(exc, state_machine.state, context)
                state_machine.transition(LoopState.FAILED)
                return "failed", ExitReason.MIDDLEWARE_ERROR, str(exc)

            self._append_tool_result_message(messages, result)
            if bool(result.metadata.get("interrupted")):
                self._record_interrupt_metadata(
                    context,
                    was_in_tool_call=True,
                )
            if self._is_permission_abort_result(result):
                self._record_interrupt_metadata(
                    context,
                    was_in_tool_call=True,
                )
                state_machine.transition(LoopState.INTERRUPTED)
                return "interrupted", ExitReason.INTERRUPTED, None

        self._apply_pending_steer_to_tool_results(
            messages,
            session_id=request.session_id,
            start_idx=tool_result_start_idx,
            context=context,
        )
        return "continue", None, None

    def _build_stream_callback(self, request: EngineRequest, context: TurnContext):
        callback = request.stream_callback

        # Emergency rollback: if the operator has flipped
        # ``EngineConfig.streaming_enabled`` off (e.g. because a gateway is
        # serving broken SSE), pretend the request had no callback at all.
        # The provider then takes the non-streaming path and the user
        # gets a single final-response chunk instead of a stream.  No
        # exception is raised — graceful degradation is the whole point.
        if not getattr(self.config, "streaming_enabled", True):
            self.services.logger.debug(
                "streaming_enabled=False; suppressing stream_callback for session %s",
                request.session_id,
            )
            return None

        if callback is None and not getattr(self.config, "empty_response_partial_stream_recovery_enabled", True):
            return None

        def _wrapped(delta: str) -> None:
            if not isinstance(delta, str) or not delta:
                return

            # fast-path interrupt check.  Polled at
            # the *top* of every delta so the latency between the user
            # pressing ESC and the stream actually stopping is bounded
            # by one chunk arrival (typically <50 ms for any modern
            # provider).  Cost is one dict lookup + one RLock acquire
            # — ~100 ns per call, ~0.01 % CPU even at 1000 chunks/s.
            #
            # The exception is a ``BaseException`` subclass so the
            # ``except Exception:`` clauses in providers / middleware
            # do not accidentally swallow it.  See
            # ``runtime/exceptions.py`` for the full rationale.
            if self._is_interrupted(request.session_id, context):
                partial = str(
                    context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""
                )
                raise EngineInterrupted(
                    reason="user-interrupt",
                    partial_text=partial,
                    was_in_tool_call=False,
                )

            if getattr(self.config, "empty_response_partial_stream_recovery_enabled", True):
                current = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
                context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = current + delta
            if callback is None:
                return
            try:
                callback(delta)
                context.metadata["streamed_output"] = True
                context.metadata["stream_callback_calls"] = int(context.metadata.get("stream_callback_calls", 0)) + 1
            except Exception:
                self.services.logger.exception("stream callback failed")

        return _wrapped

    def _build_stream_silent_callback(self, request: EngineRequest, context: TurnContext):
        """Wrap :attr:`EngineRequest.stream_silent_callback` for provider use.

        the silent counterpart of
        :meth:`_build_stream_callback`.  Providers forward count-only
        chunks (tool-arg JSON, signatures) here; the wrapped callback
        bumps a separate ``stream_silent_callback_calls`` metadata
        counter so observability can tell how often the live token
        estimator was advanced via this path.

        Same rollback semantics: when ``streaming_enabled`` is off we
        return ``None`` so providers don't try to emit silent deltas
        either.  Same isolation guarantee: any exception from the
        downstream callback is logged and swallowed — a UI render
        failure must never poison the model call.
        """
        callback = request.stream_silent_callback
        if callback is None:
            return None

        if not getattr(self.config, "streaming_enabled", True):
            return None

        def _wrapped_silent(delta: str) -> None:
            if not isinstance(delta, str) or not delta:
                return
            try:
                callback(delta)
                context.metadata["stream_silent_callback_calls"] = (
                    int(context.metadata.get("stream_silent_callback_calls", 0)) + 1
                )
            except Exception:
                self.services.logger.exception("stream silent callback failed")

        return _wrapped_silent

    def _safe_call_hook(self, name: str, **kwargs: Any) -> Any:
        hook = getattr(self._hooks, name, None)
        if hook is None:
            return None
        try:
            return hook(**kwargs)
        except Exception:
            self.services.logger.exception("Engine hook failed: %s", name)
            return None

    def _collect_pre_llm_hook_outcome(self, name: str, **kwargs: Any) -> HookOutcome:
        outcome = self._safe_call_hook(name, **kwargs)
        if outcome is None:
            return HookOutcome()
        if isinstance(outcome, HookOutcome):
            return outcome
        self.services.logger.warning(
            "Engine hook %s returned unsupported outcome type: %s",
            name,
            type(outcome).__name__,
        )
        return HookOutcome()

    def _memory_runtime_enabled(self) -> bool:
        if not bool(getattr(self.config, "memory_enabled", False)):
            return False
        mode = normalize_memory_mode(
            getattr(self.config, "memory_mode", "off"),
            default=MemoryMode.OFF,
        )
        return mode is not MemoryMode.OFF

    def _observe_memory_turn(
        self,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> None:
        if not self._memory_runtime_enabled():
            return
        try:
            self.services.memory_provider.observe_turn(
                session_id=context.session_id,
                task_id=context.task_id,
                messages=copy.deepcopy(messages),
                metadata=dict(context.metadata),
            )
        except Exception as exc:  # noqa: BLE001 - memory observation is best effort
            self.services.logger.exception("Memory observe_turn failed")
            memory_meta = dict(context.metadata.get("memory") or {})
            memory_meta["error"] = type(exc).__name__
            if not memory_meta.get("skipped_reason"):
                memory_meta["skipped_reason"] = "observe_error"
            context.metadata["memory"] = memory_meta

    def _memory_before_compaction(
        self,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> None:
        if not self._memory_runtime_enabled():
            return
        try:
            self.services.memory_provider.before_compaction(
                session_id=context.session_id,
                task_id=context.task_id,
                messages=copy.deepcopy(messages),
                metadata=dict(context.metadata),
            )
        except Exception as exc:  # noqa: BLE001 - compaction must not depend on memory
            self.services.logger.exception("Memory before_compaction failed")
            memory_meta = dict(context.metadata.get("memory") or {})
            memory_meta["error"] = type(exc).__name__
            if not memory_meta.get("skipped_reason"):
                memory_meta["skipped_reason"] = "before_compaction_error"
            context.metadata["memory"] = memory_meta

    def _merge_memory_context_into_hook_outcome(
        self,
        messages: List[Dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> HookOutcome:
        memory_context = self._build_memory_context_for_messages(
            messages,
            outcome=outcome,
            context=context,
        )
        if not memory_context:
            return outcome
        return HookOutcome(
            inject_user_context=append_memory_context(
                outcome.inject_user_context,
                memory_context,
            ),
            inject_system_addendum=outcome.inject_system_addendum,
            short_circuit_response=outcome.short_circuit_response,
        )

    def _build_memory_context_for_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        outcome: HookOutcome,
        context: TurnContext,
    ) -> str:
        mode_value = str(getattr(self.config, "memory_mode", "off") or "off")
        enabled = bool(getattr(self.config, "memory_enabled", False))
        default_stats = default_memory_metadata(enabled=enabled, mode=mode_value)

        if not enabled:
            context.metadata["memory"] = {**default_stats, "skipped_reason": "disabled"}
            return ""

        mode = normalize_memory_mode(mode_value, default=MemoryMode.OFF)
        if mode is MemoryMode.OFF:
            context.metadata["memory"] = {**default_stats, "skipped_reason": "mode_off"}
            return ""

        if outcome.short_circuit_response is not None:
            context.metadata["memory"] = {
                **default_stats,
                "skipped_reason": "short_circuit",
            }
            return ""

        user_message = self._latest_user_message_text(messages)
        if not user_message.strip():
            context.metadata["memory"] = {
                **default_stats,
                "skipped_reason": "no_query_signal",
            }
            return ""

        estimated_prompt_tokens = estimate_messages_tokens(messages)
        estimated_prompt_tokens += estimate_text_tokens(outcome.inject_user_context or "")
        estimated_prompt_tokens += estimate_text_tokens(outcome.inject_system_addendum or "")
        budget = resolve_memory_token_budget(
            model_window=self._resolve_model_window(),
            estimated_prompt_tokens=estimated_prompt_tokens,
            memory_token_budget_pct=float(
                getattr(self.config, "memory_token_budget_pct", 0.0) or 0.0
            ),
            memory_token_budget_max=int(
                getattr(self.config, "memory_token_budget_max", 0) or 0
            ),
            compression_threshold_pct=float(
                getattr(self.config, "compression_pre_llm_pct", 0.85) or 0.85
            ),
        )
        if budget.skipped_reason:
            context.metadata["memory"] = {
                **default_stats,
                "skipped_reason": budget.skipped_reason,
            }
            return ""

        query = MemoryQuery(
            session_id=context.session_id,
            task_id=context.task_id,
            user_message=user_message,
            recent_messages=copy.deepcopy(messages[-8:]),
            working_directory=str(Path.cwd()),
            active_files=self._memory_active_files_from_metadata(context.metadata),
            mode=mode,
            token_budget=budget.effective_budget,
            metadata={
                "iteration": context.iteration,
                "turn_id": context.turn_id,
                "budget_base": budget.base_budget,
            },
        )

        started = time.perf_counter()
        try:
            retrieved = self.services.memory_provider.retrieve(query)
        except Exception as exc:  # noqa: BLE001 - memory is best-effort context
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.services.logger.exception("Memory retrieval failed")
            context.metadata["memory"] = metadata_from_bundle(
                MemoryBundle.skipped(
                    "provider_error",
                    latency_ms=elapsed_ms,
                    provider_errors=(type(exc).__name__,),
                ),
                enabled=enabled,
                mode=mode.value,
                injected_count=0,
                injected_tokens=0,
                skipped_reason="provider_error",
                error=type(exc).__name__,
            )
            return ""

        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieved = MemoryBundle(
            blocks=tuple(retrieved.blocks),
            token_estimate=int(retrieved.token_estimate),
            skipped_reason=retrieved.skipped_reason,
            latency_ms=elapsed_ms,
            provider_errors=tuple(retrieved.provider_errors),
        )
        if not retrieved.blocks:
            context.metadata["memory"] = metadata_from_bundle(
                retrieved,
                enabled=enabled,
                mode=mode.value,
                injected_count=0,
                injected_tokens=0,
            )
            return ""

        include_user_scope = (
            mode is MemoryMode.PERSONAL_ASSISTANT
            and bool(getattr(self.config, "memory_user_profile_enabled", False))
        )
        allowed_scopes = set(
            scopes_for_mode(
                mode,
                user_profile_enabled=include_user_scope,
            )
        )
        allowed_blocks = tuple(
            block for block in retrieved.blocks if block.scope in allowed_scopes
        )
        if not allowed_blocks:
            context.metadata["memory"] = metadata_from_bundle(
                MemoryBundle.skipped("no_relevant_blocks", latency_ms=elapsed_ms),
                enabled=enabled,
                mode=mode.value,
                injected_count=0,
                injected_tokens=0,
                candidate_count=len(retrieved.blocks),
                skipped_reason="no_relevant_blocks",
            )
            return ""

        packed = pack_memory_blocks(
            allowed_blocks,
            token_budget=budget.effective_budget,
            block_token_max=int(getattr(self.config, "memory_block_token_max", 500) or 500),
        )
        packed = MemoryBundle(
            blocks=tuple(packed.blocks),
            token_estimate=int(packed.token_estimate),
            skipped_reason=packed.skipped_reason,
            latency_ms=elapsed_ms,
            provider_errors=retrieved.provider_errors,
        )
        rendered = render_memory_bundle(
            packed,
            include_user_scope=include_user_scope,
        )
        if not rendered:
            skipped_reason = packed.skipped_reason or "no_relevant_blocks"
            context.metadata["memory"] = metadata_from_bundle(
                packed,
                enabled=enabled,
                mode=mode.value,
                injected_count=0,
                injected_tokens=0,
                candidate_count=len(retrieved.blocks),
                skipped_reason=skipped_reason,
            )
            return ""

        context.metadata["memory"] = metadata_from_bundle(
            packed,
            enabled=enabled,
            mode=mode.value,
            candidate_count=len(retrieved.blocks),
        )
        return rendered

    @staticmethod
    def _latest_user_message_text(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(part for part in parts if part)
            return str(content)
        return ""

    @staticmethod
    def _memory_active_files_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
        raw = metadata.get("active_files") or metadata.get("active_paths") or ()
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, (list, tuple, set)):
            return tuple(str(item) for item in raw if item)
        return ()

    def _apply_hook_outcome_to_messages(
        self,
        messages: List[Dict[str, Any]],
        outcome: HookOutcome,
        *,
        context: TurnContext,
    ) -> List[Dict[str, Any]]:
        if not outcome.inject_user_context and not outcome.inject_system_addendum:
            return messages

        outbound = copy.deepcopy(messages)
        if outcome.inject_user_context:
            applied = self._append_hook_user_context(
                outbound,
                outcome.inject_user_context,
            )
            context.metadata["hook_injected_user_context"] = applied
        if outcome.inject_system_addendum:
            self._append_hook_system_addendum(
                outbound,
                outcome.inject_system_addendum,
            )
            context.metadata["hook_injected_system_addendum"] = True
        return outbound

    @staticmethod
    def _append_hook_user_context(
        messages: List[Dict[str, Any]],
        context_text: str,
    ) -> bool:
        cleaned = context_text.strip()
        if not cleaned:
            return False
        marker = f"\n\n<hook_context>\n{cleaned}\n</hook_context>"
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                message["content"] = content + marker
            elif isinstance(content, list):
                content.append({"type": "text", "text": marker.lstrip()})
            else:
                message["content"] = f"{content}{marker}"
            return True
        return False

    @staticmethod
    def _append_hook_system_addendum(
        messages: List[Dict[str, Any]],
        addendum: str,
    ) -> None:
        cleaned = addendum.strip()
        if not cleaned:
            return
        marker = f"\n\n<hook_system_addendum>\n{cleaned}\n</hook_system_addendum>"
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                message["content"] = content + marker
            elif isinstance(content, list):
                content.append({"type": "text", "text": marker.lstrip()})
            else:
                message["content"] = f"{content}{marker}"
            return
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"<hook_system_addendum>\n{cleaned}\n</hook_system_addendum>",
            },
        )

    @staticmethod
    def _next_api_request_attempt_count(context: TurnContext) -> int:
        next_count = int(context.metadata.get("_api_request_attempt_count", 0)) + 1
        context.metadata["_api_request_attempt_count"] = next_count
        return next_count

    def _build_api_hook_payload(
        self,
        *,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        tools: list[Any],
        call_config: ModelCallConfig,
        context: TurnContext,
        provider: ModelProvider,
        api_call_count: int,
    ) -> Dict[str, Any]:
        model = call_config.extra.get("model") if isinstance(call_config.extra, dict) else None
        if model is None:
            model = getattr(provider, "model", None)
        if model is None:
            model = "unknown"

        try:
            approx_input_tokens = estimate_messages_tokens(messages)
        except Exception:
            approx_input_tokens = 0

        try:
            request_char_count = len(json.dumps(messages, ensure_ascii=False, default=str))
        except Exception:
            request_char_count = sum(len(str(message)) for message in messages)

        return {
            "session_id": request.session_id,
            "iteration": context.iteration,
            "model": str(model),
            "provider": str(getattr(provider, "provider_name", type(provider).__name__)),
            "api_mode": str(getattr(provider, "api_mode", "chat")),
            "api_call_count": api_call_count,
            "message_count": len(messages),
            "tool_count": len(tools or []),
            "approx_input_tokens": int(approx_input_tokens),
            "request_char_count": int(request_char_count),
            "max_tokens": call_config.max_tokens,
            "context_metadata": context.metadata,
        }

    @staticmethod
    def _build_post_api_hook_payload(
        pre_payload: Dict[str, Any],
        *,
        elapsed_ms: float,
        response: NormalizedResponse | None,
        error: Exception | None,
    ) -> Dict[str, Any]:
        return {
            "session_id": pre_payload["session_id"],
            "iteration": pre_payload["iteration"],
            "model": pre_payload["model"],
            "provider": pre_payload["provider"],
            "api_mode": pre_payload["api_mode"],
            "api_call_count": pre_payload["api_call_count"],
            "elapsed_ms": elapsed_ms,
            "response_finish_reason": response.finish_reason if response else None,
            "error": error,
            "context_metadata": pre_payload["context_metadata"],
        }

    def _hydrate_todo_snapshot(self, messages: List[Dict[str, Any]], context: TurnContext) -> None:
        snapshot = self._extract_todo_snapshot(messages)
        if not snapshot:
            context.metadata["todo_restored_count"] = 0
            return
        context.metadata["todo_snapshot"] = snapshot
        context.metadata["todo_restored_count"] = len(snapshot)

    def _extract_todo_snapshot(self, messages: List[Dict[str, Any]]) -> list[dict[str, Any]]:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue

            metadata = message.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("todos"), list):
                todos = metadata.get("todos")
                if todos:
                    return [todo for todo in todos if isinstance(todo, dict)]

            content = message.get("content")
            if isinstance(content, str) and content.strip().startswith("["):
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
                    return payload
        return []

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._sanitize_structure(msg) for msg in messages]

    def _sanitize_structure(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_text(value)
        if isinstance(value, list):
            return [self._sanitize_structure(item) for item in value]
        if isinstance(value, dict):
            return {k: self._sanitize_structure(v) for k, v in value.items()}
        return value

    @staticmethod
    def _sanitize_text(text: str | None) -> str:
        if not text:
            return ""
        return strip_surrogates(text)

    def _tool_descriptors_for_provider_call(self, context: TurnContext) -> list[Any]:
        sanitized = context.metadata.get("_schema_sanitized_tool_descriptors")
        if isinstance(sanitized, list):
            return sanitized
        return self.services.tool_registry.list_descriptors()

    def _maybe_apply_schema_sanitizer_retry(
        self,
        *,
        decision: RecoveryDecision,
        tools: list[Any],
        context: TurnContext,
        state: _WithholdingState,
    ) -> bool:
        if decision.classified_reason != FailoverReason.llama_cpp_grammar_pattern.value:
            return False
        if not decision.retry or context.metadata.get("_schema_sanitizer_retry_attempted"):
            return False

        sanitized_tools, removed = sanitize_tool_descriptors(tools)
        if removed <= 0:
            context.metadata.setdefault("schema_sanitizer_removed_count", 0)
            return False

        context.metadata["_schema_sanitized_tool_descriptors"] = sanitized_tools
        context.metadata["_schema_sanitizer_retry_attempted"] = True
        context.metadata["schema_sanitizer_applied"] = True
        context.metadata["schema_sanitizer_removed_count"] = removed
        state.cascade_log.append(f"schema_sanitizer(removed={removed})")
        self.services.logger.warning(
            "Recovered from local grammar schema error by stripping %d unsupported schema keys; retrying.",
            removed,
        )
        return True

    def _maybe_apply_image_shrink_retry(
        self,
        *,
        decision: RecoveryDecision,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        state: _WithholdingState,
    ) -> bool:
        if decision.classified_reason != FailoverReason.image_too_large.value:
            return False
        if not decision.retry or context.metadata.get("_image_shrink_retry_attempted"):
            return False

        context.metadata["_image_shrink_retry_attempted"] = True
        shrunk_messages, stats = shrink_image_parts_in_messages(
            messages,
            max_base64_bytes=int(
                getattr(self.config, "image_shrink_max_base64_bytes", 5 * 1024 * 1024)
            ),
            target_base64_bytes=int(
                getattr(self.config, "image_shrink_target_base64_bytes", 4 * 1024 * 1024)
            ),
        )
        context.metadata["image_shrink"] = stats.to_metadata()
        context.metadata["image_shrink_attempted"] = True
        if not stats.changed:
            return False

        messages[:] = shrunk_messages
        context.metadata["image_shrink_applied"] = True
        context.metadata["image_shrink_count"] = stats.changed_count
        state.cascade_log.append(f"image_shrink(count={stats.changed_count})")
        self.services.logger.warning(
            "Recovered from provider image_too_large error by shrinking %d image payload(s); retrying.",
            stats.changed_count,
        )
        return True

    def _prepare_unicode_safe_payload(
        self,
        *,
        canonical_messages: List[Dict[str, Any]],
        prepared_messages: List[Dict[str, Any]],
        tools: list[Any],
        provider: ModelProvider,
        context: TurnContext,
    ) -> None:
        changed = False
        if context.metadata.get("_unicode_strip_surrogates_payload"):
            changed = sanitize_structure_surrogates(canonical_messages) or changed
            changed = sanitize_structure_surrogates(prepared_messages) or changed
            changed = sanitize_structure_surrogates(tools) or changed
        if context.metadata.get("force_ascii_payload"):
            changed = sanitize_structure_non_ascii(canonical_messages) or changed
            changed = sanitize_structure_non_ascii(prepared_messages) or changed
            changed = sanitize_structure_non_ascii(tools) or changed
            changed = sanitize_provider_credentials_non_ascii(provider) or changed
        if changed:
            context.metadata["unicode_payload_sanitized"] = True

    def _maybe_recover_unicode_error(
        self,
        error: Exception,
        *,
        canonical_messages: List[Dict[str, Any]],
        prepared_messages: List[Dict[str, Any]],
        tools: list[Any],
        provider: ModelProvider,
        context: TurnContext,
    ) -> bool:
        if not isinstance(error, UnicodeEncodeError):
            return False

        passes = int(context.metadata.get("unicode_recovery_passes", 0))
        if passes >= 2:
            return False

        error_text = str(error).lower()
        is_ascii_codec = "ascii" in error_text
        is_surrogate_error = (
            "surrogate" in error_text
            or ("utf-8" in error_text and not is_ascii_codec)
        )
        if not is_ascii_codec and not is_surrogate_error:
            return False

        changed = False
        if is_ascii_codec:
            context.metadata["force_ascii_payload"] = True
            context.metadata["unicode_recovery_reason"] = "ascii_codec"
            changed = sanitize_structure_non_ascii(canonical_messages) or changed
            changed = sanitize_structure_non_ascii(prepared_messages) or changed
            changed = sanitize_structure_non_ascii(tools) or changed
            credentials_changed = sanitize_provider_credentials_non_ascii(provider)
            changed = credentials_changed or changed
            if credentials_changed:
                self.services.logger.warning(
                    "Provider credential/header contained non-ASCII characters; stripped them before retry."
                )
        else:
            context.metadata["_unicode_strip_surrogates_payload"] = True
            context.metadata["unicode_recovery_reason"] = "surrogate"
            changed = sanitize_structure_surrogates(canonical_messages) or changed
            changed = sanitize_structure_surrogates(prepared_messages) or changed
            changed = sanitize_structure_surrogates(tools) or changed

        context.metadata["unicode_recovery_passes"] = passes + 1
        context.metadata["unicode_payload_sanitized"] = bool(
            context.metadata.get("unicode_payload_sanitized") or changed
        )
        self.services.logger.warning(
            "Recovered from %s UnicodeEncodeError by sanitizing provider payload; retrying.",
            context.metadata["unicode_recovery_reason"],
        )
        return True

    def _maybe_dump_failed_request(
        self,
        *,
        error: Exception,
        reason: str,
        request: EngineRequest,
        prepared_messages: List[Dict[str, Any]],
        tools: list[Any],
        call_config: ModelCallConfig,
        provider: ModelProvider,
        context: TurnContext,
    ) -> None:
        if not getattr(self.config, "dump_failed_requests", False):
            return
        if context.metadata.get("_failed_request_dump_written"):
            return
        try:
            api_kwargs = {
                "messages": prepared_messages,
                "tools": [asdict(tool) if is_dataclass(tool) else tool for tool in tools],
                "model_config": asdict(call_config),
            }
            model = call_config.extra.get("model") if isinstance(call_config.extra, dict) else None
            if model is None:
                model = getattr(provider, "model", None)
            path = dump_api_request_debug(
                api_kwargs,
                model=str(model or "unknown"),
                provider=str(getattr(provider, "provider_name", type(provider).__name__)),
                base_url=getattr(provider, "base_url", None),
                reason=reason,
                error=error,
                dump_dir=Path(self.config.request_dump_dir),
                session_id=request.session_id,
            )
            context.metadata["_failed_request_dump_written"] = True
            context.metadata["request_dump"] = {
                "path": str(path),
                "reason": reason,
            }
        except Exception as dump_error:  # noqa: BLE001 - observability must not mask root cause
            context.metadata["request_dump"] = {
                "path": None,
                "reason": reason,
                "error": str(dump_error),
            }
            self.services.logger.exception("failed request dump failed")

    def _save_trajectory_if_enabled(
        self,
        *,
        result: EngineResult,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> None:
        trajectory_meta = {"saved": False, "path": None, "error": None}
        if not getattr(self.config, "save_trajectories", False):
            result.metadata["trajectory"] = trajectory_meta
            return
        completed = result.status in {EngineStatus.COMPLETED, EngineStatus.MAX_ITERATIONS}
        provider = self.services.provider
        try:
            record = build_trajectory_record(
                messages=messages,
                session_id=result.session_id,
                turn_id=context.turn_id,
                task_id=context.task_id,
                model=str(getattr(provider, "model", "unknown") or "unknown"),
                provider=str(getattr(provider, "provider_name", type(provider).__name__)),
                completed=completed,
            )
            path = save_trajectory_record(
                record,
                trajectory_dir=Path(self.config.trajectory_dir),
                session_id=result.session_id,
                completed=completed,
            )
            trajectory_meta = {"saved": True, "path": str(path), "error": None}
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            trajectory_meta = {"saved": False, "path": None, "error": str(exc)}
            self.services.logger.exception("trajectory persistence failed")
        context.metadata["trajectory"] = trajectory_meta
        result.metadata["trajectory"] = trajectory_meta

    def _cleanup_task_resources_if_needed(
        self,
        *,
        result: EngineResult,
        context: TurnContext,
    ) -> None:
        cleanup_meta = self._cleanup_task_resources(
            context=context,
            completed=result.status in {EngineStatus.COMPLETED, EngineStatus.MAX_ITERATIONS},
            interrupted=result.status == EngineStatus.INTERRUPTED,
        )
        result.metadata["resource_cleanup"] = cleanup_meta
        turn_meta = result.metadata.get("turn")
        if isinstance(turn_meta, dict):
            turn_meta["resource_cleanup"] = cleanup_meta

    def _cleanup_task_resources(
        self,
        *,
        context: TurnContext,
        completed: bool,
        interrupted: bool,
    ) -> dict[str, Any]:
        existing = context.metadata.get("resource_cleanup")
        cleanup_meta: dict[str, Any] = existing if isinstance(existing, dict) else {}
        cleanup_meta.update(
            {
                "completed": bool(completed),
                "interrupted": bool(interrupted),
            }
        )
        normalize_task_cleanup_metadata(cleanup_meta)

        if context.metadata.get("_task_cleanup_done"):
            return cleanup_meta

        task_id = context.task_id
        if task_id:
            resources = context.metadata.get("_task_resource_handles")
            if not isinstance(resources, list):
                resources = []
            release_task_resources(
                resources,
                task_id=task_id,
                cleanup_meta=cleanup_meta,
                logger=self.services.logger,
            )
            self._call_task_cleanup_hook(
                task_id=task_id,
                session_id=context.session_id,
                completed=completed,
                interrupted=interrupted,
                context_metadata=context.metadata,
                cleanup_meta=cleanup_meta,
            )

        context.metadata["resource_cleanup"] = cleanup_meta
        context.metadata["_task_cleanup_done"] = True
        return cleanup_meta

    def _call_task_cleanup_hook(
        self,
        *,
        task_id: str,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
        cleanup_meta: dict[str, Any],
    ) -> None:
        hook = getattr(self._hooks, "on_task_cleanup", None)
        if hook is None:
            return
        try:
            hook(
                task_id=task_id,
                session_id=session_id,
                completed=completed,
                interrupted=interrupted,
                context_metadata=context_metadata,
            )
            cleanup_meta["hook_called"] = True
        except Exception as exc:  # noqa: BLE001 - cleanup hooks are best-effort
            cleanup_meta.setdefault("errors", []).append(
                {"resource": "hook:on_task_cleanup", "error": str(exc)}
            )
            self.services.logger.exception("Engine hook failed: on_task_cleanup")

    def _inject_system_prompt(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._sanitize_text(system_prompt).strip()
        if not prompt:
            return messages

        merged = list(messages)
        if merged and isinstance(merged[0], dict) and merged[0].get("role") == "system":
            existing = merged[0].get("content")
            if isinstance(existing, str) and existing.strip() == prompt:
                return merged

        merged.insert(0, {"role": "system", "content": prompt})
        return merged

    def _signal_for_request(self, request: EngineRequest) -> InterruptSignal:
        if request.interrupt_signal is not None:
            return request.interrupt_signal
        return self.services.interrupt_controller.signal_for(request.session_id)

    def _signal_for_context(self, context: TurnContext) -> InterruptSignal | None:
        signal = context.interrupt_signal
        if signal is not None:
            return signal
        cached = context.metadata.get("_interrupt_signal")
        return cached if isinstance(cached, InterruptSignal) else None

    def _is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool:
        if context is not None and context.interrupt_signal is not None:
            return context.interrupt_signal.is_aborted()
        return self.services.interrupt_controller.is_interrupted(session_id)

    @staticmethod
    def _resolve_terminal_exit_reason(
        terminal_hint: Any,
        exc: Exception,
    ) -> ExitReason:
        """Map a recovery-strategy hint string (or fallback exc) to ``ExitReason``.

        the recovery strategy can pin a precise
        terminal cause via ``context.metadata['recovery_terminal_exit_reason']``
        (using the ``ExitReason.<NAME>.value`` string, NOT the enum
        instance, so the metadata bag stays JSON-serialisable).  This
        helper consumes that hint with safe fallbacks:

        * Recognised hint string → matching ``ExitReason``.
        * Hint missing or unrecognised -> fall back to the default
          length-based mapping.
          rule: ``ResponseInvalidError`` → ``RESPONSE_INVALID``,
          else ``PROVIDER_ERROR``.

        The defensive fallback preserves the existing contract for
        callers that have not migrated to the newer strategy shape
        yet (e.g. bespoke recovery strategies in tests).
        """
        if isinstance(terminal_hint, str) and terminal_hint:
            try:
                return ExitReason(terminal_hint)
            except ValueError:
                # Unknown hint — fall through to the default mapping rather
                # than fail the turn.  Logged at the call site if
                # needed; here we stay defensive.
                pass
        if isinstance(exc, ResponseInvalidError):
            return ExitReason.RESPONSE_INVALID
        return ExitReason.PROVIDER_ERROR

    def _handle_pipeline_error(self, error: Exception, state: LoopState, context: TurnContext) -> None:
        # Pass ``exc_info=error`` explicitly: this method is sometimes called
        # *outside* the original ``except`` block (e.g. after the recovery
        # loop in ``_invoke_provider_with_recovery``), where
        # ``sys.exc_info()`` would be cleared and ``logger.exception`` would
        # otherwise emit a misleading "NoneType: None" stack trace.
        self.services.logger.error(
            "AgentEngine error at %s: %s",
            state,
            error,
            exc_info=error,
        )
        self.services.middleware_pipeline.run_on_error(error, state, context)

    # ------------------------------------------------------------------
    # Provider invocation with engine-side recovery
    # ------------------------------------------------------------------

    @dataclass(slots=True)
    class _ProviderInvocationOutcome:
        """Tri-state result of ``_invoke_provider_with_recovery``.

        Exactly one of (``response``, ``error``) is non-None unless
        ``interrupted`` is True (in which case both are None — the engine
        treats that as an INTERRUPTED exit reason without consulting either).
        """

        response: NormalizedResponse | None = None
        error: Exception | None = None
        interrupted: bool = False

    def _invoke_provider_with_recovery(
        self,
        *,
        request: EngineRequest,
        canonical_messages: List[Dict[str, Any]],
        prepared_messages: List[Dict[str, Any]],
        stream_callback,
        stream_silent_callback,
        context: TurnContext,
    ) -> "_ProviderInvocationOutcome":
        """Issue ``provider.generate`` and apply the recovery strategy on failure.

        Loop semantics ( base +  extensions):

        * On success → returns ``_ProviderInvocationOutcome(response=...)``.
        * On ``ProviderInvocationError`` → bumps the per-turn provider-error
          counter, asks the configured ``RecoveryStrategy``.  The decision
          is then **dispatched along three orthogonal axes**:

          1. ``decision.activate_fallback`` — if the engine has a
             ``FallbackChain`` (with budget left and
             ``EngineConfig.fallback_chain_enabled``), rotate the
             active provider before the next attempt.  When the
             rotation succeeds the attempt budget is **reset** so the
             new provider gets its own retry budget.  When the chain
             is exhausted (or fallback is disabled), we treat the
             decision as a give-up tagged ``FALLBACK_EXHAUSTED`` so
             observers can tell "we tried everyone" apart from "the
             single provider blew up".
          2. ``decision.compress_context`` —  stop-gap: we
             have no compressor yet ( lands it), so we
             surface this as an early give-up tagged
             ``CONTEXT_EXHAUSTED`` / ``PAYLOAD_TOO_LARGE`` rather
             than silently retrying the same oversized payload.
          3. ``decision.retry`` — sleep ``wait_seconds`` interruptibly,
             then re-issue the call.  The wait is interrupt-aware
             so user cancellations preempt long backoffs.

        * On any other ``Exception`` → bumps the counter once and returns
          immediately with that exception (no retry).  This preserves the
          old behaviour for non-provider bugs (e.g. the test suite's
          ``ScriptedProvider`` raising ``RuntimeError``).
        * If an interrupt arrives during a retry wait → returns
          ``interrupted=True``; the caller transitions to LoopState.INTERRUPTED.

        The retry trail is recorded in
        ``context.metadata['recovery_decisions']`` as a list of
        ``{"reason": ..., "wait_seconds": ..., "attempt": ...}`` dicts so
        callers and tests can inspect what the strategy did without parsing
        logs.   added the ``classified_reason`` /
        ``activate_fallback`` / ``compress_context`` keys to that schema —
        see ``RecoveryDecision`` for the full shape.
        """
        attempt_state = AttemptState()
        last_error: Exception | None = None
        decisions_log = context.metadata.setdefault("recovery_decisions", [])
        # Stash the active provider's name (when known) so the
        # classifier can use it as observability metadata.  The chain
        # API exposes ``current_slot_name``; non-chained providers
        # don't ship a name yet, so we leave the field empty.
        if self.services.fallback_chain is not None:
            context.metadata["active_provider_name"] = (
                self.services.fallback_chain.current_slot_name
            )
        withholding_state = _WithholdingState()
        max_provider_attempts = max(
            1,
            int(getattr(self.config, "max_provider_recovery_attempts", 8)),
        )

        provider: ModelProvider = self.services.provider
        tools: list[Any] = []
        call_config = request.model_config

        while attempt_state.attempt < max_provider_attempts:
            context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT] = ""
            try:
                call_config = request.model_config
                ephemeral_max_output_tokens = context.metadata.pop("_ephemeral_max_output_tokens", None)
                if isinstance(ephemeral_max_output_tokens, int) and ephemeral_max_output_tokens > 0:
                    #  length continuation temporarily raises the output
                    # budget without mutating EngineRequest.model_config in-place.
                    call_config = ModelCallConfig(
                        temperature=request.model_config.temperature,
                        max_tokens=ephemeral_max_output_tokens,
                        extra=dict(request.model_config.extra),
                    )

                provider = self.services.provider
                rate_guard_check = self._check_rate_guard_before_provider_call(
                    provider=provider,
                    context=context,
                )
                if rate_guard_check is not None and rate_guard_check.blocked:
                    if self._activate_rate_guard_fallback(
                        context=context,
                        state=withholding_state,
                    ):
                        attempt_state = AttemptState()
                        continue
                    error = self._build_rate_guard_blocked_error(rate_guard_check)
                    context.metadata["recovery_terminal_exit_reason"] = (
                        ExitReason.RATE_LIMITED.value
                    )
                    return AgentEngine._ProviderInvocationOutcome(error=error)

                tools = self._tool_descriptors_for_provider_call(context)
                self._prepare_unicode_safe_payload(
                    canonical_messages=canonical_messages,
                    prepared_messages=prepared_messages,
                    tools=tools,
                    provider=provider,
                    context=context,
                )
                api_call_count = self._next_api_request_attempt_count(context)
                api_hook_payload = self._build_api_hook_payload(
                    request=request,
                    messages=prepared_messages,
                    tools=tools,
                    call_config=call_config,
                    context=context,
                    provider=provider,
                    api_call_count=api_call_count,
                )
                self._safe_call_hook("pre_api_request", **api_hook_payload)
                api_start = time.perf_counter()
                response: NormalizedResponse | None = None
                try:
                    response = provider.generate(
                        prepared_messages,
                        tools,
                        call_config,
                        context,
                        stream_callback=stream_callback,
                        stream_silent_callback=stream_silent_callback,
                    )
                    # post-LLM response-shape validation.
                    # ``validate_response`` is non-mutating; if it returns False
                    # we lift the structured failure into the recovery loop so
                    # the existing retry / give-up machinery handles it.  The
                    # default base-class implementation always returns valid,
                    # so providers that don't care pay zero cost here.
                    ok, reasons = provider.validate_response(response)
                    if not ok:
                        raise ResponseInvalidError(
                            validation_errors=list(reasons),
                            body_summary="invalid response: " + "; ".join(reasons[:5]),
                            metadata={"phase": "validate_response"},
                        )
                    self._clear_rate_guard_after_success(provider=provider, context=context)
                except Exception as exc:
                    self._safe_call_hook(
                        "post_api_request",
                        **self._build_post_api_hook_payload(
                            api_hook_payload,
                            elapsed_ms=(time.perf_counter() - api_start) * 1000,
                            response=response,
                            error=exc,
                        ),
                    )
                    raise
                self._safe_call_hook(
                    "post_api_request",
                    **self._build_post_api_hook_payload(
                        api_hook_payload,
                        elapsed_ms=(time.perf_counter() - api_start) * 1000,
                        response=response,
                        error=None,
                    ),
                )
                if withholding_state.pending_errors or withholding_state.cascade_log:
                    self._observe_recovery_cascade(
                        context,
                        withholding_state,
                        terminal="success",
                    )
                return AgentEngine._ProviderInvocationOutcome(response=response)
            except EngineInterrupted as exc:
                # the stream callback observed a
                # user-interrupt mid-stream and raised through the
                # provider stack.  Package it into the existing
                # ``interrupted=True`` channel so ``run_loop`` does not
                # need a separate handler, and stash the partial text +
                # tool-call flag on context.metadata for ``_build_result``
                # to surface under ``EngineResult.metadata["interrupt"]``.
                self._record_interrupt_metadata(
                    context,
                    reason=exc.reason,
                    partial_text=exc.partial_text,
                    was_in_tool_call=exc.was_in_tool_call,
                )
                return AgentEngine._ProviderInvocationOutcome(interrupted=True)
            except ProviderInvocationError as exc:
                # Bump the observability counter once per failed attempt.
                context.metadata[TURN_KEY_PROVIDER_ERROR_RETRIES] = (
                    int(context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)) + 1
                )
                attempt_state.attempt += 1
                attempt_state.errors.append(exc)
                last_error = exc
                withholding_state.pending_errors.append(exc)

                raw_error = exc.raw if isinstance(exc.raw, UnicodeEncodeError) else exc
                if self._maybe_recover_unicode_error(
                    raw_error,
                    canonical_messages=canonical_messages,
                    prepared_messages=prepared_messages,
                    tools=tools,
                    provider=provider,
                    context=context,
                ):
                    continue

                decision = self.services.recovery_strategy.decide(
                    error=exc,
                    attempt_state=attempt_state,
                    context=context,
                )
                if getattr(self.config, "error_withholding_enabled", True):
                    decision = self._maybe_upgrade_decision_for_repeat_withholding(
                        decision,
                        state=withholding_state,
                        context=context,
                    )

                self._maybe_write_rate_guard_lock(
                    provider=provider,
                    error=exc,
                    decision=decision,
                    context=context,
                )

                image_shrink_retry = False
                if decision.classified_reason == FailoverReason.image_too_large.value:
                    image_shrink_retry = self._maybe_apply_image_shrink_retry(
                        decision=decision,
                        messages=prepared_messages,
                        context=context,
                        state=withholding_state,
                    )
                    if not image_shrink_retry:
                        decision = RecoveryDecision.give_up(
                            "image-too-large:shrink-unavailable",
                            classified_reason=FailoverReason.image_too_large.value,
                        )

                decisions_log.append(
                    {
                        "attempt": attempt_state.attempt,
                        "retry": decision.retry,
                        "wait_seconds": decision.wait_seconds,
                        "reason": decision.reason,
                        "status_code": exc.status_code,
                        "is_network_error": exc.is_network_error,
                        # surface the classifier
                        # verdict + new dispatch hints in the trail
                        # so observers (and tests) can reason about
                        # the chain of decisions without re-parsing.
                        "classified_reason": decision.classified_reason,
                        "activate_fallback": decision.activate_fallback,
                        "compress_context": decision.compress_context,
                        "strip_thinking": decision.strip_thinking,
                    }
                )

                # ── Strip-thinking hint ( will consume) ───
                # Surface the breadcrumb now so observability sees
                # the request even though the message-builder rewrite
                # lands in .
                if decision.strip_thinking:
                    context.metadata["recovery_strip_thinking_requested"] = True

                if image_shrink_retry:
                    continue

                if self._maybe_apply_schema_sanitizer_retry(
                    decision=decision,
                    tools=tools,
                    context=context,
                    state=withholding_state,
                ):
                    continue

                if getattr(self.config, "error_withholding_enabled", True):
                    applied = self._apply_recovery_decision_cascade(
                        decision=decision,
                        messages=prepared_messages,
                        context=context,
                        state=withholding_state,
                    )
                else:
                    applied = self._apply_recovery_decision_singleshot(
                        decision=decision,
                        messages=prepared_messages,
                        context=context,
                        state=withholding_state,
                    )
                if applied:
                    if context.metadata.pop("_recovery_reset_attempt_state", False):
                        attempt_state = AttemptState()
                    continue

                terminal_after_recovery = context.metadata.get(
                    "recovery_terminal_exit_reason"
                )
                if terminal_after_recovery and (
                    decision.compress_context or decision.activate_fallback
                ):
                    self._observe_recovery_cascade(
                        context,
                        withholding_state,
                        terminal="surface",
                    )
                    self._maybe_dump_failed_request(
                        error=exc,
                        reason="recovery_terminal",
                        request=request,
                        prepared_messages=prepared_messages,
                        tools=tools,
                        call_config=call_config,
                        provider=provider,
                        context=context,
                    )
                    return AgentEngine._ProviderInvocationOutcome(error=exc)

                if not decision.retry:
                    # Pin RATE_LIMITED as the terminal when the strategy
                    # gave up specifically on a rate-limit failure — the
                    # generic PROVIDER_ERROR mask would lose that signal.
                    if (
                        decision.classified_reason == FailoverReason.rate_limit.value
                        and not context.metadata.get("recovery_terminal_exit_reason")
                    ):
                        context.metadata["recovery_terminal_exit_reason"] = (
                            ExitReason.RATE_LIMITED.value
                        )
                    self._observe_recovery_cascade(
                        context,
                        withholding_state,
                        terminal="surface",
                    )
                    self._maybe_dump_failed_request(
                        error=exc,
                        reason="non_retryable_provider_error",
                        request=request,
                        prepared_messages=prepared_messages,
                        tools=tools,
                        call_config=call_config,
                        provider=provider,
                        context=context,
                    )
                    return AgentEngine._ProviderInvocationOutcome(error=exc)

                # Interruptible wait — if the user cancels mid-retry we abort
                # the whole turn rather than serve a stale recovery attempt.
                if decision.wait_seconds > 0:
                    completed = wait_interruptible(
                        decision.wait_seconds,
                        interrupt_controller=self.services.interrupt_controller,
                        session_id=request.session_id,
                        interrupt_signal=self._signal_for_context(context),
                    )
                    attempt_state.total_wait_seconds += decision.wait_seconds
                    if not completed:
                        return AgentEngine._ProviderInvocationOutcome(interrupted=True)

                # Loop and retry the provider call.
                continue
            except Exception as exc:
                # Non-structured errors (programming errors, ScriptedProvider
                # exhaustion, etc.) bypass the recovery strategy entirely —
                # one bump + immediate hand-off to middleware.
                context.metadata[TURN_KEY_PROVIDER_ERROR_RETRIES] = (
                    int(context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)) + 1
                )
                if self._maybe_recover_unicode_error(
                    exc,
                    canonical_messages=canonical_messages,
                    prepared_messages=prepared_messages,
                    tools=tools,
                    provider=provider,
                    context=context,
                ):
                    continue
                last_error = exc
                self._maybe_dump_failed_request(
                    error=last_error,
                    reason="non_retryable_client_error",
                    request=request,
                    prepared_messages=prepared_messages,
                    tools=tools,
                    call_config=call_config,
                    provider=provider,
                    context=context,
                )
                return AgentEngine._ProviderInvocationOutcome(error=last_error)

        self._observe_recovery_cascade(
            context,
            withholding_state,
            terminal="exhausted",
        )
        if last_error is not None:
            self._maybe_dump_failed_request(
                error=last_error,
                reason="max_retries_exhausted",
                request=request,
                prepared_messages=prepared_messages,
                tools=tools,
                call_config=call_config,
                provider=provider,
                context=context,
            )
        return AgentEngine._ProviderInvocationOutcome(error=last_error)

    def _check_rate_guard_before_provider_call(
        self,
        *,
        provider: ModelProvider,
        context: TurnContext,
    ) -> RateGuardCheck | None:
        if not getattr(self.config, "rate_guard_enabled", True):
            return None
        try:
            check = RateGuard(getattr(self.config, "rate_guard_dir", None)).check(provider)
        except Exception as exc:  # noqa: BLE001 - guard must never break a turn
            check = RateGuardCheck(checked=True, blocked=False, error=str(exc))
        self._record_rate_guard_check(check, context=context)
        return check

    def _record_rate_guard_check(
        self,
        check: RateGuardCheck,
        *,
        context: TurnContext,
    ) -> None:
        md = context.metadata.setdefault("rate_guard", {})
        md["checked"] = bool(md.get("checked")) or bool(check.checked)
        md["blocked"] = bool(md.get("blocked")) or bool(check.blocked)
        md["fallback_activated"] = bool(md.get("fallback_activated", False))
        md["last_blocked"] = bool(check.blocked)
        if check.key is not None:
            md["last_provider"] = check.key.provider
            md["last_base_url_hash"] = check.key.namespace_hash
        if check.blocked:
            md["until_unix"] = check.until_unix
            if check.lock is not None:
                md["reason"] = check.lock.reason
                md["source_session_id"] = check.lock.source_session_id
        else:
            md.setdefault("until_unix", None)
        if check.error:
            md["error"] = check.error

    def _activate_rate_guard_fallback(
        self,
        *,
        context: TurnContext,
        state: _WithholdingState,
    ) -> bool:
        md = context.metadata.setdefault("rate_guard", {})
        chain = self.services.fallback_chain
        if chain is None or not getattr(self.config, "fallback_chain_enabled", False):
            return False
        max_rotations = max(
            0,
            int(getattr(self.config, "max_fallback_activations_per_turn", 4)),
        )
        rotations = int(context.metadata.get("fallback_activations_this_turn", 0))
        if rotations >= max_rotations or not chain.has_next():
            return False
        if not chain.activate_next():
            return False
        context.metadata["fallback_activations_this_turn"] = rotations + 1
        context.metadata["active_provider_name"] = chain.current_slot_name
        md["fallback_activated"] = True
        md["fallback_slot"] = chain.current_slot_name
        state.cascade_log.append(f"rate_guard:fallback({chain.current_slot_name})")
        return True

    def _build_rate_guard_blocked_error(
        self,
        check: RateGuardCheck,
    ) -> ProviderInvocationError:
        until = check.until_unix
        retry_after = max(0.0, until - time.time()) if until is not None else None
        provider_name = check.key.provider if check.key is not None else "provider"
        body = "rate guard blocked provider"
        if until is not None:
            body = f"{body} until {until:.3f}"
        metadata: dict[str, Any] = {
            "rate_guard_blocked": True,
            "provider": provider_name,
        }
        if check.key is not None:
            metadata["base_url_hash"] = check.key.namespace_hash
        if until is not None:
            metadata["until_unix"] = until
        return ProviderInvocationError(
            status_code=429,
            retry_after_seconds=retry_after,
            body_summary=body,
            metadata=metadata,
        )

    def _clear_rate_guard_after_success(
        self,
        *,
        provider: ModelProvider,
        context: TurnContext,
    ) -> None:
        if not getattr(self.config, "rate_guard_enabled", True):
            return
        try:
            cleared = RateGuard(getattr(self.config, "rate_guard_dir", None)).clear(provider)
        except Exception as exc:  # noqa: BLE001 - guard must never break a turn
            context.metadata.setdefault("rate_guard", {})["clear_error"] = str(exc)
            return
        if cleared:
            context.metadata.setdefault("rate_guard", {})["cleared"] = True

    def _maybe_write_rate_guard_lock(
        self,
        *,
        provider: ModelProvider,
        error: ProviderInvocationError,
        decision: RecoveryDecision,
        context: TurnContext,
    ) -> None:
        if not getattr(self.config, "rate_guard_enabled", True):
            return
        if not (
            error.status_code == 429
            or decision.classified_reason == FailoverReason.rate_limit.value
        ):
            return
        until = self._rate_guard_until_from_error(error)
        if until is None:
            return

        md = context.metadata.setdefault("rate_guard", {})
        md["write_attempted"] = True
        md["lock_until_unix"] = until
        try:
            lock = RateGuard(getattr(self.config, "rate_guard_dir", None)).block(
                provider,
                until_unix=until,
                reason=FailoverReason.rate_limit.value,
                source_session_id=context.session_id,
            )
        except Exception as exc:  # noqa: BLE001 - guard must never break a turn
            md["write_error"] = str(exc)
            return
        md["lock_written"] = lock is not None
        if lock is not None:
            md["until_unix"] = lock.until_unix
            md["blocked"] = bool(md.get("blocked", False))

    @staticmethod
    def _rate_guard_until_from_error(error: ProviderInvocationError) -> float | None:
        now = time.time()
        candidates: list[float] = []

        retry_after = AgentEngine._positive_float(error.retry_after_seconds)
        if retry_after is not None:
            candidates.append(now + retry_after)

        metadata = error.metadata if isinstance(error.metadata, dict) else {}
        retry_after_meta = AgentEngine._positive_float(metadata.get("retry_after_seconds"))
        if retry_after_meta is not None:
            candidates.append(now + retry_after_meta)

        for key in (
            "rate_limit_reset_unix",
            "reset_unix",
            "x_ratelimit_reset_unix",
            "x-ratelimit-reset-unix",
        ):
            reset_unix = AgentEngine._positive_float(metadata.get(key))
            if reset_unix is not None:
                if reset_unix > 1_000_000_000_000:
                    reset_unix = reset_unix / 1000.0
                if reset_unix > now:
                    candidates.append(reset_unix)

        for key in (
            "rate_limit_reset_after_seconds",
            "reset_after_seconds",
            "x_ratelimit_reset_after_seconds",
            "x-ratelimit-reset-after-seconds",
        ):
            reset_after = AgentEngine._positive_float(metadata.get(key))
            if reset_after is not None:
                candidates.append(now + reset_after)

        return max(candidates) if candidates else None

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed) or parsed <= 0:
            return None
        return parsed

    def _apply_recovery_decision_cascade(
        self,
        *,
        decision: RecoveryDecision,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        state: _WithholdingState,
    ) -> bool:
        applied = False
        if decision.compress_context:
            applied = self._apply_compression_recovery(
                decision=decision,
                messages=messages,
                context=context,
                state=state,
            ) or applied
        if decision.strip_thinking:
            if self._strip_thinking_metadata(messages):
                state.cascade_log.append(
                    f"strip_thinking({decision.classified_reason or decision.reason})"
                )
                applied = True
        if decision.activate_fallback:
            applied = self._apply_fallback_recovery(
                decision=decision,
                context=context,
                state=state,
            ) or applied
        return applied

    def _apply_recovery_decision_singleshot(
        self,
        *,
        decision: RecoveryDecision,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        state: _WithholdingState,
    ) -> bool:
        if decision.activate_fallback and self._apply_fallback_recovery(
            decision=decision,
            context=context,
            state=state,
            prefix="singleshot:",
        ):
            return True
        if decision.compress_context and self._apply_compression_recovery(
            decision=decision,
            messages=messages,
            context=context,
            state=state,
            prefix="singleshot:",
        ):
            return True
        if decision.strip_thinking and self._strip_thinking_metadata(messages):
            state.cascade_log.append(
                f"singleshot:strip_thinking({decision.classified_reason or decision.reason})"
            )
            return True
        return False

    def _apply_compression_recovery(
        self,
        *,
        decision: RecoveryDecision,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        state: _WithholdingState,
        prefix: str = "",
    ) -> bool:
        context.metadata["recovery_compress_required"] = True
        reason = decision.classified_reason or "withholding"
        state.compression_attempted_for.add(reason)
        if not getattr(self.config, "compression_enabled", False):
            self._pin_compression_terminal(reason, compressed_but_exhausted=False, context=context)
            return False

        # Memory boundary: recovery compaction must not see
        # retrieved memory.  Strip any ``<memory_context>`` blocks
        # before forking the summariser so the canonical/outbound split
        # holds even on the recovery path.  The retry within this
        # ``_invoke_provider_with_recovery`` uses the stripped (and
        # compacted) messages; a fresh memory injection only happens
        # on a subsequent outer-loop iteration, where the budget is
        # recomputed against the now-smaller prompt.
        if strip_memory_context_from_messages(messages):
            memory_meta = dict(context.metadata.get("memory") or {})
            memory_meta["injected_count"] = 0
            memory_meta["injected_tokens"] = 0
            if not memory_meta.get("skipped_reason"):
                memory_meta["skipped_reason"] = "recovery_compaction_stripped"
            context.metadata["memory"] = memory_meta

        trigger_reason = (
            "payload_too_large"
            if reason == FailoverReason.payload_too_large.value
            else "context_overflow"
        )
        try:
            compacted = self._maybe_compact_messages(
                messages,
                context=context,
                trigger_reason=trigger_reason,
            )
        except Exception:
            self.services.logger.exception("recovery compression failed")
            state.cascade_log.append(f"{prefix}compress_context(failed,reason={reason})")
            return False
        if compacted is not None and compacted.tokens_after < compacted.tokens_before:
            messages[:] = compacted.compressed_messages
            freed = compacted.tokens_before - compacted.tokens_after
            state.cascade_log.append(
                f"{prefix}compress_context(freed={freed},reason={reason})"
            )
            return True
        state.cascade_log.append(f"{prefix}compress_context(freed=0,reason={reason})")
        self._pin_compression_terminal(reason, compressed_but_exhausted=True, context=context)
        return False

    def _pin_compression_terminal(
        self,
        reason: str,
        *,
        compressed_but_exhausted: bool,
        context: TurnContext,
    ) -> None:
        if reason == FailoverReason.payload_too_large.value:
            context.metadata["recovery_terminal_exit_reason"] = ExitReason.PAYLOAD_TOO_LARGE.value
        elif compressed_but_exhausted:
            context.metadata["recovery_terminal_exit_reason"] = ExitReason.COMPRESSION_EXHAUSTED.value
        else:
            context.metadata["recovery_terminal_exit_reason"] = ExitReason.CONTEXT_EXHAUSTED.value

    def _apply_fallback_recovery(
        self,
        *,
        decision: RecoveryDecision,
        context: TurnContext,
        state: _WithholdingState,
        prefix: str = "",
    ) -> bool:
        chain = self.services.fallback_chain
        if chain is None or not getattr(self.config, "fallback_chain_enabled", False):
            return False
        max_rotations = max(
            0,
            int(getattr(self.config, "max_fallback_activations_per_turn", 4)),
        )
        rotations = int(context.metadata.get("fallback_activations_this_turn", 0))
        if rotations >= max_rotations or not chain.has_next():
            context.metadata["recovery_terminal_exit_reason"] = ExitReason.FALLBACK_EXHAUSTED.value
            return False
        if not chain.activate_next():
            context.metadata["recovery_terminal_exit_reason"] = ExitReason.FALLBACK_EXHAUSTED.value
            return False
        context.metadata["fallback_activations_this_turn"] = rotations + 1
        context.metadata["active_provider_name"] = chain.current_slot_name
        context.metadata["_recovery_reset_attempt_state"] = True
        state.cascade_log.append(f"{prefix}fallback({chain.current_slot_name})")
        return True

    def _maybe_upgrade_decision_for_repeat_withholding(
        self,
        decision: RecoveryDecision,
        *,
        state: _WithholdingState,
        context: TurnContext,
    ) -> RecoveryDecision:
        reason = decision.classified_reason or ""
        withholdable = {
            FailoverReason.context_overflow.value,
            FailoverReason.payload_too_large.value,
            FailoverReason.long_context_tier.value,
        }
        if (
            decision.activate_fallback
            or reason not in withholdable
            or reason not in state.compression_attempted_for
        ):
            return decision
        chain = self.services.fallback_chain
        if (
            chain is None
            or not getattr(self.config, "fallback_chain_enabled", False)
            or not chain.has_next()
        ):
            return decision
        state.cascade_log.append(
            f"force_fallback_upgrade(reason={reason},after_compression=true)"
        )
        return RecoveryDecision.retry_after(
            0.0,
            reason=f"{decision.reason}|forced-fallback-after-compress-no-progress",
            activate_fallback=True,
            compress_context=decision.compress_context,
            strip_thinking=decision.strip_thinking,
            classified_reason=decision.classified_reason,
        )

    @staticmethod
    def _strip_thinking_metadata(messages: List[Dict[str, Any]]) -> bool:
        changed = False
        for message in messages:
            if not isinstance(message, dict):
                continue
            metadata = message.get("metadata")
            if isinstance(metadata, dict):
                for key in ("reasoning_details", "reasoning_content"):
                    if key in metadata:
                        metadata.pop(key, None)
                        changed = True
        return changed

    def _observe_recovery_cascade(
        self,
        context: TurnContext,
        state: _WithholdingState,
        *,
        terminal: str,
    ) -> None:
        md = context.metadata.setdefault("recovery", {})
        md["cascade_log"] = list(state.cascade_log)
        md["pending_errors_count"] = len(state.pending_errors)
        md["terminal"] = terminal
        md["suppressed_callback_notifications"] = state.suppressed_callback_notifications

    @dataclass(slots=True)
    class _LengthHandlingOutcome:
        action: str
        messages: List[Dict[str, Any]]
        final_response: str | None = None
        exit_reason: ExitReason = ExitReason.TEXT_RESPONSE

    @dataclass(slots=True)
    class _ToolCallValidationOutcome:
        """Verdict from ``_validate_tool_call_arguments``.

        ``action`` is one of:

        * ``"ok"``               — every tool_call has parseable args; the
                                   engine proceeds with the existing
                                   dispatch path.
        * ``"retry"``            — at least one tool_call had truncated /
                                   invalid args but we still have retry
                                   budget; the engine continues without
                                   appending to history (re-issues the
                                   provider call).
        * ``"truncated"``        — args looked truncated and we are out of
                                   retry budget; the engine finalises the
                                   turn with TOOL_CALL_TRUNCATED.
        * ``"inject_error"``    — args were unparseable JSON (not
                                   truncated) and we exhausted the silent
                                   retry budget; the caller appends the
                                   assistant message + a tool-error stub
                                   per failing call so the model can
                                   self-correct on the next iteration.
        """

        action: str
        invalid_json_args: List[tuple[str, str]] = field(default_factory=list)
        # When action="inject_error", the messages the caller should append
        # before continuing the loop — assistant tool_calls message followed
        # by one ``role="tool"`` error stub per failing tool_call.
        injection_messages: List[Dict[str, Any]] = field(default_factory=list)

    def _handle_length_finish_reason(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        request: EngineRequest,
        context: TurnContext,
    ) -> "_LengthHandlingOutcome":
        """Handle ``finish_reason == "length"`` before normal tool/text split.

        This covers the text-response side of length handling:

        1. **Thinking-budget exhaustion** — if the model spent its
           output budget entirely on hidden reasoning-like text
           (``<think>`` / ``<reasoning>``) and produced no visible
           answer, stop politely instead of returning an empty response.
        2. **Continuation retry** — append the partial assistant
           message plus a user continuation instruction, raise the
           retry token budget, and re-enter PRE_LLM up to
           ``max_length_continue_retries`` times.
        3. **Partial return** — once retries are exhausted, drop the
           continuation scaffolding and return the best visible prefix
           we have, marking the turn as ``LENGTH_EXHAUSTED``.

        Tool-call truncation is handled separately by
        :meth:`_handle_length_with_tool_calls` and
        :meth:`_validate_tool_call_arguments`.
        """
        if not getattr(self.config, "length_continuation_enabled", True):
            merged = list(messages)
            self._append_assistant_text_message(merged, response)
            return AgentEngine._LengthHandlingOutcome(
                action="finalize",
                messages=merged,
                final_response=response.content or "",
                exit_reason=ExitReason.TEXT_RESPONSE,
            )

        visible_text = self._extract_visible_text(response.content or "")
        attempts = int(context.metadata.get("length_continue_attempts", 0))

        if self._looks_like_thinking_only_length_response(response):
            merged = list(messages)
            friendly = (
                "The model ran out of output budget while reasoning and did not produce a visible answer. "
                "Please try again with a lower reasoning effort or a larger max token limit."
            )
            merged.append(
                {
                    "role": "assistant",
                    "content": friendly,
                    "finish_reason": response.finish_reason,
                    "metadata": {"partial": True, "length_reason": "thinking_budget"},
                    "_aether_meta": self._assistant_aether_meta(),
                }
            )
            context.metadata["partial"] = True
            context.metadata["length_exit_reason"] = "thinking_budget"
            return AgentEngine._LengthHandlingOutcome(
                action="finalize",
                messages=merged,
                final_response=friendly,
                exit_reason=ExitReason.LENGTH_EXHAUSTED,
            )

        max_retries = max(0, int(getattr(self.config, "max_length_continue_retries", 3)))
        if attempts < max_retries:
            continuation_messages = list(messages)
            continuation_messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "finish_reason": response.finish_reason,
                    "metadata": {
                        "partial": True,
                        "length_continue_attempt": attempts + 1,
                    },
                    "_aether_meta": self._assistant_aether_meta(),
                }
            )
            continuation_messages.append(
                {
                    "role": "user",
                    "content": (
                        "[System: Your previous response was truncated by the output limit. "
                        "Continue exactly where you left off without repeating earlier text.]"
                    ),
                    "metadata": {"_length_continue_prompt": True},
                }
            )
            context.metadata["length_continue_attempts"] = attempts + 1
            if visible_text:
                prefix_parts = context.metadata.setdefault("truncated_response_prefix_parts", [])
                if isinstance(prefix_parts, list):
                    prefix_parts.append(visible_text)
            base_max_tokens = request.model_config.max_tokens or 0
            if base_max_tokens > 0:
                context.metadata["_ephemeral_max_output_tokens"] = min(32000, base_max_tokens * (attempts + 2))
            context.metadata["_messages_override"] = continuation_messages
            return AgentEngine._LengthHandlingOutcome(action="continue", messages=continuation_messages)

        rollback_messages = self._get_messages_up_to_last_assistant(messages)
        prefix_parts = context.metadata.get("truncated_response_prefix_parts", [])
        prefix_items = [part.strip() for part in prefix_parts if isinstance(part, str) and part.strip()]
        if visible_text:
            prefix_items.append(visible_text)
        prefix = " ".join(prefix_items).strip()
        context.metadata["partial"] = True
        context.metadata["length_exit_reason"] = "retries_exhausted"
        context.metadata["truncated_response_prefix"] = prefix
        if prefix:
            rollback_messages.append(
                {
                    "role": "assistant",
                    "content": prefix,
                    "finish_reason": response.finish_reason,
                    "metadata": {"partial": True, "length_exhausted": True},
                    "_aether_meta": self._assistant_aether_meta(),
                }
            )
        return AgentEngine._LengthHandlingOutcome(
            action="finalize",
            messages=rollback_messages,
            final_response=prefix or (response.content or ""),
            exit_reason=ExitReason.LENGTH_EXHAUSTED,
        )

    @staticmethod
    def _strip_reasoning_markup(text: str) -> str:
        cleaned = text.replace("<think>", "").replace("</think>", "")
        cleaned = cleaned.replace("<reasoning>", "").replace("</reasoning>", "")
        return cleaned

    def _extract_visible_text(self, text: str) -> str:
        """Remove reasoning blocks entirely, then trim remaining visible text."""
        cleaned = text
        while True:
            lowered = cleaned.lower()
            start = lowered.find("<think>")
            if start == -1:
                break
            end = lowered.find("</think>", start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len("</think>"):]
        while True:
            lowered = cleaned.lower()
            start = lowered.find("<reasoning>")
            if start == -1:
                break
            end = lowered.find("</reasoning>", start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len("</reasoning>"):]
        return cleaned.strip()

    def _looks_like_thinking_only_length_response(self, response: NormalizedResponse) -> bool:
        if response.finish_reason != "length":
            return False
        raw_text = (response.content or "").strip()
        if not raw_text:
            return False
        visible = self._extract_visible_text(raw_text)
        if visible:
            return False
        lowered = raw_text.lower()
        return "<think>" in lowered or "<reasoning>" in lowered

    # ------------------------------------------------------------------
    # Truncated tool-call detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_truncated_tool_call(tool_calls: List[Any]) -> bool:
        """Heuristic for "the model ran out of tokens while emitting tool args".

        We look at each tool_call's ``arguments`` payload after
        stripping trailing whitespace. A
        well-formed JSON object/array always ends with ``}`` or ``]``; if
        we see anything else the args were almost certainly cut mid-stream.

        Why a heuristic rather than ``json.loads``?  Because some models
        emit args in slightly non-canonical JSON (single quotes, trailing
        commas) that ``json.loads`` rejects but that downstream tools are
        happy with after a normalisation pass.  We only want to refuse
        dispatch when the *shape* looks incomplete, not when the *syntax*
        is loose.

        We also tolerate already-parsed dict/list arguments — those came
        in fully decoded by the provider so they are by definition
        non-truncated.  Empty / whitespace-only strings are ignored
        (treated as "{}" elsewhere) and never count as truncation.
        """
        for call in tool_calls:
            raw = getattr(call, "arguments", None)
            if isinstance(raw, (dict, list)):
                continue
            text = "" if raw is None else str(raw)
            stripped = text.rstrip()
            if not stripped:
                continue
            if not stripped.endswith(("}", "]")):
                return True
        return False

    def _validate_tool_call_arguments(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> "AgentEngine._ToolCallValidationOutcome":
        """Normalise tool_call args, detect truncation, decide retry vs dispatch.

        Implements the full validation path:

        1. **Type normalisation.**  ``arguments`` arrives as ``str`` 99% of
           the time, but providers occasionally hand us a pre-parsed
           ``dict`` / ``list`` (already-decoded JSON), or a non-string scalar
           (e.g. integer mistakenly serialised as the args field), or an
           empty string (treated as ``"{}"``).  We coerce all of these into
           their canonical string form before any further checks so the
           rest of the pipeline only sees well-typed args.
        2. **JSON parse.**  We attempt ``json.loads`` on each arg string and
           collect every (tool_name, error_message) pair that fails.  An
           empty ``invalid_json_args`` list is the dispatch-OK fast path.
        3. **Truncation heuristic.**  If *any* parse failure looks truncated
           (``_detect_truncated_tool_call``) we treat the whole response
           as a length-class failure: refuse to dispatch and either retry
           silently (if the caller has budget left) or finalise as
           TOOL_CALL_TRUNCATED.  This branch is critical because some
           routers rewrite ``finish_reason`` from ``"length"`` to
           ``"tool_calls"``, hiding the real cause from the length
           handler — the heuristic catches that case.
        4. **Pure-syntax JSON error.**  When the args don't look truncated
           but still don't parse, we fall back to silently re-issuing the
           call up to ``max_invalid_json_retries`` times. After that we
           give up the silent path and ask the model to fix it: we append
           the assistant tool_calls message followed by one ``role=tool``
           error stub per failing call. This keeps the original model
           output in history while nudging it to resend valid JSON.
           Callers that want to opt out of this injection can set
           ``invalid_json_recovery_enabled=False``.

        The method **mutates** ``response.tool_calls[*].arguments`` in
        place to install the normalised representation so downstream
        dispatch sees consistent types.  The mutation is safe: by the
        time we reach this point the response object is owned by this
        turn and not shared.

        ``context.metadata`` is updated with the running counters so the
        recovery decisions trail and ``EngineResult.metadata.runtime``
        view stay accurate.
        """
        invalid_json_args: list[tuple[str, str]] = []
        invalid_json_by_id: dict[str, tuple[str, json.JSONDecodeError]] = {}

        for call in response.tool_calls:
            raw = getattr(call, "arguments", None)

            # 1) Pre-parsed dict / list — leave alone.  These are
            # produced by ``OpenAICompatibleModel._parse_tool_call`` after
            # a successful upstream JSON parse, or by ScriptedProvider in
            # tests.  They are never truncated by definition.
            if isinstance(raw, (dict, list)):
                continue

            # 2) Non-string scalars: coerce to ``str()``.  A model can
            # technically emit ``"arguments": 42`` if the provider does
            # not enforce the schema; falling back to ``str()`` keeps the
            # downstream JSON parse honest about what we received.
            if raw is not None and not isinstance(raw, str):
                call.arguments = str(raw)
                raw = call.arguments

            # 3) Empty / whitespace-only string: treat as empty object.
            # OpenAI sometimes emits this for tools with no required
            # parameters; rejecting it would create needless noise.
            args_text = "" if raw is None else str(raw)
            if not args_text or not args_text.strip():
                call.arguments = {}
                continue

            try:
                parsed = json.loads(args_text)
            except json.JSONDecodeError as exc:
                invalid_json_args.append((call.name, str(exc)))
                invalid_json_by_id[call.id] = (args_text, exc)
                # Keep the raw string on ``call.arguments`` so the
                # truncation heuristic and the error-injection path can
                # both see it.  Dispatch will not see this call —
                # ``invalid_json_args`` gates that.
                continue

            # Successful parse: store the decoded form so dispatch sees
            # the canonical dict/list payload regardless of how the wire
            # protocol delivered it.
            call.arguments = parsed if isinstance(parsed, (dict, list)) else {}

        # 4) Fast path: no JSON errors, dispatch.
        if not invalid_json_args:
            context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = 0
            return AgentEngine._ToolCallValidationOutcome(action="ok")

        # 5) Truncation guard.  If the failing args don't terminate with
        # ``}`` / ``]`` we treat this as a length-class failure regardless
        # of what ``finish_reason`` said — see the docstring for why.
        invalid_ids = set(invalid_json_by_id)
        truncated = any(
            (not (str(getattr(c, "arguments", "")) or "").rstrip().endswith(("}", "]")))
            and c.id in invalid_ids
            for c in response.tool_calls
        )
        if truncated and getattr(self.config, "truncated_tool_call_detection_enabled", True):
            attempts = int(context.metadata.get(TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES, 0))
            max_attempts = max(0, int(getattr(self.config, "max_truncated_tool_call_retries", 1)))
            if attempts < max_attempts:
                context.metadata[TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES] = attempts + 1
                # Reset the JSON retry counter — we burned this attempt on
                # the truncation path, not on a JSON-format mistake.
                context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = 0
                return AgentEngine._ToolCallValidationOutcome(
                    action="retry", invalid_json_args=invalid_json_args
                )
            return AgentEngine._ToolCallValidationOutcome(
                action="truncated", invalid_json_args=invalid_json_args
            )

        # 6) Pure-syntax JSON error path: silent retry, then injection.
        attempts = int(context.metadata.get(TURN_KEY_INVALID_JSON_RETRIES, 0)) + 1
        context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = attempts
        max_attempts = max(0, int(getattr(self.config, "max_invalid_json_retries", 3)))
        if attempts < max_attempts:
            return AgentEngine._ToolCallValidationOutcome(
                action="retry", invalid_json_args=invalid_json_args
            )

        # Exhausted silent retries — fall through to injection.  The
        # caller is responsible for appending these messages to history
        # and continuing the loop; we surface them here rather than
        # mutating ``messages`` so the validation routine has zero
        # side-effects on the turn message list.
        if not getattr(self.config, "invalid_json_recovery_enabled", True):
            # Operator opted out of injection — fail the turn instead.
            return AgentEngine._ToolCallValidationOutcome(
                action="truncated", invalid_json_args=invalid_json_args
            )

        injection: list[Dict[str, Any]] = []
        # Build the assistant message first so role-alternation is
        # preserved (assistant tool_calls → tool result(s)).  The args
        # we send back are the **raw** strings — that is what the model
        # actually produced and rebroadcasting them lets the model see
        # exactly what was wrong.
        injection.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": c.arguments
                            if isinstance(c.arguments, str)
                            else json.dumps(c.arguments, ensure_ascii=False),
                        },
                    }
                    for c in response.tool_calls
                ],
                "finish_reason": response.finish_reason,
                "metadata": {"_invalid_json_recovery": True},
                "_aether_meta": self._assistant_aether_meta(),
            }
        )
        for c in response.tool_calls:
            if c.id in invalid_json_by_id:
                raw_args, exc = invalid_json_by_id[c.id]
                if getattr(self.config, "tool_error_structured_format_enabled", True):
                    formatted = format_invalid_tool_args_error(
                        tool_name=c.name,
                        exc=exc,
                        raw_args=raw_args,
                    )
                    content_text = formatted.text
                    error_category = formatted.category
                else:
                    err_msg = str(exc)
                    content_text = (
                        f"Error: Invalid JSON arguments. {err_msg}. "
                        "For tools with no required parameters, use an empty object: {}. "
                        "Please retry with valid JSON."
                    )
                    error_category = "json_syntax_plain"
                self._record_tool_error(context, c.name, error_category)
                injection.append(
                    {
                        "role": "tool",
                        "name": c.name,
                        "tool_call_id": c.id,
                        "content": content_text,
                        "is_error": True,
                        "metadata": {
                            "_invalid_json_recovery": True,
                            "_tool_error_category": error_category,
                        },
                    }
                )
            else:
                error_category = "skipped_due_to_sibling"
                self._record_tool_error(context, c.name, error_category)
                injection.append(
                    {
                        "role": "tool",
                        "name": c.name,
                        "tool_call_id": c.id,
                        "content": "Skipped: another tool call in this response had invalid JSON.",
                        "is_error": True,
                        "metadata": {
                            "_invalid_json_recovery": True,
                            "_tool_error_category": error_category,
                        },
                    }
                )

        # Reset the silent-retry budget so a *future* JSON error in the
        # same turn gets its own clean count.  Without this, a model that
        # emits one bad JSON, gets injection, then emits another bad JSON
        # would skip the silent-retry path entirely and immediately
        # inject again.
        context.metadata[TURN_KEY_INVALID_JSON_RETRIES] = 0

        return AgentEngine._ToolCallValidationOutcome(
            action="inject_error",
            invalid_json_args=invalid_json_args,
            injection_messages=injection,
        )

    def _maybe_inject_schema_errors(
        self,
        *,
        dispatch_plan: ToolDispatchPlan,
        response: NormalizedResponse,
        context: TurnContext,
    ) -> List[Dict[str, Any]] | None:
        """Inject structured tool errors for obvious descriptor/schema mismatches.

        The assistant tool-call message has already been appended by the
        caller.  This helper therefore returns only the matching
        ``role="tool"`` messages needed to keep role alternation valid.
        """
        if not getattr(self.config, "tool_error_structured_format_enabled", True):
            return None
        if not getattr(self.config, "tool_schema_precheck_enabled", True):
            return None

        issues_by_call_id: dict[str, tuple[Any, FormattedToolError]] = {}
        for prepared in dispatch_plan.prepared:
            if prepared.synthetic_result is not None:
                continue
            call = prepared.call
            descriptor = self.services.tool_registry.get_descriptor(call.name)
            if descriptor is None:
                continue
            formatted = format_schema_error(
                call.name,
                descriptor.parameters,
                call.arguments,
            )
            if formatted.category == "schema_unknown":
                continue
            issues_by_call_id[call.id] = (call, formatted)

        if not issues_by_call_id:
            return None

        injection: list[Dict[str, Any]] = []
        for original_call in response.tool_calls:
            issue = issues_by_call_id.get(original_call.id)
            if issue is not None:
                call, formatted = issue
                self._record_tool_error(context, call.name, formatted.category)
                injection.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "tool_call_id": call.id,
                        "content": formatted.text,
                        "is_error": True,
                        "metadata": {
                            "_schema_error_recovery": True,
                            "_tool_error_category": formatted.category,
                        },
                    }
                )
                continue

            error_category = "skipped_due_to_sibling_schema"
            self._record_tool_error(context, original_call.name, error_category)
            injection.append(
                {
                    "role": "tool",
                    "name": original_call.name,
                    "tool_call_id": original_call.id,
                    "content": (
                        "Skipped: another tool call in this response had a schema error."
                    ),
                    "is_error": True,
                    "metadata": {
                        "_schema_error_recovery": True,
                        "_tool_error_category": error_category,
                    },
                }
            )
        return injection

    def _record_tool_error(
        self,
        context: TurnContext,
        tool_name: str,
        category: str,
    ) -> None:
        """Count structured tool errors for the current turn."""
        if not category:
            return
        md = context.metadata.setdefault("tool_errors", {})
        if not isinstance(md, dict):
            md = {}
            context.metadata["tool_errors"] = md
        by_category = md.setdefault("by_category", {})
        if not isinstance(by_category, dict):
            by_category = {}
            md["by_category"] = by_category
        by_tool = md.setdefault("by_tool", {})
        if not isinstance(by_tool, dict):
            by_tool = {}
            md["by_tool"] = by_tool
        by_category[category] = int(by_category.get(category, 0)) + 1
        by_tool[tool_name] = int(by_tool.get(tool_name, 0)) + 1
        md["total"] = int(md.get("total", 0)) + 1

    def _record_tool_result_error(
        self,
        context: TurnContext,
        result: ToolResult,
    ) -> None:
        if not result.is_error:
            return
        category = result.metadata.get("_tool_error_category")
        if isinstance(category, str) and category:
            self._record_tool_error(context, result.name, category)

    def _format_unknown_tool_content(
        self,
        tool_name: str,
        *,
        context: TurnContext,
    ) -> str:
        if not getattr(self.config, "tool_error_structured_format_enabled", True):
            return f"Unknown tool: {tool_name}"
        return format_unknown_tool_error(
            tool_name,
            self.services.tool_registry.list_names(),
        ).text

    def _handle_length_with_tool_calls(
        self,
        *,
        response: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> "AgentEngine._LengthHandlingOutcome":
        """``finish_reason="length"`` AND ``tool_calls`` non-empty.

        When a model exhausts its output budget *while*
        producing tool_calls, the canonical recovery is **not** to
        continue (the next chunk would be more tool args, not prose) but
        rather to give the model a single chance to produce complete
        arguments by re-issuing the same request.  We deliberately do
        NOT append the broken assistant message to history during this
        retry: the goal is to remove the bad attempt from the model's
        view of the conversation rather than ask it to recover from it.

        After ``max_truncated_tool_call_retries`` (default 1) we surrender
        and finalise the turn with ExitReason.TOOL_CALL_TRUNCATED.
        """
        attempts = int(context.metadata.get(TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "max_truncated_tool_call_retries", 1)))

        if attempts < max_attempts:
            context.metadata[TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES] = attempts + 1
            return AgentEngine._LengthHandlingOutcome(
                action="continue",
                # Re-issue the same request from the current message
                # state — the broken assistant tool_calls do NOT enter
                # history.
                messages=messages,
            )

        # Out of retries — finalise as TOOL_CALL_TRUNCATED.  Any visible
        # text the model emitted before the truncation is preserved on
        # ``truncated_response_prefix_parts`` so downstream finalisation
        # (or upstream observability) can still surface it.
        visible_text = self._extract_visible_text(response.content or "")
        if visible_text:
            prefix_parts = context.metadata.setdefault("truncated_response_prefix_parts", [])
            if isinstance(prefix_parts, list):
                prefix_parts.append(visible_text)
        context.metadata["partial"] = True
        context.metadata["length_exit_reason"] = "tool_call_truncated"

        rollback = self._get_messages_up_to_last_assistant(messages)
        return AgentEngine._LengthHandlingOutcome(
            action="finalize",
            messages=rollback,
            final_response=visible_text or None,
            exit_reason=ExitReason.TOOL_CALL_TRUNCATED,
        )

    @staticmethod
    def _get_messages_up_to_last_assistant(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a copy of ``messages`` with trailing continuation scaffolding removed.

        We only want to roll back the artificial continuation user prompts
        and partial assistant fragments introduced by the
        length-continuation path. The last fully-
        committed assistant turn from the *original* conversation should stay.
        """
        trimmed = list(messages)
        while trimmed:
            tail = trimmed[-1]
            if not isinstance(tail, dict):
                trimmed.pop()
                continue
            metadata = tail.get("metadata") if isinstance(tail.get("metadata"), dict) else {}
            if metadata.get("_length_continue_prompt") or metadata.get("partial"):
                trimmed.pop()
                continue
            break
        return trimmed

    @staticmethod
    def _pop_context_response(context: TurnContext, key: str) -> NormalizedResponse | None:
        candidate = context.metadata.pop(key, None)
        if isinstance(candidate, NormalizedResponse):
            return candidate
        return None

    @staticmethod
    def _assistant_aether_meta() -> Dict[str, Any]:
        """Per-message metadata stamped on every assistant message we emit.

        Populates ``_aether_meta.timestamp`` so
        :class:`TimeBasedMicrocompactor` can compute the gap since the
        last assistant message and decide whether the prompt cache is
        cold enough to clear stale ``tool_result`` payloads.

        The key sits under a private (``_``-prefixed) namespace so
        provider serialisation layers (``OpenAICompatible._convert_messages``
        and friends) don't mistake it for a wire-level field — they
        whitelist ``role`` / ``content`` / ``tool_calls`` and silently
        drop everything else.

        Always returns a fresh dict so callers can mutate without
        accidentally aliasing across messages.
        """
        return {"timestamp": time.time()}

    def _append_assistant_tool_message(
        self,
        messages: List[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in response.tool_calls
        ]
        message = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls,
            "finish_reason": response.finish_reason,
            "_aether_meta": self._assistant_aether_meta(),
        }
        reasoning_metadata = self._response_reasoning_metadata(response)
        if reasoning_metadata:
            message["metadata"] = reasoning_metadata
        messages.append(message)

    def _append_assistant_text_message(
        self,
        messages: List[Dict[str, Any]],
        response: NormalizedResponse,
    ) -> None:
        message = {
            "role": "assistant",
            "content": response.content or "",
            "finish_reason": response.finish_reason,
            "_aether_meta": self._assistant_aether_meta(),
        }
        reasoning_metadata = self._response_reasoning_metadata(response)
        if reasoning_metadata:
            message["metadata"] = reasoning_metadata
        messages.append(message)

    @staticmethod
    def _response_reasoning_metadata(response: NormalizedResponse) -> Dict[str, Any]:
        metadata = response.metadata or {}
        reasoning_metadata: Dict[str, Any] = {}
        for key in ("reasoning_content", "reasoning_details"):
            value = metadata.get(key)
            if value:
                reasoning_metadata[key] = value
        return reasoning_metadata

    def _append_tool_result_message(self, messages: List[Dict[str, Any]], result: ToolResult) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": result.content,
                "is_error": result.is_error,
                "metadata": dict(result.metadata),
            }
        )

    @staticmethod
    def _is_permission_abort_result(result: ToolResult) -> bool:
        return (
            bool(result.metadata.get("permission_denied"))
            and result.metadata.get("permission_decision")
            == ToolPermissionDecisionType.ABORT.value
        )

    def _apply_tool_permission_gate(
        self,
        call: ToolCall,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> ToolCall | ToolResult:
        if not bool(getattr(self.config, "tool_permissions_enabled", True)):
            return call
        if not is_dangerous_tool(call.name):
            return call

        plan_refusal = check_plan_mode_block(call.name, context)
        if plan_refusal is not None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=plan_refusal,
                is_error=True,
                metadata={"plan_mode_blocked": True, "tool_executed": False},
            )

        executor = self.services.tool_registry.get(call.name)
        executor.validate(call)

        preview_or_result = self._build_tool_permission_preview(
            executor=executor,
            call=call,
            context=context,
        )
        if isinstance(preview_or_result, ToolResult):
            return preview_or_result

        permission_request = make_permission_request(
            call,
            session_id=request.session_id,
            preview=preview_or_result,
            allow_session=bool(
                getattr(self.config, "tool_permission_session_allow_enabled", True)
            ),
        )
        decision = self._decide_tool_permission(
            permission_request,
            request=request,
            context=context,
        )

        if decision.type in {
            ToolPermissionDecisionType.ALLOW_ONCE,
            ToolPermissionDecisionType.ALLOW_SESSION,
        }:
            args = (
                dict(decision.updated_arguments)
                if isinstance(decision.updated_arguments, dict)
                else dict(call.arguments or {})
            )
            if decision.type == ToolPermissionDecisionType.ALLOW_SESSION:
                self._add_tool_permission_session_rule(
                    permission_request,
                    decision=decision,
                    context=context,
                )
            return ToolCall(id=call.id, name=call.name, arguments=args)

        return build_permission_denied_result(permission_request, decision)

    def _build_tool_permission_preview(
        self,
        *,
        executor: object,
        call: ToolCall,
        context: TurnContext,
    ) -> ToolPermissionPreview | ToolResult:
        builder = getattr(executor, "build_permission_preview", None)
        if callable(builder):
            try:
                preview = builder(call, context)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"Tool permission preview error: {exc}",
                    is_error=True,
                    metadata={"permission_preview_error": True},
                )
            if isinstance(preview, (ToolPermissionPreview, ToolResult)):
                return preview
        return build_fallback_preview(call)

    def _decide_tool_permission(
        self,
        permission_request: ToolPermissionRequest,
        *,
        request: EngineRequest,
        context: TurnContext,
    ) -> ToolPermissionDecision:
        state = self._session_runtime.get(context.session_id)
        rules = [
            rule
            for rule in getattr(state, "tool_permission_rules", [])
            if hasattr(rule, "tool_name")
        ]
        matched_rule = find_matching_rule(permission_request, rules)
        if matched_rule is not None:
            if matched_rule.behavior == ToolPermissionMode.DENY:
                self._bump_tool_permission_stat(context, "denied")
                return ToolPermissionDecision(
                    type=ToolPermissionDecisionType.DENY,
                    rule=matched_rule,
                    source="session_rule",
                )
            self._bump_tool_permission_stat(context, "allowed_session")
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.ALLOW_ONCE,
                rule=matched_rule,
                source="session_rule",
            )

        default_mode = normalize_permission_mode(
            getattr(self.config, "tool_permission_default", "ask"),
            default=ToolPermissionMode.ASK,
        )
        if default_mode == ToolPermissionMode.ALLOW:
            self._bump_tool_permission_stat(context, "allowed_once")
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.ALLOW_ONCE,
                source="config",
            )
        if default_mode == ToolPermissionMode.DENY:
            self._bump_tool_permission_stat(context, "denied")
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                source="config",
            )

        prompter = getattr(request, "tool_permission_prompter", None)
        interactive = False
        if prompter is not None:
            try:
                interactive = bool(prompter.is_interactive())
            except Exception:  # noqa: BLE001
                interactive = False

        if not interactive:
            non_interactive_mode = normalize_permission_mode(
                getattr(self.config, "tool_permission_non_interactive_default", "deny"),
                default=ToolPermissionMode.DENY,
            )
            if non_interactive_mode == ToolPermissionMode.ALLOW:
                self._bump_tool_permission_stat(context, "allowed_once")
                return ToolPermissionDecision(
                    type=ToolPermissionDecisionType.ALLOW_ONCE,
                    source="non_interactive",
                )
            self._bump_tool_permission_stat(context, "denied")
            self._bump_tool_permission_stat(context, "non_interactive_denied")
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                source="non_interactive",
            )

        self._bump_tool_permission_stat(context, "asked")
        try:
            decision = prompter.request_tool_permission(permission_request)
        except Exception as exc:  # noqa: BLE001
            self._bump_tool_permission_stat(context, "denied")
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                feedback=f"permission prompter failed: {exc}",
                source="prompter_error",
            )
        if not isinstance(decision, ToolPermissionDecision):
            self._bump_tool_permission_stat(context, "denied")
            return ToolPermissionDecision(
                type=ToolPermissionDecisionType.DENY,
                feedback="permission prompter returned an invalid decision",
                source="prompter_error",
            )
        if decision.type == ToolPermissionDecisionType.ALLOW_ONCE:
            self._bump_tool_permission_stat(context, "allowed_once")
        elif decision.type == ToolPermissionDecisionType.ALLOW_SESSION:
            self._bump_tool_permission_stat(context, "allowed_session")
        elif decision.type == ToolPermissionDecisionType.ABORT:
            self._bump_tool_permission_stat(context, "aborted")
        else:
            self._bump_tool_permission_stat(context, "denied")
        return decision

    def _add_tool_permission_session_rule(
        self,
        permission_request: ToolPermissionRequest,
        *,
        decision: ToolPermissionDecision,
        context: TurnContext,
    ) -> None:
        if not bool(getattr(self.config, "tool_permission_session_allow_enabled", True)):
            return
        state = self._session_runtime.get(context.session_id)
        rule = decision.rule or build_session_rule_for_request(permission_request)
        rules = getattr(state, "tool_permission_rules", None)
        if not isinstance(rules, list):
            state.tool_permission_rules = []
            rules = state.tool_permission_rules
        if rule not in rules:
            rules.append(rule)
            self._bump_tool_permission_stat(context, "session_rules_added")

    @staticmethod
    def _bump_tool_permission_stat(
        context: TurnContext,
        key: str,
        *,
        amount: int = 1,
    ) -> None:
        stats = context.metadata.setdefault(
            "tool_permissions",
            default_permission_stats(enabled=True),
        )
        if not isinstance(stats, dict):
            stats = default_permission_stats(enabled=True)
            context.metadata["tool_permissions"] = stats
        stats[key] = int(stats.get(key, 0) or 0) + int(amount)

    def _record_interrupt_metadata(
        self,
        context: TurnContext,
        *,
        reason: str = "user-interrupt",
        partial_text: str = "",
        was_in_tool_call: bool = False,
    ) -> None:
        interrupt_meta = context.metadata.get("interrupt")
        if not isinstance(interrupt_meta, dict):
            interrupt_meta = {}
        existing_partial = str(interrupt_meta.get("partial_text") or "")
        if partial_text and not existing_partial:
            interrupt_meta["partial_text"] = partial_text
        else:
            interrupt_meta["partial_text"] = existing_partial
        interrupt_meta["reason"] = str(interrupt_meta.get("reason") or reason)
        interrupt_meta["was_in_tool_call"] = bool(
            interrupt_meta.get("was_in_tool_call", False) or was_in_tool_call
        )
        interrupt_meta.setdefault("triggered_at", time.time())
        context.metadata["interrupt"] = interrupt_meta

    def _append_interrupted_tool_results(
        self,
        messages: List[Dict[str, Any]],
    ) -> int:
        assistant_idx: int | None = None
        assistant_message: Dict[str, Any] | None = None
        for idx in range(len(messages) - 1, -1, -1):
            candidate = messages[idx]
            if (
                isinstance(candidate, dict)
                and candidate.get("role") == "assistant"
                and isinstance(candidate.get("tool_calls"), list)
                and candidate.get("tool_calls")
            ):
                assistant_idx = idx
                assistant_message = candidate
                break
        if assistant_idx is None or assistant_message is None:
            return 0

        tool_calls = assistant_message.get("tool_calls") or []
        expected_ids = {
            str(call.get("id") or ""): str(
                ((call.get("function") or {}).get("name")) or call.get("name") or ""
            )
            for call in tool_calls
            if isinstance(call, dict) and str(call.get("id") or "")
        }
        if not expected_ids:
            return 0

        responded_ids: set[str] = set()
        for message in messages[assistant_idx + 1 :]:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue
            tool_call_id = str(message.get("tool_call_id") or "")
            if tool_call_id:
                responded_ids.add(tool_call_id)

        appended = 0
        for tool_call_id, tool_name in expected_ids.items():
            if tool_call_id in responded_ids:
                continue
            self._append_tool_result_message(
                messages,
                ToolResult(
                    tool_call_id=tool_call_id,
                    name=tool_name or "tool",
                    content=(
                        "Tool execution interrupted by user before completion. "
                        "Treat any partial side effects as incomplete."
                    ),
                    is_error=True,
                    metadata={
                        "interrupted": True,
                        "synthetic_interrupt_result": True,
                    },
                ),
            )
            appended += 1
        return appended

    def _append_interrupt_marker_message(self, messages: List[Dict[str, Any]], marker: str) -> None:
        messages.append({"role": "user", "content": marker})

    def _preserve_interrupt_context(self, messages: List[Dict[str, Any]], context: TurnContext) -> None:
        self._record_interrupt_metadata(context)
        interrupt_meta = context.metadata.get("interrupt")
        assert isinstance(interrupt_meta, dict)
        partial_text = str(interrupt_meta.get("partial_text") or "").strip()
        appended_tool_results = self._append_interrupted_tool_results(messages)
        was_in_tool_call = bool(
            interrupt_meta.get("was_in_tool_call", False) or appended_tool_results > 0
        )
        interrupt_meta["was_in_tool_call"] = was_in_tool_call
        marker = str(interrupt_meta.get("marker") or select_interrupt_marker(was_in_tool_call=was_in_tool_call))
        interrupt_meta["marker"] = marker
        interrupt_meta["partial_assistant_chars"] = len(partial_text)

        if partial_text:
            last = messages[-1] if messages else None
            if not (
                isinstance(last, dict)
                and last.get("role") == "assistant"
                and str(last.get("content") or "").strip() == partial_text
            ):
                self._append_assistant_text_message(
                    messages,
                    NormalizedResponse(content=partial_text, finish_reason="interrupt"),
                )

        last = messages[-1] if messages else None
        if not (
            isinstance(last, dict)
            and last.get("role") == "user"
            and str(last.get("content") or "").strip() == marker
        ):
            self._append_interrupt_marker_message(messages, marker)

    def _apply_pending_steer_to_tool_results(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str,
        start_idx: int,
        context: TurnContext,
    ) -> None:
        steer_text = self.services.steer_inbox.drain(session_id)
        if not steer_text:
            return

        target_idx: int | None = None
        lower_bound = max(0, start_idx)
        for idx in range(len(messages) - 1, lower_bound - 1, -1):
            message = messages[idx]
            if isinstance(message, dict) and message.get("role") == "tool":
                target_idx = idx
                break

        if target_idx is None:
            self.services.steer_inbox.put_back(session_id, steer_text)
            return

        marker = f"\n\nUser guidance: {steer_text}"
        target = messages[target_idx]
        content = target.get("content", "")
        if isinstance(content, str):
            target["content"] = content + marker
        elif isinstance(content, list):
            content.append({"type": "text", "text": marker.lstrip()})
        else:
            target["content"] = f"{content}{marker}"

        context.metadata["steer_injected_count"] = (
            int(context.metadata.get("steer_injected_count", 0)) + 1
        )

    def _finalize_empty_response(
        self,
        *,
        response: NormalizedResponse,
        response_to_store: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        request: EngineRequest,
        phantom_outcome: str,
        prefix: str,
    ) -> _FinalizeResponseOutcome:
        raw_content = response_to_store.content or ""
        final_response = raw_content if strip_thinking_tags(raw_content).strip() else ""
        stream_callback_wrapped = self._build_stream_callback(request, context)
        if (
            request.stream_callback is not None
            and stream_callback_wrapped
            and final_response
            and not context.metadata.get("streamed_output")
        ):
            stream_callback_wrapped(final_response)
            context.metadata["stream_fallback_emitted"] = True

        classification = is_legitimate_empty(
            response_to_store,
            streamed_assistant_text=str(
                context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or ""
            ),
        )
        empty_md = context.metadata.setdefault("empty_recovery", {})
        empty_md["classification"] = classification.kind.value

        if final_response:
            self._append_assistant_text_message(messages, response_to_store)
            removed = self._pop_thinking_prefill_messages(messages)
            if removed:
                empty_md["thinking_prefill_cleaned"] = removed
            context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
            context.metadata[TURN_KEY_THINKING_PREFILL_RETRIES] = 0
            codex_outcome = self._maybe_handle_codex_intermediate_ack(
                response=response,
                response_to_store=response_to_store,
                messages=messages,
                context=context,
            )
            if codex_outcome is not None and codex_outcome.continue_loop:
                empty_md["last_step"] = codex_outcome.step.value
                context.metadata[TURN_KEY_EMPTY_RECOVERY_LAST_STEP] = codex_outcome.step.value
                return _FinalizeResponseOutcome(_CONTINUE_LOOP_SENTINEL, None, None)
            if phantom_outcome == "exhausted":
                exit_reason = ExitReason.PHANTOM_TOOL_INTENT
            else:
                exit_reason = ExitReason.LENGTH_RECOVERED if prefix else ExitReason.TEXT_RESPONSE
            return _FinalizeResponseOutcome(final_response, exit_reason, None)

        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = (
            int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)) + 1
        )

        legitimate_enabled = getattr(self.config, "legitimate_empty_passthrough_enabled", True)
        recovery_enabled = getattr(self.config, "empty_response_recovery_enabled", True)
        if (
            classification.kind == EmptyKind.LEGITIMATE_END_TURN
            and legitimate_enabled
        ) or (
            classification.kind == EmptyKind.THINKING_ONLY
            and legitimate_enabled
            and not recovery_enabled
        ):
            empty_md["last_step"] = "legitimate_empty_passthrough"
            context.metadata[TURN_KEY_EMPTY_RECOVERY_LAST_STEP] = "legitimate_empty_passthrough"
            return _FinalizeResponseOutcome("", ExitReason.TEXT_RESPONSE, None)

        if not legitimate_enabled and not recovery_enabled:
            empty_md["last_step"] = "no_recovery_yet"
            context.metadata[TURN_KEY_EMPTY_RECOVERY_LAST_STEP] = "no_recovery_yet"
            return _FinalizeResponseOutcome("", ExitReason.EMPTY_RESPONSE, None)

        outcome = self._handle_empty_response(
            response=response,
            response_to_store=response_to_store,
            messages=messages,
            context=context,
            request=request,
            classification=classification,
        )
        if outcome.continue_loop:
            return _FinalizeResponseOutcome(_CONTINUE_LOOP_SENTINEL, None, None)
        if outcome.final_response is not None:
            return _FinalizeResponseOutcome(
                outcome.final_response,
                outcome.exit_reason or ExitReason.TEXT_RESPONSE,
                None,
            )
        return _FinalizeResponseOutcome(
            "",
            outcome.exit_reason or ExitReason.EMPTY_RESPONSE,
            None,
        )

    def _handle_empty_response(
        self,
        *,
        response: NormalizedResponse,
        response_to_store: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        request: EngineRequest,  # noqa: ARG002 - kept for future hooks
        classification: ResponseClassification,
    ) -> _EmptyRecoveryOutcome:
        if not getattr(self.config, "empty_response_recovery_enabled", True):
            return self._record_empty_recovery_outcome(
                self._step7_terminal_empty(), context
            )

        pipeline = [
            lambda: self._step1_truncated_prefix_concat(
                response_to_store=response_to_store,
                context=context,
                classification=classification,
            ),
            lambda: self._step2_partial_stream_recovery(
                context=context,
                classification=classification,
            ),
            lambda: self._step3_housekeeping_fallback(context=context),
            lambda: self._step4_post_tool_empty_nudge(
                messages=messages,
                context=context,
            ),
            lambda: self._step5_thinking_prefill(
                response_to_store=response_to_store,
                messages=messages,
                context=context,
                classification=classification,
            ),
            lambda: self._step6_retry_or_fallback(context=context),
        ]
        for step in pipeline:
            outcome = step()
            if outcome is not None:
                return self._record_empty_recovery_outcome(outcome, context)
        return self._record_empty_recovery_outcome(
            self._step7_terminal_empty(), context
        )

    def _record_empty_recovery_outcome(
        self,
        outcome: _EmptyRecoveryOutcome,
        context: TurnContext,
    ) -> _EmptyRecoveryOutcome:
        md = context.metadata.setdefault("empty_recovery", {})
        md["last_step"] = outcome.step.value
        context.metadata[TURN_KEY_EMPTY_RECOVERY_LAST_STEP] = outcome.step.value
        self.services.logger.info(
            "empty_response: step=%s continue=%s exit=%s",
            outcome.step.value,
            outcome.continue_loop,
            outcome.exit_reason.value if outcome.exit_reason else None,
        )
        return outcome

    def _step1_truncated_prefix_concat(
        self,
        *,
        response_to_store: NormalizedResponse,
        context: TurnContext,
        classification: ResponseClassification,
    ) -> _EmptyRecoveryOutcome | None:
        prefix = str(context.metadata.get(TURN_KEY_TRUNCATED_RESPONSE_PREFIX, "") or "")
        if not prefix:
            return None
        body = strip_thinking_tags(response_to_store.content or "").strip()
        streamed = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
        streamed_clean = strip_thinking_tags(streamed).strip()
        if not body and not streamed_clean and not classification.has_streamed_partial:
            return None
        payload = body if len(body) >= len(streamed_clean) else streamed_clean
        if not payload:
            return None
        context.metadata[TURN_KEY_TRUNCATED_RESPONSE_PREFIX] = ""
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.TRUNCATED_PREFIX_CONCAT,
            final_response=f"{prefix}{payload}",
            exit_reason=ExitReason.LENGTH_RECOVERED,
        )

    def _step2_partial_stream_recovery(
        self,
        *,
        context: TurnContext,
        classification: ResponseClassification,
    ) -> _EmptyRecoveryOutcome | None:
        if not getattr(self.config, "empty_response_partial_stream_recovery_enabled", True):
            return None
        if not classification.has_streamed_partial:
            return None
        raw = str(context.metadata.get(TURN_KEY_STREAMED_ASSISTANT_TEXT, "") or "")
        cleaned = strip_thinking_tags(raw).strip()
        if not cleaned:
            return None
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.PARTIAL_STREAM_RECOVERY,
            final_response=cleaned,
            exit_reason=ExitReason.PARTIAL_STREAM_RECOVERY,
        )

    def _step3_housekeeping_fallback(
        self,
        *,
        context: TurnContext,
    ) -> _EmptyRecoveryOutcome | None:
        if not getattr(self.config, "housekeeping_fallback_enabled", True):
            return None
        state = self._session_runtime.get(context.session_id)
        if not state.last_assistant_tools_all_housekeeping:
            return None
        last = state.last_assistant_text_with_tools.strip()
        if not last:
            return None
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.HOUSEKEEPING_FALLBACK,
            final_response=last,
            exit_reason=ExitReason.FALLBACK_PRIOR_TURN_CONTENT,
        )

    def _step4_post_tool_empty_nudge(
        self,
        *,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> _EmptyRecoveryOutcome | None:
        if not getattr(self.config, "post_tool_empty_nudge_enabled", True):
            return None
        if context.metadata.get(TURN_KEY_POST_TOOL_EMPTY_RETRIED):
            return None
        if not any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages[-5:]
        ):
            return None
        messages.append(
            {
                "role": "assistant",
                "content": "(empty)",
                "_aether_meta": self._assistant_aether_meta(),
                "metadata": {"_post_tool_empty_nudge": "placeholder"},
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "[System: You just executed tool calls but returned an empty response. "
                    "Please continue with your task using the tool results above, or explain "
                    "what you intend to do next.]"
                ),
                "metadata": {"_post_tool_empty_nudge": "user_nudge"},
            }
        )
        context.metadata[TURN_KEY_POST_TOOL_EMPTY_RETRIED] = True
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.POST_TOOL_EMPTY_NUDGE,
            continue_loop=True,
        )

    def _step5_thinking_prefill(
        self,
        *,
        response_to_store: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
        classification: ResponseClassification,
    ) -> _EmptyRecoveryOutcome | None:
        if not getattr(self.config, "thinking_prefill_enabled", True):
            return None
        if not classification.has_thinking:
            return None
        attempts = int(context.metadata.get(TURN_KEY_THINKING_PREFILL_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "thinking_prefill_max_retries", 2)))
        if attempts >= max_attempts:
            return None
        messages.append(self._build_prefill_assistant_message(response_to_store))
        context.metadata[TURN_KEY_THINKING_PREFILL_RETRIES] = attempts + 1
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.THINKING_PREFILL,
            continue_loop=True,
        )

    def _build_prefill_assistant_message(
        self,
        response_to_store: NormalizedResponse,
    ) -> Dict[str, Any]:
        md = response_to_store.metadata or {}
        return {
            "role": "assistant",
            "content": response_to_store.content or "",
            "_thinking_prefill": True,
            "_aether_meta": self._assistant_aether_meta(),
            "metadata": {
                "reasoning_content": md.get("reasoning_content"),
                "reasoning_details": md.get("reasoning_details"),
            },
        }

    @staticmethod
    def _pop_thinking_prefill_messages(messages: List[Dict[str, Any]]) -> int:
        removed = 0
        index = 0
        while index < len(messages):
            if isinstance(messages[index], dict) and messages[index].get("_thinking_prefill"):
                del messages[index]
                removed += 1
            else:
                index += 1
        return removed

    def _step6_retry_or_fallback(
        self,
        *,
        context: TurnContext,
    ) -> _EmptyRecoveryOutcome | None:
        attempts = int(context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "empty_response_max_retries", 3)))
        if attempts <= max_attempts and max_attempts > 0:
            return _EmptyRecoveryOutcome(
                step=_EmptyRecoveryStep.RETRY_OR_FALLBACK,
                continue_loop=True,
            )
        chain = self.services.fallback_chain
        if (
            chain is None
            or not getattr(self.config, "fallback_chain_enabled", False)
            or not chain.has_next()
        ):
            return None
        max_rotations = max(
            0,
            int(getattr(self.config, "max_fallback_activations_per_turn", 4)),
        )
        rotations = int(context.metadata.get("fallback_activations_this_turn", 0))
        if rotations >= max_rotations:
            return None
        if not chain.activate_next():
            return None
        context.metadata["fallback_activations_this_turn"] = rotations + 1
        context.metadata["active_provider_name"] = chain.current_slot_name
        context.metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES] = 0
        context.metadata.setdefault("empty_recovery", {})["activated_fallback"] = (
            chain.current_slot_name
        )
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.RETRY_OR_FALLBACK,
            continue_loop=True,
        )

    @staticmethod
    def _step7_terminal_empty() -> _EmptyRecoveryOutcome:
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.TERMINAL_EMPTY,
            final_response=None,
            exit_reason=ExitReason.EMPTY_RESPONSE,
        )

    def _maybe_handle_codex_intermediate_ack(
        self,
        *,
        response: NormalizedResponse,  # noqa: ARG002 - kept for future provider-specific checks
        response_to_store: NormalizedResponse,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> _EmptyRecoveryOutcome | None:
        if not getattr(self.config, "codex_intermediate_ack_enabled", True):
            return None
        if not self._is_codex_responses_provider(self.services.provider):
            return None
        if response_to_store.tool_calls:
            return None
        if not self._looks_like_codex_intermediate_ack(response_to_store.content or ""):
            return None
        attempts = int(context.metadata.get(TURN_KEY_CODEX_ACK_RETRIES, 0))
        max_attempts = max(0, int(getattr(self.config, "codex_intermediate_ack_max_retries", 2)))
        if attempts >= max_attempts:
            return None
        messages.append(
            {
                "role": "system",
                "content": (
                    "[System: You said you would perform an action but did not emit any "
                    "tool_calls. Please proceed by issuing the actual tool call now.]"
                ),
                "metadata": {"_codex_intermediate_ack": True},
            }
        )
        context.metadata[TURN_KEY_CODEX_ACK_RETRIES] = attempts + 1
        return _EmptyRecoveryOutcome(
            step=_EmptyRecoveryStep.CODEX_INTERMEDIATE_ACK,
            continue_loop=True,
        )

    @staticmethod
    def _is_codex_responses_provider(provider: ModelProvider) -> bool:
        return (
            getattr(provider, "provider_name", None) == "codex"
            and getattr(provider, "api_mode", None) == "responses"
        )

    @staticmethod
    def _looks_like_codex_intermediate_ack(content: str) -> bool:
        if not content:
            return False
        head = content.strip().lower()[:64]
        en = ("okay", "ok,", "sure,", "alright", "i'll", "let me", "one moment")
        zh = ("好的", "好的，", "好的我", "好，", "我来", "我会", "马上", "稍等")
        return any(item in head for item in en) or any(item in head for item in zh)

    def _accumulate_usage(self, response: NormalizedResponse, context: TurnContext) -> None:
        """Add this LLM call's usage to the per-turn accumulator.

        Tolerant by design: any failure to extract or
        normalise usage degrades to a no-op rather than crashing the turn —
        accumulation is observability, not correctness-critical.

        Reads ``response.metadata["usage"]`` (the raw provider dict) and
        normalises via ``aether.runtime.observability.usage.normalize_usage`` using the
        provider's ``provider_name`` / ``api_mode`` class attributes.
        Writes the running ``CanonicalUsage`` to
        ``context.metadata["usage_accumulator"]`` and bumps
        ``context.metadata["api_calls"]``.
        """
        try:
            raw = (response.metadata or {}).get("usage") if response else None
            provider_name = getattr(self.services.provider, "provider_name", "openai")
            api_mode = getattr(self.services.provider, "api_mode", "chat")
            this_call = normalize_usage(raw, provider=provider_name, api_mode=api_mode)
            acc = context.metadata.get("usage_accumulator")
            if not isinstance(acc, CanonicalUsage):
                acc = CanonicalUsage()
            context.metadata["usage_accumulator"] = acc.add(this_call)
            context.metadata["api_calls"] = int(context.metadata.get("api_calls", 0)) + 1
        except Exception:  # noqa: BLE001 — observability path, never crash a turn
            self.services.logger.debug(
                "usage accumulation failed; leaving accumulator unchanged",
                exc_info=True,
            )

    def _is_cheap_tool(self, tool_name: str) -> bool:
        """Whether ``tool_name`` is in the cheap-tool refund whitelist.

        Names are compared with the same normalisation the tool-repair
        path uses (case-fold + dash↔underscore + namespace strip via
        :func:`aether.agents.core.phantom_tool._normalize_name`), so
        ``UpdateTodo``, ``update-todo`` and
        ``mcp__router__update_todo`` all match the configured
        ``update_todo`` entry. Empty or missing config disables the
        whitelist and makes every tool call consume iteration budget.
        """
        cheap_names = getattr(self.config, "cheap_tool_names", ())
        if not cheap_names:
            return False
        from aether.agents.core.phantom_tool import _normalize_name

        target = _normalize_name(tool_name or "")
        if not target:
            return False
        return any(target == _normalize_name(n) for n in cheap_names)

    def _is_housekeeping_tool(self, tool_name: str) -> bool:
        names = getattr(self.config, "housekeeping_tool_names", ())
        if not names:
            return False
        from aether.agents.core.phantom_tool import _normalize_name

        target = _normalize_name(tool_name or "")
        return bool(target) and any(target == _normalize_name(name) for name in names)

    @staticmethod
    def _extract_assistant_visible_text(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return strip_thinking_tags(content).strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return strip_thinking_tags("".join(parts)).strip()
        return ""

    def _record_last_content_for_housekeeping_fallback(
        self,
        *,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> None:
        state = self._session_runtime.get(context.session_id)
        last_assistant = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "assistant"
            ),
            None,
        )
        if last_assistant is None:
            state.last_assistant_text_with_tools = ""
            state.last_assistant_tools_all_housekeeping = False
            return
        tool_calls = last_assistant.get("tool_calls") or []
        tool_names: list[str] = []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = call.get("name") or function.get("name")
                if name:
                    tool_names.append(str(name))
        state.last_assistant_text_with_tools = self._extract_assistant_visible_text(last_assistant)
        state.last_assistant_tools_all_housekeeping = bool(tool_names) and all(
            self._is_housekeeping_tool(name) for name in tool_names
        )

    def _handle_max_iterations(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        context: TurnContext,
    ) -> str | None:
        """Generate a summary when the iteration budget is exhausted.

        Called by ``run_loop`` after a break with
        ``exit_reason = MAX_ITERATIONS``. The method:

        1. Returns ``None`` when
           ``EngineConfig.summary_on_budget_exhausted`` is ``False``.
        2. Returns ``None`` when the budget's grace round was already
           consumed.
        3. Builds a one-shot prompt from the current message history
           asking the model to summarize progress and remaining work.
        4. Calls ``provider.generate`` with tools disabled and
           streaming turned off.
        5. Logs a warning and returns ``None`` on any failure.
        """
        if not getattr(self.config, "summary_on_budget_exhausted", True):
            return None

        budget = context.metadata.get("_iteration_budget_obj")
        if not isinstance(budget, IterationBudget):
            return None
        if not budget.grace_call():
            return None
        # Snapshot the new grace_consumed=True state so EngineResult
        # observers see the grace round happened even before we know
        # whether the summary text itself succeeded.
        context.metadata["iteration_budget"] = budget.to_dict()

        summary_messages = list(messages)
        summary_messages.append(
            {
                "role": "user",
                "content": (
                    "[System: You've used your iteration budget. Please "
                    "provide a clean summary of:\n"
                    "  1. What you accomplished in this turn,\n"
                    "  2. What remains to be done,\n"
                    "  3. Any important findings the user should know.\n"
                    "Be concise but specific. Reference actual files / "
                    "actions taken.]"
                ),
            }
        )

        try:
            summary_response = self.services.provider.generate(
                summary_messages,
                [],
                request.model_config,
                context,
                stream_callback=None,
                stream_silent_callback=None,
            )
        except Exception as exc:  # noqa: BLE001 — summary must never crash a turn
            self.services.logger.warning(
                "max_iterations summary generation failed: %s", exc
            )
            return None

        text = (getattr(summary_response, "content", None) or "").strip()
        if not text:
            return None
        context.metadata["summary_provided"] = True
        return text

    def _build_result(
        self,
        request: EngineRequest,
        messages: List[Dict[str, Any]],
        iterations: int,
        final_response: str | None,
        error_text: str | None,
        exit_reason: ExitReason,
        *,
        context: TurnContext,
        active_system_prompt: str | None,
    ) -> EngineResult:
        self._record_last_content_for_housekeeping_fallback(
            messages=messages,
            context=context,
        )
        if exit_reason == ExitReason.INTERRUPTED:
            self._preserve_interrupt_context(messages, context)

        if exit_reason == ExitReason.INTERRUPTED:
            status = EngineStatus.INTERRUPTED
        elif exit_reason == ExitReason.MAX_ITERATIONS:
            status = EngineStatus.MAX_ITERATIONS
        elif exit_reason in {
            ExitReason.PROVIDER_ERROR,
            ExitReason.TOOL_ERROR,
            ExitReason.MIDDLEWARE_ERROR,
            ExitReason.UNKNOWN_TOOL,
            # response-shape validation failure after
            # all retries.  Same terminal class as PROVIDER_ERROR but with
            # a distinct ExitReason so observers can branch.
            ExitReason.RESPONSE_INVALID,
            # the model repeatedly hit its output budget
            # and we had to stop with a partial / rolled-back answer.
            ExitReason.LENGTH_EXHAUSTED,
            # tool_call arguments were cut off
            # mid-stream and we refused to dispatch them.  Treat as
            # FAILED so callers can branch (the alternative — COMPLETED —
            # would silently look like success, hiding a real recovery
            # failure).
            ExitReason.TOOL_CALL_TRUNCATED,
            # Phantom-tool recovery: model wrote tool
            # invocations as prose for the entire retry budget.  The
            # turn produced text but no actual work was done — the
            # user explicitly asked for action and got narration.
            # Classifying this as FAILED ensures the CLI footer flips
            # to a warning state by default, even when the engine
            # produced diagnostic text instead of a normal reply.
            ExitReason.PHANTOM_TOOL_INTENT,
            # These are also FAILED-class because the engine could not
            # produce a usable assistant response for the turn. The
            # exit reason itself carries enough signal for the UI and
            # observability layer to render a precise footer.
            ExitReason.EMPTY_RESPONSE,
            ExitReason.RATE_LIMITED,
            ExitReason.CONTEXT_EXHAUSTED,
            ExitReason.PAYLOAD_TOO_LARGE,
            ExitReason.COMPRESSION_EXHAUSTED,
            ExitReason.FALLBACK_EXHAUSTED,
            # The model emitted unrepairable tool names for the entire
            # retry budget. Keep the same status class as
            # ``PHANTOM_TOOL_INTENT`` so the CLI footer reads
            # consistently.
            ExitReason.INVALID_TOOL_REPEATED,
        }:
            status = EngineStatus.FAILED
        else:
            status = EngineStatus.COMPLETED

        # The ``runtime`` block here is a flat snapshot of the per-turn retry
        # counters that turn-scoped recovery paths use to drive
        # recovery. We read them straight from ``context.metadata``
        # instead of from ``self.*`` — the latter would re-introduce
        # the multi-session leak this change was supposed to fix.
        #
        # The top-level keys ``usage`` / ``api_calls`` /
        # ``iteration_budget`` / ``exit`` / ``reasoning`` /
        # ``compaction`` form the stable metadata schema documented in
        # ``EngineResult``. See that docstring for consumer guarantees.
        usage_acc = context.metadata.get("usage_accumulator")
        if not isinstance(usage_acc, CanonicalUsage):
            usage_acc = CanonicalUsage()

        last_reasoning = extract_last_reasoning(
            messages,
            int(context.metadata.get("turn_start_idx", 0) or 0),
        )
        if last_reasoning:
            context.metadata["last_reasoning_text"] = last_reasoning

        last_msg = messages[-1] if messages else None
        last_msg_role = (
            last_msg.get("role", "unknown") if isinstance(last_msg, dict) else "unknown"
        )
        stuck_after_tool = (
            isinstance(last_msg, dict)
            and last_msg.get("role") == "tool"
            and exit_reason in {ExitReason.MAX_ITERATIONS, ExitReason.EMPTY_RESPONSE}
        )

        # ``turn`` is a flat snapshot of context.metadata for
        # downstream observability.  We exclude internal accumulator objects
        # (CanonicalUsage instances etc.) that are not JSON-serializable —
        # their normalised dict form is exposed at the top level under ``usage``.
        turn_snapshot = {
            k: v
            for k, v in context.metadata.items()
            if k not in _METADATA_INTERNAL_KEYS
        }
        empty_recovery_metadata = dict(context.metadata.get("empty_recovery") or {})
        if not empty_recovery_metadata.get("classification"):
            empty_recovery_metadata["classification"] = (
                EmptyKind.NOT_EMPTY.value
                if strip_thinking_tags(final_response or "").strip()
                else EmptyKind.BUG_EMPTY.value
                if exit_reason == ExitReason.EMPTY_RESPONSE
                else "n/a"
            )
        if not empty_recovery_metadata.get("last_step"):
            empty_recovery_metadata["last_step"] = (
                context.metadata.get(TURN_KEY_EMPTY_RECOVERY_LAST_STEP) or "n/a"
            )
        recovery_metadata = context.metadata.get("recovery")
        if not isinstance(recovery_metadata, dict):
            recovery_metadata = {}
        tool_errors_metadata = context.metadata.get("tool_errors")
        if not isinstance(tool_errors_metadata, dict):
            tool_errors_metadata = {}

        metadata = {
            "request": {
                "session_id": request.session_id,
                "model_config": asdict(request.model_config),
                "system_message_present": bool(request.system_message),
                "stream_callback_present": bool(request.stream_callback),
            },
            "turn": turn_snapshot,
            "runtime": {
                "empty_response_retries": int(
                    context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)
                ),
                "provider_error_retries": int(
                    context.metadata.get(TURN_KEY_PROVIDER_ERROR_RETRIES, 0)
                ),
                # exposed so observability dashboards
                # and tests can confirm the truncated-tool-call detector
                # actually fired.  Both reset to 0 at turn entry.
                "truncated_tool_call_retries": int(
                    context.metadata.get(TURN_KEY_TRUNCATED_TOOL_CALL_RETRIES, 0)
                ),
                "invalid_json_retries": int(
                    context.metadata.get(TURN_KEY_INVALID_JSON_RETRIES, 0)
                ),
                "thinking_prefill_retries": int(
                    context.metadata.get(TURN_KEY_THINKING_PREFILL_RETRIES, 0)
                ),
                "codex_ack_retries": int(
                    context.metadata.get(TURN_KEY_CODEX_ACK_RETRIES, 0)
                ),
                "post_tool_empty_retried": bool(
                    context.metadata.get(TURN_KEY_POST_TOOL_EMPTY_RETRIED, False)
                ),
                # Count of synthesized phantom
                # ``ToolCall``s dispatched this turn (prose intents
                # that mapped cleanly to registered tools).  ``0``
                # for a normal turn; non-zero whenever the engine
                # had to repair a Kimi-class "tool call as prose"
                # response into structured calls inline.
                "phantom_tool_synthesized": int(
                    context.metadata.get(TURN_KEY_PHANTOM_TOOL_SYNTHESIZED, 0)
                ),
                # Phantom-tool recovery counter: how many corrective
                # ``role=user`` nudges this turn sent
                # before the model produced a structured ``tool_calls``
                # (or PHANTOM_TOOL_INTENT exited).  ``0`` means the
                # turn never triggered the detector.
                "phantom_tool_retries": int(
                    context.metadata.get(TURN_KEY_PHANTOM_TOOL_RETRIES, 0)
                ),
            },
            # ----------------- stable v1 ----------------- #
            # Aggregated token usage across every LLM call this turn.
            # Always present (zero-valued on turns with no LLM call).
            "usage": usage_acc.to_dict(),
            "api_calls": int(context.metadata.get("api_calls", 0)),
            "pending_steer": context.metadata.get("pending_steer"),
            "tool_permissions": dict(
                context.metadata.get("tool_permissions")
                or default_permission_stats(
                    enabled=bool(getattr(self.config, "tool_permissions_enabled", True))
                )
            ),
            "memory": dict(
                context.metadata.get("memory")
                or default_memory_metadata(
                    enabled=bool(getattr(self.config, "memory_enabled", False)),
                    mode=str(getattr(self.config, "memory_mode", "off") or "off"),
                )
            ),
            "request_dump": context.metadata.get(
                "request_dump",
                {"path": None, "reason": None},
            ),
            "trajectory": context.metadata.get(
                "trajectory",
                {"saved": False, "path": None, "error": None},
            ),
            # Populated when the turn ended via a
            # user-interrupt that was observed inside the stream
            # callback / tool path.  Schema:
            #   {"reason": str,
            #    "partial_text": str,
            #    "was_in_tool_call": bool,
            #    "triggered_at": float}
            # ``None`` on every non-interrupted turn so consumers can
            # branch on truthiness instead of catching KeyErrors.
            "interrupt": context.metadata.get("interrupt"),
            # structured iteration budget snapshot.
            # Filled by the live ``IterationBudget`` instance the loop
            # carries on ``context.metadata['_iteration_budget_obj']``.
            # Falls back to a synthesized snapshot keyed off
            # ``max_iterations`` for the (rare) early-exit paths that
            # short-circuit before the loop installs the budget — this
            # keeps the ``EngineResult.metadata['iteration_budget']``
            # field unconditionally present in the v1 schema.
            "iteration_budget": (
                context.metadata["_iteration_budget_obj"].to_dict()
                if isinstance(
                    context.metadata.get("_iteration_budget_obj"),
                    IterationBudget,
                )
                else {
                    "used": iterations,
                    "max": int(getattr(self.config, "max_iterations", 0) or 0),
                    "remaining": max(
                        0,
                        int(getattr(self.config, "max_iterations", 0) or 0)
                        - iterations,
                    ),
                    "grace_consumed": False,
                    "consume_count": iterations,
                    "refund_count": 0,
                }
            ),
            # Why the loop terminated, in a structured form distinct from
            # the top-level ``exit_reason`` enum (which is the canonical
            # source — ``exit`` is the auxiliary diagnostic block).
            "exit": {
                "reason": exit_reason.value,
                "last_msg_role": last_msg_role,
                "stuck_after_tool": bool(stuck_after_tool),
            },
            "empty_recovery": {
                **empty_recovery_metadata,
                "classification": empty_recovery_metadata.get("classification"),
                "last_step": empty_recovery_metadata.get("last_step"),
                "retries": {
                    "empty": int(
                        context.metadata.get(TURN_KEY_EMPTY_RESPONSE_RETRIES, 0)
                    ),
                    "thinking_prefill": int(
                        context.metadata.get(TURN_KEY_THINKING_PREFILL_RETRIES, 0)
                    ),
                    "codex_ack": int(
                        context.metadata.get(TURN_KEY_CODEX_ACK_RETRIES, 0)
                    ),
                },
                "post_tool_empty_retried": bool(
                    context.metadata.get(TURN_KEY_POST_TOOL_EMPTY_RETRIED, False)
                ),
            },
            "recovery": {
                "cascade_log": list(recovery_metadata.get("cascade_log") or []),
                "pending_errors_count": int(
                    recovery_metadata.get("pending_errors_count", 0)
                ),
                "terminal": recovery_metadata.get("terminal", "n/a"),
                "withheld": int(
                    recovery_metadata.get("suppressed_callback_notifications", 0)
                ),
            },
            "tool_errors": dict(tool_errors_metadata),
            # Reasoning block. Provider-emitted thinking extraction
            # populates ``last_reasoning``; the schema stays present so
            # consumers can rely on its shape.
            "reasoning": {
                "last_reasoning": context.metadata.get("last_reasoning_text"),
            },
            # Compaction counters. The five-tier pipeline increments
            # these as it fires; the schema stays present so downstream
            # consumers can already render zero-valued footers.
            "compaction": {
                "tier1_spilled_count": int(
                    context.metadata.get("tier1_spilled_count", 0)
                ),
                "tier2_snipped_count": int(
                    context.metadata.get("tier2_snipped_count", 0)
                ),
                "tier3_cleared_count": int(
                    context.metadata.get("tier3_cleared_count", 0)
                ),
                "tier4_collapse_segments": int(
                    context.metadata.get("tier4_collapse_segments", 0)
                ),
                "tier5_summaries_generated": int(
                    context.metadata.get("tier5_summaries_generated", 0)
                ),
            },
        }

        return EngineResult(
            session_id=request.session_id,
            status=status,
            exit_reason=exit_reason,
            messages=messages,
            iterations=iterations,
            final_response=final_response,
            error=error_text,
            task_id=context.task_id,
            turn_id=context.turn_id,
            system_prompt=active_system_prompt,
            streamed=bool(context.metadata.get("streamed_output")),
            metadata=metadata,
        )
