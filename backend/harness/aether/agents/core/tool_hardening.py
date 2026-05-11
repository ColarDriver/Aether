"""Sprint 2 / PR 2.3 — tool dispatch hardening primitives.

Three independent failure modes share a common pre-dispatch sanitiser:

1. **Tool name typos / aliasing** (P0-5).  Models routinely write
   ``readFile`` / ``read-file`` / ``tool_read_file`` / ``reed_file``
   when they mean ``read_file``.  Without repair, the engine either
   fails the entire turn (``fail_on_unknown_tool=True``) or feeds the
   model a sterile ``Unknown tool: readFile`` error and the model
   confidently retries with the same typo.

2. **Recursive subagent fan-out** (P0-6 part 1).  A model that
   discovers ``delegate_task`` will sometimes emit ``delegate_task``
   ``× 10`` in a single turn, spawning ten subagents and exploding
   the cost / latency budget.  We cap the per-turn count and
   replace overflow with synthetic error ToolResults so the model
   sees the limit and has a chance to consolidate.

3. **Same-arg duplicate calls** (P0-6 part 2).  Models emit
   ``read_file({"path":"/etc/hosts"})`` ``× 5`` when they re-read the
   same file inside a single turn.  Dispatching all five wastes
   tokens and tool-execution time; deduplication runs the call once
   and stubs the rest with a "(duplicate; see call_<id>)" reference
   so the model knows the result was reused.

This module is **pure functions over data** — no engine state, no
side effects, no logging.  The engine wires it in by:

::

    plan = prepare_tool_calls(
        response.tool_calls,
        registry=self.services.tool_registry,
        config=self.config,
        context=context,
    )
    if plan.exit_reason is not None:
        # INVALID_TOOL_REPEATED — too many unrepairable names in a row.
        ...
    for prepared in plan.prepared:
        if prepared.synthetic_result is not None:
            # cap / dedup / unrepairable name → skip dispatch and
            # use the pre-built ToolResult.
            ...
        else:
            registry.dispatch(prepared.call, context)

The split (sanitise → dispatch) keeps the existing dispatch loop
mostly unchanged: it only learns to honour ``synthetic_result`` and
to bail when ``exit_reason`` is set.  Every other behaviour
(middleware ``before_tool``, error recovery, metadata bookkeeping)
keeps working with no diff.

Design choices
--------------
* **Reuse ``phantom_tool._normalize_name``.**  That helper already
  handles namespace-stripping (``functions.X``, ``mcp__server__X``)
  and case/dash folding.  Repair shouldn't ship a parallel copy that
  drifts out of sync — bugs found in one place must propagate.

* **Levenshtein, not edit-distance-N.**  Distance ≤ 2 catches the
  vast majority of single-character typos (``read_fle`` →
  ``read_file``, ``writ_file`` → ``write_file``) without
  accidentally mapping ``read_files`` (distance 1) onto ``read_file``
  in a registry that doesn't have a list-files tool.  Tests cover
  the boundary.  Implemented inline (~30 lines) rather than depending
  on ``rapidfuzz`` because the dispatch path is hot and the registry
  rarely exceeds a few dozen entries; the constant factor matters.

* **Cap is per-turn, not per-session.**  A 10-iteration turn that
  legitimately needs 5 ``delegate_task`` calls (one per subtask)
  should not be artificially capped just because *iteration 1*
  emitted 4.  The recovery counter resets every PRE_LLM cycle.

* **Dedup keys on JSON-serialised, key-sorted args.**  Two calls with
  ``{"a":1,"b":2}`` and ``{"b":2,"a":1}`` are semantically identical
  and must dedupe.  Falling back to ``str(args)`` would be order-
  dependent.  Falling back to ``hash(frozenset(args.items()))``
  fails for nested dicts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence

from aether.agents.core.phantom_tool import _normalize_name
from aether.runtime.contracts import ToolCall, ToolResult
from aether.runtime.tool_error_format import format_unknown_tool_error

if TYPE_CHECKING:        # pragma: no cover
    from aether.config.schema import EngineConfig
    from aether.runtime.contracts import TurnContext
    from aether.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Default tool-name set treated as "delegate-class" for the cap.
# Operators can override via ``EngineConfig.delegate_tool_names``;
# the default covers the names hermes-agent and Aether's own
# subagent layer use.  Lower-case here because we compare against
# ``_normalize_name(call.name)``.
DEFAULT_DELEGATE_TOOL_NAMES: tuple[str, ...] = (
    "delegate_task",
    "delegate",
    "subagent",
    "subagent_dispatch",
    "spawn_subagent",
)


# Default name-repair strip prefixes.  When the model writes
# ``tool_read_file`` or ``mcp__filesystem__read_file`` we want to peel
# the namespace before fuzzy-matching.  Tested boundary: do NOT strip
# ``builtin_`` because some registries genuinely register a
# ``builtin_shell`` distinct from ``shell``.
_REPAIR_STRIP_PREFIXES: tuple[str, ...] = (
    "tool_",
    "tools_",
    "function_",
    "functions_",
    "fn_",
    "func_",
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreparedToolCall:
    """One sanitised entry in the pre-dispatch plan.

    * ``call`` — the (possibly repaired-name) ``ToolCall`` to dispatch.
      When ``synthetic_result`` is set this is still populated for
      observability (the engine logs the call regardless).
    * ``synthetic_result`` — when set, the engine SKIPS actual
      dispatch and appends this ``ToolResult`` instead.  Used for:
        - duplicate calls (stub points to the first call's id)
        - capped delegate calls (error stub explaining the cap)
        - unrepairable names with a queued role=tool error stub
    * ``repaired_from`` — the original ``call.name`` before any
      fuzzy-repair rewrite.  ``None`` means the name passed through
      unchanged.  Used by the engine to surface a one-line
      "↻ repaired tool name: foo → foo_bar" hint.
    """

    call: ToolCall
    synthetic_result: Optional[ToolResult] = None
    repaired_from: Optional[str] = None


@dataclass(slots=True)
class ToolDispatchPlan:
    """Pre-dispatch plan produced by ``prepare_tool_calls``.

    ``prepared`` mirrors the original ``response.tool_calls`` length
    (cap / dedup don't drop entries — they replace them with stubs)
    so the engine's iteration count and observability metrics stay
    1:1 with what the model originally emitted.  Each unrepairable
    name carries its own ``synthetic_result`` containing the
    "available tools: …" advice, so the engine's normal
    ``_append_tool_result_message`` path is enough — no parallel
    "extra messages to inject" list is needed.

    ``exit_reason`` is non-None ONLY when the per-turn invalid-tool
    retry budget is exhausted; the engine then surfaces
    ``ExitReason.INVALID_TOOL_REPEATED`` and finalises with
    ``partial=True``.
    """

    prepared: List[PreparedToolCall] = field(default_factory=list)
    exit_reason: Optional[str] = None
    # Observability counters — surfaced in TurnContext.metadata so the
    # CLI / logs can summarise what the sanitiser did this iteration.
    repaired_count: int = 0
    deduped_count: int = 0
    capped_count: int = 0
    unresolved_count: int = 0


# ---------------------------------------------------------------------------
# Levenshtein helper
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str, *, max_distance: int = 2) -> int:
    """Edit distance with early-out at ``max_distance + 1``.

    Returns the actual distance if it is ≤ ``max_distance``, else
    ``max_distance + 1`` (the exact value above the threshold is
    irrelevant for repair purposes).  The early-out means we never
    compute more than ``max_distance + 1`` rows of the DP matrix —
    cheap enough to call against every tool descriptor for every
    typo'd name.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    # Single-row DP; this is the standard "two-row Levenshtein"
    # collapsed to one row + a scratch.  We keep it inline because
    # the algorithm is small and the dispatch path is hot.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            ins = current[j - 1] + 1
            dele = previous[j] + 1
            sub = previous[j - 1] + (0 if ca == cb else 1)
            current[j] = min(ins, dele, sub)
            row_min = min(row_min, current[j])
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Tool name repair (P0-5)
# ---------------------------------------------------------------------------


def repair_tool_name(
    candidate: str,
    registry: "ToolRegistry",
    *,
    max_distance: int = 2,
) -> Optional[str]:
    """Return the registry-canonical name for *candidate*, or ``None``.

    Pipeline (each step short-circuits on a hit):

    1. Exact match: ``registry.has(candidate)`` returns the name as-is.
    2. Strip common prefixes (``tool_``, ``functions_``, …) and retry
       exact match.
    3. Normalise via ``_normalize_name`` (case-fold + dash↔underscore
       + namespace strip) and look up against every descriptor's
       normalised form.
    4. Fuzzy Levenshtein against every descriptor's normalised form,
       returning the unique closest match within ``max_distance``.
       If two descriptors tie at the same distance, return ``None``
       — better to surface "unknown" than dispatch the wrong tool.

    The return value is the **registry-canonical** name (the casing
    and underscore convention the registry was registered with), so
    the caller can ``ToolCall(name=repaired, ...)`` and dispatch
    immediately without any further translation.

    A return of ``None`` means the engine should treat the call as an
    unknown tool — inject the role=tool error message and bump the
    invalid-tool retry counter.
    """
    if not candidate:
        return None

    # 1. exact match
    if registry.has(candidate):
        return candidate

    # 2. prefix strip + exact match
    for prefix in _REPAIR_STRIP_PREFIXES:
        if candidate.startswith(prefix):
            stripped = candidate[len(prefix):]
            if stripped and registry.has(stripped):
                return stripped

    # 3. normalised match
    normalized = _normalize_name(candidate)
    if normalized:
        for descriptor in registry.list_descriptors():
            if _normalize_name(descriptor.name) == normalized:
                return descriptor.name

    # 4. Levenshtein on normalised form
    if not normalized:
        return None
    best_name: Optional[str] = None
    best_distance: int = max_distance + 1
    tied: bool = False
    for descriptor in registry.list_descriptors():
        norm = _normalize_name(descriptor.name)
        if not norm:
            continue
        distance = _levenshtein(normalized, norm, max_distance=max_distance)
        if distance > max_distance:
            continue
        if distance < best_distance:
            best_distance = distance
            best_name = descriptor.name
            tied = False
        elif distance == best_distance and best_name != descriptor.name:
            # Two different registered tools tie at the same distance.
            # Refuse to guess — the caller will surface unknown-tool
            # so the model picks one explicitly.
            tied = True

    if tied or best_name is None:
        return None
    return best_name


# ---------------------------------------------------------------------------
# Cap (P0-6 part 1)
# ---------------------------------------------------------------------------


def _is_delegate_call(call: ToolCall, delegate_names: Iterable[str]) -> bool:
    """True iff this call's name normalises to one in ``delegate_names``."""
    if not call.name:
        return False
    normalised = _normalize_name(call.name)
    return normalised in {_normalize_name(n) for n in delegate_names}


# ---------------------------------------------------------------------------
# Dedup (P0-6 part 2)
# ---------------------------------------------------------------------------


def _dedup_key(call: ToolCall) -> str:
    """Stable JSON-serialised key for the (name, args) pair.

    Args are dumped with ``sort_keys=True`` so semantically equal
    dicts produce the same key regardless of insertion order.  When
    args contain non-JSON-serialisable values (rare — args from the
    LLM are always JSON), we fall back to ``repr`` which is order-
    deterministic for ``dict``s in Python 3.7+ but loses the
    semantic-equality guarantee.  That fallback is acceptable
    because a tool-call carrying non-JSON args is itself a bug
    elsewhere.
    """
    try:
        args_blob = json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args_blob = repr(call.arguments)
    return f"{_normalize_name(call.name)}::{args_blob}"


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


# Sentinel context-metadata keys.  Surfaced as constants so tests +
# observability tooling don't need to hardcode the strings.
TURN_KEY_INVALID_TOOL_RETRIES = "invalid_tool_retries"
TURN_KEY_DELEGATE_CALLS = "delegate_calls_this_turn"


def prepare_tool_calls(
    calls: Sequence[ToolCall],
    *,
    registry: "ToolRegistry",
    config: "EngineConfig",
    context: "TurnContext",
) -> ToolDispatchPlan:
    """Sanitise a freshly-emitted batch of ``ToolCall`` for dispatch.

    Order of operations matters:

    1. **Repair** runs first because cap and dedup compare against
       *registered* names.  Repairing ``readFile`` → ``read_file``
       lets dedup spot a follow-up duplicate-by-args correctly even
       if the model spelled the name differently across calls.
    2. **Cap** runs next so dedup doesn't let a runaway 10x
       ``delegate_task`` blow past the limit just because they all
       had different goal strings.
    3. **Dedup** runs last; it's the cheapest signal and the one
       most sensitive to upstream sanitisation.

    Counters in ``TurnContext.metadata``:

    * ``TURN_KEY_INVALID_TOOL_RETRIES`` — incremented once per
      ``ToolDispatchPlan`` that contains at least one unrepairable
      name.  When the value crosses
      ``EngineConfig.invalid_tool_max_retries`` we set
      ``plan.exit_reason = ExitReason.INVALID_TOOL_REPEATED.value``
      (string, not enum, to keep metadata JSON-safe).
    * ``TURN_KEY_DELEGATE_CALLS`` — running per-turn count of
      delegate-class calls actually dispatched.  Combined with the
      cap to short-circuit the rest of the iteration.
    """
    plan = ToolDispatchPlan()

    # Phase 1: name repair.
    repaired_calls: List[PreparedToolCall] = []
    invalid_names: List[str] = []
    for call in calls:
        if registry.has(call.name):
            repaired_calls.append(PreparedToolCall(call=call))
            continue
        repaired = repair_tool_name(call.name, registry)
        if repaired is None:
            invalid_names.append(call.name)
            structured_unknown = getattr(
                config,
                "tool_error_structured_format_enabled",
                True,
            )
            stub = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=_unknown_tool_message(
                    call.name,
                    registry,
                    structured=structured_unknown,
                ),
                is_error=True,
                metadata={
                    "_unknown_tool_recovery": True,
                    "_tool_error_category": "unknown_tool",
                }
                if structured_unknown
                else {"_unknown_tool_recovery": True},
            )
            repaired_calls.append(
                PreparedToolCall(call=call, synthetic_result=stub)
            )
            plan.unresolved_count += 1
            continue
        if repaired != call.name:
            repaired_call = ToolCall(
                id=call.id,
                name=repaired,
                arguments=call.arguments,
            )
            logger.info(
                "tool name repaired: %r → %r (call_id=%s)",
                call.name,
                repaired,
                call.id,
            )
            repaired_calls.append(
                PreparedToolCall(
                    call=repaired_call,
                    repaired_from=call.name,
                )
            )
            plan.repaired_count += 1
        else:
            repaired_calls.append(PreparedToolCall(call=call))

    # Phase 2: delegate-cap.  Counts only *successful* dispatches
    # (i.e. entries that didn't already get a synthetic result).
    delegate_names = tuple(
        getattr(config, "delegate_tool_names", DEFAULT_DELEGATE_TOOL_NAMES)
    )
    max_delegate = max(0, int(getattr(config, "max_delegate_calls_per_turn", 4)))
    delegate_count_so_far = int(context.metadata.get(TURN_KEY_DELEGATE_CALLS, 0))
    capped_calls: List[PreparedToolCall] = []
    for prepared in repaired_calls:
        if prepared.synthetic_result is not None:
            capped_calls.append(prepared)
            continue
        if not _is_delegate_call(prepared.call, delegate_names):
            capped_calls.append(prepared)
            continue
        if delegate_count_so_far >= max_delegate:
            stub = ToolResult(
                tool_call_id=prepared.call.id,
                name=prepared.call.name,
                content=(
                    f"Skipped: exceeds delegate cap "
                    f"({max_delegate} per turn). Consolidate the work "
                    "into fewer delegations and retry."
                ),
                is_error=True,
            )
            prepared.synthetic_result = stub
            plan.capped_count += 1
        else:
            delegate_count_so_far += 1
        capped_calls.append(prepared)
    context.metadata[TURN_KEY_DELEGATE_CALLS] = delegate_count_so_far

    # Phase 3: dedup.  First occurrence (per dedup key) dispatches;
    # later occurrences become stubs that reference the first.
    # Calls already carrying a synthetic_result skip the dedup
    # bookkeeping entirely (they wouldn't be dispatched anyway).
    # Skipping the entire phase when ``tool_dedup_enabled=False`` is
    # the supported escape hatch for diagnostic / replay use cases
    # where duplicate dispatch is intentional.
    if getattr(config, "tool_dedup_enabled", True):
        seen: dict[str, str] = {}        # dedup_key → first ToolCall.id
        final_calls: List[PreparedToolCall] = []
        for prepared in capped_calls:
            if prepared.synthetic_result is not None:
                final_calls.append(prepared)
                continue
            key = _dedup_key(prepared.call)
            if key in seen:
                stub = ToolResult(
                    tool_call_id=prepared.call.id,
                    name=prepared.call.name,
                    content=(
                        f"Duplicate of earlier call (id={seen[key]}); result "
                        "reused. Avoid issuing identical tool calls in a single "
                        "turn — read the earlier result before retrying."
                    ),
                    is_error=False,
                )
                prepared.synthetic_result = stub
                plan.deduped_count += 1
            else:
                seen[key] = prepared.call.id
            final_calls.append(prepared)
        plan.prepared = final_calls
    else:
        plan.prepared = capped_calls

    # Invalid-tool retry counter (P0-5 part 3).
    # Bumped once per *iteration* that surfaced any unrepairable name,
    # NOT once per name — multiple unknown names in the same iteration
    # are still a single confused turn.
    if invalid_names:
        retry_count = int(context.metadata.get(TURN_KEY_INVALID_TOOL_RETRIES, 0)) + 1
        context.metadata[TURN_KEY_INVALID_TOOL_RETRIES] = retry_count
        max_retries = int(getattr(config, "invalid_tool_max_retries", 3))
        if retry_count >= max_retries:
            from aether.runtime.contracts import ExitReason

            plan.exit_reason = ExitReason.INVALID_TOOL_REPEATED.value

    return plan


def _unknown_tool_message(
    name: str,
    registry: "ToolRegistry",
    *,
    structured: bool = True,
) -> str:
    """Build the role=tool error body for an unknown / unrepairable name.

    Includes the registry's available tool names so the model has the
    list it needs to self-correct on the next attempt.  Truncates the
    list at 25 entries to keep the message under a couple of hundred
    tokens — registries with hundreds of MCP-imported tools would
    otherwise crowd out the rest of the prompt.
    """
    if structured:
        return format_unknown_tool_error(name, registry.list_names()).text

    descriptors = registry.list_descriptors()
    names = sorted(d.name for d in descriptors)
    if len(names) > 25:
        listed = ", ".join(names[:25]) + f", … (+{len(names) - 25} more)"
    else:
        listed = ", ".join(names) if names else "(none registered)"
    return (
        f"Unknown tool: {name!r}. Available tools: {listed}. "
        "Re-issue the call using exactly one of those names."
    )


__all__ = [
    "DEFAULT_DELEGATE_TOOL_NAMES",
    "PreparedToolCall",
    "ToolDispatchPlan",
    "TURN_KEY_DELEGATE_CALLS",
    "TURN_KEY_INVALID_TOOL_RETRIES",
    "prepare_tool_calls",
    "repair_tool_name",
]
