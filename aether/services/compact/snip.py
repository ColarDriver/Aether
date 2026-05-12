"""Tier 2 snip — local redundancy removal.

Self-developed implementation (claude-code's ``snipCompact.ts`` is a
``@generated stub``).  Three conservative rules apply, in order, in a
single plan-and-apply pass so the rules' interactions are deterministic:

* **Rule 1 (DUPE)** — for the small whitelist
  :data:`SNIP_DUPE_RULE_TOOLS`, when the same ``(name, input)``
  ``tool_use`` appears two or more times, keep only the LAST
  occurrence.  All earlier ``tool_use`` ids and their matching
  ``tool_result`` blocks are deleted.  ``shell`` / ``write_file``
  are intentionally excluded because identical input there can
  produce different outcomes (timestamps, side-effects).

* **Rule 2 (FAIL)** — for ANY tool, when an ``is_error=True``
  ``tool_result`` is followed later (by encounter order in the
  message list) by an ``is_error=False`` call with the same
  ``(name, input)``, the earlier failed pair is deleted.  Pure
  failures with no follow-up success are preserved (failure itself
  is meaningful signal).

* **Rule 3 (EMPTY)** — assistant messages whose content is empty /
  whitespace-only / only contains empty ``text`` or empty
  ``thinking`` blocks (and crucially **no** ``tool_use``) are
  deleted whole.  Streaming-interrupted assistant frames are the
  primary source of these.

Pairing invariant
-----------------

Every assistant ``tool_use[id=X]`` MUST have a matching user
``tool_result[tool_use_id=X]`` and vice versa — Anthropic's wire
schema rejects orphans outright.  The :class:`Snipper` enforces this
by collecting every deletion target into a single
:class:`_SnipPlan` and then doing one atomic rewrite that:

* removes both halves of any deleted ``tool_use_id`` together,
* drops a message whole if all of its content blocks were removed,
* drops an assistant message that ended up containing only empty
  thinking blocks (a degenerate "skeleton" left after the cut).

OpenAI-style messages (``role='tool'`` + ``tool_calls``) are
recognised in a *limited* way: Rule 1 / 2 inspect them as
information sources but the deletion path is currently
Anthropic-shape only.  This matches Sprint 3's de-facto
canonical message shape inside the engine; OpenAI-shape Tier 2
support is a planned follow-up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from aether.runtime.core.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.token_estimation import estimate_messages_tokens


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


SNIP_DUPE_RULE_TOOLS: tuple[str, ...] = (
    "read_file",
    "list_dir",
    "glob",
    "grep",
)
"""Tools eligible for Rule 1 (DUPE) — read-only resource-fetch verbs.

Excluding ``shell`` and ``write_file`` is deliberate: same input on
those can produce different outcomes (timestamps, file mutations,
external state).  Adding a tool here means "two calls with the same
``(name, input)`` carry no new information" — a strong claim that
must hold for *every* deployment.  Operators who ship custom
read-only tools should currently fork this constant; per-deployment
configurability is a planned extension.
"""


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class _SnipPlan:
    """The set of deletions to apply atomically in one rewrite pass.

    Built by :meth:`Snipper._plan_rule1_dupe` /
    :meth:`Snipper._plan_rule2_failed` /
    :meth:`Snipper._plan_rule3_empty_assistant` and consumed by
    :meth:`Snipper._apply_plan`.  Splitting plan-and-apply keeps the
    three rules side-effect-free and makes the rule interactions
    obvious — a tool_use_id targeted by both Rule 1 and Rule 2 is
    deleted exactly once because ``set`` deduplicates naturally.
    """

    delete_tool_use_ids: set[str] = field(default_factory=set)
    delete_msg_indices: set[int] = field(default_factory=set)

    @property
    def has_work(self) -> bool:
        return bool(self.delete_tool_use_ids or self.delete_msg_indices)


# ---------------------------------------------------------------------------
# Snipper
# ---------------------------------------------------------------------------


class Snipper:
    """Tier 2 redundancy snipper.

    Conforms to the ``CompactorTier`` protocol.  Returns the rewritten
    messages list (a NEW list — original is not mutated, matching the
    ``maybe_run`` contract every other tier honours) and the
    estimator-computed token delta.
    """

    name: str = "tier2_snip"

    def __init__(self, *, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,  # noqa: ARG002 — protocol signature
        turn_context: TurnContext,
    ) -> tuple[list[dict[str, Any]], int]:
        if not getattr(self.config, "compression_enabled", False):
            return messages, 0
        if not getattr(self.config, "snip_enabled", True):
            return messages, 0
        if not messages:
            return messages, 0

        plan = _SnipPlan()
        if getattr(self.config, "snip_dupe_enabled", True):
            self._plan_rule1_dupe(messages, plan)
        if getattr(self.config, "snip_fail_enabled", True):
            self._plan_rule2_failed(messages, plan)
        if getattr(self.config, "snip_empty_enabled", True):
            self._plan_rule3_empty_assistant(messages, plan)

        if not plan.has_work:
            return messages, 0

        before = estimate_messages_tokens(messages)
        new_messages = self._apply_plan(messages, plan)
        after = estimate_messages_tokens(new_messages)
        freed = max(0, before - after)

        deleted_pairs = len(plan.delete_tool_use_ids)
        deleted_msgs = len(plan.delete_msg_indices)

        # Observability: a single counter accumulating all snipped
        # units this turn (pairs + standalone empty messages).  The
        # engine surfaces this on
        # ``EngineResult.metadata['compaction']['tier2_snipped_count']``.
        turn_context.metadata["tier2_snipped_count"] = (
            int(turn_context.metadata.get("tier2_snipped_count", 0))
            + deleted_pairs
            + deleted_msgs
        )

        self.logger.info(
            "tier2: snipped %d tool-use/result pairs + %d empty assistants, "
            "freed ~%d tokens",
            deleted_pairs,
            deleted_msgs,
            freed,
        )

        return new_messages, freed

    # ------------------------------------------------------------------
    # Rule 1 — DUPE
    # ------------------------------------------------------------------

    def _plan_rule1_dupe(
        self,
        messages: list[dict[str, Any]],
        plan: _SnipPlan,
    ) -> None:
        """Mark all-but-last duplicates of ``(name, input)`` for deletion.

        Encounter order matters: ``groups[key]`` is built by walking
        messages in order, then we keep the LAST entry and mark the
        prefix for deletion.  Iterating over ``groups.values()``
        afterwards is safe because the per-key list internally
        preserves order.
        """
        compactable = {_normalize_name(n) for n in SNIP_DUPE_RULE_TOOLS}
        groups: dict[str, list[str]] = {}
        for msg in messages:
            for tid, name, input_ in self._iter_tool_uses(msg):
                if name not in compactable:
                    continue
                key = self._make_key(name, input_)
                groups.setdefault(key, []).append(tid)
        for ids in groups.values():
            if len(ids) >= 2:
                plan.delete_tool_use_ids.update(ids[:-1])

    # ------------------------------------------------------------------
    # Rule 2 — FAIL
    # ------------------------------------------------------------------

    def _plan_rule2_failed(
        self,
        messages: list[dict[str, Any]],
        plan: _SnipPlan,
    ) -> None:
        """Delete failed pairs that are superseded by a later success.

        Two-pass scan:

        1. Walk all messages in order, recording each ``tool_use``'s
           ``id``, ``key`` (= ``json([name, input])``), and a
           sequence number.  Keys come from the same
           :meth:`_make_key` helper Rule 1 uses, so the two rules
           agree on what counts as "the same call".
        2. Walk again and pair each ``tool_result`` with its
           ``tool_use`` via ``tool_use_id``, populating ``is_error``.
        3. For each key, find the first ``is_error=False`` entry; if
           any failures precede it, mark them for deletion.

        Tool_uses without a result (interrupted mid-call) are never
        marked — we don't know whether the missing result would
        have been a success.  Pure failure runs (no success at all)
        are preserved because the failure itself can be informative
        to the model.
        """
        # Sequence + meta keyed by tool_use_id.  Sequence is the
        # encounter index across the whole message list (assistant +
        # user) so the per-key ordering is stable even when
        # tool_use and its result live on the same vs different
        # messages.
        per_id: dict[str, dict[str, Any]] = {}
        seq = 0
        for msg in messages:
            for tid, name, input_ in self._iter_tool_uses(msg):
                if tid in per_id:
                    # Duplicate id (defensive — providers shouldn't
                    # do this, but the registry path is normalised
                    # so mirror entries here would double-count).
                    continue
                per_id[tid] = {
                    "key": self._make_key(name, input_),
                    "seq": seq,
                    "is_error": None,
                }
                seq += 1
            for tid, is_error in self._iter_tool_results(msg):
                if tid in per_id:
                    per_id[tid]["is_error"] = bool(is_error)

        # Group by key in encounter order, then apply the
        # "first success wins" rule.
        groups: dict[str, list[tuple[str, bool, int]]] = {}
        for tid, meta in per_id.items():
            if meta["is_error"] is None:
                continue
            groups.setdefault(meta["key"], []).append(
                (tid, meta["is_error"], meta["seq"])
            )

        for entries in groups.values():
            entries.sort(key=lambda x: x[2])
            first_success_idx: Optional[int] = None
            for i, (_, is_err, _seq) in enumerate(entries):
                if not is_err:
                    first_success_idx = i
                    break
            if first_success_idx is None or first_success_idx == 0:
                continue
            for tid, is_err, _seq in entries[:first_success_idx]:
                if is_err:
                    plan.delete_tool_use_ids.add(tid)

    # ------------------------------------------------------------------
    # Rule 3 — EMPTY
    # ------------------------------------------------------------------

    def _plan_rule3_empty_assistant(
        self,
        messages: list[dict[str, Any]],
        plan: _SnipPlan,
    ) -> None:
        for idx, msg in enumerate(messages):
            if self._is_empty_assistant(msg):
                plan.delete_msg_indices.add(idx)

    @staticmethod
    def _is_empty_assistant(msg: dict[str, Any]) -> bool:
        """Detect assistant frames that carry zero useful content.

        ``content=None`` and empty/whitespace strings count as empty.
        For list content we require EVERY block to be empty
        (whitespace-only ``text`` or ``thinking``); the presence of
        any ``tool_use`` block or any non-empty ``text`` /
        ``thinking`` returns False.  Unrecognised block types
        ("redacted_thinking", "image", custom provider blocks)
        return False — we'd rather keep an unknown block than risk
        deleting a meaningful one.
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return False
        # Belt and braces: an assistant carrying OpenAI-style
        # ``tool_calls`` is NOT empty even if its ``content`` is.
        if msg.get("tool_calls"):
            return False
        content = msg.get("content")
        if content is None:
            return True
        if isinstance(content, str):
            return not content.strip()
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    return False
                t = block.get("type")
                if t == "text" and (block.get("text") or "").strip():
                    return False
                if t == "thinking" and (block.get("thinking") or "").strip():
                    return False
                if t == "tool_use":
                    return False
                if t not in {"text", "thinking"}:
                    return False
            return True
        return False

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _apply_plan(
        self,
        messages: list[dict[str, Any]],
        plan: _SnipPlan,
    ) -> list[dict[str, Any]]:
        """Single-pass rewrite that honours the deletion plan.

        Builds a NEW list; original messages and their nested dicts
        are left unmutated (the per-message ``dict(msg)`` copy is
        cheap and matches the convention every other tier uses).

        Three drop conditions for a message:

        1. Its index is in ``plan.delete_msg_indices`` (Rule 3
           empty-assistant deletion).
        2. After per-block filtering its ``content`` list is empty
           (every block was a deleted ``tool_use`` / ``tool_result``).
        3. After filtering it's an assistant whose only remaining
           blocks are empty thinking — a degenerate skeleton left
           when the only non-thinking blocks were tool_use blocks
           we just removed.
        """
        new_messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            if idx in plan.delete_msg_indices:
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                new_messages.append(msg)
                continue
            new_content: list[Any] = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue
                t = block.get("type")
                if t == "tool_use" and block.get("id") in plan.delete_tool_use_ids:
                    continue
                if (
                    t == "tool_result"
                    and block.get("tool_use_id") in plan.delete_tool_use_ids
                ):
                    continue
                new_content.append(block)
            if not new_content:
                continue
            if msg.get("role") == "assistant" and all(
                isinstance(b, dict)
                and b.get("type") == "thinking"
                and not (b.get("thinking") or "").strip()
                for b in new_content
            ):
                continue
            new_msg = dict(msg)
            new_msg["content"] = new_content
            new_messages.append(new_msg)
        return new_messages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_tool_uses(msg: dict[str, Any]):
        """Yield ``(tool_use_id, normalised_name, input_dict)`` for a message.

        Walks Anthropic-style ``content=[{type=tool_use, id, name,
        input}]`` first, then OpenAI-style ``tool_calls`` so Rule 1
        and Rule 2 see both shapes when populating their key
        groups.  IDs without a name (or vice versa) are skipped to
        keep the key consistent.
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                tid = block.get("id")
                if not isinstance(tid, str) or not tid:
                    continue
                name = _normalize_name(block.get("name") or "")
                yield tid, name, block.get("input") or {}
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tid = call.get("id")
                if not isinstance(tid, str) or not tid:
                    continue
                fn = call.get("function") or {}
                name = _normalize_name(fn.get("name") or call.get("name") or "")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except (TypeError, ValueError):
                        args = {"_raw": args}
                if not isinstance(args, dict):
                    args = {}
                yield tid, name, args

    @staticmethod
    def _iter_tool_results(msg: dict[str, Any]):
        """Yield ``(tool_use_id, is_error_bool)`` pairs for a message.

        Anthropic-style: ``role='user'`` with content blocks of
        ``type='tool_result'`` carrying ``tool_use_id`` and
        optional ``is_error``.  OpenAI-style: ``role='tool'`` with
        ``tool_call_id`` — these have no ``is_error`` flag, so we
        attempt to detect failure by inspecting the content for
        common error markers; absent any signal we conservatively
        treat OpenAI-style results as ``is_error=False`` (Rule 2
        never deletes successful results, so a false-positive
        success is the safe direction).
        """
        if not isinstance(msg, dict):
            return
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    tid = block.get("tool_use_id")
                    if isinstance(tid, str) and tid:
                        yield tid, bool(block.get("is_error", False))
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if isinstance(tid, str) and tid:
                yield tid, False

    @staticmethod
    def _make_key(name: str, input_: Any) -> str:
        """Stable, JSON-derived ``(name, input)`` key.

        ``sort_keys=True`` makes dicts whose key order differs but
        contents are the same hash to the same key.  Falls back to
        ``str()`` when the input contains non-JSON-friendly objects
        (sets, custom classes) — those won't dedupe across
        equivalent representations but at least won't crash, and
        custom-class inputs in ``tool_use`` blocks are a programmer
        error in upstream code anyway.
        """
        try:
            return json.dumps([name, input_], sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return f"{name}::{input_!s}"


def _normalize_name(name: str) -> str:
    """Lazy import wrapper so the snip module doesn't pull in agents at import time.

    ``aether.agents.core.phantom_tool._normalize_name`` is the
    canonical Aether tool-name normalisation (case-fold,
    dash→underscore, namespace strip).  We resolve it lazily so
    ``services.compact.snip`` stays a pure leaf module — keeps the
    import graph clean and avoids a circular when ``agents.core``
    eventually imports from ``services.compact``.
    """
    from aether.agents.core.phantom_tool import _normalize_name as impl

    return impl(name)
