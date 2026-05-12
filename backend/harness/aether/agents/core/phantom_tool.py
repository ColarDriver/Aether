"""Phantom-tool intent detection and corrective-message construction.

Some model providers (notably Kimi-class routers) occasionally emit a
response that *looks* like a tool invocation but is actually plain
prose — the model writes ``\u0060\u0060\u0060bash <cmd>`` or ``<function=NAME> <cmd>`` /
``<invoke name="…">…</invoke>`` into the assistant content stream
instead of populating the structured ``tool_calls`` field.  The engine
sees an empty ``tool_calls`` array and, without recovery, falls
straight through to the TEXT_RESPONSE finalise path: the user's turn
ends with the model "describing" what it would have done and nothing
actually executed.

This module owns the detection heuristics and produces a corrective
``role=user`` message that the run-loop appends to ``messages`` before
issuing the next LLM call.  The message tells the model exactly what
it wrote, why it didn't run, and asks it to retry with proper
``tool_calls``.  Hermes ships an analogous "self-correct on bad JSON"
nudge as part of its tool-error injection layer; see P0-4 for the
parallel pattern in the engine.

Design notes
------------

* **Pure-function module.**  All helpers are stateless; the engine
  threads its retry counter through ``TurnContext.metadata`` and never
  caches anything in this module.  Tests can call the helpers directly
  without a fixture.

* **Shared with the CLI.**  ``aether.cli.ui`` imports the same
  primitives so the user-visible "└ attempted: $ <cmd>" hint and the
  engine's corrective message stay in lock-step.  Drift between the
  two would silently confuse users — the diagnostic might list 3
  attempted commands while the engine retried with a 2-command list.

* **Conservative.**  We only detect *clear* phantom intent (fenced
  shell block, ``<function=>`` tag, or ``<invoke>`` XML).  A polite
  greeting that happens to mention "ls" in passing should not trigger
  a corrective retry — false positives turn an answer into a
  confusing back-and-forth.

* **No execution.**  We *parse* the attempted command(s) but never
  run them.  Auto-running raw shell commands the model wrote in prose
  bypasses every guardrail (tool registry, permissions, etc.) and is
  exactly the failure mode the structured ``tool_calls`` field exists
  to avoid.  The corrective retry pattern keeps the model in charge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

from aether.runtime.core.contracts import ToolCall

if TYPE_CHECKING:
    from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fenced shell-block extraction (multi-block, language-tag aware)
# ---------------------------------------------------------------------------
#
# The regex is shared with ``aether.cli.ui._COMMAND_FENCE_RE``.  Body
# capture is non-greedy and ends at the *earliest* of:
#   1. ``\u0060\u0060\u0060`` *not* followed by a shell-language tag — a real
#      closing fence; consume it (so ``sub`` also drops the fence text).
#   2. (lookahead) a blank line — paragraph break, treated as the
#      implicit end of an unclosed phantom-tool block since models
#      never embed real prose-paragraphs inside a shell command.
#   3. (lookahead) ``\u0060\u0060\u0060<lang>`` — opener of the next fenced block.
#   4. End-of-string — models routinely truncate the line they were
#      about to call a tool on.
#
# The negative lookahead in branch 1 is critical: without it the
# non-greedy ``.*?`` would happily stop at the *opening* ``\u0060\u0060\u0060`` of
# the *next* shell fence and substitution would leave a stranded
# ``bash\n...`` behind that fails to re-match.
_SHELL_LANG_GROUP = r"(?:bash|sh|shell|zsh|console|cmd|ps1?|powershell)"
_COMMAND_FENCE_RE = re.compile(
    r"(?is)```" + _SHELL_LANG_GROUP + r"\b"
    r"[ \t]*\n?(.*?)"
    r"(?:\n?```(?!" + _SHELL_LANG_GROUP + r"\b)"
    r"|(?=\n[ \t]*\n)"
    r"|(?=\n?```" + _SHELL_LANG_GROUP + r"\b)"
    r"|\Z)"
)

# The non-standard ``<function=NAME> <body>`` tag (Kimi-style).  Body
# extends until the next ``<function=`` / ``</function`` / end-of-text.
_FUNCTION_EQ_TAG_RE = re.compile(
    r"(?is)<function=([^>\s/]+)>(.*?)(?=<function=|</function|\Z)"
)

# Anthropic-style ``<invoke name="…">…<parameter>…</parameter>…</invoke>``
# inline tags.  Some models emit these inside content rather than
# populating ``tool_calls``.
_INVOKE_BLOCK_RE = re.compile(
    r'(?is)<invoke\b[^>]*name="([^"]+)"[^>]*>(.*?)</invoke\s*>'
)
_INVOKE_PARAM_RE = re.compile(
    r'(?is)<parameter\b[^>]*name="([^"]+)"[^>]*>(.*?)</parameter\s*>'
)
# OpenAI-ish ``<tool_call>{"name":..., "arguments":...}</tool_call>``.
_TOOL_CALL_JSON_RE = re.compile(
    r"(?is)<tool_call\b[^>]*>\s*(\{.*?\})\s*</tool_call\s*>"
)
# Moonshot / Kimi template residue: ``<functions.NAME:N>{json}</functions.NAME:N>``
# where ``N`` is a slot index (we don't care which one).  Three failure
# modes seen in the wild:
#
#   * Properly bracketed: ``<functions.shell:0>{json}</functions.shell:0>``.
#   * Leading ``<`` dropped, trailing ``>`` kept:
#     ``functions.shell:0> {json} </functions.shell:0>``.
#   * Both angle brackets dropped on the open tag:
#     ``functions.shell:0 {json} </functions.shell:0>``.
#
# The leading-anchor ``(?:^|(?<=[<\s\n]))`` matches all three forms
# while *rejecting* the close tag — at the position of the close
# tag's ``functions``, the character before is ``/`` which is not in
# the allowed lookbehind set.  Without that the regex would happily
# match ``functions.shell:0>`` inside ``</functions.shell:0>`` and
# produce a phantom empty body.  The trailing ``>?`` is required to
# accept the bare form where the model never wrote the close-of-open
# angle bracket at all.  The body terminator ``(?=</?\s*functions\.``
# stops the match at the next phantom tag (open or close), giving us
# clean per-call separation when the model emits multiple in a row.
_FUNCTIONS_SLOT_TAG_RE = re.compile(
    r"(?is)(?:^|(?<=[<\s\n]))\s*functions\.([A-Za-z_][A-Za-z0-9_]*):\d+\s*>?"
    r"(.*?)"
    r"(?=</?\s*functions\.[A-Za-z_]|\Z)"
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PhantomToolIntent:
    """What the model *attempted* to invoke when it bypassed ``tool_calls``.

    Three optional fields are populated independently:

    * ``shell_commands`` — every fenced shell block we could parse,
      one entry per block, multi-line bodies collapsed to a single
      space-joined line so the corrective message lists each attempt
      compactly.

    * ``invoke_calls`` — parsed ``(name, args)`` pairs for every
      ``<invoke>``/``<tool_call>``/``<function=>`` style tag.  When the
      model mixed XML/JSON inline tags with ``\u0060\u0060\u0060bash`` blocks
      (yes, this happens) we collect both.

    * ``raw_intents_count`` — total inline intent markers we found
      across all syntaxes.  Used by the diagnostic / footer to say
      "3 attempted commands" without re-counting on the consumer side.

    A populated instance with ``raw_intents_count > 0`` is the trigger
    for the engine to send a corrective message instead of finalising
    the turn.  An empty instance (every field default) means the
    response was prose and the engine should fall through to the
    normal TEXT_RESPONSE / EMPTY_RESPONSE branch.
    """

    shell_commands: List[str] = field(default_factory=list)
    invoke_calls: List[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def raw_intents_count(self) -> int:
        return len(self.shell_commands) + len(self.invoke_calls)

    def is_empty(self) -> bool:
        return self.raw_intents_count == 0


# ---------------------------------------------------------------------------
# Public detection API
# ---------------------------------------------------------------------------


def detect_phantom_tool_intent(content: str) -> PhantomToolIntent:
    """Return every parsed phantom-tool intent in *content*.

    Combines the three known failure-mode parsers (fenced shell
    blocks, ``<function=NAME>`` inline tags, Anthropic-style
    ``<invoke>`` / OpenAI-ish ``<tool_call>`` XML).  Returns an empty
    :class:`PhantomToolIntent` for a normal prose response — callers
    should test with ``is_empty()`` before deciding whether to send a
    corrective message.

    The function is intentionally conservative: a fenced shell block
    body of pure whitespace, an ``<invoke>`` tag with an empty name,
    or a malformed JSON ``<tool_call>`` payload all contribute zero
    intents.  False positives would turn legitimate answers into
    confusing self-correction loops.
    """
    intent = PhantomToolIntent()
    if not content:
        return intent

    # 1) Fenced shell blocks.
    if "```" in content:
        for match in _COMMAND_FENCE_RE.finditer(content):
            body = (match.group(1) or "").strip()
            if not body:
                continue
            flat = " ".join(
                line.strip() for line in body.splitlines() if line.strip()
            )
            if flat:
                intent.shell_commands.append(flat)

    # 2) <function=NAME> ... inline tags.
    for match in _FUNCTION_EQ_TAG_RE.finditer(content):
        name = (match.group(1) or "").strip()
        body = (match.group(2) or "").strip()
        if not name:
            continue
        # Strip a stray ``</function>`` close tag that the body match
        # may have absorbed when the model didn't follow up with
        # another ``<function=`` immediately.
        body = re.sub(r"</?\s*function\b[^>]*>\s*$", "", body).strip()

        args: dict[str, Any] = {}
        if body:
            # Most observed bodies are JSON objects (the model is
            # mimicking the OpenAI tool-call shape).  Fall back to
            # treating the whole body as a single ``command`` string
            # for the legacy "model just dumped the command" form.
            try:
                parsed = json.loads(body)
            except Exception:  # noqa: BLE001
                parsed = None
            if isinstance(parsed, dict):
                args = parsed
            else:
                args["command"] = " ".join(body.split())
        intent.invoke_calls.append((name, args))

    # 3) Anthropic-style <invoke>...</invoke>.
    for invoke in _INVOKE_BLOCK_RE.finditer(content):
        name = (invoke.group(1) or "").strip()
        if not name:
            continue
        params: dict[str, Any] = {}
        for param in _INVOKE_PARAM_RE.finditer(invoke.group(2) or ""):
            key = (param.group(1) or "").strip()
            value = (param.group(2) or "").strip()
            if key:
                params[key] = value
        intent.invoke_calls.append((name, params))

    # 4) OpenAI-ish <tool_call>{json}</tool_call>.
    for call in _TOOL_CALL_JSON_RE.finditer(content):
        try:
            payload = json.loads(call.group(1))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or payload.get("tool") or "").strip()
        raw_args = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:  # noqa: BLE001
                raw_args = {"_raw": raw_args}
        if not isinstance(raw_args, dict):
            raw_args = {"_raw": raw_args}
        if name:
            intent.invoke_calls.append((name, raw_args))

    # 5) Moonshot/Kimi <functions.NAME:N>{json}</functions.NAME:N>.
    for match in _FUNCTIONS_SLOT_TAG_RE.finditer(content):
        name = (match.group(1) or "").strip()
        body = (match.group(2) or "").strip()
        if not name:
            continue
        # Strip a trailing close tag if the body match ate it whole (the
        # outer regex uses a lookahead, but a few Kimi variants don't
        # close cleanly and the body includes the close-tag anyway).
        body = re.sub(r"</?\s*functions\.[A-Za-z_][A-Za-z0-9_]*:\d+\s*>\s*$", "", body).strip()

        args: dict[str, Any] = {}
        if body:
            # Most Kimi templates write a JSON object, but a stray model
            # sometimes types the command directly.  Try JSON first, then
            # fall back to "treat the whole body as a shell command".
            try:
                parsed = json.loads(body)
            except Exception:  # noqa: BLE001
                parsed = None
            if isinstance(parsed, dict):
                args = parsed
            else:
                args = {"command": " ".join(body.split())}
        intent.invoke_calls.append((name, args))

    return intent


# ---------------------------------------------------------------------------
# Corrective-message construction
# ---------------------------------------------------------------------------


# Bilingual corrective wording — Chinese first because the operator
# baseline our diagnostic UI targets is bilingual zh/en, and putting
# the Chinese instruction first reads more naturally for our primary
# users.  English follows so models trained more heavily on English
# instruction-tuning data still get an unambiguous signal.
_CORRECTIVE_HEADLINE = (
    "你刚才把工具调用写成了散文（"
    "```bash`, `<function=NAME>`, 或 `<invoke>` 标签），"
    "但没有填写结构化 `tool_calls` 字段，所以**没有任何工具被实际执行**。"
)
_CORRECTIVE_FALLBACK_EN = (
    "You just wrote tool invocations as prose (```bash, <function=NAME>, "
    "or <invoke> tags) without populating the structured `tool_calls` "
    "field, so **no tool actually ran**."
)
_CORRECTIVE_INSTRUCTION = (
    "请重新尝试：从你可用的工具列表中选择合适的工具（例如 bash / shell / "
    "execute_command），并通过结构化 `tool_calls` 字段调用它们。不要"
    "把命令写在 markdown 代码块或 `<function=...>` 标签里。"
)
_CORRECTIVE_INSTRUCTION_EN = (
    "Please retry: pick the appropriate tool from your tool list "
    "(e.g. bash / shell / execute_command) and invoke it via the "
    "structured `tool_calls` field.  Do NOT write the command in a "
    "markdown code fence or in `<function=…>` tags."
)


def build_corrective_user_message(
    intent: PhantomToolIntent,
    *,
    attempt_index: int,
) -> dict[str, Any]:
    """Build the ``role=user`` corrective message for the run-loop.

    Returned shape matches the messages list the run-loop already
    appends to (``{"role": "user", "content": "…"}``), so the caller
    can ``messages.append(build_corrective_user_message(...))`` without
    further wrapping.

    *attempt_index* is the 1-based number of this corrective cycle.
    The wording is identical regardless of attempt number — repeated
    nudges with novel phrasings would just confuse the model further
    — but the index is included in a leading bracket so transcript
    inspection makes the retry pattern obvious.

    Always includes a compact "attempted:" digest of every parsed
    intent so the model sees exactly what it wrote and can lift the
    same arguments straight into a structured ``tool_calls`` block on
    the next turn.  Truncates the digest to keep the corrective
    message under a couple of hundred tokens — Hermes-style nudges
    that ballooned past that point started crowding out the actual
    user request from the context window.
    """
    parts: list[str] = []
    parts.append(f"[Aether note · attempt {attempt_index}]")
    parts.append(_CORRECTIVE_HEADLINE)
    parts.append(_CORRECTIVE_FALLBACK_EN)
    parts.append("")
    parts.append(_CORRECTIVE_INSTRUCTION)
    parts.append(_CORRECTIVE_INSTRUCTION_EN)

    digest = _format_attempt_digest(intent)
    if digest:
        parts.append("")
        parts.append("Detected attempted invocations:")
        parts.append(digest)

    return {"role": "user", "content": "\n".join(parts)}


def _format_attempt_digest(intent: PhantomToolIntent, limit: int = 6) -> str:
    """Render a bullet list of what the model tried to invoke.

    Caps the entries at *limit* so a runaway response with 20 inline
    intents doesn't balloon the corrective message.  Each shell
    command is shown as ``- $ <cmd>``; each invoke / function-eq /
    tool_call entry is shown as ``- <name>(arg=value, …)`` so the
    model recognises exactly what it tried.  Truncated lines end in
    a single ``…``.
    """
    lines: list[str] = []
    for cmd in intent.shell_commands:
        if len(lines) >= limit:
            break
        lines.append(f"- $ {_truncate(cmd)}")
    for name, args in intent.invoke_calls:
        if len(lines) >= limit:
            break
        rendered_args = _format_args(args)
        if rendered_args:
            lines.append(f"- {name}({rendered_args})")
        else:
            lines.append(f"- {name}()")
    extra = intent.raw_intents_count - len(lines)
    if extra > 0:
        lines.append(f"- … (+{extra} more)")
    return "\n".join(lines)


def _format_args(args: dict[str, Any], limit: int = 96) -> str:
    if not args:
        return ""
    pairs: list[str] = []
    for key, value in args.items():
        if isinstance(value, (dict, list)):
            try:
                rendered = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:  # noqa: BLE001
                rendered = repr(value)
        else:
            rendered = str(value)
        rendered = _truncate(rendered, limit=limit)
        pairs.append(f"{key}={rendered}")
    out = ", ".join(pairs)
    return _truncate(out, limit=160)


def _truncate(text: str, limit: int = 96) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Phantom → structured ToolCall synthesis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SynthesisOutcome:
    """Result of attempting to turn a :class:`PhantomToolIntent` into real ``ToolCall``s.

    * ``tool_calls`` — synthesized calls, ready to be assigned to
      :attr:`NormalizedResponse.tool_calls` and dispatched through the
      normal tool-execution path.

    * ``notes`` — short human-readable strings (one per synthesized
      call) that the UI surfaces as e.g. ``↻ synthesized shell(ls -la)
      from prose``.  Listing them keeps the user informed that the
      model misbehaved without making them read the loud
      ``inline tool tags`` warning.

    * ``unresolved`` — the (name-or-prose, reason) pairs we *couldn't*
      synthesize because the registry didn't carry a matching tool
      (or the args were unrecoverable).  Caller can use this to decide
      whether to fall through to the corrective-message path.
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    unresolved: List[tuple[str, str]] = field(default_factory=list)


# Names a model commonly writes when it means the registered ``shell``
# tool but doesn't know what we called it.  Keeping this list small +
# explicit (rather than fuzz-matching everything) makes synthesis
# predictable: a registered ``ssh_run`` tool will never accidentally
# claim a phantom ``shell`` block.
_SHELL_ALIASES: tuple[str, ...] = (
    "shell",
    "bash",
    "sh",
    "zsh",
    "execute_command",
    "exec_command",
    "run_command",
    "run_bash",
    "run_shell",
    "execute",
    "exec",
    "terminal",
    "cli",
    "command",
)


def _normalize_name(name: str) -> str:
    """Lower-case + replace hyphens with underscores; strip namespace prefixes."""
    raw = (name or "").strip()
    if not raw:
        return raw
    # Drop ``functions.X`` / ``mcp__server__X`` / ``ns:X`` / ``ns.X`` prefixes.
    if "__" in raw:
        raw = raw.split("__")[-1]
    if "." in raw:
        raw = raw.split(".")[-1]
    if ":" in raw:
        raw = raw.split(":")[0]  # strip slot index ``shell:0`` → ``shell``
    raw = raw.replace("-", "_").lower()
    return raw


def _resolve_registered(
    candidate: str,
    registry: "ToolRegistry",
) -> str | None:
    """Return the registry's canonical name for *candidate*, or ``None``.

    Strategy mirrors the spirit of P0-5 fuzzy repair but stays
    intentionally conservative: case-fold + underscore-normalisation
    only.  Levenshtein-class fuzzy match would risk the model writing
    ``read_files`` and us happily mapping it to ``read_file`` — better
    to surface that as unresolved than to dispatch the wrong tool.
    """
    if not candidate:
        return None
    if registry.has(candidate):
        return candidate
    normalized = _normalize_name(candidate)
    if not normalized:
        return None
    # Direct hit on the normalised form.
    for descriptor in registry.list_descriptors():
        if _normalize_name(descriptor.name) == normalized:
            return descriptor.name
    return None


def _resolve_shell_tool(registry: "ToolRegistry") -> str | None:
    """Return whichever name the registry uses for shell-style execution."""
    descriptors = registry.list_descriptors()
    by_normalized = {_normalize_name(d.name): d.name for d in descriptors}
    for alias in _SHELL_ALIASES:
        if alias in by_normalized:
            return by_normalized[alias]
    # Last-resort: any descriptor whose declared parameters obviously
    # accept a free-form command string.
    for descriptor in descriptors:
        props = (descriptor.parameters or {}).get("properties") or descriptor.parameters or {}
        if isinstance(props, dict) and "command" in props:
            return descriptor.name
    return None


def synthesize_tool_calls_from_phantom(
    intent: PhantomToolIntent,
    registry: "ToolRegistry",
    *,
    id_prefix: str = "phantom",
) -> SynthesisOutcome | None:
    """Try to turn a :class:`PhantomToolIntent` into real :class:`ToolCall`s.

    Returns ``None`` when nothing in *intent* could be safely mapped to
    a registered tool — the caller should fall through to the
    corrective-message retry path in that case.  When at least one
    call could be synthesized, returns a :class:`SynthesisOutcome`
    with the parsed calls *and* (separately) any unresolved entries
    so the UI can surface a partial-recovery hint.

    Synthesis rules:

    * **Fenced shell blocks** — emit a :class:`ToolCall` for the
      registry's shell-style tool (resolved via
      :data:`_SHELL_ALIASES`), passing the full body as ``command``.
      Each block becomes one call so the model can recover from
      typos in any individual block independently.

    * **``<function=NAME>`` / ``<invoke>`` / ``<tool_call>`` /
      ``<functions.NAME:N>``** — try to resolve ``NAME`` against the
      registry (case-fold, underscore-normalise, namespace-strip).
      On a hit, emit a :class:`ToolCall` with the parsed args.
      Special case: when the resolved alias is shell-flavoured but
      args don't include ``command``, but include exactly one string
      value, treat that value as the command.
    """
    if intent.is_empty():
        return None

    calls: List[ToolCall] = []
    notes: List[str] = []
    unresolved: List[tuple[str, str]] = []

    shell_tool = _resolve_shell_tool(registry)

    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{id_prefix}_{counter}"

    # --- fenced shell blocks ------------------------------------------------
    for cmd in intent.shell_commands:
        if shell_tool is None:
            unresolved.append((cmd, "no shell-style tool registered"))
            continue
        calls.append(
            ToolCall(id=_next_id(), name=shell_tool, arguments={"command": cmd})
        )
        notes.append(f"{shell_tool}({_truncate(cmd, limit=80)})")

    # --- inline tags --------------------------------------------------------
    for raw_name, args in intent.invoke_calls:
        resolved = _resolve_registered(raw_name, registry)
        if resolved is None:
            # Last-ditch: treat shell-aliased phantoms as the shell tool.
            if _normalize_name(raw_name) in _SHELL_ALIASES and shell_tool is not None:
                resolved = shell_tool
            else:
                unresolved.append((raw_name, "no tool by that name registered"))
                continue

        normalized_args = _normalize_invoke_args(resolved, args, shell_tool)
        if normalized_args is None:
            unresolved.append((raw_name, "could not parse arguments"))
            continue

        calls.append(ToolCall(id=_next_id(), name=resolved, arguments=normalized_args))
        notes.append(_format_synth_note(resolved, normalized_args))

    if not calls:
        return None
    return SynthesisOutcome(tool_calls=calls, notes=notes, unresolved=unresolved)


def _normalize_invoke_args(
    resolved: str,
    args: dict[str, Any],
    shell_tool: str | None,
) -> dict[str, Any] | None:
    """Massage parsed inline-tag args into something the executor accepts.

    Three observed Kimi failure modes:

    1. ``{"command": "ls -la"}`` — pass-through, common form.
    2. ``{"_raw": "ls -la"}`` — JSON parse failed in the detector, the
       raw body was preserved.  When the resolved tool is shell-style,
       lift it into ``command``.
    3. ``{}`` — empty body for shell-style call.  We can't synthesize
       a useful call; let the corrective path retry.
    """
    if not isinstance(args, dict):
        return None

    cleaned = dict(args)

    if shell_tool is not None and resolved == shell_tool:
        # Promote ``_raw`` / sole-string values into ``command``.
        if "command" not in cleaned:
            if "_raw" in cleaned and isinstance(cleaned["_raw"], str):
                cleaned["command"] = cleaned.pop("_raw")
            elif len(cleaned) == 1:
                only_key, only_value = next(iter(cleaned.items()))
                if isinstance(only_value, str) and only_key.lower() in {
                    "cmd",
                    "input",
                    "script",
                    "shell",
                    "bash",
                    "code",
                }:
                    cleaned = {"command": only_value}
        if not cleaned.get("command"):
            return None
        return cleaned

    # For non-shell tools we only synthesize when the args look
    # plausible (at least one non-meta key).
    cleaned.pop("_raw", None)
    if not cleaned:
        return None
    return cleaned


def _format_synth_note(name: str, args: dict[str, Any]) -> str:
    if not args:
        return f"{name}()"
    rendered = _format_args(args, limit=64)
    rendered = _truncate(rendered, limit=96)
    return f"{name}({rendered})"


__all__ = [
    "PhantomToolIntent",
    "SynthesisOutcome",
    "detect_phantom_tool_intent",
    "build_corrective_user_message",
    "synthesize_tool_calls_from_phantom",
]
