"""Engine configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    max_iterations: int = 8
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
