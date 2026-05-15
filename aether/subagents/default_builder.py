"""Default subagent builder implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from aether.runtime.control.interrupt_signal import InterruptSignal
from aether.config.schema import EngineConfig
from aether.subagents.builder import SubagentBuilder
from aether.subagents.contracts import SubagentTask
from aether.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aether.agents.core.agent import AgentEngine


# Tools every typed subagent gets by default when the type sets a
# whitelist (``tools=...``).  These are the meta-tools a child needs to
# stay controllable: ``task_stop`` so the parent can interrupt it,
# ``task_output`` so it can read peer task results, ``send_message`` so
# it can talk back, and ``skill`` so it can load on-demand playbooks.
# Not adding them to the whitelist would silently strand the child.
#
# A type can still ban them explicitly via ``disallowed_tools`` —
# ``Verifier`` does this for ``send_message`` since "verdict IS the
# message".  Explicit deny always wins over default allow.
META_TOOLS_DEFAULT_ALLOWED: frozenset[str] = frozenset({
    "task",
    "task_stop",
    "task_output",
    "send_message",
    "skill",
})


# Public aliases the model can pass to ``task(model="sonnet")``.  Resolved
# to a concrete model string before being forwarded to the provider via
# ``ModelCallConfig.extra["model"]``.  Unknown values pass through as-is
# so callers that already know the exact model id can use it directly.
_MODEL_ALIAS_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
    "haiku":  "claude-haiku-4-5-20251001",
}


def _resolve_model_alias(name: str) -> str:
    return _MODEL_ALIAS_MAP.get(name.strip().lower(), name.strip())


def _format_preloaded_skills_hint(skill_names: tuple[str, ...]) -> str:
    bullets = "\n".join(f"- {name}" for name in skill_names)
    return (
        "<system-reminder>\n"
        "You have been hand-picked for this task because the following "
        "skills are likely relevant. Consider invoking them early via "
        "the `skill` tool:\n"
        f"{bullets}\n"
        "</system-reminder>"
    )


class DefaultSubagentBuilder(SubagentBuilder):
    """Build child agents by inheriting parent dependencies and config."""

    def __init__(self, inherit_tools: bool = True, inherit_middlewares: bool = True) -> None:
        self.inherit_tools = inherit_tools
        self.inherit_middlewares = inherit_middlewares

    def build_child(
        self, parent: AgentEngine, task: SubagentTask, child_depth: int
    ) -> AgentEngine:
        from aether.agents.core.agent import AgentEngine

        provider = task.provider or parent.services.provider
        agent_type_def = task.metadata.get("_agent_type_def")
        tool_registry = parent.services.tool_registry if self.inherit_tools else None
        if tool_registry is not None:
            allowed_tools = getattr(agent_type_def, "tools", None)
            disallowed_tools = tuple(getattr(agent_type_def, "disallowed_tools", ()) or ())
            if allowed_tools is not None or disallowed_tools:
                tool_registry = _filter_tool_registry(
                    tool_registry,
                    allowed_tools=allowed_tools,
                    disallowed_tools=disallowed_tools,
                )
        system_prompt = getattr(agent_type_def, "system_prompt", None)
        if isinstance(system_prompt, str) and system_prompt.strip():
            existing = task.request.system_message
            task.request.system_message = (
                f"{system_prompt.strip()}\n\n{existing.strip()}"
                if isinstance(existing, str) and existing.strip()
                else system_prompt.strip()
            )

        type_skills = tuple(getattr(agent_type_def, "skills", ()) or ())
        if type_skills:
            hint = _format_preloaded_skills_hint(type_skills)
            existing = task.request.system_message
            task.request.system_message = (
                f"{existing.rstrip()}\n\n{hint}"
                if isinstance(existing, str) and existing.strip()
                else hint
            )

        # Model override: caller-supplied ``model`` arg (set by
        # ``AgentTool``) wins over the type definition's own model.
        # Both are forwarded via ``ModelCallConfig.extra["model"]`` so
        # the Claude provider picks them up at request-build time
        # without mutating the shared provider instance.
        caller_model = task.metadata.get("model_override")
        type_model = getattr(agent_type_def, "model", None)
        effective_model = (
            caller_model
            if isinstance(caller_model, str) and caller_model.strip() and caller_model.strip().lower() != "inherit"
            else type_model
        )
        if isinstance(effective_model, str) and effective_model.strip():
            resolved = _resolve_model_alias(effective_model)
            if task.request.model_config is not None:
                task.request.model_config.extra["model"] = resolved

        # ``max_turns`` clamps the child's iteration budget but never
        # raises it above the inherited ceiling — type definitions only
        # tighten, never loosen.
        max_iterations = (
            task.max_iterations
            if task.max_iterations is not None
            else parent.config.max_iterations
        )
        type_max_turns = getattr(agent_type_def, "max_turns", None)
        if isinstance(type_max_turns, int) and type_max_turns > 0:
            max_iterations = min(max_iterations, type_max_turns)

        child_config = EngineConfig(
            max_iterations=max_iterations,
            fail_on_tool_error=parent.config.fail_on_tool_error,
            raise_on_middleware_error=parent.config.raise_on_middleware_error,
            fail_on_unknown_tool=parent.config.fail_on_unknown_tool,
            enable_todo_hydration=parent.config.enable_todo_hydration,
            memory_nudge_interval=parent.config.memory_nudge_interval,
            skill_nudge_interval=parent.config.skill_nudge_interval,
            skill_listing_enabled=parent.config.skill_listing_enabled,
            skill_listing_token_budget=parent.config.skill_listing_token_budget,
            verification_directive_enabled=parent.config.verification_directive_enabled,
            faithful_reporting_enabled=parent.config.faithful_reporting_enabled,
            verifier_gate_enabled=parent.config.verifier_gate_enabled,
            verifier_gate_file_threshold=parent.config.verifier_gate_file_threshold,
            memory_enabled=parent.config.memory_enabled,
            memory_mode=parent.config.memory_mode,
            memory_token_budget_pct=parent.config.memory_token_budget_pct,
            memory_token_budget_max=parent.config.memory_token_budget_max,
            memory_block_token_max=parent.config.memory_block_token_max,
            memory_retrieval_timeout_ms=parent.config.memory_retrieval_timeout_ms,
            memory_project_store_enabled=parent.config.memory_project_store_enabled,
            memory_user_profile_enabled=parent.config.memory_user_profile_enabled,
            memory_auto_write_enabled=parent.config.memory_auto_write_enabled,
            memory_llm_rerank_enabled=parent.config.memory_llm_rerank_enabled,
            memory_debug_log_content=parent.config.memory_debug_log_content,
            agent_type_search_paths=parent.config.agent_type_search_paths,
            agent_type_registry_enabled=parent.config.agent_type_registry_enabled,
            task_store_enabled=parent.config.task_store_enabled,
            task_store_path=parent.config.task_store_path,
            task_store_stale_seconds=parent.config.task_store_stale_seconds,
        )

        child = AgentEngine(
            provider=provider,
            tool_registry=tool_registry,
            middleware_pipeline=parent.services.middleware_pipeline if self.inherit_middlewares else None,
            config=child_config,
            interrupt_controller=parent.services.interrupt_controller,
            logger=parent.services.logger,
            delegate_depth=child_depth,
            subagent_id=f"subagent-{uuid4().hex[:8]}",
            parent_subagent_id=parent.subagent_id,
            subagent_manager=parent.subagent_manager,
            session_store=parent._session_store,
            hooks=parent._hooks,
            memory_provider=parent.services.memory_provider,
            skill_catalog=parent._skill_catalog,
            agent_type_registry=parent._agent_type_registry,
            task_store=parent._task_store,
            diagnostic_tracker=getattr(parent, "_diagnostic_tracker", None),
        )
        parent_signal = None
        if getattr(task.request, "interrupt_signal", None) is not None:
            parent_signal = task.request.interrupt_signal
        else:
            parent_session_id = getattr(parent, "_current_session_id", None)
            if parent_session_id is not None:
                parent_signal = parent.services.interrupt_controller.signal_for(parent_session_id)
        if parent_signal is not None:
            async_mode = bool(task.metadata.get("run_in_background", False))
            task.request.interrupt_signal = InterruptSignal() if async_mode else InterruptSignal(parent=parent_signal)
        return child


def _filter_tool_registry(
    registry: ToolRegistry,
    *,
    allowed_tools: tuple[str, ...] | None,
    disallowed_tools: tuple[str, ...],
) -> ToolRegistry:
    available = set(registry.list_names())
    if allowed_tools is None:
        # No whitelist: keep everything except explicit denies.
        allowed = set(available)
    else:
        # Whitelist set: union with the meta-tool default-allow set so a
        # forgotten ``task_stop`` / ``task_output`` / ``send_message`` /
        # ``skill`` doesn't silently strand the child.  Only inject meta
        # tools that actually exist in the source registry.
        allowed = set(allowed_tools) | (META_TOOLS_DEFAULT_ALLOWED & available)
    denied = set(disallowed_tools)
    filtered = ToolRegistry()
    for name in registry.list_names():
        if name not in allowed or name in denied:
            continue
        filtered.register(registry.get(name))
    return filtered
