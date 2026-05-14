"""Built-in ``skill`` tool.

Loads a SKILL.md file from the configured catalog and returns its
body to the model so the calling turn can follow the playbook
described by the skill.

Catalog resolution order (first non-``None`` wins):

1. Constructor injection (``SkillTool(catalog=...)``).
2. ``context.metadata['_skill_catalog']`` (set by
   ``AgentEngine._prepare_turn_entry`` from
   ``AgentEngine._skill_catalog``).
3. Lazy default — build a :class:`SkillCatalog` from
   ``EngineConfig.skill_search_paths`` (or sane defaults if empty).

Placeholder substitution: skill bodies may reference
``$ARGUMENTS`` / ``${ARGUMENTS}`` (replaced with the model-supplied
``args`` string) and ``${AETHER_SESSION_ID}`` (replaced with
``context.session_id``).  This mirrors the ``$ARGUMENTS`` convention
used by Anthropic's bundled skills and by the ``open-claude-code``
skill loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.skill_catalog import SkillCatalog
from aether.tools.base import ToolDescriptor, ToolExecutor, maybe_spill_for_tool


class SkillTool(ToolExecutor):
    """Load a SKILL.md and return its rendered body."""

    NAME = "skill"
    MAX_RESULT_CHARS = 60_000
    _MAX_LIST_PREVIEW = 30

    def __init__(self, catalog: SkillCatalog | None = None) -> None:
        self._catalog = catalog
        self._descriptor = ToolDescriptor(
            name=self.NAME,
            description=(
                "Load a SKILL.md from the configured skill catalog and "
                "return its body so you can follow the playbook it "
                "describes. Skill names are derived from their directory "
                "path (e.g. 'software-development-test-driven-development'). "
                "Optional `args` is substituted into $ARGUMENTS / "
                "${ARGUMENTS} placeholders inside the skill body."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Skill name (case-insensitive; leading '/' tolerated).",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments substituted into the skill body.",
                    },
                },
                "required": ["skill"],
            },
            required=["skill"],
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        args_dict = call.arguments or {}
        skill_name = args_dict.get("skill")
        if not isinstance(skill_name, str) or not skill_name.strip():
            return _error(call, "'skill' is required and must be a non-empty string")
        skill_name = skill_name.strip()
        substitution = args_dict.get("args", "")
        if substitution is None:
            substitution = ""
        if not isinstance(substitution, str):
            return _error(call, "'args' must be a string when provided")

        config = context.metadata.get("_engine_config") if context.metadata else None
        if not bool(getattr(config, "skill_tool_enabled", True)):
            return _error(call, "skill tool is disabled by configuration")

        catalog = self._resolve_catalog(context, config)
        if catalog is None:
            return _error(
                call,
                "skill tool unavailable: no catalog configured "
                "(set EngineConfig.skill_search_paths or pass "
                "skill_catalog=... to AgentEngine).",
            )

        skill = catalog.get(skill_name)
        if skill is None:
            available = ", ".join(
                s.name for s in catalog.list_all()[: self._MAX_LIST_PREVIEW]
            )
            suffix = (
                f"\nAvailable (first {self._MAX_LIST_PREVIEW}): {available}"
                if available
                else "\nThe catalog is empty."
            )
            return _error(call, f"unknown skill: {skill_name!r}{suffix}")

        body = skill.body or ""
        if substitution:
            body = body.replace("$ARGUMENTS", substitution).replace(
                "${ARGUMENTS}", substitution
            )
        body = body.replace("${AETHER_SESSION_ID}", context.session_id or "")

        full_output = self._render(skill, body, args=substitution)
        content = maybe_spill_for_tool(
            full_output,
            call=call,
            context=context,
            max_chars=self.MAX_RESULT_CHARS,
            extension="md",
            full_lines=full_output.count("\n") + 1,
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            is_error=False,
            metadata={
                "skill_name": skill.name,
                "skill_path": str(skill.path),
                "skill_version": skill.version,
            },
        )

    # ----------------------------------------------------------- helpers

    def _resolve_catalog(
        self, context: TurnContext, config: Any
    ) -> Optional[SkillCatalog]:
        if self._catalog is not None:
            return self._catalog
        injected = context.metadata.get("_skill_catalog") if context.metadata else None
        if isinstance(injected, SkillCatalog):
            return injected
        # Lazy default: build from configured search paths or fall back
        # to the project-root ``skills/`` directory.  Cache on ``self``
        # so repeated invocations within the same engine reuse the
        # catalog without re-walking the filesystem.
        search_paths: list[Path] = []
        configured = tuple(getattr(config, "skill_search_paths", ()) or ())
        if configured:
            search_paths.extend(Path(p) for p in configured)
        else:
            cwd = Path.cwd()
            for candidate in (cwd / "skills", Path.home() / ".aether" / "skills"):
                search_paths.append(candidate)
        if not search_paths:
            return None
        catalog = SkillCatalog(search_paths=search_paths)
        catalog.discover()
        self._catalog = catalog
        return catalog

    @staticmethod
    def _render(skill: Any, body: str, *, args: str) -> str:
        header = [f"# Loaded skill: {skill.name}", f"Source: {skill.path}"]
        if skill.description:
            header.append(f"Description: {skill.description}")
        if skill.when_to_use:
            header.append(f"When to use: {skill.when_to_use}")
        if skill.version:
            header.append(f"Version: {skill.version}")
        if args:
            header.append(f"Arguments: {args}")
        header.append("")
        header.append("--- BEGIN SKILL ---")
        return "\n".join(header) + "\n" + body + "\n--- END SKILL ---\n"


def _error(
    call: ToolCall,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=message,
        is_error=True,
        metadata=metadata or {},
    )


__all__ = ["SkillTool"]
