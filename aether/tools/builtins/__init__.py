"""Built-in tool kit shipped with the Aether harness.

This package exposes the executors any ``AgentEngine`` can use out of
the box. The bundled registry includes:

* Filesystem and shell tools: ``shell``, ``read_file``, ``write_file``,
  ``list_dir``, ``grep``, ``glob``, ``file_edit``, ``notebook_edit``,
  ``todo_write``.
* Web access tools: ``web_fetch``, ``web_search``.
* Subagent dispatch tools: ``task``, ``task_output``, ``task_stop``.
* Interaction tools: ``enter_plan_mode``, ``exit_plan_mode``,
  ``ask_user_question``.
* Skill loading: ``skill``.
* Semantic code navigation: ``lsp``.
* Headless browser automation: ``web_browser``.
* Memory tools.

Each new tool family is gated by an ``EngineConfig`` switch
(``web_fetch_enabled`` / ``allow_subagent_dispatch`` /
``plan_mode_enabled`` / ``ask_user_question_enabled`` /
``skill_tool_enabled`` / ``lsp_tool_enabled`` /
``web_browser_enabled``); flipping any of these to ``False`` makes
the corresponding tool refuse with a structured error rather than
disappearing from the registry, so the model gets a clear message and
the surrounding observability/telemetry stays consistent.

The :func:`build_default_tool_registry` factory is the canonical entry
point used by :class:`aether.agents.core.agent.AgentEngine` when no
explicit ``tool_registry`` is passed and ``EngineConfig.use_builtin_tools``
is ``True``.  Callers that want a different mix can either pass their
own :class:`~aether.tools.registry.ToolRegistry` to ``AgentEngine`` or
flip ``EngineConfig.use_builtin_tools`` off and register tools
manually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether.tools.builtins.agent_tool import AgentTool
from aether.tools.builtins.ask_user_question import AskUserQuestionTool
from aether.tools.builtins.enter_plan_mode import EnterPlanModeTool
from aether.tools.builtins.exit_plan_mode import ExitPlanModeTool
from aether.tools.builtins.file_edit import FileEditTool
from aether.tools.builtins.glob import GlobTool
from aether.tools.builtins.grep import GrepTool
from aether.tools.builtins.list_dir import ListDirTool
from aether.tools.builtins.lsp import LSPTool
from aether.tools.builtins.memory import (
    MemoryForgetTool,
    MemoryListTool,
    MemoryReadTool,
    MemoryUpdateTool,
    MemoryWriteTool,
)
from aether.tools.builtins.notebook_edit import NotebookEditTool
from aether.tools.builtins.read_file import ReadFileTool
from aether.tools.builtins.send_message import SendMessageTool
from aether.tools.builtins.shell import ShellTool
from aether.tools.builtins.skill import SkillTool
from aether.tools.builtins.task_output import TaskOutputTool
from aether.tools.builtins.task_stop import TaskStopTool
from aether.tools.builtins.todo_write import TodoWriteTool
from aether.tools.builtins.web_browser import WebBrowserTool
from aether.tools.builtins.web_fetch import WebFetchTool
from aether.tools.builtins.web_search import WebSearchTool
from aether.tools.builtins.write_file import WriteFileTool
from aether.tools.registry import ToolRegistry


def build_default_tool_registry(
    *,
    cwd: Path | None = None,
    skill_catalog: Any | None = None,
    agent_type_registry: Any | None = None,
    approval_prompter: Any | None = None,
    lsp_manager: Any | None = None,
    browser_manager: Any | None = None,
) -> ToolRegistry:
    """Return a :class:`ToolRegistry` populated with the bundled tool kit.

    Parameters
    ----------
    cwd:
        Default working directory passed into the file-aware tools.
        When ``None`` the tools resolve relative paths against
        :func:`pathlib.Path.cwd` at call time.
    skill_catalog:
        Optional :class:`~aether.runtime.tools.skill_catalog.SkillCatalog`
        injected into :class:`SkillTool`.  When ``None`` the tool
        falls back to ``context.metadata['_skill_catalog']`` (set by
        the engine) or builds a lazy default from
        ``EngineConfig.skill_search_paths``.
    agent_type_registry:
        Optional registry injected into :class:`AgentTool` so the
        ``subagent_type`` schema can expose an enum and runtime
        dispatch can validate requested types.
    approval_prompter:
        Optional CLI prompter pre-bound onto ``ExitPlanModeTool`` /
        ``AskUserQuestionTool``.  In production the engine forwards
        ``EngineRequest.approval_prompter`` via metadata, so passing
        ``None`` here is the common case.
    lsp_manager:
        Optional :class:`~aether.runtime.resources.lsp_manager.LSPManager`
        pre-bound onto :class:`LSPTool`.  When omitted the tool builds
        a lazy default the first time it runs.
    browser_manager:
        Optional :class:`~aether.runtime.resources.browser_manager.BrowserManager`
        pre-bound onto :class:`WebBrowserTool`.  When omitted the tool
        builds one on demand using ``EngineConfig`` settings.
    """
    registry = ToolRegistry()
    registry.register(ShellTool(default_cwd=cwd))
    registry.register(ReadFileTool(default_cwd=cwd))
    registry.register(WriteFileTool(default_cwd=cwd))
    registry.register(ListDirTool(default_cwd=cwd))
    registry.register(GrepTool(default_cwd=cwd))
    registry.register(GlobTool(default_cwd=cwd))
    # Filesystem and shell tools.
    registry.register(FileEditTool(default_cwd=cwd))
    registry.register(NotebookEditTool(default_cwd=cwd))
    registry.register(TodoWriteTool())
    # Web, subagent, interaction, and skill tools.
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())
    registry.register(AgentTool(agent_type_registry=agent_type_registry))
    registry.register(TaskOutputTool())
    registry.register(TaskStopTool())
    registry.register(SendMessageTool())
    registry.register(EnterPlanModeTool())
    registry.register(ExitPlanModeTool(prompter=approval_prompter))
    registry.register(AskUserQuestionTool(prompter=approval_prompter))
    registry.register(SkillTool(catalog=skill_catalog))
    # Semantic navigation and browser tools.
    registry.register(LSPTool(manager=lsp_manager))
    registry.register(WebBrowserTool(manager=browser_manager))
    # Memory tools.
    registry.register(MemoryReadTool())
    registry.register(MemoryListTool())
    registry.register(MemoryWriteTool())
    registry.register(MemoryUpdateTool())
    registry.register(MemoryForgetTool())
    return registry


__all__ = [
    "ShellTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "GrepTool",
    "GlobTool",
    "FileEditTool",
    "NotebookEditTool",
    "TodoWriteTool",
    "WebFetchTool",
    "WebSearchTool",
    "AgentTool",
    "TaskOutputTool",
    "TaskStopTool",
    "SendMessageTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "AskUserQuestionTool",
    "SkillTool",
    "LSPTool",
    "MemoryForgetTool",
    "MemoryListTool",
    "MemoryReadTool",
    "MemoryUpdateTool",
    "MemoryWriteTool",
    "WebBrowserTool",
    "build_default_tool_registry",
]
