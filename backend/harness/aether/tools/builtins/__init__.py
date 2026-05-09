"""Built-in tool kit shipped with the Aether harness.

This package exposes the executors any ``AgentEngine`` can use out of
the box.  Sprint 3.5 / PR 3.5.1 expanded the kit from six tools to
nine by adding ``file_edit``, ``todo_write`` and ``notebook_edit``,
and upgraded the original six (``shell`` / ``read_file`` / ``grep`` /
``glob`` / ``list_dir``) to spill large outputs to disk via
:mod:`aether.runtime.tool_result_storage`.

Why these nine?  They cover the claude-code default surface that
applies to every project (Bash / Read / Write / LS / Grep / Glob /
Edit / NotebookEdit / TodoWrite) \u2014 enough to read, search, edit and
maintain a task checklist in any workspace.  More specialised tools
(web fetch, subagent dispatch, plan mode, skill, LSP, browser) come
in subsequent PRs of Sprint 3.5.

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

from aether.tools.builtins.file_edit import FileEditTool
from aether.tools.builtins.glob import GlobTool
from aether.tools.builtins.grep import GrepTool
from aether.tools.builtins.list_dir import ListDirTool
from aether.tools.builtins.notebook_edit import NotebookEditTool
from aether.tools.builtins.read_file import ReadFileTool
from aether.tools.builtins.shell import ShellTool
from aether.tools.builtins.todo_write import TodoWriteTool
from aether.tools.builtins.write_file import WriteFileTool
from aether.tools.registry import ToolRegistry


def build_default_tool_registry(
    *,
    cwd: Path | None = None,
) -> ToolRegistry:
    """Return a :class:`ToolRegistry` populated with the bundled tool kit.

    Parameters
    ----------
    cwd:
        Default working directory passed into the file-aware tools
        (``shell`` / ``read_file`` / ``write_file`` / ``list_dir`` /
        ``grep`` / ``glob`` / ``file_edit`` / ``notebook_edit``).
        When ``None`` the tools resolve relative paths against
        :func:`pathlib.Path.cwd` at call time.
    """
    registry = ToolRegistry()
    registry.register(ShellTool(default_cwd=cwd))
    registry.register(ReadFileTool(default_cwd=cwd))
    registry.register(WriteFileTool(default_cwd=cwd))
    registry.register(ListDirTool(default_cwd=cwd))
    registry.register(GrepTool(default_cwd=cwd))
    registry.register(GlobTool(default_cwd=cwd))
    # Sprint 3.5 / PR 3.5.1 \u2014 new tools.  Order is alphabetical-ish
    # within the new block; doesn't matter functionally but makes
    # ``list_descriptors()`` output readable.
    registry.register(FileEditTool(default_cwd=cwd))
    registry.register(NotebookEditTool(default_cwd=cwd))
    registry.register(TodoWriteTool())
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
    "build_default_tool_registry",
]
