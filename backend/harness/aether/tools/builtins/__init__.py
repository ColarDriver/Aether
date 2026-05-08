"""Built-in tool kit shipped with the Aether harness.

This package exposes a small set of executors that any
``AgentEngine`` can use out of the box: ``shell``, ``read_file``,
``write_file``, ``list_dir``, ``grep``, ``glob``.

Why these six?  Together they cover claude-code's day-one tool
surface (``Bash`` / ``Read`` / ``Write`` / ``LS`` / ``Grep`` /
``Glob``) — enough to read, search, edit, and execute commands in a
workspace.  More specialised tools (web fetch, image read, MCP
bridges) belong in their own packages so callers can opt in.

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

from aether.tools.builtins.glob import GlobTool
from aether.tools.builtins.grep import GrepTool
from aether.tools.builtins.list_dir import ListDirTool
from aether.tools.builtins.read_file import ReadFileTool
from aether.tools.builtins.shell import ShellTool
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
        ``grep`` / ``glob``).  When ``None`` the tools resolve relative
        paths against :func:`pathlib.Path.cwd` at call time.
    """
    registry = ToolRegistry()
    registry.register(ShellTool(default_cwd=cwd))
    registry.register(ReadFileTool(default_cwd=cwd))
    registry.register(WriteFileTool(default_cwd=cwd))
    registry.register(ListDirTool(default_cwd=cwd))
    registry.register(GrepTool(default_cwd=cwd))
    registry.register(GlobTool(default_cwd=cwd))
    return registry


__all__ = [
    "ShellTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "GrepTool",
    "GlobTool",
    "build_default_tool_registry",
]
