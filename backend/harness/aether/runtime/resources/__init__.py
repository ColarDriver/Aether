"""Runtime-managed external resources and local service clients."""

from .browser_manager import BrowserManager, BrowserUnavailable
from .lsp_client import LSPClient, LSPProtocolError
from .lsp_manager import LSPManager
from .lsp_servers import language_for, resolve_server_for
from .web_safety import is_url_safe

__all__ = [
    "BrowserManager",
    "BrowserUnavailable",
    "LSPClient",
    "LSPProtocolError",
    "LSPManager",
    "language_for",
    "resolve_server_for",
    "is_url_safe",
]
