"""Per-language singleton manager for LSP clients.

Lazily constructs an :class:`LSPClient` the first time a tool asks for
one for a given language.  Subsequent calls return the cached client so
we pay the (sometimes substantial) initialisation cost — rust-analyzer
indexing a large repo can take 10+ seconds — exactly once per session.

Failure handling
----------------
* If the launch command's binary is not on ``PATH`` we mark the
  language as failed and refuse to retry for the rest of the process.
* If ``LSPClient.start`` raises (network errors, init handshake
  failure) we tear down the half-built client and add the language to
  ``_failed`` so we don't burn another startup attempt next call.
* Tools surface a clear "no LSP server available" message to the
  model when ``get_client_for`` returns ``None``; the model is free to
  fall back to grep / read_file.

Lifecycle
---------
The CLI registers ``LSPManager.shutdown_all`` with ``atexit`` so a
process-level Ctrl-C never leaves orphan server processes behind.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Mapping, Optional

from aether.runtime.diagnostics.types import Diagnostic
from aether.runtime.resources.lsp_client import LSPClient
from aether.runtime.resources.lsp_servers import language_for, resolve_server_for

logger = logging.getLogger(__name__)


__all__ = ["LSPManager"]


class LSPManager:
    def __init__(
        self,
        *,
        project_root: Path,
        init_timeout: float = 10.0,
        overrides: Optional[Mapping[str, list[list[str]]]] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.init_timeout = float(init_timeout)
        self.overrides = dict(overrides) if overrides else None
        self._clients: dict[str, LSPClient] = {}
        self._failed: set[str] = set()
        self._lock = threading.Lock()

    # ----------------------------------------------------------- public API

    def get_client_for(self, file_path: Path) -> Optional[LSPClient]:
        lang = language_for(file_path)
        if lang is None:
            return None
        with self._lock:
            if lang in self._failed:
                return None
            cached = self._clients.get(lang)
            if cached is not None and cached.is_running:
                return cached
            if cached is not None:
                # Server died — wipe and try a fresh launch below.
                logger.info("lsp[%s] cached client is dead; restarting", lang)
                try:
                    cached.shutdown(grace=False)
                except Exception:
                    pass
                self._clients.pop(lang, None)
            commands = resolve_server_for(file_path, overrides=self.overrides)
            if not commands:
                self._failed.add(lang)
                return None
            for cmd in commands:
                if not cmd:
                    continue
                if shutil.which(cmd[0]) is None:
                    continue
                client = LSPClient(
                    command=cmd,
                    project_root=self.project_root,
                    language=lang,
                    init_timeout=self.init_timeout,
                )
                try:
                    client.start()
                except FileNotFoundError as exc:
                    logger.warning("lsp[%s] FileNotFound for %s: %s", lang, cmd, exc)
                    continue
                except Exception as exc:
                    logger.warning("lsp[%s] failed to start %s: %s", lang, cmd, exc)
                    try:
                        client.shutdown(grace=False)
                    except Exception:
                        pass
                    continue
                self._clients[lang] = client
                return client
            self._failed.add(lang)
            return None

    def any_initialized_client(self) -> Optional[LSPClient]:
        """Return any started client (used by ``workspaceSymbol`` which
        is not file-bound).  Picks the most recently-started running
        client; returns ``None`` when no client has been started."""
        with self._lock:
            for client in self._clients.values():
                if client.is_running:
                    return client
        return None

    def known_failed_languages(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._failed))

    def known_running_languages(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(lang for lang, c in self._clients.items() if c.is_running))

    # ------------------------------------------------- high-level helpers
    #
    # These wrap the ``LSPClient`` document-sync primitives in a
    # "no LSP available → silent no-op" envelope so callers (the
    # :class:`DiagnosticTracker` and edit-tool hooks can invoke them
    # unconditionally.

    def change_file(self, path: Path, content: str) -> None:
        """Notify the appropriate LSP server that *path* has new content.

        Sends ``textDocument/didOpen`` the first time, ``didChange`` on
        subsequent calls.  Silently returns when no language server is
        configured for this file's language or when the server cannot
        be launched — the model still gets the edited file via the
        tool result; we just won't be able to surface diagnostics.
        """
        client = self.get_client_for(path)
        if client is None:
            return
        try:
            if client.is_open(path):
                client.did_change(path, content)
            else:
                client.did_open(path, content)
        except Exception:
            logger.warning(
                "lsp[%s] change_file failed for %s",
                getattr(client, "language", "?"),
                path,
                exc_info=True,
            )

    def save_file(self, path: Path, *, content: str | None = None) -> None:
        """Notify the LSP server that *path* has been saved on disk.

        Many servers (pyright, tsserver) only run a full re-lint pass on
        ``didSave``, not ``didChange`` — so this notification is what
        actually surfaces newly-introduced diagnostics most of the time.
        """
        client = self.get_client_for(path)
        if client is None:
            return
        try:
            client.did_save(path, content=content)
        except Exception:
            logger.warning(
                "lsp[%s] save_file failed for %s",
                getattr(client, "language", "?"),
                path,
                exc_info=True,
            )

    def pull_diagnostics(
        self,
        path: Path,
        *,
        deadline: float,
    ) -> list[Diagnostic]:
        """Block (with deadline) for the next ``publishDiagnostics`` push.

        *deadline* is a ``time.perf_counter()`` value; this method
        returns no later than that moment.  Returns the freshest known
        diagnostic snapshot — possibly empty when no LSP server exists,
        when the push hasn't happened yet, or when the file is clean.
        """
        client = self.get_client_for(path)
        if client is None:
            return []
        try:
            return client.wait_for_diagnostics(path, deadline=deadline)
        except Exception:
            logger.warning(
                "lsp[%s] pull_diagnostics failed for %s",
                getattr(client, "language", "?"),
                path,
                exc_info=True,
            )
            return []

    def shutdown_all(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.shutdown(grace=True)
            except Exception:
                pass
