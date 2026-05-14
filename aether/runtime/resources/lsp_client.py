"""Minimal LSP client over stdio.

A single :class:`LSPClient` instance owns one ``subprocess.Popen``
running a Language Server Protocol (LSP) server.  It speaks the
JSON-RPC 2.0 envelope LSP defines:

    Content-Length: <N>\\r\\n
    \\r\\n
    {"jsonrpc": "2.0", "id": ..., "method": ..., "params": ...}

We deliberately do **not** pull in ``pylsp-jsonrpc`` or
``python-lsp-jsonrpc`` — both add weight (asyncio loops, server-side
helpers we don't need) and using a thin homemade transport is what
``open-claude-code``'s LSP plumbing does.  The full surface we need is
``request`` (await response), ``notify`` (fire-and-forget) and
``shutdown`` (graceful exit), totalling ~250 lines.

Threading model
---------------
The reader runs on a background daemon thread.  Both ``request`` and
``notify`` are safe to call from arbitrary threads — they hold a
``Lock`` while writing the envelope to ``stdin``.  Responses are
demultiplexed by the integer JSON-RPC ``id`` field; pending requests
park on a per-id ``threading.Event``.

Error model
-----------
* Any I/O failure during ``start`` is caught by :class:`LSPManager`
  and the language is added to the ``_failed`` set so we never retry.
* Per-call timeouts raise :class:`TimeoutError`; the surrounding tool
  surfaces this to the model as a structured ``ToolResult``.
* A server crash mid-request is surfaced as :class:`LSPProcessExited`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


__all__ = ["LSPClient", "LSPProcessExited", "LSPProtocolError"]


_HEADER_TERMINATOR = b"\r\n\r\n"


class LSPProcessExited(RuntimeError):
    """Raised when the underlying server process has exited unexpectedly."""


class LSPProtocolError(RuntimeError):
    """Raised when the server sends a malformed JSON-RPC envelope."""


@dataclass
class _PendingRequest:
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[dict[str, Any]] = None


class LSPClient:
    """Synchronous JSON-RPC LSP client backed by ``subprocess.Popen``."""

    DEFAULT_INIT_TIMEOUT = 10.0
    DEFAULT_REQUEST_TIMEOUT = 15.0

    def __init__(
        self,
        *,
        command: list[str],
        project_root: Path,
        language: str,
        env: Optional[dict[str, str]] = None,
        init_timeout: float = DEFAULT_INIT_TIMEOUT,
    ) -> None:
        if not command:
            raise ValueError("command must be non-empty")
        self.command = list(command)
        self.project_root = Path(project_root).resolve()
        self.language = language
        self.env = env or os.environ.copy()
        self.init_timeout = float(init_timeout)

        self._process: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, _PendingRequest] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._closed = False
        self._initialized = False
        self.server_capabilities: dict[str, Any] = {}

    # ------------------------------------------------------------- lifecycle

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and not self._closed
        )

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.project_root),
            env=self.env,
            bufsize=0,
        )
        # Reader threads must be daemons so an unclean Aether shutdown
        # never blocks on a hung server.
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"lsp-reader-{self.language}",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name=f"lsp-stderr-{self.language}",
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            self._initialize()
        except Exception:
            self.shutdown(grace=False)
            raise

    def _initialize(self) -> None:
        params = {
            "processId": os.getpid(),
            "clientInfo": {"name": "Aether", "version": "0.1"},
            "rootUri": self.project_root.as_uri(),
            "workspaceFolders": [
                {"uri": self.project_root.as_uri(), "name": self.project_root.name}
            ],
            "capabilities": _CLIENT_CAPABILITIES,
        }
        result = self.request("initialize", params, timeout=self.init_timeout)
        self.server_capabilities = result.get("capabilities", {}) if isinstance(result, dict) else {}
        self.notify("initialized", {})
        self._initialized = True

    def shutdown(self, *, grace: bool = True, timeout: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        try:
            if grace and self._initialized:
                try:
                    self.request("shutdown", None, timeout=timeout)
                except Exception:
                    pass
                try:
                    self.notify("exit", None)
                except Exception:
                    pass
        finally:
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
            except Exception:
                pass
            try:
                self._process.wait(timeout=timeout)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            # Close the read pipes so the OS file descriptors and the
            # Python file objects are released — otherwise they leak
            # and surface as ``ResourceWarning`` during test teardown.
            for stream in (self._process.stdout, self._process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except Exception:
                        pass

    # ----------------------------------------------------------- public API

    def request(
        self,
        method: str,
        params: Any,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._process is None:
            raise LSPProcessExited("LSP client has not been started")
        if self._process.poll() is not None:
            raise LSPProcessExited(f"LSP server exited (rc={self._process.returncode})")
        timeout = float(timeout) if timeout is not None else self.DEFAULT_REQUEST_TIMEOUT

        with self._id_lock:
            request_id = self._next_id
            self._next_id += 1

        pending = _PendingRequest()
        with self._pending_lock:
            self._pending[request_id] = pending

        envelope = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            envelope["params"] = params
        try:
            self._send(envelope)
            if not pending.event.wait(timeout=timeout):
                raise TimeoutError(f"LSP {method} timed out after {timeout:.1f}s")
            response = pending.response or {}
            if "error" in response:
                err = response["error"]
                raise LSPProtocolError(
                    f"LSP {method} error code={err.get('code')} message={err.get('message')}"
                )
            return response.get("result")
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def notify(self, method: str, params: Any) -> None:
        if self._process is None:
            raise LSPProcessExited("LSP client has not been started")
        envelope = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            envelope["params"] = params
        self._send(envelope)

    # ------------------------------------------------------- transport guts

    def _send(self, envelope: dict[str, Any]) -> None:
        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            assert self._process is not None and self._process.stdin is not None
            try:
                self._process.stdin.write(header + body)
                self._process.stdin.flush()
            except BrokenPipeError as exc:
                raise LSPProcessExited(f"LSP stdin closed: {exc}") from exc

    def _reader_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        buffer = bytearray()
        while not self._closed:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                term = buffer.find(_HEADER_TERMINATOR)
                if term == -1:
                    break
                header_bytes = bytes(buffer[:term])
                content_length = self._parse_content_length(header_bytes)
                total = term + len(_HEADER_TERMINATOR) + (content_length or 0)
                if content_length is None or len(buffer) < total:
                    break
                body = bytes(buffer[term + len(_HEADER_TERMINATOR) : total])
                del buffer[:total]
                self._dispatch_message(body)
        # Stream closed; resolve every pending request with an error.
        with self._pending_lock:
            for pending in self._pending.values():
                pending.response = {"error": {"code": -1, "message": "server stream closed"}}
                pending.event.set()
            self._pending.clear()

    def _stderr_loop(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        stream = self._process.stderr
        try:
            for line in iter(stream.readline, b""):
                if not line:
                    break
                logger.debug(
                    "lsp[%s] stderr: %s", self.language, line.decode("utf-8", errors="replace").rstrip()
                )
        except Exception:
            pass

    @staticmethod
    def _parse_content_length(header_bytes: bytes) -> Optional[int]:
        try:
            text = header_bytes.decode("ascii")
        except UnicodeDecodeError:
            return None
        for raw in text.split("\r\n"):
            key, _, value = raw.partition(":")
            if key.strip().lower() == "content-length":
                try:
                    return int(value.strip())
                except ValueError:
                    return None
        return None

    def _dispatch_message(self, body: bytes) -> None:
        try:
            envelope = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("lsp[%s] dropped malformed message: %s", self.language, exc)
            return
        if not isinstance(envelope, dict):
            return
        msg_id = envelope.get("id")
        if msg_id is None:
            # server -> client notification or window/showMessage etc.;
            # we don't handle these in v1 (just log at debug).
            logger.debug("lsp[%s] notification: %s", self.language, envelope.get("method"))
            return
        with self._pending_lock:
            pending = self._pending.get(msg_id)
        if pending is None:
            logger.debug("lsp[%s] response for unknown id=%s", self.language, msg_id)
            return
        pending.response = envelope
        pending.event.set()


# Minimal client capabilities — enough for the 9 ops we expose.  We
# advertise empty objects for most groups so the server returns
# results in the simplest legal shape.
_CLIENT_CAPABILITIES: dict[str, Any] = {
    "workspace": {
        "applyEdit": False,
        "workspaceEdit": {"documentChanges": False},
        "didChangeConfiguration": {"dynamicRegistration": False},
        "symbol": {"dynamicRegistration": False},
    },
    "textDocument": {
        "synchronization": {
            "dynamicRegistration": False,
            "willSave": False,
            "didSave": False,
        },
        "definition": {"linkSupport": False},
        "references": {"dynamicRegistration": False},
        "hover": {
            "dynamicRegistration": False,
            "contentFormat": ["markdown", "plaintext"],
        },
        "documentSymbol": {
            "dynamicRegistration": False,
            "hierarchicalDocumentSymbolSupport": True,
        },
        "implementation": {"dynamicRegistration": False},
        "callHierarchy": {"dynamicRegistration": False},
    },
}
