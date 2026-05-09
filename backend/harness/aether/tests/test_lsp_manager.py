"""Tests for ``aether.runtime.lsp_manager.LSPManager``.

Sprint 3.5 / PR-3 (PR 3.5.9).

We don't depend on a real LSP binary being installed.  Instead we
patch :class:`LSPClient` with a stub class that always succeeds (or
fails on demand) so we can exercise the manager's caching + failure
memory in isolation.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from aether.runtime import lsp_manager as lm
from aether.runtime.lsp_manager import LSPManager


class _FakeClient:
    """Just enough of :class:`LSPClient` for the manager to use it."""

    def __init__(
        self,
        *,
        command: list[str],
        project_root: Path,
        language: str,
        init_timeout: float = 5.0,
        env: Optional[dict] = None,
        start_raises: Optional[Exception] = None,
        running: bool = True,
    ) -> None:
        self.command = command
        self.project_root = project_root
        self.language = language
        self.init_timeout = init_timeout
        self._running = running
        self.start_raises = start_raises
        self.start_called = 0
        self.shutdown_called = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.start_called += 1
        if self.start_raises is not None:
            raise self.start_raises

    def shutdown(self, *, grace: bool = True, timeout: float = 2.0) -> None:
        self.shutdown_called += 1
        self._running = False


class ManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lsp-mgr-"))
        # Ensure the language's first candidate command resolves in
        # ``shutil.which``; we patch it instead of relying on the host.
        self._which_patch = mock.patch.object(lm.shutil, "which", lambda _name: "/usr/bin/" + _name)
        self._which_patch.start()
        self._fake_clients: list[_FakeClient] = []

        def _factory(*args, **kwargs):
            client = _FakeClient(*args, **kwargs)
            self._fake_clients.append(client)
            return client

        self._client_patch = mock.patch.object(lm, "LSPClient", _factory)
        self._client_patch.start()

    def tearDown(self) -> None:
        self._which_patch.stop()
        self._client_patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a1_returns_none_for_unknown_language(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        self.assertIsNone(mgr.get_client_for(self.tmp / "x.unknown"))

    def test_a2_caches_per_language(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        c1 = mgr.get_client_for(self.tmp / "a.py")
        c2 = mgr.get_client_for(self.tmp / "b.py")
        self.assertIs(c1, c2)
        self.assertEqual(len(self._fake_clients), 1)

    def test_a3_separate_languages_get_separate_clients(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        py = mgr.get_client_for(self.tmp / "a.py")
        ts = mgr.get_client_for(self.tmp / "a.ts")
        self.assertIsNot(py, ts)
        self.assertEqual(len(self._fake_clients), 2)

    def test_a4_failed_language_is_remembered(self) -> None:
        # Make every fake client raise during ``start``.
        with mock.patch.object(
            lm,
            "LSPClient",
            lambda **kw: _FakeClient(start_raises=RuntimeError("nope"), **kw),
        ):
            mgr = LSPManager(project_root=self.tmp)
            self.assertIsNone(mgr.get_client_for(self.tmp / "a.py"))
            # Second call should *not* attempt to start again.
            self.assertIsNone(mgr.get_client_for(self.tmp / "b.py"))
            self.assertIn("python", mgr.known_failed_languages())

    def test_a5_skips_command_when_binary_not_on_path(self) -> None:
        with mock.patch.object(lm.shutil, "which", lambda _n: None):
            mgr = LSPManager(project_root=self.tmp)
            self.assertIsNone(mgr.get_client_for(self.tmp / "a.py"))
            self.assertIn("python", mgr.known_failed_languages())

    def test_a6_uses_overrides_table(self) -> None:
        overrides = {"python": [["my-lsp"]]}
        mgr = LSPManager(project_root=self.tmp, overrides=overrides)
        client = mgr.get_client_for(self.tmp / "a.py")
        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(client.command, ["my-lsp"])

    def test_a7_known_running_languages_reflects_state(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        mgr.get_client_for(self.tmp / "a.py")
        mgr.get_client_for(self.tmp / "a.ts")
        self.assertEqual(set(mgr.known_running_languages()), {"python", "typescript"})

    def test_a8_dead_client_is_replaced_on_next_call(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        first = mgr.get_client_for(self.tmp / "a.py")
        assert first is not None
        first._running = False
        second = mgr.get_client_for(self.tmp / "b.py")
        self.assertIsNot(first, second)
        # Old client got shut down.
        self.assertEqual(first.shutdown_called, 1)

    def test_a9_any_initialized_client_returns_running_one(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        self.assertIsNone(mgr.any_initialized_client())
        client = mgr.get_client_for(self.tmp / "a.py")
        assert client is not None
        self.assertIs(mgr.any_initialized_client(), client)

    def test_a10_shutdown_all_closes_every_client(self) -> None:
        mgr = LSPManager(project_root=self.tmp)
        mgr.get_client_for(self.tmp / "a.py")
        mgr.get_client_for(self.tmp / "a.ts")
        mgr.shutdown_all()
        for client in self._fake_clients:
            self.assertEqual(client.shutdown_called, 1)
        self.assertEqual(mgr.known_running_languages(), ())


if __name__ == "__main__":
    unittest.main()
