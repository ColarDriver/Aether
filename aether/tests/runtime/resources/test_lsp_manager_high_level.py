"""High-level method tests for LSPManager.

These tests substitute a fake :class:`LSPClient` so they don't depend
on an actual ``pyright-langserver`` / ``tsserver`` binary being on
``PATH``.  Integration with a real LSP belongs in an opt-in marker
that's outside the default CI cycle.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from typing import Optional

from aether.runtime.diagnostics.types import Diagnostic
from aether.runtime.resources.lsp_manager import LSPManager


class _FakeClient:
    language = "python"

    def __init__(
        self,
        *,
        diagnostics: list[Diagnostic] | None = None,
        change_raises: Exception | None = None,
        save_raises: Exception | None = None,
        wait_raises: Exception | None = None,
    ) -> None:
        self._opened: set[str] = set()
        self.did_open_calls: list[tuple[Path, str]] = []
        self.did_change_calls: list[tuple[Path, str]] = []
        self.did_save_calls: list[tuple[Path, str | None]] = []
        self.wait_calls: list[Path] = []
        self._diagnostics = list(diagnostics or [])
        self._change_raises = change_raises
        self._save_raises = save_raises
        self._wait_raises = wait_raises

    @property
    def is_running(self) -> bool:
        return True

    def is_open(self, path: Path) -> bool:
        return str(Path(path).resolve()) in self._opened

    def did_open(self, path: Path, content: str, *, language_id: str | None = None) -> None:
        if self._change_raises is not None:
            raise self._change_raises
        self._opened.add(str(Path(path).resolve()))
        self.did_open_calls.append((Path(path), content))

    def did_change(self, path: Path, content: str) -> None:
        if self._change_raises is not None:
            raise self._change_raises
        self.did_change_calls.append((Path(path), content))

    def did_save(self, path: Path, *, content: str | None = None) -> None:
        if self._save_raises is not None:
            raise self._save_raises
        self.did_save_calls.append((Path(path), content))

    def wait_for_diagnostics(
        self,
        path: Path,
        *,
        deadline: float,
    ) -> list[Diagnostic]:
        if self._wait_raises is not None:
            raise self._wait_raises
        self.wait_calls.append(Path(path))
        return list(self._diagnostics)


class _StubManager(LSPManager):
    """LSPManager subclass that overrides ``get_client_for`` to return
    a pre-built fake.  Sidesteps the binary lookup + subprocess launch
    so these tests stay hermetic."""

    def __init__(self, client_factory) -> None:  # type: ignore[no-untyped-def]
        super().__init__(project_root=Path.cwd())
        self._client_factory = client_factory

    def get_client_for(self, file_path: Path) -> Optional[_FakeClient]:  # type: ignore[override]
        return self._client_factory(Path(file_path))


class ChangeFileTests(unittest.TestCase):
    def test_first_call_invokes_did_open(self) -> None:
        client = _FakeClient()
        manager = _StubManager(lambda _p: client)
        manager.change_file(Path("/tmp/x.py"), "print(1)")
        self.assertEqual(len(client.did_open_calls), 1)
        self.assertEqual(len(client.did_change_calls), 0)

    def test_second_call_invokes_did_change(self) -> None:
        client = _FakeClient()
        manager = _StubManager(lambda _p: client)
        manager.change_file(Path("/tmp/x.py"), "print(1)")
        manager.change_file(Path("/tmp/x.py"), "print(2)")
        self.assertEqual(len(client.did_open_calls), 1)
        self.assertEqual(len(client.did_change_calls), 1)
        _, content = client.did_change_calls[0]
        self.assertEqual(content, "print(2)")

    def test_no_client_is_silent_noop(self) -> None:
        manager = _StubManager(lambda _p: None)
        # Must not raise even when there's no LSP server for the
        # language (e.g. plain markdown).
        manager.change_file(Path("/tmp/x.md"), "hello")

    def test_client_exception_is_swallowed(self) -> None:
        client = _FakeClient(change_raises=RuntimeError("boom"))
        manager = _StubManager(lambda _p: client)
        # Must not propagate; LSP failures never break edit pipeline.
        manager.change_file(Path("/tmp/x.py"), "print(1)")


class SaveFileTests(unittest.TestCase):
    def test_sends_did_save(self) -> None:
        client = _FakeClient()
        manager = _StubManager(lambda _p: client)
        manager.save_file(Path("/tmp/x.py"), content="print(1)")
        self.assertEqual(len(client.did_save_calls), 1)
        _, sent_content = client.did_save_calls[0]
        self.assertEqual(sent_content, "print(1)")

    def test_no_client_is_silent_noop(self) -> None:
        manager = _StubManager(lambda _p: None)
        manager.save_file(Path("/tmp/x.md"))

    def test_client_exception_is_swallowed(self) -> None:
        client = _FakeClient(save_raises=RuntimeError("boom"))
        manager = _StubManager(lambda _p: client)
        manager.save_file(Path("/tmp/x.py"))


class PullDiagnosticsTests(unittest.TestCase):
    def test_returns_diagnostics_from_client(self) -> None:
        diag = Diagnostic(
            message="undefined name",
            severity="error",
            line=2,
            column=4,
            source="pyright",
        )
        client = _FakeClient(diagnostics=[diag])
        manager = _StubManager(lambda _p: client)
        result = manager.pull_diagnostics(
            Path("/tmp/x.py"), deadline=time.perf_counter() + 0.1
        )
        self.assertEqual(result, [diag])

    def test_no_client_returns_empty(self) -> None:
        manager = _StubManager(lambda _p: None)
        result = manager.pull_diagnostics(
            Path("/tmp/x.md"), deadline=time.perf_counter() + 0.1
        )
        self.assertEqual(result, [])

    def test_client_exception_returns_empty(self) -> None:
        client = _FakeClient(wait_raises=RuntimeError("boom"))
        manager = _StubManager(lambda _p: client)
        result = manager.pull_diagnostics(
            Path("/tmp/x.py"), deadline=time.perf_counter() + 0.1
        )
        self.assertEqual(result, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
