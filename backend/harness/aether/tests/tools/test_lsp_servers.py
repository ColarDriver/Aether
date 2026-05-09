"""Tests for ``aether.runtime.lsp_servers``.

Sprint 3.5 / PR-3 (PR 3.5.9).

The module is pure data + a tiny resolver, so the tests are short.
The point is to lock the contract so a future refactor (e.g. adding
``lua`` support) can't accidentally break the table.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from aether.runtime.lsp_servers import (
    EXT_TO_LANG,
    LANGUAGE_SERVERS,
    language_for,
    resolve_server_for,
    supported_extensions,
    supported_languages,
)


class TableShapeTests(unittest.TestCase):
    def test_a1_every_known_extension_maps_to_a_known_language(self) -> None:
        for ext, lang in EXT_TO_LANG.items():
            self.assertIn(lang, LANGUAGE_SERVERS, f"orphan ext={ext} lang={lang}")

    def test_a2_every_command_is_a_non_empty_argv(self) -> None:
        for lang, cmds in LANGUAGE_SERVERS.items():
            self.assertGreater(len(cmds), 0, lang)
            for cmd in cmds:
                self.assertIsInstance(cmd, list, f"{lang}: not a list")
                self.assertGreater(len(cmd), 0, f"{lang}: empty argv")
                for token in cmd:
                    self.assertIsInstance(token, str, f"{lang}: non-str token {token!r}")
                    self.assertTrue(token, f"{lang}: empty token in {cmd}")

    def test_a3_supported_languages_matches_table(self) -> None:
        self.assertEqual(set(supported_languages()), set(LANGUAGE_SERVERS.keys()))

    def test_a4_supported_extensions_matches_table(self) -> None:
        self.assertEqual(set(supported_extensions()), set(EXT_TO_LANG.keys()))


class LanguageForTests(unittest.TestCase):
    def test_b1_python_extensions(self) -> None:
        self.assertEqual(language_for(Path("a/b.py")), "python")
        self.assertEqual(language_for(Path("c.pyi")), "python")

    def test_b2_typescript_family(self) -> None:
        for ext in (".ts", ".tsx", ".mts", ".cts"):
            self.assertEqual(language_for(Path("file" + ext)), "typescript", ext)

    def test_b3_unknown_extension_returns_none(self) -> None:
        self.assertIsNone(language_for(Path("a.unknown")))

    def test_b4_extensionless_file_returns_none(self) -> None:
        self.assertIsNone(language_for(Path("Makefile")))

    def test_b5_uppercase_extension_resolves(self) -> None:
        self.assertEqual(language_for(Path("X.PY")), "python")


class ResolveServerForTests(unittest.TestCase):
    def test_c1_returns_default_table_for_known_language(self) -> None:
        cmds = resolve_server_for(Path("x.py"))
        self.assertIsNotNone(cmds)
        assert cmds is not None
        self.assertIn(["pylsp"], cmds)

    def test_c2_returns_none_for_unknown_extension(self) -> None:
        self.assertIsNone(resolve_server_for(Path("x.unknown")))

    def test_c3_overrides_replace_default(self) -> None:
        overrides = {"python": [["my-lsp", "--foo"]]}
        cmds = resolve_server_for(Path("x.py"), overrides=overrides)
        self.assertEqual(cmds, [["my-lsp", "--foo"]])

    def test_c4_overrides_for_unrelated_language_dont_leak(self) -> None:
        overrides = {"rust": [["my-rust-lsp"]]}
        cmds = resolve_server_for(Path("x.py"), overrides=overrides)
        self.assertIsNotNone(cmds)
        assert cmds is not None
        self.assertNotIn(["my-rust-lsp"], cmds)
        self.assertIn(["pylsp"], cmds)

    def test_c5_returns_independent_list(self) -> None:
        # Mutating the returned list must not affect the global table.
        cmds = resolve_server_for(Path("x.py"))
        assert cmds is not None
        cmds.append(["sentinel"])
        self.assertNotIn(["sentinel"], LANGUAGE_SERVERS["python"])


if __name__ == "__main__":
    unittest.main()
