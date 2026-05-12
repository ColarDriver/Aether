"""Tests for the persistent CLI preferences module (``cli/prefs.py``)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aether.cli import prefs as prefs_module
from aether.cli.prefs import (
    PREFS_FORMAT_VERSION,
    Prefs,
    get_last_model,
    load_prefs,
    save_prefs,
    set_last_model,
)


class PrefsRoundTripTests(unittest.TestCase):
    """Save → load → save preserves both known and unknown fields."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._home_patch = mock.patch.dict(os.environ, {"AETHER_HOME": self._tmp.name})
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmp.cleanup()

    def test_load_returns_empty_when_file_missing(self) -> None:
        loaded = load_prefs()
        self.assertEqual(loaded.last_model_by_provider, {})
        self.assertEqual(loaded.version, PREFS_FORMAT_VERSION)

    def test_save_then_load_roundtrip(self) -> None:
        save_prefs(Prefs(last_model_by_provider={"openai": "kimi-k2.6"}))
        loaded = load_prefs()
        self.assertEqual(loaded.last_model_by_provider, {"openai": "kimi-k2.6"})

    def test_set_last_model_helper_persists(self) -> None:
        set_last_model("openai", "kimi-k2.6")
        self.assertEqual(get_last_model("openai"), "kimi-k2.6")

    def test_set_last_model_overwrites_previous_entry(self) -> None:
        set_last_model("openai", "gpt-5.4")
        set_last_model("openai", "kimi-k2.6")
        self.assertEqual(get_last_model("openai"), "kimi-k2.6")

    def test_set_last_model_keeps_other_providers(self) -> None:
        set_last_model("openai", "gpt-5.4")
        set_last_model("claude", "sonnet-4-6")
        self.assertEqual(get_last_model("openai"), "gpt-5.4")
        self.assertEqual(get_last_model("claude"), "sonnet-4-6")

    def test_get_last_model_returns_none_when_unknown(self) -> None:
        set_last_model("openai", "kimi-k2.6")
        self.assertIsNone(get_last_model("nonexistent"))

    def test_set_last_model_silent_on_empty_args(self) -> None:
        # Defensive: a buggy caller passing "" must not wipe an entry.
        set_last_model("openai", "kimi-k2.6")
        set_last_model("", "garbage")
        set_last_model("openai", "")
        self.assertEqual(get_last_model("openai"), "kimi-k2.6")

    def test_unknown_fields_round_trip_unchanged(self) -> None:
        # Newer aether builds may add fields older ones don't recognise —
        # the older build must not strip them on save.
        path = Path(self._tmp.name) / "prefs.json"
        path.write_text(
            json.dumps(
                {
                    "version": PREFS_FORMAT_VERSION,
                    "last_model_by_provider": {"openai": "gpt-5.4"},
                    "future_field_we_dont_know_yet": {"theme": "dracula"},
                }
            ),
            encoding="utf-8",
        )

        loaded = load_prefs()
        self.assertIn("future_field_we_dont_know_yet", loaded.unknown)

        # Save again; the unknown field should still be there.
        loaded.last_model_by_provider["openai"] = "kimi-k2.6"
        save_prefs(loaded)

        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["last_model_by_provider"]["openai"], "kimi-k2.6")
        self.assertEqual(
            on_disk["future_field_we_dont_know_yet"], {"theme": "dracula"}
        )

    def test_load_returns_empty_on_corrupt_file(self) -> None:
        path = Path(self._tmp.name) / "prefs.json"
        path.write_text("not valid json {", encoding="utf-8")
        loaded = load_prefs()
        self.assertEqual(loaded.last_model_by_provider, {})

    def test_load_returns_empty_when_file_is_a_list(self) -> None:
        # File exists but is the wrong shape — we must not crash.
        path = Path(self._tmp.name) / "prefs.json"
        path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        loaded = load_prefs()
        self.assertEqual(loaded.last_model_by_provider, {})


class PrefsAtomicWriteTests(unittest.TestCase):
    """Verify save_prefs doesn't leave a half-written file behind."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._home_patch = mock.patch.dict(os.environ, {"AETHER_HOME": self._tmp.name})
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmp.cleanup()

    def test_no_temp_file_left_after_successful_write(self) -> None:
        save_prefs(Prefs(last_model_by_provider={"openai": "gpt-5.4"}))
        files = list(Path(self._tmp.name).iterdir())
        # Only the final prefs.json should remain — no orphan ``.prefs-*.json``.
        self.assertEqual([f.name for f in files], ["prefs.json"])

    def test_save_silently_no_ops_when_home_is_unwritable(self) -> None:
        # Point AETHER_HOME at a read-only path; save_prefs must not raise.
        os.environ["AETHER_HOME"] = "/proc/this-cannot-be-written"
        try:
            save_prefs(Prefs(last_model_by_provider={"openai": "gpt-5.4"}))
        finally:
            os.environ["AETHER_HOME"] = self._tmp.name


if __name__ == "__main__":
    unittest.main()
