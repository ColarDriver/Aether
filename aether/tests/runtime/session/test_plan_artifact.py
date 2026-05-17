"""Tests for plan artifact storage — Sprint 12 PR 12.4."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aether.runtime.session.plan_artifact import (
    clear_plan,
    get_plan_path,
    plans_dir,
    read_plan,
    write_plan,
)


class _ArtifactCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ, {"AETHER_HOME": self._tmp.name}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)


# ---------------------------------------------------------------- pathing

class PathTests(_ArtifactCase):
    def test_plans_dir_under_aether_home(self) -> None:
        self.assertEqual(plans_dir(), Path(self._tmp.name) / "plans")

    def test_get_plan_path_uses_first_eight_chars(self) -> None:
        path = get_plan_path("abc12345-6789-0000-0000-deadbeefcafe")
        self.assertEqual(path.name, "abc12345.md")
        self.assertEqual(path.parent, plans_dir())

    def test_get_plan_path_is_stable_per_session(self) -> None:
        sid = "session-xyz-1234"
        self.assertEqual(get_plan_path(sid), get_plan_path(sid))

    def test_get_plan_path_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            get_plan_path("")

    def test_get_plan_path_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            get_plan_path("../escape")
        with self.assertRaises(ValueError):
            get_plan_path("foo/bar")
        with self.assertRaises(ValueError):
            get_plan_path("a.b")  # dot not allowed in our slug

    def test_get_plan_path_rejects_non_string(self) -> None:
        with self.assertRaises(ValueError):
            get_plan_path(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- write/read

class WriteReadTests(_ArtifactCase):
    def test_write_plan_creates_directory(self) -> None:
        self.assertFalse(plans_dir().exists())
        path = write_plan("session-abc", "# Plan\n\n- step 1")
        self.assertTrue(plans_dir().exists())
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "# Plan\n\n- step 1")

    def test_write_plan_overwrites(self) -> None:
        sid = "session-abc"
        write_plan(sid, "first")
        write_plan(sid, "second")
        self.assertEqual(read_plan(sid), "second")

    def test_read_plan_missing_returns_none(self) -> None:
        self.assertIsNone(read_plan("nope-1234"))

    def test_read_plan_empty_file_returns_empty_string(self) -> None:
        sid = "session-abc"
        write_plan(sid, "")
        self.assertEqual(read_plan(sid), "")

    def test_read_plan_round_trips_unicode(self) -> None:
        sid = "session-uni"
        plan = "# 计划\n\n- 步骤 1\n- 步骤 2"
        write_plan(sid, plan)
        self.assertEqual(read_plan(sid), plan)

    def test_read_plan_on_invalid_session_returns_none(self) -> None:
        # Should not raise; bad input → no artifact.
        self.assertIsNone(read_plan(""))
        self.assertIsNone(read_plan("../escape"))


# ---------------------------------------------------------------- clear

class ClearTests(_ArtifactCase):
    def test_clear_plan_idempotent_on_missing(self) -> None:
        clear_plan("never-existed")  # must not raise

    def test_clear_plan_removes_existing(self) -> None:
        sid = "session-abc"
        write_plan(sid, "to be erased")
        clear_plan(sid)
        self.assertIsNone(read_plan(sid))

    def test_clear_plan_on_invalid_session_is_noop(self) -> None:
        clear_plan("")  # must not raise


# ---------------------------------------------------------------- AETHER_HOME

class AetherHomeFallbackTests(unittest.TestCase):
    def test_falls_back_to_home_dot_aether_when_env_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AETHER_HOME", None)
            expected = Path.home() / ".aether" / "plans"
            self.assertEqual(plans_dir(), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
