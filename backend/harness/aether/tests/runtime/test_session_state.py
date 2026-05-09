"""Unit tests for ``aether.runtime.session_state``.

Sprint 3.5 / PR-2 (PR 3.5.7).
"""

from __future__ import annotations

import threading
import unittest

from aether.runtime.session_state import (
    SessionMode,
    all_sessions,
    clear_mode,
    get_mode,
    set_mode,
)


class SessionStateBasicsTests(unittest.TestCase):
    def setUp(self) -> None:
        for sid in list(all_sessions().keys()):
            clear_mode(sid)

    def tearDown(self) -> None:
        for sid in list(all_sessions().keys()):
            clear_mode(sid)

    def test_a1_default_mode_is_agent(self) -> None:
        self.assertEqual(get_mode("session-A"), "agent")
        self.assertEqual(get_mode("session-A"), SessionMode.AGENT.value)

    def test_a2_set_then_get_round_trip(self) -> None:
        set_mode("session-A", SessionMode.PLAN)
        self.assertEqual(get_mode("session-A"), "plan")

    def test_a3_set_via_string_value(self) -> None:
        set_mode("session-A", "plan")
        self.assertEqual(get_mode("session-A"), "plan")

    def test_a4_clear_returns_to_default(self) -> None:
        set_mode("session-A", "plan")
        clear_mode("session-A")
        self.assertEqual(get_mode("session-A"), "agent")

    def test_a5_isolation_across_sessions(self) -> None:
        set_mode("session-A", SessionMode.PLAN)
        set_mode("session-B", SessionMode.AGENT)
        self.assertEqual(get_mode("session-A"), "plan")
        self.assertEqual(get_mode("session-B"), "agent")
        clear_mode("session-A")
        self.assertEqual(get_mode("session-A"), "agent")
        self.assertEqual(get_mode("session-B"), "agent")

    def test_a6_empty_session_id_get_returns_default(self) -> None:
        self.assertEqual(get_mode(""), "agent")

    def test_a7_empty_session_id_set_raises(self) -> None:
        with self.assertRaises(ValueError):
            set_mode("", SessionMode.PLAN)

    def test_a8_clear_unknown_session_is_idempotent(self) -> None:
        clear_mode("never-set")  # must not raise
        self.assertEqual(get_mode("never-set"), "agent")

    def test_a9_all_sessions_snapshot_returns_copy(self) -> None:
        set_mode("session-A", "plan")
        snap = all_sessions()
        snap["session-A"] = "agent"
        self.assertEqual(get_mode("session-A"), "plan")

    def test_a10_thread_safety_smoke(self) -> None:
        # 50 threads alternately flipping modes on shared sessions —
        # the test does not assert order, only that the lock prevents
        # KeyError / dict mutation explosions.
        def worker(i: int) -> None:
            sid = f"shared-{i % 5}"
            for _ in range(40):
                set_mode(sid, SessionMode.PLAN if i % 2 else SessionMode.AGENT)
                _ = get_mode(sid)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for sid in (f"shared-{i}" for i in range(5)):
            self.assertIn(get_mode(sid), {"plan", "agent"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
