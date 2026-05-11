from __future__ import annotations

import threading
import time
import unittest

from aether.runtime.interrupt_signal import InterruptSignal


class InterruptSignalTests(unittest.TestCase):
    def test_abort_sets_reason_and_flag(self) -> None:
        signal = InterruptSignal()
        self.assertTrue(signal.abort("user-interrupt"))
        self.assertTrue(signal.is_aborted())
        self.assertEqual(signal.reason(), "user-interrupt")

    def test_listener_fires_on_abort(self) -> None:
        signal = InterruptSignal()
        seen: list[str | None] = []
        signal.add_listener(lambda reason: seen.append(reason))
        signal.abort("boom")
        self.assertEqual(seen, ["boom"])

    def test_listener_added_after_abort_fires_immediately(self) -> None:
        signal = InterruptSignal()
        signal.abort("done")
        seen: list[str | None] = []
        signal.add_listener(lambda reason: seen.append(reason))
        self.assertEqual(seen, ["done"])

    def test_child_inherits_parent_abort(self) -> None:
        parent = InterruptSignal()
        child = InterruptSignal(parent=parent)
        parent.abort("parent-stop")
        self.assertTrue(child.is_aborted())
        self.assertEqual(child.reason(), "parent-stop")

    def test_wait_returns_true_when_aborted(self) -> None:
        signal = InterruptSignal()
        threading.Timer(0.05, lambda: signal.abort("later")).start()
        self.assertTrue(signal.wait(0.5))

    def test_remove_listener_prevents_future_fire(self) -> None:
        signal = InterruptSignal()
        seen: list[str | None] = []
        def listener(reason: str | None) -> None:
            seen.append(reason)
        signal.add_listener(listener)
        signal.remove_listener(listener)
        signal.abort("x")
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
