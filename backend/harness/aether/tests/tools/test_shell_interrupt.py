from __future__ import annotations

import threading
import time
import unittest

from aether.runtime.contracts import ToolCall, TurnContext
from aether.runtime.interrupt_signal import InterruptSignal
from aether.tools.builtins.shell import ShellTool


class ShellInterruptTests(unittest.TestCase):
    def test_long_sleep_interrupted_quickly(self) -> None:
        signal = InterruptSignal()
        ctx = TurnContext(session_id="shell-int", iteration=0, metadata={}, interrupt_signal=signal)
        tool = ShellTool()
        timer = threading.Timer(0.1, lambda: signal.abort("user-interrupt"))
        timer.start()
        started = time.monotonic()
        result = tool.execute(
            ToolCall(id="c1", name="shell", arguments={"command": "python -c 'import time; time.sleep(5)'", "timeout_sec": 10}),
            ctx,
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.5)
        self.assertTrue(result.is_error)
        self.assertTrue(result.metadata.get("interrupted"))


if __name__ == "__main__":
    unittest.main()
