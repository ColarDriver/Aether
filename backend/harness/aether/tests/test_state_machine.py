from __future__ import annotations

import unittest

from aether.runtime.contracts import LoopState
from aether.runtime.state_machine import EngineStateMachine, StateTransitionError


class StateMachineTests(unittest.TestCase):
    def test_valid_transition_path(self) -> None:
        sm = EngineStateMachine()
        sm.transition(LoopState.PREPARE)
        sm.transition(LoopState.PRE_LLM)
        sm.transition(LoopState.LLM_CALL)
        sm.transition(LoopState.POST_LLM)
        sm.transition(LoopState.FINALIZE)
        sm.transition(LoopState.DONE)
        self.assertTrue(sm.terminal)

    def test_invalid_transition_raises(self) -> None:
        sm = EngineStateMachine()
        with self.assertRaises(StateTransitionError):
            sm.transition(LoopState.LLM_CALL)


if __name__ == "__main__":
    unittest.main()
