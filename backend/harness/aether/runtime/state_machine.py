"""Explicit loop state machine for the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet

from .contracts import LoopState


class StateTransitionError(RuntimeError):
    """Raised when an illegal state transition is attempted."""


_TRANSITIONS: Dict[LoopState, FrozenSet[LoopState]] = {
    LoopState.INIT: frozenset({LoopState.PREPARE}),
    LoopState.PREPARE: frozenset({LoopState.PRE_LLM, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.PRE_LLM: frozenset({LoopState.LLM_CALL, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.LLM_CALL: frozenset({LoopState.POST_LLM, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.POST_LLM: frozenset({LoopState.TOOL_DISPATCH, LoopState.FINALIZE, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.TOOL_DISPATCH: frozenset({LoopState.TOOL_EXECUTE, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.TOOL_EXECUTE: frozenset({LoopState.CHECK_EXIT, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.CHECK_EXIT: frozenset({LoopState.PRE_LLM, LoopState.FINALIZE, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.FINALIZE: frozenset({LoopState.DONE, LoopState.INTERRUPTED, LoopState.FAILED}),
    LoopState.DONE: frozenset(),
    LoopState.FAILED: frozenset(),
    LoopState.INTERRUPTED: frozenset(),
}


@dataclass(slots=True)
class EngineStateMachine:
    """Minimal explicit state machine used by AgentEngine."""

    state: LoopState = LoopState.INIT

    def transition(self, next_state: LoopState) -> None:
        allowed = _TRANSITIONS[self.state]
        if next_state not in allowed:
            raise StateTransitionError(f"Invalid transition: {self.state} -> {next_state}")
        self.state = next_state

    @property
    def terminal(self) -> bool:
        return self.state in {LoopState.DONE, LoopState.FAILED, LoopState.INTERRUPTED}
