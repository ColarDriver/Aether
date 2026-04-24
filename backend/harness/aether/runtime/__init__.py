"""Runtime loop primitives for Aether."""

from .contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    LoopState,
    ToolCall,
    ToolResult,
    NormalizedResponse,
)
from .interrupts import InterruptController
from .hooks import EngineHooks
from .session_store import InMemorySessionStore, SessionStore
from .state_machine import EngineStateMachine, StateTransitionError

__all__ = [
    "EngineRequest",
    "EngineResult",
    "EngineStatus",
    "ExitReason",
    "LoopState",
    "ToolCall",
    "ToolResult",
    "NormalizedResponse",
    "InterruptController",
    "EngineStateMachine",
    "StateTransitionError",
    "EngineHooks",
    "SessionStore",
    "InMemorySessionStore",
]
