"""Core runtime contracts and loop primitives."""

from .contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    LoopState,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from .exceptions import EngineInterrupted
from .hooks import EngineHooks, HookOutcome
from .iteration_budget import IterationBudget
from .services import EngineServices
from .state_machine import EngineStateMachine, StateTransitionError

__all__ = [
    "EngineRequest",
    "EngineResult",
    "EngineStatus",
    "ExitReason",
    "LoopState",
    "NormalizedResponse",
    "ToolCall",
    "ToolResult",
    "TurnContext",
    "EngineInterrupted",
    "EngineHooks",
    "HookOutcome",
    "IterationBudget",
    "EngineServices",
    "EngineStateMachine",
    "StateTransitionError",
]
