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
from .hooks import EngineHooks, HookOutcome
from .session_store import InMemorySessionStore, SessionStore
from .state_machine import EngineStateMachine, StateTransitionError
from .steer import SteerInbox
from .unicode_sanitizer import strip_non_ascii, strip_surrogates
from .reasoning import extract_last_reasoning
from .trajectory import convert_to_trajectory_format
from .task_cleanup import acquire_task_resource_for_executor, release_task_resources
from .schema_sanitizer import strip_pattern_and_format
from .image_shrink import shrink_image_parts_in_messages
from .rate_guard import RateGuard, default_rate_guard_dir, provider_rate_guard_key

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
    "HookOutcome",
    "SteerInbox",
    "strip_non_ascii",
    "strip_surrogates",
    "extract_last_reasoning",
    "convert_to_trajectory_format",
    "acquire_task_resource_for_executor",
    "release_task_resources",
    "strip_pattern_and_format",
    "shrink_image_parts_in_messages",
    "RateGuard",
    "default_rate_guard_dir",
    "provider_rate_guard_key",
    "SessionStore",
    "InMemorySessionStore",
]
