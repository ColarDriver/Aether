"""Aether harness package."""

from .config import load_aether_dotenv

# Load local project .env once on package import.
load_aether_dotenv()

from .agents.core.agent import AgentEngine
from .runtime.contracts import (
    EngineRequest,
    EngineResult,
    EngineStatus,
    ExitReason,
    ToolCall,
    ToolResult,
    NormalizedResponse,
)
from .config.schema import EngineConfig, ModelCallConfig

__all__ = [
    "AgentEngine",
    "EngineRequest",
    "EngineResult",
    "EngineStatus",
    "ExitReason",
    "ToolCall",
    "ToolResult",
    "NormalizedResponse",
    "EngineConfig",
    "ModelCallConfig",
]
