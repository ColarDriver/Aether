"""FastAPI app factory for the local Aether web console."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from aether.services.analytics import AnalyticsService
from aether.services.config import ConfigService, PrefsService
from aether.services.diagnostics import DiagnosticsService
from aether.services.docs import DocsService
from aether.services.environment import EnvironmentService
from aether.services.health import HealthService
from aether.services.logs import LogService
from aether.services.providers import ModelSelectionService, ProviderService
from aether.services.runs import AgentRunService
from aether.services.sessions import SessionService
from aether.services.skills import SkillService
from aether.services.tasks import TaskService
from aether.services.tools import ToolService
from aether.services.workspace import WorkspaceService
from aether.web.errors import install_error_handlers
from aether.web.routes.analytics import router as analytics_router
from aether.web.routes.bootstrap import router as bootstrap_router
from aether.web.routes.commands import router as commands_router
from aether.web.routes.config import router as config_router
from aether.web.routes.diagnostics import router as diagnostics_router
from aether.web.routes.docs import router as docs_router
from aether.web.routes.environment import router as environment_router
from aether.web.routes.health import router as health_router
from aether.web.routes.logs import router as logs_router
from aether.web.routes.plan import router as plan_router
from aether.web.routes.providers import router as providers_router
from aether.web.routes.runs import router as runs_router
from aether.web.routes.sessions import router as sessions_router
from aether.web.routes.skills import router as skills_router
from aether.web.routes.tasks import router as tasks_router
from aether.web.routes.tools import router as tools_router
from aether.web.routes.workspace import router as workspace_router
from aether.web.security import create_session_token, install_security
from aether.web.static import mount_spa
from aether.web.ws.hub import WebRunSocketHub
from aether.web.ws.prompts import WebPromptBroker
from aether.web.ws.runs import router as run_ws_router


@dataclass(slots=True)
class WebServices:
    health: HealthService
    sessions: SessionService
    config: ConfigService
    prefs: PrefsService
    providers: ProviderService
    model_selection: ModelSelectionService
    tools: ToolService
    skills: SkillService
    tasks: TaskService
    diagnostics: DiagnosticsService
    environment: EnvironmentService
    runs: AgentRunService
    logs: LogService
    analytics: AnalyticsService
    docs: DocsService
    workspace: WorkspaceService


def create_app(
    *,
    auth_enabled: bool = True,
    session_token: str | None = None,
    bound_host: str | None = None,
    web_dist: str | Path | None = None,
    health_service: HealthService | None = None,
    session_service: SessionService | None = None,
    config_service: ConfigService | None = None,
    prefs_service: PrefsService | None = None,
    provider_service: ProviderService | None = None,
    model_selection_service: ModelSelectionService | None = None,
    tool_service: ToolService | None = None,
    skill_service: SkillService | None = None,
    task_service: TaskService | None = None,
    diagnostics_service: DiagnosticsService | None = None,
    environment_service: EnvironmentService | None = None,
    run_service: AgentRunService | None = None,
    log_service: LogService | None = None,
    analytics_service: AnalyticsService | None = None,
    docs_service: DocsService | None = None,
    workspace_service: WorkspaceService | None = None,
) -> FastAPI:
    """Create a configured FastAPI app.

    The app is intentionally built from dependency-injectable services so tests
    and future adapters can replace individual service objects without importing
    gateway handlers.
    """

    token = session_token or create_session_token()
    app = FastAPI(title="Aether Web Console", version=_package_version())
    sessions = session_service or SessionService()
    app.state.aether_session_token = token
    app.state.aether_auth_enabled = bool(auth_enabled)
    app.state.aether_bound_host = bound_host
    app.state.aether_run_socket_hub = WebRunSocketHub()
    app.state.aether_web_prompt_broker = WebPromptBroker(send_frame=app.state.aether_run_socket_hub.broadcast)
    app.state.aether_run_tasks = set()
    app.state.aether_services = WebServices(
        health=health_service or HealthService(),
        sessions=sessions,
        config=config_service or ConfigService(),
        prefs=prefs_service or PrefsService(),
        providers=provider_service or ProviderService(),
        model_selection=model_selection_service or ModelSelectionService(),
        tools=tool_service or ToolService(),
        skills=skill_service or SkillService(),
        tasks=task_service or TaskService(),
        diagnostics=diagnostics_service or DiagnosticsService(),
        environment=environment_service or EnvironmentService(),
        runs=run_service or AgentRunService(session_service=sessions),
        logs=log_service or LogService(),
        analytics=analytics_service or AnalyticsService(session_service=sessions),
        docs=docs_service or DocsService(),
        workspace=workspace_service or WorkspaceService(),
    )

    install_security(
        app,
        auth_enabled=auth_enabled,
        session_token=token,
        bound_host=bound_host,
    )
    install_error_handlers(app)

    app.include_router(health_router)
    app.include_router(bootstrap_router)
    app.include_router(analytics_router)
    app.include_router(commands_router)
    app.include_router(docs_router)
    app.include_router(workspace_router)
    app.include_router(sessions_router)
    app.include_router(plan_router)
    app.include_router(config_router)
    app.include_router(providers_router)
    app.include_router(tools_router)
    app.include_router(skills_router)
    app.include_router(tasks_router)
    app.include_router(diagnostics_router)
    app.include_router(environment_router)
    app.include_router(logs_router)
    app.include_router(runs_router)
    app.include_router(run_ws_router)

    if web_dist is not None:
        mount_spa(app, Path(web_dist), session_token=token)

    return app


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("aether-harness")
    except Exception:
        return "unknown"


__all__ = ["WebServices", "create_app"]
