"""FastAPI app factory for the local Aether web console."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from aether.services.health import HealthService
from aether.web.errors import install_error_handlers
from aether.web.routes.health import router as health_router
from aether.web.security import create_session_token, install_security
from aether.web.static import mount_spa


@dataclass(slots=True)
class WebServices:
    health: HealthService


def create_app(
    *,
    auth_enabled: bool = True,
    session_token: str | None = None,
    bound_host: str | None = None,
    web_dist: str | Path | None = None,
    health_service: HealthService | None = None,
) -> FastAPI:
    """Create a configured FastAPI app.

    The app is intentionally built from dependency-injectable services so tests
    and future adapters can replace individual service objects without importing
    gateway handlers.
    """

    token = session_token or create_session_token()
    app = FastAPI(title="Aether Web Console", version=_package_version())
    app.state.aether_session_token = token
    app.state.aether_auth_enabled = bool(auth_enabled)
    app.state.aether_bound_host = bound_host
    app.state.aether_services = WebServices(health=health_service or HealthService())

    install_security(
        app,
        auth_enabled=auth_enabled,
        session_token=token,
        bound_host=bound_host,
    )
    install_error_handlers(app)

    app.include_router(health_router)

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
