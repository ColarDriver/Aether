from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aether.web.app import create_app
from aether.web.entry import build_parser
from aether.web.security import SESSION_HEADER_NAME, is_accepted_host
from aether.web.static import inject_bootstrap


def test_create_app_status_and_health_public() -> None:
    app = create_app(auth_enabled=False)
    client = TestClient(app)

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["ok"] is True
    assert status.json()["name"] == "Aether"
    assert status.json()["web"]["enabled"] is True

    health = client.get("/api/health")
    assert health.status_code == 200
    assert "services" in health.json()


def test_auth_middleware_rejects_and_accepts_protected_api_route() -> None:
    app = create_app(auth_enabled=True, session_token="test-token")
    _add_private_route(app)
    client = TestClient(app)

    denied = client.get("/api/private")
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "unauthorized"

    allowed = client.get("/api/private", headers={SESSION_HEADER_NAME: "test-token"})
    assert allowed.status_code == 200
    assert allowed.json() == {"ok": True}

    bearer = client.get("/api/private", headers={"Authorization": "Bearer test-token"})
    assert bearer.status_code == 200


def test_host_validation_rejects_invalid_host_for_loopback_bind() -> None:
    app = create_app(auth_enabled=False, bound_host="127.0.0.1")
    client = TestClient(app)

    invalid = client.get("/api/status", headers={"host": "evil.test"})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_host"

    valid = client.get("/api/status", headers={"host": "127.0.0.1"})
    assert valid.status_code == 200


def test_host_validation_helper_accepts_loopback_aliases() -> None:
    assert is_accepted_host("localhost:9120", "127.0.0.1")
    assert is_accepted_host("127.0.0.1:9120", "localhost")
    assert is_accepted_host("[::1]:9120", "::1")
    assert not is_accepted_host("example.com", "127.0.0.1")


def test_static_bootstrap_injects_token_and_base_path(tmp_path: Path) -> None:
    html = "<html><head><title>Aether</title></head><body></body></html>"

    injected = inject_bootstrap(html, session_token="abc", base_path="/console")

    assert "window.__AETHER_SESSION_TOKEN__='abc'" in injected
    assert "window.__AETHER_BASE_PATH__='/console'" in injected
    assert injected.index("<script>") < injected.index("</head>")


def test_entry_parser_imports_without_starting_server() -> None:
    parser = build_parser()
    args = parser.parse_args(["--host", "127.0.0.1", "--port", "9121", "--no-open"])
    assert args.host == "127.0.0.1"
    assert args.port == 9121
    assert args.no_open is True


def _add_private_route(app: FastAPI) -> None:
    @app.get("/api/private")
    async def private() -> dict[str, bool]:
        return {"ok": True}



def test_static_spa_mount_serves_index_with_bootstrap(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<html><head></head><body><div id='root'></div></body></html>",
        encoding="utf-8",
    )
    app = create_app(auth_enabled=True, session_token="static-token", web_dist=dist)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "window.__AETHER_SESSION_TOKEN__='static-token'" in response.text
    assert "window.__AETHER_BASE_PATH__=''" in response.text
