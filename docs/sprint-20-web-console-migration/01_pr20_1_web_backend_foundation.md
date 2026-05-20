# PR20.1 - Web Backend Foundation

## Goal

Create the Python web backend foundation for Aether's local browser console.
This PR must make it possible to import a FastAPI app, run it through a CLI
entrypoint, serve health/status JSON, and later mount the built TypeScript SPA.

## Current Problem

Aether has no HTTP server. The existing TUI speaks to Python through the stdio
gateway, and Sprint 19 created service objects that can now be called by future
adapters. Without a dedicated web adapter package, later REST and WebSocket work
would either duplicate gateway logic or grow another monolithic server file.

Hermes proves FastAPI is a good fit, but `hermes_cli/web_server.py` is too broad
for Aether's first web backend. Aether needs a smaller package structure with
security and routing boundaries from the start.

## Required Changes

- Add runtime dependencies:
  - `fastapi`
  - `uvicorn[standard]`
- Add project script:
  - `aether-web = "aether.web.entry:main"`
- Create `aether/web/__init__.py`.
- Create `aether/web/app.py` with `create_app(...) -> FastAPI`.
- Create `aether/web/entry.py` for CLI invocation.
- Create `aether/web/security.py` for:
  - ephemeral local session token generation
  - token header validation
  - bearer fallback for browser clients
  - localhost CORS defaults
  - Host-header validation for loopback binds
  - test-mode bypasses only through explicit app factory options
- Create `aether/web/errors.py` for service exception to HTTP error mapping.
- Create `aether/web/serializers.py` for dataclass and enum serialization.
- Create `aether/web/static.py` for optional SPA mount and `index.html` token
  injection hooks.
- Create `aether/web/routes/health.py` with:
  - `GET /api/status`
  - `GET /api/health`
- Keep the app factory dependency-injectable so tests can use fake services.

## Route Semantics

`GET /api/status` should return a compact console status:

```json
{
  "ok": true,
  "name": "Aether",
  "version": "1.0.0",
  "web": {"enabled": true}
}
```

`GET /api/health` should return the serialized `HealthService.status()` result.

The exact payload can grow, but the fields above must remain stable once added.

## Security Details

Use Hermes as the reference for local-dashboard security, but keep Aether names:

- Header: `X-Aether-Session-Token`
- Bearer fallback: `Authorization: Bearer <token>`
- Public endpoints:
  - `GET /api/status`
  - `GET /api/health`
- Protected endpoints:
  - all other `/api/*`
  - all `/api/*` WebSocket endpoints unless tests disable auth
- Default host: `127.0.0.1`
- Default port: `9120`
- Accept CORS origins matching `localhost` or `127.0.0.1`.

Do not expose the generated token in JSON status endpoints. Token injection is
only for served SPA HTML and test app introspection.

## CLI Entry

`aether-web` should accept:

- `--host`
- `--port`
- `--reload`
- `--web-dist`
- `--no-open` reserved for future browser opening behavior

The command should call `uvicorn.run(...)` with the app factory. It must not
import frontend build tooling.

## Tests

Add `aether/tests/web/test_web_app_foundation.py`:

- `create_app(auth_enabled=False)` imports and returns a FastAPI app.
- `/api/status` returns 200 and stable status fields.
- `/api/health` returns 200 and serializes service data.
- Protected route middleware rejects a fake `/api/private` route without token.
- The same route accepts `X-Aether-Session-Token`.
- Host validation rejects an invalid Host header when bound to loopback.
- `aether.web.entry` imports without starting a server.

## Non-Goals

- Do not implement session/model/tool REST routes in this PR.
- Do not implement agent run WebSockets in this PR.
- Do not scaffold the React app in this PR.
- Do not add remote auth or user accounts.

## Acceptance

- `python -m pytest aether/tests/web/test_web_app_foundation.py` passes.
- `python -m aether.web.entry --help` or the script import path is valid.
- `python -m pytest aether/tests/services aether/tests/gateway` still passes
  or known failures are documented before merge.
- No `aether/services/**` module imports `aether.web`.
