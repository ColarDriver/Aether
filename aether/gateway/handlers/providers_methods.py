"""``providers.*`` RPC methods.

For OpenAI-compatible endpoints we now attempt live discovery via the
provider's ``/v1/models`` endpoint (matching Python ``_cmd_model`` in
``aether/cli/commands.py``) and only fall back to the static catalog
when credentials are missing or the call fails. The static catalog
remains the authoritative list for providers without a live endpoint
(Anthropic Claude, Codex) and as a degraded-mode fallback.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

from aether.cli.providers import build_provider, list_providers, resolve_provider_name
from aether.cli.sessions import load_session
from aether.gateway.dispatcher import method
from aether.gateway.handlers.schemas import ModelInfo, ProviderInfo
from aether.gateway.handlers.state import get_current_session
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)


# ── Display + credential metadata for every supported provider ──────


_PROVIDER_DISPLAY: dict[str, dict[str, Any]] = {
    "claude": {
        "display_name": "Anthropic Claude",
        "requires_api_key": True,
        "default_base_url": None,
    },
    "openai": {
        "display_name": "OpenAI-compatible",
        "requires_api_key": True,
        "default_base_url": "https://api.openai.com/v1",
    },
    "codex": {
        "display_name": "OpenAI Codex",
        "requires_api_key": True,
        "default_base_url": None,
    },
}


# ── Static model catalog ────────────────────────────────────────────
# Keep these IDs aligned with the model strings referenced in
# `aether/cli/providers.py:_DEFAULTS` and the wider codebase.  When a
# new canonical Claude / OpenAI / Codex model lands, add it here so
# the TS picker discovers it.


_MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    "claude": [
        ModelInfo(id="claude-opus-4-7", display_name="Claude Opus 4.7", context_window=200_000),
        ModelInfo(id="claude-sonnet-4-6", display_name="Claude Sonnet 4.6", context_window=200_000),
        ModelInfo(
            id="claude-haiku-4-5-20251001",
            display_name="Claude Haiku 4.5",
            context_window=200_000,
        ),
    ],
    "openai": [
        ModelInfo(id="gpt-4o", display_name="GPT-4o", context_window=128_000),
        ModelInfo(id="gpt-4-turbo", display_name="GPT-4 Turbo", context_window=128_000),
        ModelInfo(id="gpt-4.1", display_name="GPT-4.1", context_window=128_000),
        ModelInfo(id="gpt-5", display_name="GPT-5", context_window=128_000),
    ],
    "codex": [
        ModelInfo(id="gpt-5.4", display_name="Codex GPT-5.4", context_window=128_000),
    ],
}


def providers_list(_params: dict[str, Any] | None) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for name in list_providers():
        meta = _PROVIDER_DISPLAY.get(name, {})
        info = ProviderInfo(
            name=name,
            display_name=meta.get("display_name", name),
            requires_api_key=bool(meta.get("requires_api_key", True)),
            default_base_url=meta.get("default_base_url"),
        )
        items.append(info.model_dump(mode="json", exclude_none=True))
    return {"providers": items}


def providers_models(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params or not isinstance(params.get("provider"), str) or not params["provider"].strip():
        raise GatewayError(
            "providers.models requires non-empty string 'provider'",
            code=ERROR_INVALID_PARAMS,
        )
    raw_name = params["provider"].strip()
    resolved = resolve_provider_name(raw_name)
    catalog = _MODEL_CATALOG.get(resolved)
    if catalog is None:
        raise GatewayError(
            f"unknown provider: {raw_name}",
            code=ERROR_APPLICATION,
            data={"provider": raw_name, "resolved": resolved},
        )

    # Live discovery first — this is what `_cmd_model` does in the Python TUI
    # and is what users expect from /model on a custom OpenAI-compatible
    # endpoint (Moonshot/Kimi, DeepSeek, Together, vLLM, etc.). The TS picker
    # surfaces `discovery` so the user can see at a glance whether they got
    # the live list, the static fallback, and (when fallback) WHY.
    live, discovery = _live_list_models(resolved, params)
    if live:
        merged = _merge_models(live, catalog)
        return {
            "models": [m.model_dump(mode="json", exclude_none=True) for m in merged],
            "discovery": discovery,
        }
    return {
        "models": [m.model_dump(mode="json", exclude_none=True) for m in catalog],
        "discovery": discovery,
    }


_DISCOVERY_TIMEOUT_SEC = 8.0
# Some custom OpenAI-compatible servers (vLLM, llama.cpp, Ollama gateway, …)
# do not actually root at the same path as the chat endpoint. We try a small
# set of common probes so the picker works without the user having to
# hand-edit OPENAI_BASE_URL.
_DISCOVERY_PATH_PROBES: tuple[str, ...] = ("/models", "/v1/models", "/api/models")


def _live_list_models(
    provider_name: str, params: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Best-effort fetch of the provider's `/v1/models` endpoint.

    Returns ``(ids, discovery)`` where ``discovery`` is a small dict the
    picker surfaces so users can debug their config without grepping the
    gateway log. Failures never raise — the picker stays usable on the
    static fallback.

    For OpenAI-compatible providers we do the HTTP probe **directly** in
    this module rather than going through ``provider.list_models()`` for
    two reasons:
    1. We want full visibility into what the endpoint returned (status
       code, raw body snippet) so the user knows whether the URL is wrong,
       authentication failed, or the response shape is exotic.
    2. Custom endpoints (vLLM, llama.cpp, Ollama) sometimes serve `/models`
       at different mount points than the chat endpoint, so we probe a
       small list of paths instead of the single hard-coded one.
    """
    discovery: dict[str, Any] = {"kind": "live", "source": "provider"}

    base_url = _resolve_base_url(params, discovery)
    if base_url:
        discovery["base_url"] = base_url

    resolved_name = resolve_provider_name(provider_name)
    if resolved_name == "openai":
        # Direct HTTP path with tolerant parsing + diagnostic output.
        return _live_list_models_openai(base_url, discovery)

    # Non-OpenAI providers don't have a generic discovery endpoint we can
    # probe (Claude / Codex catalogs are static client-side). Fall back to
    # the legacy `provider.list_models()` path so future provider additions
    # can opt into discovery just by overriding that method.
    try:
        provider = build_provider(
            resolved_name,
            api_key=os.getenv("AETHER_API_KEY"),
            base_url=base_url,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc) or type(exc).__name__
        _log_discovery(f"build_provider failed: {message}")
        return [], {
            "kind": "static",
            "reason": "no_credentials",
            "error": message,
            **({"base_url": base_url} if base_url else {}),
        }
    try:
        raw_ids = list(provider.list_models())
    except Exception as exc:  # noqa: BLE001
        message = str(exc) or type(exc).__name__
        _log_discovery(f"list_models() raised: {type(exc).__name__}: {message}")
        return [], {
            "kind": "static",
            "reason": "list_models_error",
            "error": message,
            **({"base_url": base_url} if base_url else {}),
        }
    cleaned = _dedupe_sorted_strings(raw_ids)
    if not cleaned:
        return [], {
            "kind": "static",
            "reason": "empty_response",
            **({"base_url": base_url} if base_url else {}),
        }
    discovery["count"] = len(cleaned)
    return cleaned, discovery


def _live_list_models_openai(
    base_url: str | None, discovery: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    api_key = (
        os.getenv("AETHER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
        or ""
    )

    # Step 1 — try the provider's own list_models() first. It's the cheapest
    # path and matches what Python's `_cmd_model` does. When the provider
    # returns a usable list we are done; the multi-shape probe below only
    # needs to run when this path returns nothing or raises.
    try:
        provider = build_provider("openai", api_key=api_key or None, base_url=base_url)
        provider_ids = list(provider.list_models())
        cleaned_quick = _dedupe_sorted_strings(provider_ids)
        if cleaned_quick:
            discovery["count"] = len(cleaned_quick)
            return cleaned_quick, discovery
    except Exception as exc:  # noqa: BLE001 — fall through to diagnostic probe.
        _log_discovery(
            f"provider.list_models() raised, falling back to direct probe: "
            f"{type(exc).__name__}: {exc}"
        )

    if not api_key:
        _log_discovery("no API key in env (set OPENAI_API_KEY)")
        return [], {
            "kind": "static",
            "reason": "no_credentials",
            "error": "no api key in env",
            **({"base_url": base_url} if base_url else {}),
        }

    # Step 2 — direct HTTP fetch with tolerant parsing + diagnostic output.
    # This catches custom endpoints (vLLM, Ollama, llama.cpp, Moonshot/Kimi)
    # that either mount /models at a different path or return a non-OpenAI
    # response shape.
    effective_base = base_url or "https://api.openai.com/v1"
    last_error: str | None = None
    last_status: int | None = None
    last_body_preview: str | None = None
    for probe in _candidate_urls(effective_base):
        try:
            with httpx.Client(timeout=_DISCOVERY_TIMEOUT_SEC) as client:
                resp = client.get(
                    probe,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _log_discovery(f"GET {probe} failed: {last_error}")
            continue

        last_status = resp.status_code
        if resp.status_code >= 400:
            last_error = f"HTTP {resp.status_code}"
            last_body_preview = _truncate_for_log(resp.text)
            _log_discovery(f"GET {probe} → {last_error} · body={last_body_preview}")
            continue

        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"json decode: {exc}"
            last_body_preview = _truncate_for_log(resp.text)
            _log_discovery(f"GET {probe} → not JSON · body={last_body_preview}")
            continue

        ids = _extract_model_ids(payload)
        cleaned = _dedupe_sorted_strings(ids)
        if cleaned:
            discovery["count"] = len(cleaned)
            discovery["url"] = probe
            # Detect the common "base_url missing /v1" trap: discovery
            # succeeded at `…/v1/models` but the configured base_url does
            # not end in `/v1`, which means the chat endpoint will hit
            # `…/chat/completions` and 404 even though /model looked fine.
            # Surface the suggested base_url so the picker can warn the
            # user before they spend 30 seconds waiting for the 404.
            suggested = _suggest_base_url(base_url, probe)
            if suggested and suggested != base_url:
                discovery["suggested_base_url"] = suggested
                discovery["warning"] = (
                    "discovery succeeded at /v1/models but base_url is missing /v1; "
                    "chat completions will likely 404. "
                    f"Restart with OPENAI_BASE_URL={suggested}"
                )
                _log_discovery(
                    f"base_url mismatch: discovered at {probe} but base_url={base_url}; "
                    f"suggesting OPENAI_BASE_URL={suggested}"
                )
            return cleaned, discovery

        last_error = "200 OK but no model ids in response"
        last_body_preview = _truncate_for_log(_short_payload(payload))
        _log_discovery(f"GET {probe} → 200 but unrecognised shape · body={last_body_preview}")

    fallback: dict[str, Any] = {
        "kind": "static",
        **({"base_url": base_url} if base_url else {}),
    }
    if last_status is None:
        fallback["reason"] = "list_models_error"
        fallback["error"] = last_error or "no response"
    elif last_status >= 400:
        fallback["reason"] = "list_models_error"
        fallback["error"] = last_error or f"HTTP {last_status}"
        if last_body_preview:
            fallback["body_preview"] = last_body_preview
    else:
        fallback["reason"] = "empty_response"
        if last_body_preview:
            fallback["body_preview"] = last_body_preview
    return [], fallback


def _suggest_base_url(base_url: str | None, working_probe: str) -> str | None:
    """Recommend an OPENAI_BASE_URL value that should make chat work too.

    Called only when /models discovery succeeded — `working_probe` is the
    full URL that returned a usable list. The suggestion is the prefix of
    `working_probe` minus the trailing `/models` segment, so it points at
    the API root the chat endpoint will hang off (e.g. `…/v1`).
    """
    if not working_probe.endswith("/models"):
        return None
    api_root = working_probe[: -len("/models")]
    if not base_url:
        return api_root
    base = base_url.rstrip("/")
    if base == api_root:
        return None
    return api_root


def _candidate_urls(base_url: str) -> list[str]:
    """Build the ordered list of URLs to probe for /models discovery.

    Tries the user-configured base_url first, then walks common probe
    suffixes so a misconfigured base_url (missing /v1, extra path) still
    finds the catalog. Never duplicates a URL.
    """
    base = base_url.rstrip("/")
    seen: set[str] = set()
    candidates: list[str] = []
    for path in _DISCOVERY_PATH_PROBES:
        url = f"{base}{path}"
        if url not in seen:
            seen.add(url)
            candidates.append(url)
    # If the base_url itself ends with /v1 we already cover /v1/models above
    # via /models on the stripped base. Try the un-stripped base too in case
    # the user pointed straight at /v1/models.
    if base.endswith("/v1"):
        url = f"{base}/models"
        if url not in seen:
            seen.add(url)
            candidates.append(url)
    return candidates


def _extract_model_ids(payload: object) -> list[str]:
    """Tolerant id extractor — handles every common /models response shape.

    Recognised shapes:
    - {"data": [{"id": "..."}]}              (OpenAI canonical)
    - {"models": [{"id": "..."}]}            (some custom servers)
    - [{"id": "..."}]                        (direct array of objects)
    - ["id1", "id2"]                         (array of strings)
    - {"data": ["id1", "id2"]}               (Ollama / a few vLLM forks)
    - {"models": ["id1"]}                    (Ollama variant)
    """

    def _coerce_list(value: object) -> list[str]:
        out: list[str] = []
        if not isinstance(value, list):
            return out
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                out.append(entry.strip())
            elif isinstance(entry, dict):
                ident = entry.get("id") or entry.get("name") or entry.get("model")
                if isinstance(ident, str) and ident.strip():
                    out.append(ident.strip())
        return out

    if isinstance(payload, list):
        return _coerce_list(payload)

    if isinstance(payload, dict):
        for key in ("data", "models", "items", "result"):
            extracted = _coerce_list(payload.get(key))
            if extracted:
                return extracted

    return []


def _dedupe_sorted_strings(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            cleaned.append(candidate)
    cleaned.sort()
    return cleaned


def _truncate_for_log(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    snippet = text.replace("\n", " ").strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 1] + "…"


def _short_payload(value: object) -> str:
    try:
        return json.dumps(value)[:200]
    except (TypeError, ValueError):
        return str(value)[:200]


def _log_discovery(message: str) -> None:
    sys.stderr.write(f"[providers.models] {message}\n")
    sys.stderr.flush()


def _resolve_base_url(params: dict[str, Any], discovery: dict[str, Any]) -> str | None:
    """Return the base URL to use, walking the precedence chain
    Python `_cmd_model` follows: explicit param > current session record >
    OPENAI_BASE_URL / ANTHROPIC_BASE_URL env > None (provider default)."""
    override = params.get("base_url")
    if isinstance(override, str) and override.strip():
        discovery["base_url_source"] = "param"
        return override.strip()
    current_session_id = get_current_session()
    if current_session_id:
        try:
            record = load_session(current_session_id)
        except Exception:  # noqa: BLE001 - never fail the picker on session-load
            record = None
        if record is not None and record.base_url:
            discovery["base_url_source"] = "session"
            return record.base_url
    env_url = os.getenv("OPENAI_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    if env_url:
        discovery["base_url_source"] = "env"
        return env_url
    discovery["base_url_source"] = "default"
    return None


def _merge_models(live_ids: list[str], catalog: list[ModelInfo]) -> list[ModelInfo]:
    """Combine the live ids with the static catalog — live wins on display
    name (we have no metadata for live ids so pass through the id as the
    label), and catalog entries not present in the live list are dropped
    when live discovery succeeded (matches the Python picker which only
    shows what /v1/models returned)."""
    catalog_by_id = {entry.id: entry for entry in catalog}
    result: list[ModelInfo] = []
    for model_id in live_ids:
        meta = catalog_by_id.get(model_id)
        if meta is not None:
            result.append(meta)
        else:
            result.append(ModelInfo(id=model_id, display_name=model_id))
    return result


def register() -> None:
    """Register ``providers.*`` handlers on the dispatcher.  Idempotent."""
    method("providers.list", long=False)(providers_list)
    method("providers.models", long=True)(providers_models)


__all__ = [
    "providers_list",
    "providers_models",
    "register",
]
