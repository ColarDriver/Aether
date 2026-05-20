"""Provider service implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from typing import Any

import httpx

from aether.cli.providers import (
    build_provider,
    get_provider_defaults,
    list_providers,
    resolve_provider_name,
)
from aether.cli.sessions import load_session
from aether.config.auxiliary_slots import resolve_auxiliary_slot
from aether.config.provider_runtime import resolve_main_provider_runtime
from aether.runtime.credentials import default_credential_lookup
from aether.services.common import ServiceValidationError
from aether.services.providers.auth import AuthService
from aether.services.providers.contracts import (
    AuxiliarySlotStatus,
    CredentialStatus,
    ModelDiscoveryStatus,
    ModelSummary,
    ProviderModelList,
    ProviderRuntimeStatus,
    ProviderSummary,
)

HttpClientFactory = Callable[[], httpx.Client]

PROVIDER_DISPLAY: dict[str, dict[str, Any]] = {
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

MODEL_CATALOG: dict[str, list[ModelSummary]] = {
    "claude": [
        ModelSummary(id="claude-opus-4-7", display_name="Claude Opus 4.7", context_window=200_000),
        ModelSummary(id="claude-sonnet-4-6", display_name="Claude Sonnet 4.6", context_window=200_000),
        ModelSummary(id="claude-haiku-4-5-20251001", display_name="Claude Haiku 4.5", context_window=200_000),
    ],
    "openai": [
        ModelSummary(id="gpt-4o", display_name="GPT-4o", context_window=128_000),
        ModelSummary(id="gpt-4-turbo", display_name="GPT-4 Turbo", context_window=128_000),
        ModelSummary(id="gpt-4.1", display_name="GPT-4.1", context_window=128_000),
        ModelSummary(id="gpt-5", display_name="GPT-5", context_window=128_000),
    ],
    "codex": [
        ModelSummary(id="gpt-5.4", display_name="Codex GPT-5.4", context_window=128_000),
    ],
}

DISCOVERY_TIMEOUT_SEC = 8.0
DISCOVERY_PATH_PROBES: tuple[str, ...] = ("/models", "/v1/models", "/api/models")


class ProviderService:
    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        http_client_factory: HttpClientFactory | None = None,
        current_session_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._environ = environ
        self._http_client_factory = http_client_factory
        self._current_session_getter = current_session_getter

    def list_providers(self) -> list[ProviderSummary]:
        items: list[ProviderSummary] = []
        for name in list_providers():
            meta = PROVIDER_DISPLAY.get(name, {})
            items.append(
                ProviderSummary(
                    name=name,
                    display_name=str(meta.get("display_name", name)),
                    requires_api_key=bool(meta.get("requires_api_key", True)),
                    default_base_url=meta.get("default_base_url"),
                )
            )
        return items

    def resolve_provider_name(self, name: str) -> str:
        return resolve_provider_name(name)

    def get_provider_defaults(self, name: str) -> dict[str, Any]:
        return get_provider_defaults(name)

    def list_models(
        self,
        provider: str,
        *,
        base_url: str | None = None,
        current_session_id: str | None = None,
    ) -> ProviderModelList:
        raw_name = _require_non_empty(provider, "provider")
        resolved = resolve_provider_name(raw_name)
        catalog = MODEL_CATALOG.get(resolved)
        if catalog is None:
            raise ServiceValidationError(
                f"unknown provider: {raw_name}",
                details={"provider": raw_name, "resolved": resolved},
            )
        live, discovery = self._live_list_models(
            resolved,
            base_url=base_url,
            current_session_id=current_session_id,
        )
        if live:
            models = _merge_models(live, catalog)
        else:
            models = list(catalog)
        return ProviderModelList(models=models, discovery=_discovery_from_dict(discovery))

    def runtime_current(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> ProviderRuntimeStatus:
        runtime = _runtime_from_values(
            provider=provider,
            model=model,
            base_url=base_url,
            environ=self._environ,
        )
        credential = default_credential_lookup(environ=self._environ).get_first(
            runtime.api_key_env_names
        )
        public = runtime.public_metadata()
        return ProviderRuntimeStatus(
            family=str(public["family"]),
            provider_name=str(public["provider_name"]),
            model=str(public["model"]),
            base_url=public.get("base_url") if isinstance(public.get("base_url"), str) else None,
            api_key_env_names=tuple(runtime.api_key_env_names),
            model_env_names=tuple(runtime.model_env_names),
            base_url_env_names=tuple(runtime.base_url_env_names),
            source=str(public["source"]),
            credential=_credential_status_from_public(
                credential.public_metadata()
                if credential is not None
                else {
                    "source": "env",
                    "name": runtime.api_key_env_names[0] if runtime.api_key_env_names else "",
                    "configured": False,
                    "redacted": "",
                }
            )
            if runtime.api_key_env_names
            else None,
            extra=dict(public.get("extra") or {}),
        )

    def auxiliary_slots(self, slots: list[str] | None = None) -> list[AuxiliarySlotStatus]:
        requested = [slot for slot in (slots or ["subagent", "compression", "verifier", "title"]) if slot.strip()]
        try:
            configs = [resolve_auxiliary_slot(slot, environ=self._environ) for slot in requested]
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc
        return [
            AuxiliarySlotStatus(
                slot=config.slot,
                provider_family=config.provider_family,
                provider_name=config.provider_name,
                model=config.model,
                inherited=config.inherited,
                source=config.source,
            )
            for config in configs
        ]

    def _live_list_models(
        self,
        provider_name: str,
        *,
        base_url: str | None,
        current_session_id: str | None,
    ) -> tuple[list[str], dict[str, Any]]:
        discovery: dict[str, Any] = {"kind": "live", "source": "provider"}
        resolved_base_url = self._resolve_base_url(
            base_url=base_url,
            current_session_id=current_session_id,
            discovery=discovery,
        )
        if resolved_base_url:
            discovery["base_url"] = resolved_base_url
        resolved_name = resolve_provider_name(provider_name)
        if resolved_name == "openai":
            return self._live_list_models_openai(resolved_base_url, discovery)
        try:
            provider = build_provider(
                resolved_name,
                api_key=_env(self._environ).get("AETHER_API_KEY"),
                base_url=resolved_base_url,
            )
        except Exception as exc:  # noqa: BLE001
            return [], {
                "kind": "static",
                "reason": "no_credentials",
                "error": str(exc) or type(exc).__name__,
                **({"base_url": resolved_base_url} if resolved_base_url else {}),
            }
        try:
            raw_ids = list(provider.list_models())
        except Exception as exc:  # noqa: BLE001
            return [], {
                "kind": "static",
                "reason": "list_models_error",
                "error": str(exc) or type(exc).__name__,
                **({"base_url": resolved_base_url} if resolved_base_url else {}),
            }
        cleaned = _dedupe_sorted_strings(raw_ids)
        if not cleaned:
            return [], {
                "kind": "static",
                "reason": "empty_response",
                **({"base_url": resolved_base_url} if resolved_base_url else {}),
            }
        discovery["count"] = len(cleaned)
        return cleaned, discovery

    def _live_list_models_openai(
        self,
        base_url: str | None,
        discovery: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        env = _env(self._environ)
        api_key = (
            env.get("AETHER_API_KEY")
            or env.get("OPENAI_API_KEY")
            or env.get("ANTHROPIC_AUTH_TOKEN")
            or ""
        )
        if not api_key:
            return [], {
                "kind": "static",
                "reason": "no_credentials",
                "error": "no api key in env",
                **({"base_url": base_url} if base_url else {}),
            }
        if self._http_client_factory is None:
            try:
                provider = build_provider("openai", api_key=api_key, base_url=base_url)
                cleaned_quick = _dedupe_sorted_strings(list(provider.list_models()))
                if cleaned_quick:
                    discovery["count"] = len(cleaned_quick)
                    return cleaned_quick, discovery
            except Exception:
                pass
        effective_base = base_url or "https://api.openai.com/v1"
        last_error: str | None = None
        last_status: int | None = None
        last_body_preview: str | None = None
        for probe in candidate_urls(effective_base):
            try:
                if self._http_client_factory is None:
                    with httpx.Client(timeout=DISCOVERY_TIMEOUT_SEC) as client:
                        resp = client.get(probe, headers={"Authorization": f"Bearer {api_key}"})
                else:
                    with self._http_client_factory() as client:
                        resp = client.get(probe, headers={"Authorization": f"Bearer {api_key}"})
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            last_status = resp.status_code
            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}"
                last_body_preview = truncate_for_log(resp.text)
                continue
            try:
                payload = resp.json()
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"json decode: {exc}"
                last_body_preview = truncate_for_log(resp.text)
                continue
            cleaned = _dedupe_sorted_strings(extract_model_ids(payload))
            if cleaned:
                discovery["count"] = len(cleaned)
                discovery["url"] = probe
                suggested = suggest_base_url(base_url, probe)
                if suggested and suggested != base_url:
                    discovery["suggested_base_url"] = suggested
                    discovery["warning"] = (
                        "discovery succeeded at /v1/models but base_url is missing /v1; "
                        "chat completions will likely 404. "
                        f"Restart with OPENAI_BASE_URL={suggested}"
                    )
                return cleaned, discovery
            last_error = "200 OK but no model ids in response"
            last_body_preview = truncate_for_log(short_payload(payload))
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

    def _resolve_base_url(
        self,
        *,
        base_url: str | None,
        current_session_id: str | None,
        discovery: dict[str, Any],
    ) -> str | None:
        if isinstance(base_url, str) and base_url.strip():
            discovery["base_url_source"] = "param"
            return base_url.strip()
        session_id = current_session_id
        if session_id is None and self._current_session_getter is not None:
            session_id = self._current_session_getter()
        if session_id:
            try:
                record = load_session(session_id)
            except Exception:  # noqa: BLE001
                record = None
            if record is not None and record.base_url:
                discovery["base_url_source"] = "session"
                return record.base_url
        env_url = _env(self._environ).get("OPENAI_BASE_URL") or _env(self._environ).get("ANTHROPIC_BASE_URL")
        if env_url:
            discovery["base_url_source"] = "env"
            return env_url
        discovery["base_url_source"] = "default"
        return None


def candidate_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    seen: set[str] = set()
    candidates: list[str] = []
    for path in DISCOVERY_PATH_PROBES:
        url = f"{base}{path}"
        if url not in seen:
            seen.add(url)
            candidates.append(url)
    if base.endswith("/v1"):
        url = f"{base}/models"
        if url not in seen:
            seen.add(url)
            candidates.append(url)
    return candidates


def extract_model_ids(payload: object) -> list[str]:
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


def suggest_base_url(base_url: str | None, working_probe: str) -> str | None:
    if not working_probe.endswith("/models"):
        return None
    api_root = working_probe[: -len("/models")]
    if not base_url:
        return api_root
    base = base_url.rstrip("/")
    if base == api_root:
        return None
    return api_root


def truncate_for_log(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    snippet = text.replace("\n", " ").strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 1] + "..."


def short_payload(value: object) -> str:
    try:
        return json.dumps(value)[:200]
    except (TypeError, ValueError):
        return str(value)[:200]


def _merge_models(live_ids: list[str], catalog: list[ModelSummary]) -> list[ModelSummary]:
    catalog_by_id = {entry.id: entry for entry in catalog}
    return [
        catalog_by_id.get(model_id, ModelSummary(id=model_id, display_name=model_id))
        for model_id in live_ids
    ]


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


def _discovery_from_dict(payload: dict[str, Any]) -> ModelDiscoveryStatus:
    known = {
        "kind",
        "source",
        "reason",
        "error",
        "base_url",
        "base_url_source",
        "count",
        "url",
        "suggested_base_url",
        "warning",
        "body_preview",
    }
    return ModelDiscoveryStatus(
        kind=str(payload.get("kind") or "static"),
        source=payload.get("source") if isinstance(payload.get("source"), str) else None,
        reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        base_url=payload.get("base_url") if isinstance(payload.get("base_url"), str) else None,
        base_url_source=payload.get("base_url_source") if isinstance(payload.get("base_url_source"), str) else None,
        count=payload.get("count") if isinstance(payload.get("count"), int) else None,
        url=payload.get("url") if isinstance(payload.get("url"), str) else None,
        suggested_base_url=payload.get("suggested_base_url") if isinstance(payload.get("suggested_base_url"), str) else None,
        warning=payload.get("warning") if isinstance(payload.get("warning"), str) else None,
        body_preview=payload.get("body_preview") if isinstance(payload.get("body_preview"), str) else None,
        extra={key: value for key, value in payload.items() if key not in known},
    )


def _runtime_from_values(
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    environ: Mapping[str, str] | None,
):
    try:
        return resolve_main_provider_runtime(
            environ=environ,
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except ValueError as exc:
        raise ServiceValidationError(str(exc)) from exc


def _credential_status_from_public(public: Mapping[str, object]) -> CredentialStatus:
    return CredentialStatus(
        source=str(public.get("source") or "env"),
        name=str(public.get("name") or ""),
        configured=bool(public.get("configured")),
        redacted=str(public.get("redacted") or ""),
    )


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return environ if environ is not None else os.environ


def _require_non_empty(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceValidationError(
            f"provider service requires non-empty string '{field}'",
            details={"field": field},
        )
    return value.strip()


__all__ = [
    "DISCOVERY_PATH_PROBES",
    "MODEL_CATALOG",
    "PROVIDER_DISPLAY",
    "ProviderService",
    "candidate_urls",
    "extract_model_ids",
    "short_payload",
    "suggest_base_url",
    "truncate_for_log",
]
