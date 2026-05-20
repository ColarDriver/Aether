"""Auxiliary provider/model slot resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any, Literal

from aether.config.provider_runtime import (
    ProviderFamily,
    normalize_provider_family,
    provider_choice,
)


AuxiliarySlotName = Literal["subagent", "compression", "verifier", "title"]

_SLOTS: set[str] = {"subagent", "compression", "verifier", "title"}
_SLOT_ENV_PREFIX: dict[str, str] = {
    "subagent": "AETHER_AUX_SUBAGENT",
    "compression": "AETHER_AUX_COMPRESSION",
    "verifier": "AETHER_AUX_VERIFIER",
    "title": "AETHER_AUX_TITLE",
}
_MODEL_ALIASES: dict[str, dict[ProviderFamily, str]] = {
    "gpt": {
        "codex": "gpt",
        "openai-compatible": "gpt",
        "claude": "gpt",
    },
    "sonnet": {
        "claude": "sonnet",
        "codex": "sonnet",
        "openai-compatible": "sonnet",
    },
}


@dataclass(frozen=True, slots=True)
class AuxiliarySlotConfig:
    slot: AuxiliarySlotName
    provider_family: ProviderFamily
    provider_name: str
    model: str
    inherited: bool = False
    source: str = "default"

    def public_metadata(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "provider_family": self.provider_family,
            "provider_name": self.provider_name,
            "model": self.model,
            "inherited": self.inherited,
            "source": self.source,
        }


def resolve_auxiliary_slot(
    slot: str,
    *,
    config: Any | None = None,
    environ: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    inherit_if_unconfigured: bool = False,
) -> AuxiliarySlotConfig:
    """Resolve provider/model for an auxiliary model slot."""

    slot_name = _normalize_slot(slot)
    env = environ if environ is not None else os.environ
    prefix = _SLOT_ENV_PREFIX[slot_name]
    raw_provider = _first_non_empty(
        provider,
        _config_slot_str(config, slot_name, "provider"),
        env.get(f"{prefix}_PROVIDER"),
        _deprecated_subagent_provider(env, slot_name),
    )
    raw_model = _first_non_empty(
        model,
        _config_slot_str(config, slot_name, "model"),
        env.get(f"{prefix}_MODEL"),
        _deprecated_subagent_model(env, slot_name),
    )
    has_explicit_slot = bool(provider or model)
    has_config_slot = _has_slot_config(config, slot_name)
    has_env_slot = bool(raw_provider or raw_model)
    source = "explicit" if has_explicit_slot else "config" if has_config_slot else "env" if has_env_slot else "default"

    if raw_provider is None:
        raw_provider = _first_non_empty(env.get("AETHER_PROVIDER"))
        if raw_provider and source == "default":
            source = "env"

    if raw_provider is None and raw_model is None and inherit_if_unconfigured:
        return AuxiliarySlotConfig(
            slot=slot_name,
            provider_family="openai-compatible",
            provider_name="openai",
            model="inherit",
            inherited=True,
            source="inherit",
        )

    family = normalize_provider_family(raw_provider) or "openai-compatible"
    choice = provider_choice(family)
    resolved_model = _resolve_model_alias(raw_model, family, choice.default_model)
    return AuxiliarySlotConfig(
        slot=slot_name,
        provider_family=family,
        provider_name=choice.provider_name,
        model=resolved_model,
        inherited=False,
        source=source,
    )


def resolve_subagent_slot(
    *,
    config: Any | None = None,
    environ: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    inherit_if_unconfigured: bool = False,
) -> AuxiliarySlotConfig:
    return resolve_auxiliary_slot(
        "subagent",
        config=config,
        environ=environ,
        provider=provider,
        model=model,
        inherit_if_unconfigured=inherit_if_unconfigured,
    )


def _normalize_slot(slot: str) -> AuxiliarySlotName:
    normalized = slot.strip().lower().replace("-", "_")
    if normalized not in _SLOTS:
        raise ValueError(f"unknown auxiliary slot: {slot!r}")
    return normalized  # type: ignore[return-value]


def _resolve_model_alias(raw_model: str | None, family: ProviderFamily, default: str) -> str:
    if raw_model is None:
        if family == "openai-compatible":
            return "gpt"
        if family == "claude":
            return "sonnet"
        if family == "codex":
            return "gpt"
        return default
    normalized = raw_model.strip()
    aliases = _MODEL_ALIASES.get(normalized.lower())
    if aliases is not None:
        return aliases.get(family, normalized)
    return normalized


def _config_slot_str(config: Any | None, slot: str, name: str) -> str | None:
    if config is None:
        return None
    direct = getattr(config, f"aux_{slot}_{name}", None)
    if isinstance(direct, str):
        return direct
    auxiliary = getattr(config, "auxiliary", None)
    if isinstance(auxiliary, Mapping):
        slot_config = auxiliary.get(slot)
        if isinstance(slot_config, Mapping):
            value = slot_config.get(name)
            if isinstance(value, str):
                return value
    return None


def _has_slot_config(config: Any | None, slot: str) -> bool:
    return _config_slot_str(config, slot, "provider") is not None or _config_slot_str(config, slot, "model") is not None


def _deprecated_subagent_provider(env: Mapping[str, str], slot: str) -> str | None:
    if slot != "subagent":
        return None
    return env.get("AETHER_SUBAGENT_PROVIDER")


def _deprecated_subagent_model(env: Mapping[str, str], slot: str) -> str | None:
    if slot != "subagent":
        return None
    return env.get("AETHER_SUBAGENT_MODEL")


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


__all__ = [
    "AuxiliarySlotConfig",
    "AuxiliarySlotName",
    "resolve_auxiliary_slot",
    "resolve_subagent_slot",
]
