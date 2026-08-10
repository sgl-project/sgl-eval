"""Strict loader for repository-owned model generation presets."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sgl_eval.preset import Sampling, reasoning_effort

MODEL_PRESET_PATH = Path(__file__).with_name("model_presets.yaml")
MODEL_PRESET_SOURCE = "sgl_eval/model_presets.yaml"
_REGISTRY_VERSION = 1


class ModelPresetRegistryError(ValueError):
    """The packaged registry is missing or violates its schema."""


class UnsupportedModelPresetError(ValueError):
    """The requested exact model ID has no built-in preset."""

    def __init__(self, model_id: str, supported_model_ids: tuple[str, ...]) -> None:
        self.model_id = model_id
        self.supported_model_ids = supported_model_ids
        supported = ", ".join(supported_model_ids) or "(none)"
        super().__init__(
            f"no built-in model preset for {model_id!r}; supported model IDs: {supported}"
        )


@dataclass(frozen=True)
class ModelPreset:
    """One built-in model identity and its recommended generation defaults."""

    model_id: str
    model: str
    sampling: Sampling


def list_model_preset_ids() -> list[str]:
    """Return every supported exact model ID in stable order."""
    return sorted(_load_registry())


def load_model_preset(model_id: str) -> ModelPreset:
    """Load one exact model ID or raise ``UnsupportedModelPresetError``."""
    registry = _load_registry()
    if model_id not in registry:
        raise UnsupportedModelPresetError(model_id, tuple(sorted(registry)))
    return registry[model_id]


def make_model_preset_meta_block(preset: ModelPreset | None) -> dict[str, str] | None:
    """Return stable ``metrics.json`` provenance for a built-in preset."""
    if preset is None:
        return None
    return {"model_id": preset.model_id, "source": MODEL_PRESET_SOURCE}


def _load_registry() -> dict[str, ModelPreset]:
    try:
        raw = yaml.safe_load(MODEL_PRESET_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelPresetRegistryError(
            f"model preset registry not found: {MODEL_PRESET_PATH}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ModelPresetRegistryError(f"invalid YAML in {MODEL_PRESET_SOURCE}: {exc}") from exc

    root = _mapping(raw, MODEL_PRESET_SOURCE)
    _reject_unknown(root, {"version", "models"}, MODEL_PRESET_SOURCE)
    version = root.get("version")
    if version != _REGISTRY_VERSION:
        raise ModelPresetRegistryError(
            f"{MODEL_PRESET_SOURCE}: unsupported version {version!r}; expected {_REGISTRY_VERSION}"
        )
    models = _mapping(root.get("models"), f"{MODEL_PRESET_SOURCE}.models")

    parsed: dict[str, ModelPreset] = {}
    for model_id, entry_raw in models.items():
        if not isinstance(model_id, str) or not model_id:
            raise ModelPresetRegistryError(
                f"{MODEL_PRESET_SOURCE}.models: model IDs must be non-empty strings"
            )
        source = f"{MODEL_PRESET_SOURCE}.models[{model_id!r}]"
        entry = _mapping(entry_raw, source)
        _reject_unknown(entry, {"model", "sampling"}, source)
        model = entry.get("model")
        if not isinstance(model, str) or not model:
            raise ModelPresetRegistryError(f"{source}.model: must be a non-empty string")
        sampling = _load_sampling(entry.get("sampling"), f"{source}.sampling")
        parsed[model_id] = ModelPreset(model_id=model_id, model=model, sampling=sampling)
    return parsed


def _load_sampling(raw: Any, source: str) -> Sampling:
    values = _mapping(raw, source)
    allowed = {field.name for field in dataclasses.fields(Sampling)}
    _reject_unknown(values, allowed, source)

    for key in ("temperature", "top_p"):
        value = values.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ModelPresetRegistryError(f"{source}.{key}: must be a number")
    max_tokens = values.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ModelPresetRegistryError(f"{source}.max_tokens: must be a positive integer")
    thinking = values.get("thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise ModelPresetRegistryError(f"{source}.thinking: must be a boolean")
    kwargs = values.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ModelPresetRegistryError(f"{source}.chat_template_kwargs: must be a mapping")
    effort = values.get("reasoning_effort")
    if effort is not None:
        try:
            values["reasoning_effort"] = reasoning_effort(str(effort))
        except argparse.ArgumentTypeError as exc:
            raise ModelPresetRegistryError(f"{source}.reasoning_effort: {exc}") from exc
    return Sampling(**values)


def _mapping(raw: Any, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ModelPresetRegistryError(f"{source}: must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def _reject_unknown(raw: dict[str, Any], allowed: set[str], source: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ModelPresetRegistryError(f"{source}: unknown fields {sorted(unknown)}")
