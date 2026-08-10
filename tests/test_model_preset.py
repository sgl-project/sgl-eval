"""Packaged model-preset registry behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from sgl_eval.model_preset import (
    ModelPresetRegistryError,
    UnsupportedModelPresetError,
    list_model_preset_ids,
    load_model_preset,
)
from sgl_eval.preset import Sampling

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_deepseek_v4_flash_0731_resolves_all_recommended_defaults() -> None:
    """A wrong or dropped registry field would send different generation requests."""
    preset = load_model_preset(MODEL_ID)

    assert preset.model_id == MODEL_ID
    assert preset.model == MODEL_ID
    assert preset.sampling == Sampling(
        temperature=1.0,
        top_p=0.95,
        max_tokens=200000,
        thinking=True,
        reasoning_effort="max",
    )


def test_supported_model_ids_exposes_the_complete_registry() -> None:
    """Callers can show every valid exact ID instead of guessing aliases."""
    assert list_model_preset_ids() == [MODEL_ID]


def test_unknown_model_id_reports_requested_and_supported_ids() -> None:
    """An unknown ID must not silently fall through to benchmark defaults."""
    with pytest.raises(UnsupportedModelPresetError) as exc_info:
        load_model_preset("deepseek-ai/not-a-model")

    assert exc_info.value.model_id == "deepseek-ai/not-a-model"
    assert exc_info.value.supported_model_ids == (MODEL_ID,)


def test_malformed_packaged_registry_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad shipped registry must not look like an unsupported user model."""
    registry = tmp_path / "model_presets.yaml"
    registry.write_text("version: 2\nmodels: {}\n", encoding="utf-8")
    monkeypatch.setattr("sgl_eval.model_preset.MODEL_PRESET_PATH", registry)

    with pytest.raises(ModelPresetRegistryError, match="unsupported version 2"):
        list_model_preset_ids()
