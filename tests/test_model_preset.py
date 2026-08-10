"""Packaged model-preset registry behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from sgl_eval.cli import build_parser
from sgl_eval.model_preset import (
    ModelPresetRegistryError,
    UnsupportedModelPresetError,
    list_model_preset_ids,
    load_model_preset,
)
from sgl_eval.pipeline import report
from sgl_eval.preset import Sampling, resolve_run_inputs
from sgl_eval.types import GenConfig

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


def _run_args(**overrides) -> argparse.Namespace:
    values = {
        "name": "aime25",
        "preset": None,
        "load_preset_from_model_id": None,
        "base_url": "http://localhost:30000/v1",
        "model": None,
        "n_repeats": None,
        "num_examples": None,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "thinking": None,
        "reasoning_effort": None,
        "chat_template_kwarg": None,
        "seed": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _spec_lookup(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        default_n_repeats=16,
        default_gen=GenConfig(
            temperature=0.0,
            top_p=0.9,
            max_tokens=None,
            chat_template_kwargs={"thinking": False},
        ),
    )


def test_cli_accepts_load_preset_from_model_id() -> None:
    """A misspelled or unwired flag would make the documented command unusable."""
    args = build_parser().parse_args(["run", "aime25", "--load-preset-from-model-id", MODEL_ID])

    assert args.load_preset_from_model_id == MODEL_ID


def test_model_preset_resolves_model_and_generation_defaults() -> None:
    """Ignoring the built-in preset would leave benchmark generation defaults in use."""
    resolved = resolve_run_inputs(
        _run_args(load_preset_from_model_id=MODEL_ID),
        _spec_lookup,
    )

    assert resolved.model == MODEL_ID
    assert resolved.n_repeats == 16
    assert resolved.gen.temperature == 1.0
    assert resolved.gen.top_p == 0.95
    assert resolved.gen.max_tokens == 200000
    assert resolved.gen.reasoning_effort == "max"
    assert resolved.gen.chat_template_kwargs == {"thinking": True}
    assert resolved.model_preset == load_model_preset(MODEL_ID)


def test_model_preset_does_not_supply_base_url() -> None:
    """Deployment-specific host and port must remain an explicit user choice."""
    with pytest.raises(SystemExit, match="model presets do not set endpoints"):
        resolve_run_inputs(
            _run_args(base_url=None, load_preset_from_model_id=MODEL_ID),
            _spec_lookup,
        )


def test_unknown_model_id_exits_with_supported_ids_and_manual_guidance() -> None:
    """Unknown IDs must fail before a run silently uses benchmark defaults."""
    with pytest.raises(SystemExit) as exc_info:
        resolve_run_inputs(
            _run_args(load_preset_from_model_id="deepseek-ai/not-a-model"),
            _spec_lookup,
        )

    message = str(exc_info.value)
    assert "deepseek-ai/not-a-model" in message
    assert MODEL_ID in message
    for flag in (
        "--model",
        "--temperature",
        "--top-p",
        "--max-tokens",
        "--thinking",
        "--reasoning-effort",
    ):
        assert flag in message


def test_cli_then_user_then_model_then_benchmark_precedence(tmp_path: Path) -> None:
    """Each higher-priority layer must replace the next layer without dropping others."""
    user_preset = tmp_path / "user.yaml"
    user_preset.write_text(
        """
benchmark: aime25
endpoint:
  base_url: http://preset:9000/v1
  model: served-user-alias
sampling:
  temperature: 0.6
  top_p: 0.8
  max_tokens: 12345
  thinking: false
  reasoning_effort: low
""",
        encoding="utf-8",
    )

    user_resolved = resolve_run_inputs(
        _run_args(base_url=None, preset=str(user_preset), load_preset_from_model_id=MODEL_ID),
        _spec_lookup,
    )
    assert user_resolved.base_url == "http://preset:9000/v1"
    assert user_resolved.model == "served-user-alias"
    assert user_resolved.gen.temperature == 0.6
    assert user_resolved.gen.top_p == 0.8
    assert user_resolved.gen.max_tokens == 12345
    assert user_resolved.gen.reasoning_effort == "low"
    assert user_resolved.gen.chat_template_kwargs == {"thinking": False}

    cli_resolved = resolve_run_inputs(
        _run_args(
            preset=str(user_preset),
            load_preset_from_model_id=MODEL_ID,
            base_url="http://cli:8000/v1",
            model="served-cli-alias",
            temperature=0.7,
            top_p=0.7,
            max_tokens=777,
            thinking=True,
            reasoning_effort="high",
        ),
        _spec_lookup,
    )
    assert cli_resolved.base_url == "http://cli:8000/v1"
    assert cli_resolved.model == "served-cli-alias"
    assert cli_resolved.gen.temperature == 0.7
    assert cli_resolved.gen.top_p == 0.7
    assert cli_resolved.gen.max_tokens == 777
    assert cli_resolved.gen.reasoning_effort == "high"
    assert cli_resolved.gen.chat_template_kwargs == {"thinking": True}


def test_report_records_model_preset_separately_from_user_preset() -> None:
    """Without source provenance, a metrics file cannot reproduce why defaults changed."""
    model_preset = load_model_preset(MODEL_ID)
    ctx = SimpleNamespace(
        stamp="20260810-120000",
        sampler=SimpleNamespace(model=MODEL_ID),
        inputs=SimpleNamespace(
            base_url="http://localhost:30000/v1",
            gen=GenConfig(),
            model_preset=model_preset,
        ),
        num_threads=64,
        bench_args={},
    )

    meta = report._build_run_meta(ctx)

    assert meta["model_preset"] == {
        "model_id": MODEL_ID,
        "source": "sgl_eval/model_presets.yaml",
    }
    assert "preset" not in meta
