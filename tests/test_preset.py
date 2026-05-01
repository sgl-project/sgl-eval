"""Preset schema, path resolution, and override-priority tests.

Covers what users depend on:
  - YAML loads into a strict schema (typo'd keys raise, not silently dropped)
  - ``benchmark`` is required; everything else optional
  - resolve_preset_path: name vs explicit path
  - CLI > preset > spec default priority via the ``pick`` chain inside
    ``apply_to_gen`` (re-implemented here so we don't have to spin up the
    full CLI to test the merge logic)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sgl_eval.preset import (
    PRESET_ROOT,
    Endpoint,
    Expected,
    Preset,
    Sampling,
    apply_to_gen,
    list_presets,
    load_preset,
    pick,
    resolve_preset_path,
)
from sgl_eval.types import GenConfig

# ---------- schema ----------


def test_minimal_preset_only_benchmark(tmp_path: Path) -> None:
    p = tmp_path / "minimal.yaml"
    p.write_text("benchmark: aime24\n")
    preset = load_preset(str(p))
    assert preset.benchmark == "aime24"
    assert preset.endpoint == Endpoint()
    assert preset.sampling == Sampling()
    assert preset.expected is None
    assert preset.n_repeats is None
    assert preset.num_examples is None


def test_full_preset(tmp_path: Path) -> None:
    p = tmp_path / "full.yaml"
    p.write_text("""
benchmark: aime24
endpoint:
  base_url: http://host:30000/v1
  model: dsv3.2
n_repeats: 16
num_examples: 30
sampling:
  temperature: 1.0
  top_p: 0.95
  max_tokens: 32768
  thinking: true
expected:
  score: 0.85
""")
    preset = load_preset(str(p))
    assert preset.benchmark == "aime24"
    assert preset.endpoint.base_url == "http://host:30000/v1"
    assert preset.endpoint.model == "dsv3.2"
    assert preset.n_repeats == 16
    assert preset.num_examples == 30
    assert preset.sampling.temperature == 1.0
    assert preset.sampling.thinking is True
    assert preset.expected is not None
    assert preset.expected.score == 0.85


def test_missing_benchmark_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_bench.yaml"
    p.write_text("n_repeats: 4\n")
    with pytest.raises(ValueError, match="missing required field 'benchmark'"):
        load_preset(str(p))


@pytest.mark.parametrize(
    "yaml_body, error_loc",
    [
        ("benchmark: aime24\nfoo: 1\n", "unknown fields"),
        ("benchmark: aime24\nendpoint:\n  bogus_url: x\n", "unknown fields"),
        ("benchmark: aime24\nsampling:\n  temp: 1.0\n", "unknown fields"),
        ("benchmark: aime24\nexpected:\n  threshold: 0.8\n", "unknown fields"),
    ],
)
def test_typoed_field_raises(tmp_path: Path, yaml_body: str, error_loc: str) -> None:
    p = tmp_path / "typo.yaml"
    p.write_text(yaml_body)
    with pytest.raises(ValueError, match=error_loc):
        load_preset(str(p))


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- benchmark: aime24\n")
    with pytest.raises(ValueError, match="top-level must be a mapping"):
        load_preset(str(p))


def test_section_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad_sampling.yaml"
    p.write_text("benchmark: aime24\nsampling: 1.0\n")
    with pytest.raises(ValueError, match="sampling: must be a mapping"):
        load_preset(str(p))


# ---------- path resolution ----------


def test_resolve_preset_path_by_name() -> None:
    assert resolve_preset_path("foo") == PRESET_ROOT / "foo.yaml"


def test_resolve_preset_path_with_slash_treated_as_path() -> None:
    p = resolve_preset_path("./presets/foo.yaml")
    assert p == Path("./presets/foo.yaml").expanduser()


def test_resolve_preset_path_yml_extension() -> None:
    p = resolve_preset_path("/abs/path/foo.yml")
    assert p == Path("/abs/path/foo.yml")


def test_resolve_preset_path_expands_home() -> None:
    p = resolve_preset_path("~/bar/foo.yaml")
    assert p.is_absolute()


def test_load_missing_preset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="preset not found"):
        load_preset(str(tmp_path / "nope.yaml"))


def test_list_presets_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sgl_eval.preset.PRESET_ROOT", tmp_path / "nonexistent")
    assert list_presets() == []


def test_list_presets_finds_yaml_and_yml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sgl_eval.preset.PRESET_ROOT", tmp_path)
    (tmp_path / "a.yaml").write_text("benchmark: x")
    (tmp_path / "b.yml").write_text("benchmark: y")
    (tmp_path / "ignore.txt").write_text("nope")
    found = [p.name for p in list_presets()]
    assert "a.yaml" in found
    assert "b.yml" in found
    assert "ignore.txt" not in found


# ---------- pick semantics ----------


def test_pick_first_non_none_wins() -> None:
    assert pick(None, "preset", "default") == "preset"
    assert pick("cli", "preset", "default") == "cli"
    assert pick(None, None, "default") == "default"
    assert pick(None, None, None) is None


def test_pick_treats_zero_as_set() -> None:
    """0 / 0.0 / False are valid signals (e.g. temperature=0.0 greedy)
    -- must not be confused with ``None`` the way ``or`` would."""
    assert pick(0.0, 1.0, 2.0) == 0.0
    assert pick(False, True, True) is False
    assert pick(0, 100, 200) == 0


# ---------- apply_to_gen priority ----------


def _args(**overrides) -> argparse.Namespace:
    base = dict(temperature=None, top_p=None, max_tokens=None, thinking=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _preset_with(**sampling) -> Preset:
    return Preset(benchmark="x", sampling=Sampling(**sampling))


def test_apply_to_gen_no_preset_falls_back_to_default() -> None:
    default = GenConfig(temperature=0.0, top_p=0.95, max_tokens=None)
    gen = apply_to_gen(default, preset=None, args=_args())
    assert gen.temperature == 0.0
    assert gen.top_p == 0.95
    assert gen.max_tokens is None


def test_apply_to_gen_preset_overrides_default() -> None:
    default = GenConfig(temperature=0.0, top_p=0.95)
    gen = apply_to_gen(default, preset=_preset_with(temperature=1.0), args=_args())
    assert gen.temperature == 1.0
    assert gen.top_p == 0.95  # preset didn't override, falls to default


def test_apply_to_gen_cli_overrides_preset() -> None:
    default = GenConfig(temperature=0.0)
    gen = apply_to_gen(
        default,
        preset=_preset_with(temperature=1.0),
        args=_args(temperature=0.6),
    )
    assert gen.temperature == 0.6


def test_apply_to_gen_thinking_priority() -> None:
    default = GenConfig(chat_template_kwargs={"thinking": False})
    # CLI explicit beats preset
    gen = apply_to_gen(default, _preset_with(thinking=True), _args(thinking=False))
    assert gen.chat_template_kwargs == {"thinking": False}
    # Preset beats default
    gen = apply_to_gen(default, _preset_with(thinking=True), _args())
    assert gen.chat_template_kwargs == {"thinking": True}
    # No preset, no CLI -> default kept
    gen = apply_to_gen(default, None, _args())
    assert gen.chat_template_kwargs == {"thinking": False}


def test_apply_to_gen_preserves_default_only_fields() -> None:
    """``reasoning_effort`` / ``extra_body`` / ``seed`` / ``system_message``
    aren't preset/CLI overridable yet -- the default must round-trip."""
    default = GenConfig(
        reasoning_effort="high",
        extra_body={"foo": "bar"},
        seed=42,
        system_message="hello",
    )
    gen = apply_to_gen(default, _preset_with(temperature=1.0), _args())
    assert gen.reasoning_effort == "high"
    assert gen.extra_body == {"foo": "bar"}
    assert gen.seed == 42
    assert gen.system_message == "hello"


# ---------- Expected merge ----------


def test_preset_expected_optional(tmp_path: Path) -> None:
    p = tmp_path / "no_expected.yaml"
    p.write_text("benchmark: aime24\n")
    preset = load_preset(str(p))
    assert preset.expected is None


def test_preset_expected_score(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text("benchmark: aime24\nexpected:\n  score: 0.85\n")
    preset = load_preset(str(p))
    assert preset.expected == Expected(score=0.85)
