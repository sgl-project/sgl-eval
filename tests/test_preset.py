"""Preset schema, path resolution, and override-priority tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sgl_eval.preset import (
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


def test_full_preset_round_trip(tmp_path: Path) -> None:
    """All sections parse and absent fields default cleanly. Covers the
    minimal-preset case via the inverse: untouched sections stay default."""
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
    assert preset.endpoint == Endpoint(base_url="http://host:30000/v1", model="dsv3.2")
    assert preset.sampling == Sampling(temperature=1.0, top_p=0.95, max_tokens=32768, thinking=True)
    assert preset.expected == Expected(score=0.85)
    assert preset.n_repeats == 16
    assert preset.num_examples == 30

    # And a minimal preset leaves everything default.
    minimal = tmp_path / "min.yaml"
    minimal.write_text("benchmark: aime24\n")
    m = load_preset(str(minimal))
    assert (m.endpoint, m.sampling, m.expected, m.n_repeats, m.num_examples) == (
        Endpoint(),
        Sampling(),
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "yaml_body, error",
    [
        ("n_repeats: 4\n", "missing required field 'benchmark'"),
        ("- benchmark: aime24\n", "top-level must be a mapping"),
        ("benchmark: aime24\nsampling: 1.0\n", "sampling: must be a mapping"),
        ("benchmark: aime24\nfoo: 1\n", "unknown fields"),
        ("benchmark: aime24\nendpoint:\n  bogus_url: x\n", "unknown fields"),
        ("benchmark: aime24\nsampling:\n  temp: 1.0\n", "unknown fields"),
        ("benchmark: aime24\nexpected:\n  threshold: 0.8\n", "unknown fields"),
    ],
)
def test_schema_rejects_bad_shapes(tmp_path: Path, yaml_body: str, error: str) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_body)
    with pytest.raises(ValueError, match=error):
        load_preset(str(p))


def test_resolve_preset_path_with_slash_treated_as_path() -> None:
    p = resolve_preset_path("./presets/foo.yaml")
    assert p == Path("./presets/foo.yaml").expanduser()


def test_load_missing_preset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="preset not found"):
        load_preset(str(tmp_path / "nope.yaml"))


def test_list_presets_finds_yaml_and_yml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sgl_eval.preset.PRESET_ROOT", tmp_path)
    (tmp_path / "a.yaml").write_text("benchmark: x")
    (tmp_path / "b.yml").write_text("benchmark: y")
    (tmp_path / "ignore.txt").write_text("nope")
    found = [p.name for p in list_presets()]
    assert "a.yaml" in found
    assert "b.yml" in found
    assert "ignore.txt" not in found


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


def _args(**overrides) -> argparse.Namespace:
    base = dict(temperature=None, top_p=None, max_tokens=None, thinking=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _preset_with(**sampling) -> Preset:
    return Preset(benchmark="x", sampling=Sampling(**sampling))


def test_apply_to_gen_priority_chain() -> None:
    """CLI > preset > spec default. Covers all three rungs in one pass."""
    default = GenConfig(temperature=0.0, top_p=0.95)

    # No preset, no CLI -> default
    gen = apply_to_gen(default, preset=None, args=_args())
    assert (gen.temperature, gen.top_p) == (0.0, 0.95)

    # Preset overrides default; un-overridden fields fall through
    gen = apply_to_gen(default, preset=_preset_with(temperature=1.0), args=_args())
    assert (gen.temperature, gen.top_p) == (1.0, 0.95)

    # CLI beats preset
    gen = apply_to_gen(default, preset=_preset_with(temperature=1.0), args=_args(temperature=0.6))
    assert gen.temperature == 0.6


def test_apply_to_gen_thinking_priority() -> None:
    """``thinking`` is the only sampling field that lives under
    ``chat_template_kwargs``; verify it follows the same priority chain."""
    default = GenConfig(chat_template_kwargs={"thinking": False})
    # CLI explicit beats preset
    gen = apply_to_gen(default, _preset_with(thinking=True), _args(thinking=False))
    assert gen.chat_template_kwargs == {"thinking": False}
    # Preset beats default
    gen = apply_to_gen(default, _preset_with(thinking=True), _args())
    assert gen.chat_template_kwargs == {"thinking": True}
