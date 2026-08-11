"""Tests for ``mmmu_pro_vision`` -- the strictly-vendored MMMU-Pro row.

Covers the three things that make it different from every other ``prepare``
benchmark: metadata is derived from upstream rather than declared, the media
sidecar has to leave ``_vendored``, and the prompt places its image first.
"""

from __future__ import annotations

import json
import types

import pytest

from sgl_eval.evals import _loader
from sgl_eval.registry import get


def test_registered_as_multichoice():
    spec = get("mmmu_pro_vision")
    assert spec.category == "multichoice"
    assert spec.default_n_repeats == 1
    assert spec.default_gen.max_tokens is None


def test_metadata_is_derived_not_declared():
    """The row must NOT hand-mirror upstream's choices: ``metrics_type`` and the
    prompt basename come from the vendored ``__init__.py``. Declaring either in
    ``_TABLE`` would silently decouple us from upstream."""
    from sgl_eval.evals._registry import _TABLE, _resolve_upstream_metadata

    [entry] = [e for e in _TABLE if e["name"] == "mmmu_pro_vision"]
    assert "metrics_type" not in entry
    assert "prompt" not in entry

    metrics_type, prompt_basename = _resolve_upstream_metadata("mmmu_pro_vision")
    assert metrics_type == "multichoice"
    # Upstream's ++prompt_config=vlm/mmmu-pro -- hyphenated, hence the
    # hyphenated vendored yaml basename.
    assert prompt_basename == "mmmu-pro"


def test_vendored_prompt_places_image_first():
    """The vision config renders the question into the image, so the text (an
    answer-format instruction) must follow it."""
    from sgl_eval.evals._prompts import prompt_media_config, vendored_prompt

    path = vendored_prompt("mmmu-pro")
    assert path.exists()
    assert prompt_media_config(path)["image_position"] == "before"


def test_vendored_dataset_module_is_upstream_verbatim():
    """The dst dir is renamed (hyphen -> underscore) but the file must still be
    upstream's, banner included."""
    from sgl_eval import VENDORED_NS_ROOT

    text = (VENDORED_NS_ROOT / "dataset" / "mmmu_pro_vision" / "__init__.py").read_text()
    assert "Source: nemo_skills/dataset/mmmu-pro/__init__.py" in text
    assert 'METRICS_TYPE = "multichoice"' in text
    assert "++prompt_config=vlm/mmmu-pro" in text


def _fake_prepare_module(tmp_path, rows, image_names):
    """A stand-in for the vendored ``prepare`` module: ``save_data`` writes a
    jsonl plus an ``images/`` sidecar next to ``__file__``, exactly as
    upstream's does."""
    vendored_dir = tmp_path / "vendored_pkg"
    vendored_dir.mkdir()
    mod = types.ModuleType("fake_prepare")
    mod.__file__ = str(vendored_dir / "prepare.py")

    def save_data(split):
        images_dir = vendored_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for name in image_names:
            (images_dir / name).write_bytes(b"\x89PNG" + name.encode())
        with (vendored_dir / f"{split}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    mod.save_data = save_data
    return mod, vendored_dir


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    rows = [
        {"id": "v1", "problem": "A) x\nB) y", "expected_answer": "A", "image_path": "images/a.png"},
        {"id": "v2", "problem": "A) p\nB) q", "expected_answer": "B", "image_path": "images/b.png"},
    ]
    mod, vendored_dir = _fake_prepare_module(tmp_path, rows, ["a.png", "b.png"])
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)
    return tmp_path, vendored_dir


def test_media_sidecar_moves_out_of_vendored(prepared):
    tmp_path, vendored_dir = prepared
    loader = _loader.load_via_prepare(
        "mmmu_pro_vision", ["test"], media_dir="images", media_field="image_path"
    )
    examples = loader(None)

    cache_dir = tmp_path / "cache" / "mmmu_pro_vision"
    assert (cache_dir / "test.jsonl").is_file()
    assert (cache_dir / "images" / "a.png").is_file()
    # _vendored must not retain generated data.
    assert not (vendored_dir / "images").exists()

    assert len(examples) == 2
    assert [m.mime for ex in examples for m in ex.media] == ["image/png", "image/png"]
    assert examples[0].media[0].data == b"\x89PNGa.png"
    assert examples[0].target == "A"


def test_media_path_kept_in_meta(prepared):
    loader = _loader.load_via_prepare(
        "mmmu_pro_vision", ["test"], media_dir="images", media_field="image_path"
    )
    [ex, _] = loader(None)
    assert ex.meta["image_path"] == "images/a.png"


def test_num_examples_truncates_before_reading_media(prepared):
    loader = _loader.load_via_prepare(
        "mmmu_pro_vision", ["test"], media_dir="images", media_field="image_path"
    )
    examples = loader(1)
    assert len(examples) == 1
    assert examples[0].id == "v1"


def test_second_load_serves_from_cache(prepared):
    """A cached run must not re-invoke save_data (whose sidecar is already
    gone from _vendored)."""
    tmp_path, vendored_dir = prepared
    loader = _loader.load_via_prepare(
        "mmmu_pro_vision", ["test"], media_dir="images", media_field="image_path"
    )
    loader(None)
    again = loader(None)
    assert len(again) == 2
    assert again[0].media[0].data == b"\x89PNGa.png"


def test_missing_media_file_raises(tmp_path, monkeypatch):
    """A row pointing at a nonexistent image is a corrupt cache -- it must fail
    loudly, never degrade to a text-only sample."""
    rows = [
        {"id": "v1", "problem": "A) x", "expected_answer": "A", "image_path": "images/gone.png"}
    ]
    mod, _ = _fake_prepare_module(tmp_path, rows, ["a.png"])
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)

    loader = _loader.load_via_prepare(
        "mmmu_pro_vision", ["test"], media_dir="images", media_field="image_path"
    )
    with pytest.raises(FileNotFoundError, match="gone.png"):
        loader(None)


def test_text_benchmarks_keep_empty_media(tmp_path, monkeypatch):
    """No media_field -> Example.media stays empty (gsm8k / mmlu / gpqa path)."""
    rows = [{"id": "t1", "problem": "2+2?", "expected_answer": "4"}]
    mod, _ = _fake_prepare_module(tmp_path, rows, [])
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _name: mod)

    [ex] = _loader.load_via_prepare("gsm8k", ["test"])(None)
    assert ex.media == []
