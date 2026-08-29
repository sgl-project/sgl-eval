"""Tests for the MMMU-Pro loader (SE-own; HF dataset mocked, no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("PIL.Image")  # pillow ships with datasets

from sgl_eval.evals._mmmu_pro import (  # noqa: E402
    _image_media,
    _normalize_answer,
    _parse_options,
    load_mmmu_pro,
)


def _fake_row(
    idx="r1",
    answer="B",
    n_options=10,
    with_image=True,
    *,
    options=None,
    question=None,
    images=None,
):
    """Build a fake MMMU_Pro row on the real schema.

    ``images`` maps an ``image_n`` column to a PIL image (referenced by
    ``<image n>`` in the question); without it, ``with_image`` toggles a plain
    ``image_1`` column. ``options`` may be a list or the HF literal-string form.
    """
    from PIL import Image

    row = {
        "id": idx,
        "question": question or f"Question {idx}?",
        "options": options if options is not None else [f"opt{i}" for i in range(n_options)],
        "answer": answer,
        "subject": "Art",
        "topic_difficulty": "Easy",
        "img_type": ["Paintings"],
    }
    if images is not None:
        for n, img in images.items():
            row[f"image_{n}"] = img
    elif with_image:
        row["image_1"] = Image.new("RGB", (8, 8))
    return row


def test_normalize_answer_letter():
    assert _normalize_answer("b") == "B"
    assert _normalize_answer("J") == "J"


def test_normalize_answer_int_index():
    assert _normalize_answer(3) == "D"


def test_normalize_answer_none():
    assert _normalize_answer(None) is None


def test_image_media_png():
    from PIL import Image

    media = _image_media(Image.new("RGB", (4, 4)))
    assert len(media) == 1
    assert media[0].kind == "image"
    assert media[0].mime == "image/png"
    assert media[0].data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_image_media_none():
    assert _image_media(None) == []


def test_load_mmmu_pro_end_to_end(monkeypatch):
    import datasets

    fake_ds = [_fake_row("r1", "B", 10), _fake_row("r2", "J", 5, with_image=False)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)

    examples = load_mmmu_pro(num_examples=None)
    assert len(examples) == 2

    ex0 = examples[0]
    assert ex0.id == "r1"
    assert ex0.target == "B"
    assert "Question r1?" in ex0.inputs["problem"]
    assert "A) opt0" in ex0.inputs["problem"]
    assert "J) opt9" in ex0.inputs["problem"]  # 10 options -> A..J
    assert len(ex0.media) == 1
    assert ex0.media[0].mime == "image/png"
    assert ex0.meta["subject"] == "Art"
    assert ex0.meta["difficulty"] == "Easy"  # reads the topic_difficulty column
    assert ex0.meta["image_type"] == ["Paintings"]  # reads the img_type column

    ex1 = examples[1]
    assert ex1.target == "J"
    assert ex1.media == []  # no image -> empty media
    assert "E) opt4" in ex1.inputs["problem"]  # 5 options -> A..E
    assert "F)" not in ex1.inputs["problem"]


def test_load_mmmu_pro_num_examples(monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: [_fake_row(f"r{i}") for i in range(20)]
    )
    examples = load_mmmu_pro(num_examples=3)
    assert len(examples) == 3


def test_options_literal_string_parsed_not_char_split(monkeypatch):
    """HF stores options as a Python-literal string; must parse to a list, not
    split per character (the original 9% bug)."""
    import datasets

    fake_ds = [_fake_row(options="['alpha','beta','gamma']", answer="C", with_image=False)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)

    [ex] = load_mmmu_pro()
    assert "A) alpha" in ex.inputs["problem"]
    assert "B) beta" in ex.inputs["problem"]
    assert "C) gamma" in ex.inputs["problem"]
    assert "D)" not in ex.inputs["problem"]  # 3 options -> A..C only


def test_image_n_marker_picks_column_and_strips(monkeypatch):
    """``<image n>`` picks the image_n column and is rewritten to ``[image]``."""
    import datasets
    from PIL import Image

    fake_ds = [
        _fake_row(
            question="<image 2> What is shown?", answer="A", images={2: Image.new("RGB", (8, 8))}
        )
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)

    [ex] = load_mmmu_pro()
    assert len(ex.media) == 1
    assert "<image" not in ex.inputs["problem"]  # marker stripped
    assert "[image]" in ex.inputs["problem"]  # rewritten to placeholder


def test_two_markers_two_images(monkeypatch):
    """Two ``<image n>`` markers resolve to two distinct images."""
    import datasets
    from PIL import Image

    fake_ds = [
        _fake_row(
            question="<image 1> and <image 2>",
            answer="A",
            images={1: Image.new("RGB", (8, 8)), 2: Image.new("RGB", (8, 8))},
        )
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)
    [ex] = load_mmmu_pro()
    assert len(ex.media) == 2


def test_missing_referenced_image_warns_and_skips(monkeypatch):
    """A ``<image n>`` whose image_n column is missing is a data error -> skip."""
    import datasets

    fake_ds = [_fake_row(question="<image 3> here?", answer="A", with_image=False)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)

    with pytest.warns(UserWarning, match="skipping MMMU-Pro row"):
        examples = load_mmmu_pro()
    assert examples == []


def test_options_not_list_warns_and_skips(monkeypatch):
    """literal_eval returning a non-list (a bare quoted string) -> skip, not
    per-character split (re-opens the 9% bug)."""
    import datasets

    fake_ds = [_fake_row(options="'just a string'", answer="A", with_image=False)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)

    with pytest.warns(UserWarning, match="skipping MMMU-Pro row"):
        examples = load_mmmu_pro()
    assert examples == []


def test_empty_options_warns_and_skips(monkeypatch):
    import datasets

    fake_ds = [_fake_row(options=[], answer="A", with_image=False)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake_ds)

    with pytest.warns(UserWarning, match="skipping MMMU-Pro row"):
        examples = load_mmmu_pro()
    assert examples == []


def test_parse_options_direct():
    assert _parse_options({"options": "['x','y']"}) == ["x", "y"]
    assert _parse_options({"options": ["a", "b"]}) == ["a", "b"]


def test_mmmu_pro_registered():
    """MMMU-Pro registers as a multichoice benchmark."""
    from sgl_eval.registry import get

    spec = get("mmmu_pro")
    assert spec.category == "multichoice"
    assert spec.default_n_repeats == 1
    assert "MMMU-Pro" in spec.description


def test_mmmu_pro_prompt_packaged():
    """The SE-own 10-choice prompt must ship in the installed package -- the
    vendored prompts are declared in pyproject package-data, but this one is
    not, and ``resolve_prompt`` returns a path ``render_prompt`` reads at
    run time. A missing file crashes the first MMMU-Pro run, not import."""
    from sgl_eval.evals._prompts import resolve_prompt

    assert resolve_prompt("mmmu-pro-cot").exists()


def test_mmmu_pro_prompt_is_not_shadowed_by_the_vendored_10choice():
    """`mcq-10choices` is MMLU-Pro's vendored prompt and hardcodes A-J;
    MMMU-Pro ships 4-10 options per question and needs its own. Since
    ``resolve_prompt`` prefers vendored, sharing the basename would silently
    swap MMMU-Pro's prompt for MMLU-Pro's."""
    from sgl_eval.evals._prompts import resolve_prompt

    assert resolve_prompt("mmmu-pro-cot") != resolve_prompt("mcq-10choices")
    mmmu = resolve_prompt("mmmu-pro-cot").read_text()
    assert "$LETTER" in mmmu and "A/B/C/D/E/F/G/H/I/J" not in mmmu


def test_max_tokens_not_pinned_for_any_benchmark():
    """No benchmark pins max_tokens; every benchmark keeps the NS-aligned
    ``max_tokens=None`` (server picks the cap)."""
    from sgl_eval.registry import get

    for name in ("gsm8k", "aime24", "aime25", "mmlu", "gpqa", "mmmu_pro"):
        assert get(name).default_gen.max_tokens is None, f"{name} should not pin max_tokens"
