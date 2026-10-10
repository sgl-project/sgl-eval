"""Tests for the MindCube benchmark (SE-own; dataset on disk mocked, no network)."""

from __future__ import annotations

import json

import pytest

from sgl_eval.evals._mindcube import (
    QUESTION_HEADER,
    RAW_QA_INSTRUCTION,
    _interleave_rows_by_setting,
    aggregate_mindcube,
    build_prompt,
    extract_answer,
    load_mindcube_examples,
    make_sample_fn,
    make_score_one_fn,
    setting_of,
)
from sgl_eval.types import Example, ExampleResult, GenConfig, Sample

# ---------- official extractor ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<answer>C. Light purple sofa</answer>", "C"),
        (
            "I think A. TV is wrong, so <answer>B. Wooden dining table</answer>",
            "B",
        ),  # last "X." wins
        ("Answer: D", "D"),
        ("The answer is B", "B"),
        ("My answer is A.", "A"),
        ("**C**", "C"),
        ("<Answer>My answer is D</Answer>", "D"),
        ("Some reasoning\nB\n", "B"),
        ("no option letter here", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_setting_of():
    assert setting_of("among_group693_q1_5_2") == "among"
    assert setting_of("around_group12_q3") == "around"
    assert setting_of("rotation_group1_q0") == "rotation"
    assert setting_of("translation_x") == "translation"
    assert setting_of("weird") == "other"
    assert setting_of("") == "other"


# ---------- prompt ----------


def test_build_prompt_is_official_raw_qa():
    q = "Based on these images ... A. TV B. Sofa C. Table D. Curtains"
    prompt = build_prompt(q)
    assert prompt == RAW_QA_INSTRUCTION + "\n" + QUESTION_HEADER + q
    assert prompt.startswith("[Task]\n")
    assert "<answer>A. Above</answer>" in prompt
    assert prompt.endswith("[Question]\n" + q)


# ---------- loader ----------


def _write_fake_dataset(root, rows):
    from PIL import Image

    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    with (raw / "MindCube_tinybench.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for row in rows:
        for rel in row["images"]:
            p = root / "data" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6)).save(p, format="PNG" if p.suffix == ".png" else "JPEG")


def _row(idx, setting, n_images, answer="B"):
    return {
        "id": f"{setting}_group{idx}_q0",
        "category": ["x"],
        "type": "1_frame",
        "meta_info": [],
        "question": f"Q{idx}? A. one B. two C. three D. four",
        "images": [
            f"other_all_image/{setting}/g{idx}/view{k}.{'jpg' if k % 2 else 'png'}"
            for k in range(n_images)
        ],
        "gt_answer": answer,
    }


def test_load_from_env_dir(tmp_path, monkeypatch):
    rows = [_row(0, "among", 4), _row(1, "around", 2, "A"), _row(2, "rotation", 3)]
    _write_fake_dataset(tmp_path, rows)
    monkeypatch.setenv("SGL_EVAL_MINDCUBE_DIR", str(tmp_path))
    examples = load_mindcube_examples(None)
    assert [e.id for e in examples] == [r["id"] for r in rows]
    ex = examples[0]
    assert ex.target == "B"
    assert ex.meta["setting"] == "among"
    assert len(ex.media) == 4
    assert ex.media[0].kind == "image" and ex.media[0].mime == "image/png"
    assert ex.media[1].mime == "image/jpeg"
    assert ex.media[0].data[:8] == b"\x89PNG\r\n\x1a\n"
    assert ex.inputs["problem"].endswith(rows[0]["question"])


def test_load_skips_row_with_missing_image(tmp_path, monkeypatch):
    rows = [_row(0, "among", 2), _row(1, "around", 2)]
    _write_fake_dataset(tmp_path, rows)
    (tmp_path / "data" / rows[1]["images"][0]).unlink()
    monkeypatch.setenv("SGL_EVAL_MINDCUBE_DIR", str(tmp_path))
    with pytest.warns(UserWarning, match="missing image"):
        examples = load_mindcube_examples(None)
    assert [e.id for e in examples] == [rows[0]["id"]]


def test_env_dir_without_bench_jsonl_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("SGL_EVAL_MINDCUBE_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        load_mindcube_examples(None)


def test_num_examples_interleaves_settings():
    rows = [_row(i, "among", 1) for i in range(5)] + [_row(i, "around", 1) for i in range(5)]
    picked = _interleave_rows_by_setting(rows, 4)
    assert [setting_of(r["id"]) for r in picked] == ["among", "around", "among", "around"]


# ---------- sample / score ----------


class _FakeSampler:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, gen):
        self.calls.append(messages)
        return Sample(text="<answer>B. two</answer>")


def test_sample_fn_puts_images_before_text():
    from sgl_eval.types import MediaItem

    ex = Example(
        id="among_g0_q0",
        inputs={"problem": build_prompt("Q? A. one B. two")},
        target="B",
        meta={"setting": "among"},
        media=[MediaItem(kind="image", data=b"\x89PNG", mime="image/png")] * 2,
    )
    sampler = _FakeSampler()
    make_sample_fn(sampler, GenConfig())(ex, 0)
    content = sampler.calls[0][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"]
    assert content[-1]["text"] == ex.inputs["problem"]


def test_score_one_official_rule():
    score_one = make_score_one_fn()
    ex = Example(id="among_g0_q0", inputs={"problem": ""}, target="B", meta={})
    assert score_one(ex, Sample(text="<answer>B. two</answer>")) == (1.0, "B")
    assert score_one(ex, Sample(text="<answer>A. one</answer>")) == (0.0, "A")
    assert score_one(ex, Sample(text="no idea")) == (0.0, None)


# ---------- aggregate ----------


def _result(setting, scores, extracted=None):
    ex = Example(id=f"{setting}_g_q", inputs={"problem": ""}, target="A", meta={"setting": setting})
    samples = [Sample(text="x") for _ in scores]
    return ExampleResult(
        example=ex, samples=samples, scores=scores, extracted=extracted or ["A"] * len(scores)
    )


def test_aggregate_excludes_translation_from_overall_but_reports_it():
    results = [
        _result("among", [1.0]),
        _result("around", [0.0]),
        _result("rotation", [1.0]),
        _result("translation", [0.0]),
    ]
    agg = aggregate_mindcube(results, n_repeats=1)
    assert agg["score"] == pytest.approx(2 / 3)  # translation excluded (official)
    assert agg["task.translation"] == 0.0
    assert agg["task.among"] == 1.0 and agg["task.around"] == 0.0
    assert agg["no_answer"] == 0.0


def test_aggregate_no_answer_rate():
    results = [_result("among", [0.0], extracted=[None]), _result("among", [1.0])]
    agg = aggregate_mindcube(results, n_repeats=1)
    assert agg["no_answer"] == 0.5


def test_aggregate_repeats_uses_pass_at_k_keys():
    results = [_result("among", [1.0, 0.0]), _result("around", [1.0, 1.0])]
    agg = aggregate_mindcube(results, n_repeats=2)
    assert agg["score"] == pytest.approx(0.75)
    assert "pass@2" in agg and "majority@2" in agg
    assert agg["task.among"] == 0.5 and agg["task.around"] == 1.0


def test_aggregate_empty():
    assert aggregate_mindcube([], 1) == {"score": 0.0}


# ---------- registration ----------


def test_registered():
    from sgl_eval import registry

    spec = registry.get("mindcube")
    assert spec.category == "multichoice"
    assert spec.default_n_repeats == 1
    assert spec.default_gen.chat_template_kwargs is None
