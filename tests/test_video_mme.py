"""Tests for the Video-MME benchmark (SE-own; dataset on disk mocked, no network)."""

from __future__ import annotations

import base64

import pytest

from sgl_eval.evals._video_mme import (
    PROMPT_INSTRUCTION,
    PROMPT_SUFFIX,
    _interleave_rows_by_duration,
    aggregate_video_mme,
    build_prompt,
    extract_characters_regex,
    load_video_mme_examples,
    make_sample_fn,
    make_score_one_fn,
    subtitle_text,
    video_url_for,
)
from sgl_eval.types import Example, ExampleResult, GenConfig, Sample

# ---------- official extractor ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("C", "C"),
        ("The best answer is: C.", "C"),
        ("The correct answer is B", "B"),
        ("(D)", "D"),
        ("b", ""),  # official regex is case-sensitive
        ("", ""),
        ("no letters here at all in this long long sentence with many words indeed ok", ""),
        # official quirk, reproduced: only listed prefixes are stripped, so the
        # "A" in "Answer:" is the first match
        ("Answer: D", "A"),
    ],
)
def test_extract_characters_regex(text, expected):
    assert extract_characters_regex(text) == expected


# ---------- prompt ----------


def test_build_prompt_without_subtitles():
    p = build_prompt("Q?", ["A. x", "B. y", "C. z", "D. w"])
    assert p == PROMPT_INSTRUCTION + "\nQ?\nA. x\nB. y\nC. z\nD. w\n" + PROMPT_SUFFIX
    assert "subtitles" not in p


def test_build_prompt_with_subtitles():
    p = build_prompt("Q?", ["A. x", "B. y"], subtitles="hello\nworld")
    assert p.startswith(
        "This video's subtitles are listed below:\nhello\nworld\n" + PROMPT_INSTRUCTION
    )
    assert p.endswith(PROMPT_SUFFIX)


def test_subtitle_text_cleaning():
    srt = (
        "1\n00:00:01,520 --> 00:00:05,040\n"
        '<font color="white" size=".72c">next track final</font>\n\n'
        "2\n00:00:03,520 --> 00:00:06,000\n"
        '<font color="white" size=".72c">is the</font>\n'
    )
    assert subtitle_text(srt) == "next track final\nis the"
    assert subtitle_text("garbage without fonts") == ""


# ---------- loader ----------


def _row(qid, video_id, duration, answer="B"):
    return {
        "video_id": qid.split("-")[0],
        "duration": duration,
        "domain": "Knowledge",
        "sub_category": "Astronomy",
        "url": "https://example.invalid",
        "videoID": video_id,
        "question_id": qid,
        "task_type": "Counting Problem",
        "question": f"Question {qid}?",
        "options": ["A. one", "B. two", "C. three", "D. four"],
        "answer": answer,
    }


def _write_fake_dataset(root, rows, subtitles=None):
    (root / "data").mkdir(parents=True)
    for row in rows:
        (root / "data" / f"{row['videoID']}.mp4").write_bytes(
            b"\x00\x00\x00\x18ftypmp42" + row["videoID"].encode()
        )
    if subtitles:
        (root / "subtitle").mkdir()
        for vid, text in subtitles.items():
            (root / "subtitle" / f"{vid}.srt").write_text(text)


def test_load_from_env_dir(tmp_path, monkeypatch):
    rows = [
        _row("001-1", "vidA", "short"),
        _row("001-2", "vidA", "short", "D"),
        _row("301-1", "vidB", "medium"),
    ]
    _write_fake_dataset(tmp_path, rows)
    monkeypatch.setenv("SGL_EVAL_VIDEO_MME_DIR", str(tmp_path))
    monkeypatch.setattr("sgl_eval.evals._video_mme._read_rows", lambda: rows)
    examples = load_video_mme_examples(None)
    assert [e.id for e in examples] == ["001-1", "001-2", "301-1"]
    ex = examples[1]
    assert ex.target == "D"
    assert ex.meta["duration"] == "short" and ex.meta["video_id"] == "vidA"
    assert ex.meta["has_subtitles"] is False
    assert len(ex.media) == 1 and ex.media[0].kind == "video"
    assert ex.media[0].url == str(tmp_path / "data" / "vidA.mp4")
    assert ex.inputs["problem"].endswith("D. four\n" + PROMPT_SUFFIX)


def test_load_with_subtitles_and_fallback(tmp_path, monkeypatch):
    rows = [_row("001-1", "vidA", "short"), _row("301-1", "vidB", "medium")]
    srt = '1\n00:00:01,000 --> 00:00:02,000\n<font color="white" size=".72c">hi there</font>\n'
    _write_fake_dataset(tmp_path, rows, subtitles={"vidA": srt})
    monkeypatch.setenv("SGL_EVAL_VIDEO_MME_DIR", str(tmp_path))
    monkeypatch.setattr("sgl_eval.evals._video_mme._read_rows", lambda: rows)
    examples = load_video_mme_examples(None, use_subtitles=True)
    assert examples[0].meta["has_subtitles"] is True
    assert "This video's subtitles are listed below:\nhi there\n" in examples[0].inputs["problem"]
    assert examples[1].meta["has_subtitles"] is False  # no .srt -> plain prompt
    assert "subtitles are listed" not in examples[1].inputs["problem"]


def test_load_skips_missing_video(tmp_path, monkeypatch):
    rows = [_row("001-1", "vidA", "short"), _row("301-1", "vidB", "medium")]
    _write_fake_dataset(tmp_path, rows)
    (tmp_path / "data" / "vidB.mp4").unlink()
    monkeypatch.setenv("SGL_EVAL_VIDEO_MME_DIR", str(tmp_path))
    monkeypatch.setattr("sgl_eval.evals._video_mme._read_rows", lambda: rows)
    with pytest.warns(UserWarning, match="missing video"):
        examples = load_video_mme_examples(None)
    assert [e.id for e in examples] == ["001-1"]


def test_duration_filter(tmp_path, monkeypatch):
    rows = [
        _row("001-1", "vidA", "short"),
        _row("301-1", "vidB", "medium"),
        _row("601-1", "vidC", "long"),
    ]
    _write_fake_dataset(tmp_path, rows)
    monkeypatch.setenv("SGL_EVAL_VIDEO_MME_DIR", str(tmp_path))
    monkeypatch.setattr("sgl_eval.evals._video_mme._read_rows", lambda: rows)
    assert [e.id for e in load_video_mme_examples(None, duration="short")] == ["001-1"]
    assert [e.id for e in load_video_mme_examples(None, duration="long")] == ["601-1"]
    assert [e.id for e in load_video_mme_examples(1, duration="medium")] == ["301-1"]


def test_env_dir_without_videos_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("SGL_EVAL_VIDEO_MME_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        load_video_mme_examples(None)


def test_num_examples_interleaves_durations():
    rows = [_row(f"{i:03d}-1", f"s{i}", "short") for i in range(4)]
    rows += [_row(f"{300+i:03d}-1", f"m{i}", "medium") for i in range(4)]
    rows += [_row(f"{600+i:03d}-1", f"l{i}", "long") for i in range(4)]
    picked = _interleave_rows_by_duration(rows, 5)
    assert [r["duration"] for r in picked] == ["short", "medium", "long", "short", "medium"]


# ---------- video transport ----------


def test_video_url_template_and_inline(tmp_path):
    p = tmp_path / "abc.mp4"
    p.write_bytes(b"\x00\x01video")
    assert (
        video_url_for(str(p), "file:///srv/vmme/data/{videoID}.mp4")
        == "file:///srv/vmme/data/abc.mp4"
    )
    assert video_url_for(str(p), "https://h/{filename}") == "https://h/abc.mp4"
    inline = video_url_for(str(p), None)
    assert inline == "data:video/mp4;base64," + base64.b64encode(b"\x00\x01video").decode()


class _FakeSampler:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, gen):
        self.calls.append(messages)
        return Sample(text="B")


def test_sample_fn_video_before_text(tmp_path):
    from sgl_eval.types import MediaItem

    p = tmp_path / "vid.mp4"
    p.write_bytes(b"x")
    ex = Example(
        id="001-1",
        inputs={"problem": build_prompt("Q?", ["A. one", "B. two"])},
        target="B",
        meta={"duration": "short"},
        media=[MediaItem(kind="video", url=str(p), mime="video/mp4")],
    )
    sampler = _FakeSampler()
    make_sample_fn(sampler, GenConfig(), "file:///mnt/{videoID}.mp4")(ex, 0)
    content = sampler.calls[0][0]["content"]
    assert [c["type"] for c in content] == ["video_url", "text"]
    assert content[0]["video_url"]["url"] == "file:///mnt/vid.mp4"
    assert content[1]["text"] == ex.inputs["problem"]


def test_score_one_official_rule():
    score_one = make_score_one_fn()
    ex = Example(id="001-1", inputs={"problem": ""}, target="B", meta={})
    assert score_one(ex, Sample(text="The best answer is: B")) == (1.0, "B")
    assert score_one(ex, Sample(text="C")) == (0.0, "C")
    assert score_one(ex, Sample(text="")) == (0.0, None)


# ---------- aggregate ----------


def _result(duration, scores, extracted):
    ex = Example(id="q", inputs={"problem": ""}, target="A", meta={"duration": duration})
    return ExampleResult(
        example=ex, samples=[Sample(text="x") for _ in scores], scores=scores, extracted=extracted
    )


def test_aggregate_official_answered_only_vs_all():
    results = [
        _result("short", [1.0], ["A"]),
        _result("short", [0.0], [None]),  # unparsable: dropped by official rule
        _result("medium", [0.0], ["B"]),
        _result("long", [1.0], ["A"]),
    ]
    agg = aggregate_video_mme(results, n_repeats=1)
    assert agg["score"] == pytest.approx(2 / 3)  # official: 3 answered, 2 correct
    assert agg["score_all"] == pytest.approx(2 / 4)
    assert agg["task.short"] == 1.0 and agg["task.medium"] == 0.0 and agg["task.long"] == 1.0
    assert agg["no_answer"] == 0.25
    assert [k for k in agg if k.startswith("task.")] == [
        "task.short",
        "task.medium",
        "task.long",
    ]


def test_aggregate_repeats_adds_pass_at_k():
    results = [_result("short", [1.0, 0.0], ["A", "B"]), _result("long", [1.0, 1.0], ["A", "A"])]
    agg = aggregate_video_mme(results, n_repeats=2)
    assert "pass@2" in agg and "majority@2" in agg
    assert agg["score"] == pytest.approx(0.75)


def test_aggregate_empty():
    assert aggregate_video_mme([], 1) == {"score": 0.0}


# ---------- registration ----------


def test_registered_with_cli_args():
    import argparse

    from sgl_eval import registry

    spec = registry.get("video_mme")
    assert spec.category == "multichoice"
    assert spec.default_num_threads == 16
    parser = argparse.ArgumentParser()
    spec.add_arguments(parser.add_argument_group("video_mme"))
    ns = parser.parse_args(
        [
            "--video-mme-video-url",
            "file:///x/{videoID}.mp4",
            "--video-mme-subtitles",
            "--video-mme-duration",
            "short",
        ]
    )
    assert ns.video_mme_video_url == "file:///x/{videoID}.mp4"
    assert ns.video_mme_duration == "short"
    assert ns.video_mme_subtitles is True
    assert parser.parse_args([]).video_mme_subtitles is None
