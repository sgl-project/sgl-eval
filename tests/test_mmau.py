"""Unit tests for the mmau benchmark glue. No live endpoint, no audio deps:
the WAV re-encode and dataset fetch are monkeypatched, rows are synthetic."""

from __future__ import annotations

import base64

import pytest

from sgl_eval.evals import _mmau
from sgl_eval.metrics import format_summary
from sgl_eval.types import Example, ExampleResult, GenConfig, MediaItem, RunResult, Sample


def _row(task: str = "sound", row_id: str = "abc") -> dict:
    return {
        "id": row_id,
        "instruction": "What do you hear?",
        "choices": ["(A) cat", "(B) dog"],
        "answer": "(A) cat",
        "context": {"bytes": b"audio"},
        "other_attributes": f'{{"task": "{task}"}}',
    }


def _example(task: str = "sound", target: str = "cat", choices=("cat", "dog")) -> Example:
    return Example(
        id=f"{task}-0",
        inputs={"problem": "What sound is this?"},
        target=target,
        meta={"task": task, "choices": list(choices)},
        media=[MediaItem(kind="audio", data=b"wav-bytes", mime="audio/wav")],
    )


# ---------- scoring ----------


def test_mmau_string_match_exact():
    assert _mmau.mmau_string_match("Yes", "Yes", ["Yes", "No"])


def test_mmau_string_match_empty_prediction():
    assert not _mmau.mmau_string_match("Yes", "", ["Yes", "No"])


def test_mmau_string_match_answer_embedded_in_sentence():
    assert _mmau.mmau_string_match("a dog", "It is clearly a dog.", ["a dog", "a cat"])


def test_mmau_string_match_wrong_choice_tokens_pollute():
    assert not _mmau.mmau_string_match("a dog", "a dog and a cat", ["a dog", "a cat"])


def test_mmau_string_match_missing_answer():
    assert not _mmau.mmau_string_match("a dog", "a cat", ["a dog", "a cat"])


def test_strip_choice_letter():
    assert _mmau._strip_choice_letter("(A) barking") == "barking"
    assert _mmau._strip_choice_letter("barking") == "barking"
    assert _mmau._strip_choice_letter("(AB) barking") == "(AB) barking"


def test_format_question_shape():
    prompt = _mmau._format_question("What do you hear?", ["cat", "dog"])
    assert prompt.startswith("What do you hear?\n\nChoice: \ncat\ndog\n")
    assert "2 choices" in prompt


# ---------- dataset shaping ----------


def test_build_example(monkeypatch):
    monkeypatch.setattr(_mmau, "_encode_mono_16k_wav", lambda b: b"wav")
    ex = _mmau._build_example(_row(), 0)
    assert ex.id == "abc"
    assert ex.target == "cat"
    assert ex.meta == {"task": "sound", "choices": ["cat", "dog"]}
    assert ex.media == [MediaItem(kind="audio", data=b"wav", mime="audio/wav")]
    assert "cat\ndog" in ex.inputs["problem"]


def test_interleave_rows_by_task_balances_tasks():
    rows = [_row(task=t, row_id=f"{t}-{i}") for t, n in [("sound", 4), ("music", 2), ("speech", 2)] for i in range(n)]
    picked = _mmau._interleave_rows_by_task(rows, 6)
    assert len(picked) == 6
    tasks = [_mmau._task_of(r) for r in picked]
    assert tasks.count("sound") == 2
    assert tasks.count("music") == 2
    assert tasks.count("speech") == 2


def test_load_skips_undecodable_rows(monkeypatch):
    rows = [_row(row_id="ok"), _row(row_id="bad")]
    rows[1]["context"] = {"bytes": b"corrupt"}
    monkeypatch.setattr(_mmau, "_fetch_dataset_rows", lambda: rows)

    def _encode(raw):
        if raw == b"audio":
            return b"wav"
        raise ValueError("bad audio")

    monkeypatch.setattr(_mmau, "_encode_mono_16k_wav", _encode)
    with pytest.warns(UserWarning, match="skipping MMAU row bad"):
        examples = _mmau.load_mmau_examples(None)
    assert [ex.id for ex in examples] == ["ok"]


# ---------- sample / score ----------


class _RecordingSampler:
    def __init__(self, text: str = "cat"):
        self.messages = None
        self.text = text

    def __call__(self, messages, gen):
        self.messages = messages
        return Sample(text=self.text, finish_reason="stop")


def test_sample_fn_builds_multipart_message():
    sampler = _RecordingSampler()
    sample_fn = _mmau.make_sample_fn(sampler, GenConfig())
    sample = sample_fn(_example(), 0)
    assert sample.text == "cat"
    content = sampler.messages[0]["content"]
    assert content[0] == {"type": "text", "text": "What sound is this?"}
    expected_url = "data:audio/wav;base64," + base64.b64encode(b"wav-bytes").decode()
    assert content[1] == {"type": "audio_url", "audio_url": {"url": expected_url}}


def test_score_one():
    score_one = _mmau.make_score_one_fn()
    assert score_one(_example(), Sample(text="cat")) == (1.0, None)
    assert score_one(_example(), Sample(text="dog")) == (0.0, None)


# ---------- aggregate ----------


def _result(task: str, score: float) -> ExampleResult:
    return ExampleResult(
        example=_example(task=task),
        samples=[Sample(text="x")],
        scores=[score],
        extracted=[None],
    )


def test_aggregate_mmau_single_shot_with_per_task():
    results = [_result("sound", 1.0), _result("sound", 0.0), _result("music", 1.0)]
    agg = _mmau.aggregate_mmau(results, n_repeats=1)
    assert agg["score"] == pytest.approx(2 / 3)
    assert agg["task.sound"] == pytest.approx(0.5)
    assert agg["task.music"] == pytest.approx(1.0)
    assert "task.speech" not in agg


def test_aggregate_mmau_empty():
    assert _mmau.aggregate_mmau([], n_repeats=1) == {"score": 0.0}


def test_summary_renders_per_task_rows():
    result = RunResult(
        name="mmau",
        per_example=[],
        aggregate={"score": 0.75, "task.music": 0.8, "task.sound": 0.7},
        latency=1.0,
        num_examples=10,
        n_repeats=1,
    )
    summary = format_summary(result)
    assert "* score" in summary
    assert "music" in summary
    assert "sound" in summary
