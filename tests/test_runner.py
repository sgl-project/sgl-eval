"""Runner integration test.

Verifies the runner: parallel sample_fn, inline per-sample score_one_fn,
default mean aggregator, completion-token tally, n_repeats * num_examples
parallelism, on_sample_scored callback contract.
"""

from __future__ import annotations

import threading

from sgl_eval.runner import run_examples
from sgl_eval.types import Example, Sample


def _fake_sample_fn(ex: Example, _rep_idx: int) -> Sample:
    return Sample(text=str(ex.target), completion_tokens=5, finish_reason="stop")


def _all_correct_score_one_fn(ex, sample):
    return 1.0, str(ex.target)


def _half_correct_score_one_fn(ex, sample):
    score = 1.0 if int(ex.id) % 2 == 0 else 0.0
    return score, "ok"


def test_runner_basic():
    examples = [Example(id=str(i), inputs={"q": i}, target=str(i)) for i in range(4)]
    result = run_examples(
        "dummy",
        examples,
        _fake_sample_fn,
        _all_correct_score_one_fn,
        num_threads=2,
        n_repeats=1,
        progress=False,
    )
    assert result.num_examples == 4
    assert result.aggregate["score"] == 1.0
    assert result.total_completion_tokens == 4 * 5
    assert result.output_throughput > 0


def test_runner_partial_correct():
    examples = [Example(id=str(i), inputs={}, target="ok") for i in range(10)]
    result = run_examples(
        "dummy",
        examples,
        _fake_sample_fn,
        _half_correct_score_one_fn,
        num_threads=4,
        n_repeats=1,
        progress=False,
    )
    assert result.aggregate["score"] == 0.5


def test_runner_n_repeats_flat_concurrency():
    """All ``num_examples * n_repeats`` tasks should be submitted to the
    threadpool, not capped at ``len(examples)``."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(4)]

    submitted = set()

    def tracking_sample_fn(ex, rep_idx):
        submitted.add((ex.id, rep_idx))
        return Sample(text="x", completion_tokens=1, finish_reason="stop")

    result = run_examples(
        "dummy",
        examples,
        tracking_sample_fn,
        _all_correct_score_one_fn,
        num_threads=16,
        n_repeats=8,
        progress=False,
    )
    assert result.num_examples == 4
    assert result.n_repeats == 8
    assert len(submitted) == 4 * 8
    assert result.total_completion_tokens == 4 * 8


def test_runner_invokes_on_sample_scored():
    """Callback fires once per ``(example, repeat)`` with the scored
    sample, score, and extracted answer -- this is the streaming-dump hook."""
    examples = [Example(id=str(i), inputs={}, target=str(i)) for i in range(3)]

    received = []
    lock = threading.Lock()

    def cb(ex, rep, sample, score, extracted):
        with lock:
            received.append((ex.id, rep, sample.text, score, extracted))

    result = run_examples(
        "dummy",
        examples,
        _fake_sample_fn,
        _all_correct_score_one_fn,
        num_threads=4,
        n_repeats=2,
        progress=False,
        on_sample_scored=cb,
    )

    assert len(received) == 3 * 2
    assert {(eid, rep) for eid, rep, _, _, _ in received} == {
        (str(i), r) for i in range(3) for r in range(2)
    }
    assert all(s == 1.0 for _, _, _, s, _ in received)
    assert all(extracted == str(eid) for eid, _, _, _, extracted in received)
    assert result.aggregate["score"] == 1.0
