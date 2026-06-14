"""Runner integration test.

Verifies the runner: parallel sample_fn, inline per-sample score_one_fn,
default mean aggregator, completion-token tally, n_repeats * num_examples
parallelism, on_sample_scored callback contract.
"""

from __future__ import annotations

import threading

import pytest

from sgl_eval.runner import WorkerAborted, run_examples
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
    assert result.partial is False
    assert result.planned_examples == 4
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


def test_runner_drops_aborted_samples_serial():
    """When ``sample_fn`` raises ``WorkerAborted`` (i.e. CLI Ctrl-C closed
    the httpx client), the serial runner stops, drops examples with no
    completed repeats, and aligns samples / scores / extracted so partial
    aggregation works."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(4)]

    counter = {"n": 0}

    def flaky_sample_fn(ex, _rep_idx):
        counter["n"] += 1
        if counter["n"] > 2:
            raise WorkerAborted()
        return Sample(text="x", completion_tokens=1, finish_reason="stop")

    result = run_examples(
        "dummy",
        examples,
        flaky_sample_fn,
        _all_correct_score_one_fn,
        num_threads=1,
        n_repeats=1,
        progress=False,
    )

    assert result.num_examples == 2
    assert result.partial is True
    assert result.planned_examples == 4
    for r in result.per_example:
        assert len(r.samples) == len(r.scores) == len(r.extracted) == 1
        assert r.samples[0].text == "x"
    assert result.aggregate["score"] == 1.0


def test_runner_partial_at_sample_level_when_n_repeats_gt_1():
    """An example that completed only some of its reps is partial too --
    even though it survives in ``per_example``, the missing reps mean
    the aggregator's pad-with-last would silently fabricate data, so
    ``partial`` must surface."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(2)]
    counter = {"n": 0}

    def flaky_sample_fn(_ex, _rep_idx):
        counter["n"] += 1
        # Serial order is (ex0,0), (ex0,1), (ex0,2), (ex1,0), ...
        # Abort after 2 calls -> ex0 keeps 2/3 reps, ex1 gets nothing.
        if counter["n"] > 2:
            raise WorkerAborted()
        return Sample(text="x", completion_tokens=1, finish_reason="stop")

    result = run_examples(
        "dummy",
        examples,
        flaky_sample_fn,
        _all_correct_score_one_fn,
        num_threads=1,
        n_repeats=3,
        progress=False,
    )

    completed = sum(len(r.samples) for r in result.per_example)
    assert result.planned_examples == 2
    assert result.num_examples == 1  # ex1 dropped (zero reps)
    assert completed == 2  # ex0 kept 2/3 reps
    # 2 < 6 planned -> partial fires. Critically, ex0 lives in
    # per_example with only 2 samples; without sample-level partial
    # this case would slip through as "complete".
    assert result.partial is True


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


def test_runner_reports_finish_reason_rates():
    """``stop_rate`` / ``truncated_rate`` over samples surface no-EOS runaways
    that ``no_answer`` (extraction failure) can't. Half the samples here hit
    the token cap (``finish_reason="length"``)."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(4)]

    def mixed_sample_fn(ex, _rep_idx):
        reason = "length" if int(ex.id) % 2 == 0 else "stop"
        return Sample(text="x", completion_tokens=1, finish_reason=reason)

    result = run_examples(
        "dummy",
        examples,
        mixed_sample_fn,
        _all_correct_score_one_fn,
        num_threads=4,
        n_repeats=1,
        progress=False,
    )
    assert result.aggregate["stop_rate"] == 0.5
    assert result.aggregate["truncated_rate"] == 0.5
    assert result.aggregate["error_rate"] == 0.0


def test_runner_finish_reason_rates_exclude_none():
    """Samples with unknown ``finish_reason`` drop out of the denominator
    rather than diluting the rates toward zero."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(3)]

    def some_none_sample_fn(ex, _rep_idx):
        # ex0: stop, ex1: length, ex2: None (excluded).
        reason = {"0": "stop", "1": "length"}.get(ex.id)
        return Sample(text="x", completion_tokens=1, finish_reason=reason)

    result = run_examples(
        "dummy",
        examples,
        some_none_sample_fn,
        _all_correct_score_one_fn,
        num_threads=3,
        n_repeats=1,
        progress=False,
    )
    # Denominator is 2 (None excluded), so each known reason is 1/2.
    assert result.aggregate["stop_rate"] == 0.5
    assert result.aggregate["truncated_rate"] == 0.5
    assert result.aggregate["error_rate"] == 0.0


def test_runner_omits_finish_reason_rates_when_all_none():
    """No known ``finish_reason`` anywhere -> keys are absent, not 0.0, so a
    backend that never reports the field doesn't fabricate a 0% stop rate."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(2)]

    def no_reason_sample_fn(_ex, _rep_idx):
        return Sample(text="x", completion_tokens=1, finish_reason=None)

    result = run_examples(
        "dummy",
        examples,
        no_reason_sample_fn,
        _all_correct_score_one_fn,
        num_threads=2,
        n_repeats=1,
        progress=False,
    )
    assert "stop_rate" not in result.aggregate
    assert "truncated_rate" not in result.aggregate


def test_runner_reports_error_rate():
    """``error_rate`` captures samples whose ``finish_reason`` is neither
    ``stop`` nor ``length`` (e.g. the sampler sets ``"error"`` on a failed
    request), so request failures aren't hidden behind stop/truncated."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(3)]

    def errored_sample_fn(ex, _rep_idx):
        # ex0: stop, ex1: length, ex2: error (request failed).
        reason = {"0": "stop", "1": "length", "2": "error"}[ex.id]
        return Sample(text="x", completion_tokens=1, finish_reason=reason)

    result = run_examples(
        "dummy",
        examples,
        errored_sample_fn,
        _all_correct_score_one_fn,
        num_threads=3,
        n_repeats=1,
        progress=False,
    )
    assert result.aggregate["stop_rate"] == pytest.approx(1 / 3)
    assert result.aggregate["truncated_rate"] == pytest.approx(1 / 3)
    assert result.aggregate["error_rate"] == pytest.approx(1 / 3)
