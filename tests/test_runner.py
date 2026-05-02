"""Runner integration test.

Verifies the runner: parallel sample_fn, inline per-sample score_one_fn,
default mean aggregator, completion-token tally, n_repeats * num_examples
parallelism, on_sample_scored callback contract.
"""

from __future__ import annotations

import threading

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


def test_runner_drops_aborted_samples_parallel():
    """Parallel path: an ``as_completed`` worker raising ``WorkerAborted``
    is skipped without poisoning sibling workers' results. We can't
    guarantee which ones complete (thread interleaving), but the invariant
    holds: every returned ExampleResult has aligned, non-empty triples."""
    examples = [Example(id=str(i), inputs={}, target="x") for i in range(8)]

    abort_event = threading.Event()
    completed = {"n": 0}
    lock = threading.Lock()

    def flaky_sample_fn(_ex, _rep_idx):
        with lock:
            completed["n"] += 1
            if completed["n"] >= 3:
                abort_event.set()
        if abort_event.is_set() and completed["n"] >= 3:
            raise WorkerAborted()
        return Sample(text="x", completion_tokens=1, finish_reason="stop")

    result = run_examples(
        "dummy",
        examples,
        flaky_sample_fn,
        _all_correct_score_one_fn,
        num_threads=4,
        n_repeats=1,
        progress=False,
    )

    assert 0 < result.num_examples <= len(examples)
    for r in result.per_example:
        assert len(r.samples) == len(r.scores) == len(r.extracted) >= 1


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
