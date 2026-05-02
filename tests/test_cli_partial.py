"""Unit tests for cli partial-run helpers (score bounds + summary line)."""

from __future__ import annotations

from sgl_eval.cli import _format_partial_summary, _partial_stats, _score_bounds
from sgl_eval.types import Example, ExampleResult, RunResult, Sample


def _make_result(per_example, planned_examples, n_repeats):
    return RunResult(
        name="dummy",
        per_example=per_example,
        aggregate={"score": 0.0},
        latency=0.0,
        num_examples=len(per_example),
        n_repeats=n_repeats,
        partial=True,
        planned_examples=planned_examples,
    )


def test_score_bounds_basic():
    """3/10 done, all correct -> worst 30% (missing all wrong),
    best 100% (missing all right)."""
    worst, best = _score_bounds(n_correct=3.0, completed_samples=3, planned_samples=10)
    assert worst == 0.3
    assert best == 1.0


def test_score_bounds_zero_correct():
    """3/10 done, none correct -> worst 0%, best 70%."""
    worst, best = _score_bounds(n_correct=0.0, completed_samples=3, planned_samples=10)
    assert worst == 0.0
    assert best == 0.7


def test_score_bounds_all_done():
    """When nothing is missing, worst == best == actual rate (sanity check
    -- partial runs typically don't hit this branch but the math should
    still collapse correctly)."""
    worst, best = _score_bounds(n_correct=7.0, completed_samples=10, planned_samples=10)
    assert worst == 0.7
    assert best == 0.7


def test_score_bounds_zero_planned():
    """Empty dataset -> bounds are zero, no division by zero."""
    worst, best = _score_bounds(n_correct=0.0, completed_samples=0, planned_samples=0)
    assert worst == 0.0
    assert best == 0.0


def _ex_result(ex_id, scores):
    ex = Example(id=ex_id, inputs={}, target="x")
    samples = [Sample(text="x", completion_tokens=1, finish_reason="stop") for _ in scores]
    return ExampleResult(
        example=ex, samples=samples, scores=list(scores), extracted=["x"] * len(scores)
    )


def test_partial_stats_single_walk():
    """One pass over ``per_example`` covers samples, scores, and the
    full/partial/dropped buckets in one go."""
    per_example = [
        _ex_result("a", [1.0, 1.0, 1.0]),  # full, 3 correct
        _ex_result("b", [1.0, 1.0]),  # part (2/3), 2 correct
        _ex_result("c", [0.0]),  # part (1/3), 0 correct
    ]
    # planned 5, only 3 in per_example -> 2 dropped, planned_samples=15
    result = _make_result(per_example, planned_examples=5, n_repeats=3)
    stats = _partial_stats(result)
    assert stats.completed_samples == 6
    assert stats.planned_samples == 15
    assert stats.unfinished_samples == 9
    assert (stats.full, stats.part, stats.dropped) == (1, 2, 2)
    # n_correct = 5; worst = 5/15, best = (5+9)/15
    assert stats.worst == 5 / 15
    assert stats.best == 14 / 15


def test_partial_stats_all_full_no_dropped():
    """Edge case: every rep done for every example -- partial flag may
    still fire from elsewhere, but the buckets all collapse correctly."""
    per_example = [_ex_result("a", [1.0, 1.0]), _ex_result("b", [0.0, 1.0])]
    result = _make_result(per_example, planned_examples=2, n_repeats=2)
    stats = _partial_stats(result)
    assert (stats.full, stats.part, stats.dropped) == (2, 0, 0)
    assert stats.unfinished_samples == 0


def test_partial_summary_n_repeats_1():
    """n_repeats=1 path uses example-level progress and shows [worst, best]."""
    per_example = [_ex_result(str(i), [1.0]) for i in range(3)]  # 3/10 examples, all correct
    result = _make_result(per_example, planned_examples=10, n_repeats=1)
    out = _format_partial_summary(result, _partial_stats(result))
    assert "3 / 10 examples completed" in out
    assert "(7 unfinished)" in out
    assert "score range: [30.00%, 100.00%]" in out
    assert "expected_vs_actual skipped" in out


def test_partial_summary_n_repeats_gt_1():
    """n_repeats>1 path uses sample-level progress + example full/partial/
    dropped breakdown; score bounds still sample-level."""
    # planned_examples=4, n_repeats=3 -> 12 planned samples
    # ex0: 3 reps done (full, 1 correct)
    # ex1: 1 rep done (partial, 1 correct)
    # ex2, ex3: missing from per_example -> dropped
    per_example = [_ex_result("0", [1.0, 0.0, 0.0]), _ex_result("1", [1.0])]
    result = _make_result(per_example, planned_examples=4, n_repeats=3)
    out = _format_partial_summary(result, _partial_stats(result))
    assert "4 / 12 samples completed" in out
    assert "(8 unfinished, n_repeats=3)" in out
    assert "examples: 1 full / 1 partial / 2 dropped (4 planned)" in out
    assert "score range: [16.67%, 83.33%]" in out
