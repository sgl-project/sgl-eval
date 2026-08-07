"""Unit tests for the partial-run record: how much ran, and the footer."""

from __future__ import annotations

from sgl_eval.pipeline.report import _format_partial_summary, _partial_stats
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


def _ex_result(ex_id, scores):
    ex = Example(id=ex_id, inputs={}, target="x")
    samples = [Sample(text="x", completion_tokens=1, finish_reason="stop") for _ in scores]
    return ExampleResult(
        example=ex, samples=samples, scores=list(scores), extracted=["x"] * len(scores)
    )


def test_partial_stats_counts_samples_not_examples():
    """Sample-level is the only count that stays meaningful across n_repeats:
    an example with 1 of 3 repeats done is neither finished nor absent."""
    per_example = [
        _ex_result("a", [1.0, 1.0, 1.0]),
        _ex_result("b", [1.0, 1.0]),
        _ex_result("c", [0.0]),
    ]
    # 5 examples planned, 3 present -> 15 planned samples, 6 completed
    stats = _partial_stats(_make_result(per_example, planned_examples=5, n_repeats=3))
    assert (stats.completed_samples, stats.planned_samples) == (6, 15)


def test_partial_summary_reports_progress_and_the_skipped_comparison():
    per_example = [_ex_result("0", [1.0, 0.0, 0.0]), _ex_result("1", [1.0])]
    result = _make_result(per_example, planned_examples=4, n_repeats=3)
    out = _format_partial_summary(result, _partial_stats(result))
    assert "4 / 12 samples completed" in out
    assert "4 examples x 3" in out
    assert "expected_vs_actual skipped" in out
