"""Live progress reports the same truncation rate as the final runner metrics."""

import io

import pytest
from tqdm import tqdm

from sgl_eval.runner import _progress as progress
from sgl_eval.runner import run_examples
from sgl_eval.types import Example, Sample


@pytest.fixture
def bars(monkeypatch):
    created = []

    def make_bar(*args, **kwargs):
        bar = tqdm(*args, **kwargs, file=io.StringIO(), mininterval=0)
        created.append(bar)
        return bar

    monkeypatch.setattr(progress, "tqdm", make_bar)
    return created


def test_truncation_updates_before_completion(bars):
    _, tick = progress._build_progress("test", 10, 1, enabled=True)
    bar = bars[0]
    try:
        tick(0, 1.0, None)
        assert "truncated=0/0 (N/A)" in bar.postfix
        tick(0, 0.0, "length")
        assert "truncated=1/1 (100.00%)" in bar.postfix
        tick(0, 1.0, "stop")
        assert "truncated=1/2 (50.00%)" in bar.postfix
        tick(0, 0.0, "error")
        assert "truncated=1/3 (33.33%)" in bar.postfix
        tick(0, 0.5, "")
        assert "truncated=1/3 (33.33%)" in bar.postfix
        assert "acc=50.00%" in bar.postfix
        assert bar.n == 5 < bar.total
    finally:
        bar.close()


@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("n_repeats", [1, 2])
def test_runner_live_truncation_matches_final_metrics(bars, num_threads, n_repeats):
    reasons = [("length", "stop", None, "error"), ("stop", "length", "length", "")]
    examples = [Example(id=str(i), inputs={}, target="ok") for i in range(4)]

    def sample(ex, rep):
        return Sample(text="ok", finish_reason=reasons[rep][int(ex.id)])

    result = run_examples(
        "test",
        examples,
        sample,
        lambda ex, sample: (0.5, "ok"),
        num_threads=num_threads,
        n_repeats=n_repeats,
    )
    assert "truncated=1/3 (33.33%)" in bars[0].postfix
    if n_repeats == 2:
        assert "truncated=2/3 (66.67%)" in bars[1].postfix
        assert "truncated=3/6 (50.00%)" in bars[-1].postfix
    final_rate = f"{result.aggregate['truncated_rate']:.2%}"
    assert f"({final_rate})" in bars[-1].postfix
    assert "acc=50.00%" in bars[-1].postfix
    assert bars[-1].n == 4 * n_repeats


def test_disabled_progress_accepts_finish_reason():
    bars, tick = progress._build_progress("test", 1, 1, enabled=False)
    tick(0, 0.0, "length")
    assert bars == []
