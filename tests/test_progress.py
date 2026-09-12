"""Live progress reports the same truncation rate as the final runner metrics."""

import io

import pytest

from sgl_eval.runner import _progress as progress
from sgl_eval.runner import run_examples
from sgl_eval.types import Example, Sample


@pytest.fixture
def bars(monkeypatch):
    created = []
    bar_class = progress._ProgressBar

    def make_bar(*args, **kwargs):
        bar = bar_class(*args, **kwargs, file=io.StringIO(), mininterval=0)
        created.append(bar)
        return bar

    monkeypatch.setattr(progress, "_ProgressBar", make_bar)
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


@pytest.mark.parametrize("n_repeats", [1, 16])
def test_rendered_statistics_survive_terminal_resize(bars, n_repeats):
    _, tick = progress._build_progress("gpqa", 198, n_repeats, enabled=True)
    try:
        for rep in range(n_repeats):
            for i in range(197):
                tick(rep, 1.0, "length" if i == 0 else "stop")
        for bar in bars:
            bar.dynamic_ncols = False
            for columns in [80, 120, 80]:
                bar.ncols = columns
                values = dict(bar.format_dict, elapsed=500, rate=0.3)
                rendered = bar.format_meter(**values)
                assert len(rendered) <= columns
                assert bar.desc.strip() in rendered
                assert f"{bar.n}/{bar.total}" in rendered
                assert "acc=100.00%" in rendered
                assert bar.postfix in rendered
                remaining = bar.format_interval((bar.total - bar.n) / 0.3)
                assert f"[08:20<{remaining}" in rendered
                if columns == 80:
                    assert values["bar_format"] is not None
                    assert "s/it" not in rendered
                else:
                    assert values["bar_format"] is None
                    assert "s/it" in rendered
    finally:
        for bar in bars:
            bar.close()
