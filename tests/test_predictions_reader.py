"""``PredictionsReader`` round-trip tests against the live writer."""

from __future__ import annotations

from pathlib import Path

from sgl_eval.predictions import PredictionsReader, PredictionsWriter
from sgl_eval.types import Example, Sample


def _ex(i: int) -> Example:
    return Example(id=str(i), inputs={"problem": f"q{i}"}, target=str(i))


def _sample(text: str = "ans") -> Sample:
    return Sample(text=text, completion_tokens=10, prompt_tokens=5, finish_reason="stop")


def _seed_run_dir(run_dir: Path, n_repeats: int) -> None:
    with PredictionsWriter(run_dir, n_repeats=n_repeats) as w:
        for ex_i in range(2):
            for rep in range(n_repeats):
                w(_ex(ex_i), rep, _sample(f"r{rep}-q{ex_i}"), float(rep == 0), f"a{ex_i}")


def test_iter_rep_yields_only_that_rep(tmp_path: Path) -> None:
    _seed_run_dir(tmp_path, n_repeats=3)
    rows = list(PredictionsReader(tmp_path).iter_rep(1))
    assert len(rows) == 2
    assert all(r["generation"].startswith("r1-") for r in rows)


def test_iter_all_covers_every_rep(tmp_path: Path) -> None:
    _seed_run_dir(tmp_path, n_repeats=3)
    pairs = list(PredictionsReader(tmp_path).iter_all())
    assert len(pairs) == 6  # 2 examples * 3 reps
    reps_seen = sorted({rep for rep, _ in pairs})
    assert reps_seen == [0, 1, 2]


def test_n_repeats_autodetect(tmp_path: Path) -> None:
    _seed_run_dir(tmp_path, n_repeats=4)
    assert PredictionsReader(tmp_path).n_repeats == 4


def test_n_repeats_override(tmp_path: Path) -> None:
    """Explicit ``n_repeats`` wins; covers a partial run that aborted
    before higher reps wrote anything."""
    _seed_run_dir(tmp_path, n_repeats=2)
    assert PredictionsReader(tmp_path, n_repeats=8).n_repeats == 8


def test_iter_rep_missing_file_is_empty(tmp_path: Path) -> None:
    """Missing rep yields nothing instead of raising, so partial-run
    analysis paths don't need to special-case it."""
    _seed_run_dir(tmp_path, n_repeats=1)
    assert list(PredictionsReader(tmp_path, n_repeats=4).iter_rep(3)) == []


def test_round_trip_preserves_fields(tmp_path: Path) -> None:
    """Every writer field is readable back verbatim -- replay relies on
    this to reconstruct ``Sample`` and ``score``."""
    with PredictionsWriter(tmp_path, n_repeats=1) as w:
        w(_ex(7), 0, _sample("response-X"), 1.0, "42")

    [row] = list(PredictionsReader(tmp_path).iter_rep(0))
    assert row["id"] == "7"
    assert row["generation"] == "response-X"
    assert row["predicted_answer"] == "42"
    assert row["symbolic_correct"] is True
    assert row["num_generated_tokens"] == 10
