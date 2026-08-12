"""Tests for ``sample_seed`` -- seeded sampling instead of first-N truncation.

The default (no ``sample_seed``) keeps first-N and stops reading there, which a
long-context split depends on. mmlu opts in because its prepared jsonl is
ordered by subject.
"""

from __future__ import annotations

import json
import types

import pytest

from sgl_eval.evals import _loader


def _write_grouped(path, n_per_group=30, groups=("politics", "math", "law")):
    """A jsonl ordered by group, like the Hendrycks tar's per-subject CSVs."""
    with path.open("w") as f:
        for g in groups:
            for i in range(n_per_group):
                f.write(
                    json.dumps(
                        {
                            "problem": f"{g}-q{i}",
                            "expected_answer": "A",
                            "subset_for_metrics": g,
                        }
                    )
                    + "\n"
                )
    return len(groups) * n_per_group


def _groups(examples):
    return {e.meta["subset_for_metrics"] for e in examples}


def test_default_keeps_first_n_and_stays_in_one_group(tmp_path):
    """Documents the behavior sample_seed exists to fix."""
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    examples = _loader._read_jsonl(path, "bench", 20)
    assert len(examples) == 20
    assert _groups(examples) == {"politics"}


def test_sample_seed_spans_groups(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    examples = _loader._read_jsonl(path, "bench", 20, sample_seed=0)
    assert len(examples) == 20
    assert len(_groups(examples)) > 1


def test_sample_seed_is_reproducible(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    a = _loader._read_jsonl(path, "bench", 15, sample_seed=0)
    b = _loader._read_jsonl(path, "bench", 15, sample_seed=0)
    assert [e.id for e in a] == [e.id for e in b]


def test_different_seeds_differ(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    a = _loader._read_jsonl(path, "bench", 15, sample_seed=0)
    b = _loader._read_jsonl(path, "bench", 15, sample_seed=1)
    assert [e.id for e in a] != [e.id for e in b]


def test_ids_track_line_number_not_sample_position(tmp_path):
    """A row must keep one id across num_examples, so a prediction dump stays
    comparable between a subsampled and a fuller run."""
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    small = {e.id: e.inputs["problem"] for e in _loader._read_jsonl(path, "b", 10, sample_seed=0)}
    big = {e.id: e.inputs["problem"] for e in _loader._read_jsonl(path, "b", 60, sample_seed=0)}
    shared = set(small) & set(big)
    assert shared, "seeded samples of different sizes should overlap"
    for k in shared:
        assert small[k] == big[k]


def test_num_examples_none_reads_everything(tmp_path):
    path = tmp_path / "test.jsonl"
    total = _write_grouped(path)
    assert len(_loader._read_jsonl(path, "b", None, sample_seed=0)) == total
    assert len(_loader._read_jsonl(path, "b", None)) == total


def test_num_examples_above_total_is_not_an_error(tmp_path):
    path = tmp_path / "test.jsonl"
    total = _write_grouped(path)
    assert len(_loader._read_jsonl(path, "b", total * 2, sample_seed=0)) == total


def test_default_path_stops_reading_early(tmp_path, monkeypatch):
    """Without sample_seed the reader must not pull the whole split into memory
    -- ruler2 rows are ~500KB each."""
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    parsed = []
    real_loads = json.loads
    monkeypatch.setattr(json, "loads", lambda s, **kw: (parsed.append(1), real_loads(s, **kw))[1])
    _loader._read_jsonl(path, "b", 5)
    assert len(parsed) == 5


def test_mmlu_declares_sample_seed(tmp_path):
    from sgl_eval.evals._registry import _TABLE

    [entry] = [e for e in _TABLE if e["name"] == "mmlu"]
    assert entry["sample_seed"] == 0


def test_long_context_rows_are_not_sampled(tmp_path):
    """ruler2 must keep the streaming first-N path; sampling it would load every
    ~500KB row."""
    from sgl_eval.evals._registry import _TABLE

    [entry] = [e for e in _TABLE if e["name"] == "ruler2"]
    assert "sample_seed" not in entry


def test_sample_seed_reaches_the_loader(tmp_path, monkeypatch):
    """load_via_prepare must forward sample_seed, not silently drop it."""
    rows = [
        {"problem": f"q{i}", "expected_answer": "A", "subset_for_metrics": "g" if i < 30 else "h"}
        for i in range(60)
    ]
    vendored = tmp_path / "pkg"
    vendored.mkdir()
    mod = types.ModuleType("fake_prepare")
    mod.__file__ = str(vendored / "prepare.py")

    def save_data(split):
        with (vendored / f"{split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    mod.save_data = save_data
    monkeypatch.setattr(_loader, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(_loader.importlib, "import_module", lambda _n: mod)

    examples = _loader.load_via_prepare("bench", ["test"], sample_seed=0)(20)
    assert len(examples) == 20
    assert len({e.meta["subset_for_metrics"] for e in examples}) == 2


@pytest.mark.parametrize("seed", [0, 7])
def test_sample_is_a_subset_of_the_split(tmp_path, seed):
    path = tmp_path / "test.jsonl"
    _write_grouped(path)
    everything = {e.inputs["problem"] for e in _loader._read_jsonl(path, "b", None)}
    sampled = {e.inputs["problem"] for e in _loader._read_jsonl(path, "b", 20, sample_seed=seed)}
    assert sampled <= everything
    assert len(sampled) == 20
