"""Tests for ``sample_seed`` (seeded sampling instead of first-N truncation)."""

from __future__ import annotations

import json

from sgl_eval.evals import _loader


def _grouped_jsonl(path, n=30, groups=("politics", "math", "law")):
    """Ordered by group, like the Hendrycks tar's per-subject CSVs."""
    with path.open("w") as f:
        for g in groups:
            for i in range(n):
                f.write(json.dumps({"problem": f"{g}{i}", "expected_answer": "A", "g": g}) + "\n")
    return len(groups) * n


def test_first_n_stays_in_one_group_but_a_sample_spans_them(tmp_path):
    path = tmp_path / "t.jsonl"
    _grouped_jsonl(path)
    first_n = _loader._read_jsonl(path, "b", 20)
    sampled = _loader._read_jsonl(path, "b", 20, sample_seed=0)
    assert {e.meta["g"] for e in first_n} == {"politics"}
    assert len(sampled) == 20 and len({e.meta["g"] for e in sampled}) > 1


def test_sample_is_reproducible(tmp_path):
    path = tmp_path / "t.jsonl"
    _grouped_jsonl(path)
    ids = [_loader._read_jsonl(path, "b", 15, sample_seed=0) for _ in range(2)]
    assert [e.id for e in ids[0]] == [e.id for e in ids[1]]


def test_ids_track_line_number_not_sample_position(tmp_path):
    """A row keeps one id across num_examples, so prediction dumps stay comparable."""
    path = tmp_path / "t.jsonl"
    _grouped_jsonl(path)
    small = {e.id: e.inputs["problem"] for e in _loader._read_jsonl(path, "b", 10, sample_seed=0)}
    big = {e.id: e.inputs["problem"] for e in _loader._read_jsonl(path, "b", 60, sample_seed=0)}
    shared = set(small) & set(big)
    assert shared and all(small[k] == big[k] for k in shared)


def test_num_examples_above_total_does_not_raise(tmp_path):
    """random.sample raises when k > len; the guard has to catch that."""
    path = tmp_path / "t.jsonl"
    total = _grouped_jsonl(path)
    assert len(_loader._read_jsonl(path, "b", total * 2, sample_seed=0)) == total


def test_default_path_stops_reading_early(tmp_path, monkeypatch):
    """Without sample_seed the reader must not parse the whole split -- ruler2
    rows are ~500KB each."""
    path = tmp_path / "t.jsonl"
    _grouped_jsonl(path)
    calls = []
    real = json.loads
    monkeypatch.setattr(json, "loads", lambda s, **kw: (calls.append(1), real(s, **kw))[1])
    _loader._read_jsonl(path, "b", 5)
    assert len(calls) == 5
