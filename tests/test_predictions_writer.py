"""Streaming prediction-writer tests.

Covers the three behaviors the runner depends on:
  - per-repeat routing into ``output-rs{i}.jsonl``
  - NS wire-shape fields on every line + every-line flush (crash recovery)
  - thread safety: concurrent writes from the runner's worker pool don't
    interleave bytes within a line and don't drop lines
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from sgl_eval.predictions import PredictionsWriter
from sgl_eval.types import Example, Sample


def _ex(i: int) -> Example:
    return Example(id=str(i), inputs={"problem": f"q{i}"}, target=str(i))


def _sample(text: str = "ans", completion: int = 10) -> Sample:
    return Sample(text=text, completion_tokens=completion, prompt_tokens=5, finish_reason="stop")


def test_writes_ns_shape_record(tmp_path: Path) -> None:
    with PredictionsWriter(tmp_path, n_repeats=2) as w:
        w(_ex(0), 0, _sample("response-A"), 1.0, "42")

    rows = [json.loads(line) for line in (tmp_path / "output-rs0.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "0"
    assert r["generation"] == "response-A"
    assert r["expected_answer"] == "0"
    assert r["problem"] == "q0"
    assert r["predicted_answer"] == "42"
    assert r["symbolic_correct"] is True
    assert r["num_generated_tokens"] == 10
    assert r["num_answer_tokens"] == 10
    assert r["num_reasoning_tokens"] == 0

    # The other repeat file is created but stays empty.
    assert (tmp_path / "output-rs1.jsonl").exists()
    assert (tmp_path / "output-rs1.jsonl").read_text() == ""


def test_routes_per_repeat(tmp_path: Path) -> None:
    with PredictionsWriter(tmp_path, n_repeats=3) as w:
        w(_ex(0), 1, _sample("rep1"), 0.0, None)
        w(_ex(0), 2, _sample("rep2"), 1.0, "ans2")
        w(_ex(1), 0, _sample("rep0"), 1.0, "ans0")

    files = {r: (tmp_path / f"output-rs{r}.jsonl").read_text().splitlines() for r in range(3)}

    assert len(files[0]) == 1 and json.loads(files[0][0])["generation"] == "rep0"
    assert len(files[1]) == 1 and json.loads(files[1][0])["generation"] == "rep1"
    assert len(files[2]) == 1 and json.loads(files[2][0])["generation"] == "rep2"

    # ``predicted_answer=None`` and ``symbolic_correct=False`` survive serialization.
    rep1 = json.loads(files[1][0])
    assert rep1["predicted_answer"] is None
    assert rep1["symbolic_correct"] is False


def test_every_line_flush_visible_before_close(tmp_path: Path) -> None:
    """Mid-run readers must see already-scored samples on disk -- this is
    the whole point of the streaming write (Ctrl-C / crash recovery)."""
    w = PredictionsWriter(tmp_path, n_repeats=1)
    try:
        w(_ex(0), 0, _sample("first"), 1.0, "a")
        # Read while writer is still open.
        content = (tmp_path / "output-rs0.jsonl").read_text()
        assert content.endswith("\n")
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == 1 and rows[0]["generation"] == "first"

        w(_ex(1), 0, _sample("second"), 0.0, None)
        rows = [
            json.loads(line) for line in (tmp_path / "output-rs0.jsonl").read_text().splitlines()
        ]
        assert [r["generation"] for r in rows] == ["first", "second"]
    finally:
        w.close()


def test_thread_safety_no_interleaving_no_drops(tmp_path: Path) -> None:
    """Concurrent writes from many threads: every line stays a valid JSON
    object (no byte-level interleaving) and the total line count matches."""
    n_repeats = 4
    n_threads = 16
    n_writes_per_thread = 64

    with PredictionsWriter(tmp_path, n_repeats=n_repeats) as w:

        def worker(tid: int) -> None:
            for i in range(n_writes_per_thread):
                rep = (tid + i) % n_repeats
                ex = _ex(tid * 1000 + i)
                w(ex, rep, _sample(f"t{tid}-{i}"), 1.0, "x")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    total_lines = 0
    seen_generations: set[str] = set()
    for r in range(n_repeats):
        lines = (tmp_path / f"output-rs{r}.jsonl").read_text().splitlines()
        for line in lines:
            obj = json.loads(line)  # interleaving would corrupt JSON -> raise
            seen_generations.add(obj["generation"])
        total_lines += len(lines)

    expected_total = n_threads * n_writes_per_thread
    assert total_lines == expected_total
    assert len(seen_generations) == expected_total  # no duplicates, no drops
