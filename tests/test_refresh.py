"""``sgl-eval refresh`` round-trip tests. Uses ``aime24`` (bundled, math
category). Refresh doesn't re-grade -- it reads ``symbolic_correct`` from
jsonl rows -- so problem text in fixtures is intentionally fake."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sgl_eval.pipeline.refresh import cmd_refresh


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(ex_id: str, correct: bool, tokens: int = 10, predicted: str = "x") -> dict:
    return {
        "id": ex_id,
        "problem": f"Q for {ex_id}",
        "expected_answer": "x",
        "generation": f"think...\\boxed{{{predicted}}}",
        "predicted_answer": predicted,
        "symbolic_correct": correct,
        "num_generated_tokens": tokens,
        "num_reasoning_tokens": 0,
        "num_answer_tokens": tokens,
    }


def test_refresh_round_trip_no_old_metrics(tmp_path: Path) -> None:
    """No metrics.json -> benchmark name parsed from dirname; aggregate
    + tokens recomputed from jsonl."""
    run_dir = tmp_path / "sgl_eval_aime24_20260502-120000"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "output-rs0.jsonl",
        [_row("aime24-0", True), _row("aime24-1", False), _row("aime24-2", True)],
    )

    rc = cmd_refresh(SimpleNamespace(run_dir=str(run_dir)))
    assert rc == 0

    payload = json.loads((run_dir / "metrics.json").read_text())
    assert payload["name"] == "aime24"
    assert payload["num_examples"] == 3
    assert payload["aggregate"]["score"] == pytest.approx(2 / 3)
    assert payload["total_completion_tokens"] == 30
    assert "partial" not in payload  # no signal of dropped examples


def test_refresh_preserves_provenance_and_clears_stale_partial(tmp_path: Path) -> None:
    """Provenance (model / latency / ns_commit_sha / total_prompt_tokens)
    survives. Aggregate / tokens recomputed. Old ``partial=True`` + bounds
    drop when the jsonl now covers the full plan."""
    run_dir = tmp_path / "sgl_eval_aime24_20260502-120000"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "name": "aime24",
                "num_examples": 1,  # stale
                "n_repeats": 1,
                "planned_examples": 2,  # full plan
                "latency_seconds": 99.5,
                "output_throughput_tps": 0.05,
                "total_completion_tokens": 5,  # stale
                "total_prompt_tokens": 200,  # not in jsonl, must survive
                "aggregate": {"score": 0.0},  # stale
                "model": "DSv3.2",
                "ns_commit_sha": "abc123",
                # Stale partial fields from a previous abort:
                "partial": True,
                "completed_samples": 1,
                "score_lower_bound": 0.5,
                "score_upper_bound": 1.0,
            }
        )
    )
    _write_jsonl(
        run_dir / "output-rs0.jsonl",
        [_row("aime24-0", True, tokens=7), _row("aime24-1", True, tokens=3)],
    )

    cmd_refresh(SimpleNamespace(run_dir=str(run_dir)))
    payload = json.loads((run_dir / "metrics.json").read_text())
    # Recomputed
    assert payload["aggregate"]["score"] == 1.0
    assert payload["total_completion_tokens"] == 10
    assert payload["num_examples"] == 2
    # Preserved
    assert payload["latency_seconds"] == 99.5
    assert payload["total_prompt_tokens"] == 200
    assert payload["model"] == "DSv3.2"
    assert payload["ns_commit_sha"] == "abc123"
    # Stale partial fields cleared
    assert "partial" not in payload
    assert "score_lower_bound" not in payload
    assert "score_upper_bound" not in payload


def test_refresh_partial_emits_score_bounds(tmp_path: Path) -> None:
    """planned=3 from old metrics, only 1 completed in jsonl ->
    partial=True + sample-level bounds."""
    run_dir = tmp_path / "sgl_eval_aime24_20260502-120000"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "name": "aime24",
                "num_examples": 3,
                "n_repeats": 1,
                "planned_examples": 3,
                "latency_seconds": 10.0,
                "output_throughput_tps": 1.0,
                "total_completion_tokens": 10,
                "total_prompt_tokens": 30,
                "aggregate": {"score": 0.0},
            }
        )
    )
    _write_jsonl(run_dir / "output-rs0.jsonl", [_row("aime24-0", True)])

    cmd_refresh(SimpleNamespace(run_dir=str(run_dir)))
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert payload["partial"] is True
    assert payload["planned_examples"] == 3
    assert payload["completed_samples"] == 1
    # 1 correct of 3 planned -> worst 1/3, best (1+2)/3 = 1.0
    assert payload["score_lower_bound"] == pytest.approx(1 / 3)
    assert payload["score_upper_bound"] == pytest.approx(1.0)
