"""Tests for ``dump_run``: folder layout + run_meta merge contract.

Covers what ``cli.cmd_run`` and any future programmatic caller depends on:
  - writes ``<out_dir>/metrics.json`` (folder layout, not flat file)
  - core RunResult fields land in payload at expected keys
  - run_meta is merged into the top level alongside core fields
  - run_meta MUST NOT silently override reserved fields -- this is the
    safety net for the ``payload.update(run_meta)`` design
  - ensure_ascii=False keeps unicode in run_meta human-readable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgl_eval.metrics import dump_run, format_summary
from sgl_eval.types import RunResult


def _result() -> RunResult:
    return RunResult(
        name="test_bench",
        per_example=[],
        aggregate={"score": 0.85, "no_answer": 0.0},
        latency=12.5,
        num_examples=10,
        n_repeats=4,
        total_completion_tokens=1000,
        total_prompt_tokens=200,
    )


def test_payload_carries_core_fields(tmp_path: Path) -> None:
    dump_run(_result(), tmp_path)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["name"] == "test_bench"
    assert payload["num_examples"] == 10
    assert payload["n_repeats"] == 4
    assert payload["aggregate"] == {"score": 0.85, "no_answer": 0.0}
    assert payload["latency_seconds"] == 12.5
    assert payload["output_throughput_tps"] > 0
    assert payload["total_completion_tokens"] == 1000
    assert payload["total_prompt_tokens"] == 200


def test_run_meta_merged_with_unicode_preserved(tmp_path: Path) -> None:
    """run_meta merges into the top-level payload; non-ASCII stays readable
    (ensure_ascii=False) and core fields aren't shadowed."""
    run_meta = {
        "model": "DSv3.2",
        "ns_commit_sha": "645cf56",
        "note": "你好 LaTeX $\\boxed{42}$",
    }
    dump_run(_result(), tmp_path, run_meta=run_meta)
    raw = (tmp_path / "metrics.json").read_text()
    payload = json.loads(raw)
    for k, v in run_meta.items():
        assert payload[k] == v
    assert payload["name"] == "test_bench"  # core field preserved
    assert "你好" in raw and "\\u" not in raw  # unicode unescaped


@pytest.mark.parametrize("clashing_key", ["name", "aggregate", "n_repeats", "latency_seconds"])
def test_run_meta_overlap_with_core_field_raises(tmp_path: Path, clashing_key: str) -> None:
    """Reserved core fields must not be silently overridden by run_meta."""
    with pytest.raises(ValueError, match="overlaps reserved metrics fields"):
        dump_run(_result(), tmp_path, run_meta={clashing_key: "bogus"})


def _result_with(aggregate: dict) -> RunResult:
    return RunResult(
        name="test_bench",
        per_example=[],
        aggregate=aggregate,
        latency=12.5,
        num_examples=10,
        n_repeats=1,
        total_completion_tokens=1000,
        total_prompt_tokens=200,
    )


def test_summary_surfaces_finish_reason_rates() -> None:
    """``stop_rate`` / ``truncated_rate`` render in the summary alongside
    ``no_answer`` when present in the aggregate."""
    summary = format_summary(
        _result_with(
            {
                "score": 0.8,
                "no_answer": 0.0,
                "stop_rate": 0.9,
                "truncated_rate": 0.1,
                "error_rate": 0.0,
            }
        )
    )
    assert "stop_rate" in summary and "90.00%" in summary
    assert "truncated_rate" in summary and "10.00%" in summary
    assert "error_rate" in summary


def test_summary_warns_on_truncation() -> None:
    """A nonzero truncation rate flags hitting max_tokens (no-EOS runaway)."""
    summary = format_summary(_result_with({"score": 0.0, "stop_rate": 0.5, "truncated_rate": 0.5}))
    assert "warn: hitting max_tokens" in summary


def test_summary_omits_finish_reason_rates_when_absent() -> None:
    """Backward compatible: an aggregate without the new keys shows no rows."""
    summary = format_summary(_result_with({"score": 0.8, "no_answer": 0.0}))
    assert "stop_rate" not in summary
    assert "truncated_rate" not in summary


def test_summary_warns_on_error() -> None:
    """A nonzero error rate flags request failures (e.g. sampler BadRequest)."""
    summary = format_summary(_result_with({"score": 0.0, "stop_rate": 0.5, "error_rate": 0.5}))
    assert "warn: request errors" in summary
