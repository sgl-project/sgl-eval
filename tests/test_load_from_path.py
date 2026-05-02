"""``--from-dataset`` loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgl_eval.evals._loader import load_from_path


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_round_trip_with_id_autogen_meta_and_slice(tmp_path: Path) -> None:
    """``id`` auto-fills as ``custom-<idx>`` when absent; extra fields go
    to ``meta``; ``num_examples`` slices the prefix."""
    p = tmp_path / "data.jsonl"
    _write(
        p,
        [
            {"id": "given-7", "problem": "Q0", "expected_answer": "42", "src": "x"},
            {"problem": "Q1", "expected_answer": "7"},  # no id
            {"problem": "Q2", "expected_answer": "13"},
        ],
    )
    examples = load_from_path(str(p))(num_examples=2)
    assert len(examples) == 2
    assert examples[0].id == "given-7"
    assert examples[0].inputs == {"problem": "Q0"}
    assert examples[0].target == "42"
    assert examples[0].meta == {"src": "x"}
    assert examples[1].id == "custom-1"  # auto, idx=1


def test_friendly_error_on_missing_required_field(tmp_path: Path) -> None:
    """Missing ``expected_answer`` -> sys.exit with file:line + field name."""
    p = tmp_path / "bad.jsonl"
    _write(p, [{"problem": "Q"}])  # no expected_answer
    with pytest.raises(SystemExit, match="missing required field 'expected_answer'"):
        load_from_path(str(p))(None)
