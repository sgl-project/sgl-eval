"""Smoke tests for the vendored math chain. ``math_equal`` / ``extract_answer``
are exhaustively covered by the vendored NS slice's own ``test_math_equal``;
we keep only what's NOT covered there: our ``relaxed`` flag config and the
``MathMetrics`` integration we actually call into.
"""

from __future__ import annotations

from sgl_eval._vendored.nemo_skills.math_grader import extract_answer, math_equal
from sgl_eval._vendored.nemo_skills.math_metrics import MathMetrics


def test_math_equal_take_modulo_param():
    """``take_modulo`` is supported by the vendored grader. sgl-eval does
    not currently pass it (upstream's aime configs don't either), but the
    parameter exists and we exercise it as a unit test."""
    assert math_equal("42", "1042", take_modulo=1000)


def test_extract_answer_relaxed_falls_back():
    """relaxed=True tries the regex first then boxed."""
    assert extract_answer("The final answer is 7", relaxed=True) == "7"


def test_math_metrics_smoke():
    m = MathMetrics()
    preds = [
        {
            "predicted_answer": "42",
            "expected_answer": "42",
            "symbolic_correct": True,
            "num_generated_tokens": 10,
            "problem": "q",
        },
        {
            "predicted_answer": "42",
            "expected_answer": "42",
            "symbolic_correct": True,
            "num_generated_tokens": 12,
            "problem": "q",
        },
    ]
    m.update(preds)
    out = m.get_metrics()
    assert "majority@2" in out
    assert "pass@2" in out
    assert "pass@1[avg-of-2]" in out
    assert out["pass@2"]["symbolic_correct"] == 100.0
