"""Smoke tests for the vendored evaluator slice: ``MathEvaluator``,
prompt yaml render, dataset metadata, bundled aime test.txt loading.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.parametrize(
    "generation, expected_answer, correct",
    [
        ("Reasoning... \\boxed{42}", "42", True),
        ("Reasoning... \\boxed{41}", "42", False),
    ],
)
def test_math_evaluator_eval_single(generation, expected_answer, correct):
    from sgl_eval._vendored.nemo_skills.evaluator.math import MathEvaluator

    ev = MathEvaluator(config={})
    out = asyncio.run(
        ev.eval_single({"generation": generation, "expected_answer": expected_answer})
    )
    assert out["symbolic_correct"] is correct


def test_dataset_metadata_aime25():
    from sgl_eval._vendored.nemo_skills.dataset import aime25

    assert aime25.METRICS_TYPE == "math"
    assert "prompt_config=generic/math" in aime25.GENERATION_ARGS


@pytest.mark.parametrize("name", ["aime24", "aime25"])
def test_aime_bundled_data_loads(name):
    from sgl_eval.evals._loader import load_bundled

    exs = load_bundled(name)(None)
    assert len(exs) == 30
    assert all(e.target.lstrip("-").isdigit() for e in exs)


def test_prompt_render_no_few_shot():
    from sgl_eval.evals._math import render_math_prompt

    rendered = render_math_prompt("What is 2+2?")
    assert "Solve the following math problem" in rendered
    assert "\\boxed{}" in rendered
    assert "What is 2+2?" in rendered
    assert "{examples}" not in rendered


def test_prompt_render_with_few_shot():
    from sgl_eval.evals._math import render_math_prompt

    rendered = render_math_prompt(
        "Find x.",
        few_shot_examples=[{"problem": "1+1=?", "solution": "\\boxed{2}"}],
    )
    assert "Here are some examples" in rendered
    assert "1+1=?" in rendered
    assert "Find x." in rendered
