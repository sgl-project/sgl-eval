"""End-to-end pipeline test against a stub sampler.

Doesn't hit any network: monkeypatches ``ChatCompletionSampler`` to return
a canned ``Sample`` and runs a tiny custom EvalSpec through the runner.
Validates that the registry / runner / aggregator wiring works without
relying on a real model or HF dataset download.
"""

from __future__ import annotations

from sgl_eval.evals._math import run_math_benchmark
from sgl_eval.evals._multichoice import run_multichoice_benchmark
from sgl_eval.types import Example, Sample


class _StubSampler:
    """Returns a canned response containing the example's target answer."""

    def __init__(self, template: str = "After thinking, the answer is \\boxed{{{target}}}."):
        self.template = template
        self.calls = 0

    def __call__(self, messages, gen):
        self.calls += 1
        target = messages[-1]["content"].split("__TARGET__:")[-1].strip()
        text = self.template.format(target=target)
        return Sample(text=text, completion_tokens=8, prompt_tokens=20, finish_reason="stop")


def test_math_pipeline_end_to_end():
    examples = [
        Example(id=f"x-{i}", inputs={"problem": f"q{i} __TARGET__: {i+1}"}, target=str(i + 1))
        for i in range(3)
    ]

    sampler = _StubSampler()

    def loader(num_examples):
        return examples[:num_examples] if num_examples else examples

    from sgl_eval.evals._prompts import vendored_prompt
    from sgl_eval.types import GenConfig

    result = run_math_benchmark(
        name="dummy_math",
        sampler=sampler,
        gen=GenConfig(),
        n_repeats=1,
        num_examples=None,
        num_threads=2,
        load_examples=loader,
        prompt_yaml=vendored_prompt("math"),
    )
    assert result.num_examples == 3
    assert result.aggregate["score"] == 1.0
    assert result.total_completion_tokens == 3 * 8


def test_multichoice_pipeline_end_to_end_with_repeats():
    from sgl_eval.evals._prompts import vendored_prompt

    examples = [
        Example(
            id=f"mc-{i}",
            inputs={"problem": "Q?\n\nA) x\nB) y\nC) z\nD) w"},
            target="A",
        )
        for i in range(2)
    ]

    class _LetterStub:
        def __init__(self):
            self.calls = 0

        def __call__(self, messages, gen):
            self.calls += 1
            return Sample(
                text="Reasoning... Answer: \\boxed{A}",
                completion_tokens=5,
                finish_reason="stop",
            )

    sampler = _LetterStub()

    def loader(num_examples):
        return examples[:num_examples] if num_examples else examples

    from sgl_eval.types import GenConfig

    result = run_multichoice_benchmark(
        name="dummy_mc",
        sampler=sampler,
        gen=GenConfig(),
        n_repeats=4,
        num_examples=None,
        num_threads=2,
        load_examples=loader,
        prompt_yaml=vendored_prompt("mcq-4choices-boxed"),
    )
    assert result.num_examples == 2
    assert result.n_repeats == 4
    assert result.aggregate.get("pass@1") == 1.0
    assert result.aggregate.get("majority@4") == 1.0
