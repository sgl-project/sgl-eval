"""``gsm8k`` ships the standard 8-shot CoT examples and ``--num-shots`` trims them.

The zero-shot ``generic/math`` wording ("put the answer (and only answer)
inside \\boxed{}") makes reasoning models loop on the answer format until
``max_tokens``; the vendored lm-eval ``gsm8k-cot`` examples are what the
benchmark is normally run with.
"""

from __future__ import annotations

import pytest

from sgl_eval.evals._prompts import load_few_shot_examples
from sgl_eval.registry import get
from sgl_eval.types import Example, GenConfig, Sample


class _PromptCapturingSampler:
    def __init__(self):
        self.prompts = []

    def __call__(self, messages, gen):
        self.prompts.append(messages[0]["content"])
        return Sample(text="\\boxed{42}", completion_tokens=3, finish_reason="stop")


def _one_example_loader(_num_examples):
    return [Example(id="p1", inputs={"problem": "Find n."}, target="42")]


def _render(num_shots=None, benchmark="gsm8k"):
    sampler = _PromptCapturingSampler()
    kwargs = {} if num_shots is None else {"num_shots": num_shots}
    get(benchmark).run(
        sampler=sampler,
        gen=GenConfig(),
        n_repeats=1,
        num_examples=1,
        num_threads=1,
        load_examples=_one_example_loader,
        **kwargs,
    )
    assert len(sampler.prompts) == 1
    return sampler.prompts[0]


def test_vendored_set_is_the_lm_eval_cot_eight():
    examples = load_few_shot_examples("gsm8k:gsm8k_standard_few_shot")
    assert len(examples) == 8
    assert examples[0]["problem"].startswith("There are 15 trees in the grove.")
    assert all(set(ex) >= {"problem", "solution"} for ex in examples)
    assert all("\\boxed{" in ex["solution"] for ex in examples)


def test_gsm8k_default_prompt_is_eight_shot():
    prompt = _render()
    assert prompt.count("Problem:\n") == 8
    assert "There are 15 trees in the grove." in prompt
    # The examples block sits between the instruction and the query, as in NS.
    assert prompt.index("inside \\boxed{}") < prompt.index("There are 15 trees")
    assert prompt.rstrip().endswith("Here is the problem you need to solve:\nFind n.")


def test_num_shots_trims_or_disables_the_examples():
    assert _render(num_shots=3).count("Problem:\n") == 3
    zero = _render(num_shots=0)
    assert "Problem:\n" not in zero
    assert "Here are some examples" not in zero
    assert zero.rstrip().endswith("\\boxed{}.\n\nFind n.")


def test_num_shots_beyond_vendored_set_fails_loudly():
    with pytest.raises(ValueError, match="exceeds the 8"):
        _render(num_shots=9)


def test_benchmarks_without_examples_reject_num_shots():
    with pytest.raises(ValueError, match="no few-shot examples"):
        _render(num_shots=5, benchmark="aime25")


def test_aime_stays_zero_shot():
    assert "Problem:\n" not in _render(benchmark="aime25")
