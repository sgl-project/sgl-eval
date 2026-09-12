"""``run --prompt``: swapping the prompt yaml a benchmark wraps questions in.

A benchmark's prompt is part of what it measures -- ``generic/math`` and
``eval/matharena/aime`` ask an AIME question differently, so their scores are
not comparable.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from sgl_eval.evals._prompts import _SE_PROMPT_DIR, render_prompt, resolve_prompt, vendored_prompt
from sgl_eval.registry import get
from sgl_eval.types import Example, GenConfig, Sample


def test_resolve_prompt_prefers_vendored_then_se_own():
    assert resolve_prompt("math") == vendored_prompt("math")
    # mmmu-pro-cot is sgl-eval's own; upstream ships no MMMU-Pro config.
    assert not vendored_prompt("mmmu-pro-cot").exists()
    assert resolve_prompt("mmmu-pro-cot") == _SE_PROMPT_DIR / "mmmu-pro-cot.yaml"


def test_resolve_prompt_takes_a_path_verbatim(tmp_path):
    """A user reproducing someone else's numbers may hold a prompt this repo
    does not ship; a path (or a .yaml basename) bypasses the two search dirs."""
    own = tmp_path / "custom.yaml"
    own.write_text("user: |-\n  {problem}\n")
    assert resolve_prompt(str(own)) == own


def test_matharena_prompt_is_vendored_and_differs_from_generic_math():
    matharena = resolve_prompt("matharena-aime")
    assert matharena.exists()
    rendered = render_prompt(matharena, problem="Find n.")
    assert "integer between 0 and 999" in rendered
    assert "Find n." in rendered
    # The distinguishing claim: generic/math never states the answer range.
    assert "integer between 0 and 999" not in render_prompt(
        vendored_prompt("math"), problem="Find n."
    )


class _PromptCapturingSampler:
    """Records the user message each request was built from."""

    def __init__(self):
        self.prompts = []

    def __call__(self, messages, gen):
        self.prompts.append(messages[0]["content"])
        return Sample(text="\\boxed{42}", completion_tokens=3, finish_reason="stop")


def _one_example_loader(_num_examples):
    return [Example(id="p1", inputs={"problem": "Find n."}, target="42")]


@pytest.mark.parametrize("benchmark", ["aime25", "gsm8k"])
def test_prompt_override_reaches_the_math_sampler(benchmark):
    sampler = _PromptCapturingSampler()
    get(benchmark).run(
        sampler=sampler,
        gen=GenConfig(),
        n_repeats=1,
        num_examples=1,
        num_threads=1,
        load_examples=_one_example_loader,
        prompt_yaml=resolve_prompt("matharena-aime"),
    )
    assert sampler.prompts and "integer between 0 and 999" in sampler.prompts[0]


def test_math_default_prompt_still_applies_without_override():
    """The registered basename -- not a module-level constant -- is what a
    math benchmark falls back to, so the two cannot drift apart."""
    sampler = _PromptCapturingSampler()
    get("aime25").run(
        sampler=sampler,
        gen=GenConfig(),
        n_repeats=1,
        num_examples=1,
        num_threads=1,
        load_examples=_one_example_loader,
        prompt_yaml=None,
    )
    assert "Solve the following math problem" in sampler.prompts[0]
    assert "integer between 0 and 999" not in sampler.prompts[0]


def _run_gpqa_capturing(prompt_yaml):
    sampler = _PromptCapturingSampler()

    def loader(_num_examples):
        return [Example(id="q1", inputs={"problem": "Q?\n\nA) x\nB) y"}, target="A")]

    get("gpqa").run(
        sampler=sampler,
        gen=GenConfig(),
        n_repeats=1,
        num_examples=1,
        num_threads=1,
        load_examples=loader,
        prompt_yaml=prompt_yaml,
    )
    assert sampler.prompts
    return sampler.prompts[0]


def test_prompt_override_reaches_the_multichoice_sampler():
    """gpqa's registered prompt is the plain `Answer: A/B/C/D` variant; the
    override swaps in the boxed one, which is the only difference between them."""
    overridden = _run_gpqa_capturing(vendored_prompt("mcq-4choices-boxed"))
    assert "Answer: \\boxed{A/B/C/D}" in overridden
    assert "Answer: \\boxed{A/B/C/D}" not in _run_gpqa_capturing(None)


def test_ruler2_refuses_a_prompt_override():
    """RULER2's prompt is `user: "{question}"` -- the context is assembled by
    the prepare scripts, so an override would silently do nothing."""
    with pytest.raises(ValueError, match="does not take a prompt override"):
        get("ruler2").run(
            sampler=_PromptCapturingSampler(),
            gen=GenConfig(),
            n_repeats=1,
            num_examples=1,
            num_threads=1,
            prompt_yaml=vendored_prompt("math"),
        )


def test_unknown_prompt_fails_before_the_first_request():
    """Resolution happens in setup, not at first render -- otherwise a typo
    only surfaces after the server is up and the dataset is loaded."""
    from sgl_eval.pipeline.setup import _resolve_prompt_override

    assert _resolve_prompt_override(None) is None
    with pytest.raises(FileNotFoundError, match="no-such-prompt"):
        _resolve_prompt_override("no-such-prompt")


def test_run_meta_records_the_override_only_when_set():
    from sgl_eval.pipeline.report import _build_run_meta

    def _ctx(prompt_yaml, spec):
        return SimpleNamespace(
            stamp="20260828-000000",
            sampler=SimpleNamespace(model="m"),
            inputs=SimpleNamespace(base_url="http://x/v1", gen=GenConfig(), model_preset=None),
            num_threads=1,
            bench_args={},
            prompt_yaml=prompt_yaml,
            args=argparse.Namespace(prompt=spec),
        )

    assert "prompt" not in _build_run_meta(_ctx(None, None))
    meta = _build_run_meta(_ctx(resolve_prompt("matharena-aime"), "matharena-aime"))
    assert meta["prompt"]["spec"] == "matharena-aime"
    assert meta["prompt"]["path"].endswith("matharena-aime.yaml")


def test_run_subcommand_exposes_prompt():
    from sgl_eval.cli import build_parser

    args = build_parser().parse_args(
        ["run", "aime25", "--base-url", "http://x/v1", "--prompt", "matharena-aime"]
    )
    assert args.prompt == "matharena-aime"
