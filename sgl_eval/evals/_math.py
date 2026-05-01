"""Glue between sgl-eval's sampler/runner and the vendored NeMo-Skills math
evaluator.

Mirrors NS pipeline stages:
  - Stage 2a (prompt render): vendored ``prompts/math.yaml`` + ``str.format``.
  - Stage 2c (extract + score): vendored ``MathEvaluator.eval_single``.
  - Stage 4 (aggregate): vendored ``MathMetrics.update`` + ``get_metrics``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from sgl_eval._vendored.nemo_skills.evaluator.math import MathEvaluator
from sgl_eval._vendored.nemo_skills.math_metrics import MathMetrics
from sgl_eval.evals._predictions import PredictionsWriter, sample_to_pred
from sgl_eval.evals._prompts import render_prompt, vendored_prompt
from sgl_eval.runner import SampleFn, ScoreOneFn, run_examples
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import Example, ExampleResult, GenConfig, RunResult, Sample

_MATH_PROMPT_YAML = vendored_prompt("math")


def render_math_prompt(problem: str, few_shot_examples: Optional[list] = None) -> str:
    return render_prompt(_MATH_PROMPT_YAML, problem=problem, few_shot_examples=few_shot_examples)


def _eval_single_sync(evaluator: MathEvaluator, data_point: Dict[str, Any]) -> Dict[str, Any]:
    """Drive ``MathEvaluator.eval_single`` (an ``async def``) synchronously.
    Body is pure-CPU for math, so per-call event-loop overhead is negligible."""
    return asyncio.run(evaluator.eval_single(data_point))


def make_sample_fn(sampler: ChatCompletionSampler, gen: GenConfig) -> SampleFn:
    def sample_fn(ex: Example, _rep_idx: int) -> Sample:
        prompt = render_math_prompt(ex.inputs["problem"])
        return sampler([{"role": "user", "content": prompt}], gen)

    return sample_fn


def make_score_one_fn(evaluator: MathEvaluator) -> ScoreOneFn:
    def score_one(ex: Example, sample: Sample) -> Tuple[float, Optional[str]]:
        data_point = {"generation": sample.text, **sample_to_pred(sample, ex)}
        data_point = _eval_single_sync(evaluator, data_point)
        score = 1.0 if data_point.get("symbolic_correct") else 0.0
        return score, data_point.get("predicted_answer")

    return score_one


def aggregate_with_math_metrics(results: List[ExampleResult], n_repeats: int) -> Dict[str, float]:
    if not results:
        return {"score": 0.0}
    metrics = MathMetrics()
    for r in results:
        preds = []
        for sample, score, ans in zip(r.samples, r.scores, r.extracted):
            pred = sample_to_pred(sample, r.example)
            pred["predicted_answer"] = ans
            pred["symbolic_correct"] = bool(score)
            preds.append(pred)
        while len(preds) < n_repeats:
            preds.append(dict(preds[-1]))
        metrics.update(preds)
    return _flatten_math_metrics(metrics.get_metrics(), n_repeats)


def _flatten_math_metrics(raw: Dict[str, Any], k: int) -> Dict[str, float]:
    """Pull headline numbers (and per-run std / SEM when ``k > 1``) out of
    ``MathMetrics``' nested output. Values normalized to [0, 1]. ``score``
    aliases the headline (``pass@1[avg-of-k]`` when ``k > 1``, plain
    ``pass@1`` when ``k == 1``)."""
    flat: Dict[str, float] = {}
    if k == 1:
        per_q = raw.get("pass@1", {})
        flat["score"] = per_q.get("symbolic_correct", 0.0) / 100.0
        flat["no_answer"] = per_q.get("no_answer", 0.0) / 100.0
        return flat
    pass_avg_key = f"pass@1[avg-of-{k}]"
    per_q = raw.get(pass_avg_key, {})
    stats = per_q.get("symbolic_correct_statistics", {})
    flat["pass@1"] = per_q.get("symbolic_correct", 0.0) / 100.0
    flat["pass@1_std"] = stats.get("std_dev_across_runs", 0.0)
    flat["pass@1_sem"] = stats.get("std_err_across_runs", 0.0)
    flat[f"pass@{k}"] = raw.get(f"pass@{k}", {}).get("symbolic_correct", 0.0) / 100.0
    flat[f"majority@{k}"] = raw.get(f"majority@{k}", {}).get("symbolic_correct", 0.0) / 100.0
    flat["no_answer"] = per_q.get("no_answer", 0.0) / 100.0
    flat["score"] = flat["pass@1"]
    return flat


def run_math_benchmark(
    *,
    name: str,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: Optional[int],
    num_threads: int,
    load_examples: Callable[[Optional[int]], List[Example]],
    evaluator_config: Optional[Dict[str, Any]] = None,
    predictions_writer: Optional[PredictionsWriter] = None,
) -> RunResult:
    examples = load_examples(num_examples)
    evaluator = MathEvaluator(config=evaluator_config or {})
    sample_fn = make_sample_fn(sampler, gen)
    score_one_fn = make_score_one_fn(evaluator)
    aggregator = (
        (lambda results: aggregate_with_math_metrics(results, n_repeats)) if n_repeats > 1 else None
    )
    return run_examples(
        name=name,
        examples=examples,
        sample_fn=sample_fn,
        score_one_fn=score_one_fn,
        num_threads=num_threads,
        n_repeats=n_repeats,
        aggregate_fn=aggregator,
        on_sample_scored=predictions_writer,
    )
