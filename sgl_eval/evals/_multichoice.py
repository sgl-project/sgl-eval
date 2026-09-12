"""Adapt the runner to vendored eval_mcq and MathMetrics.

The file-based evaluator receives one prediction per temporary JSONL.
MathMetrics aggregates its binary symbolic_correct field across repeats.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Silence vendored ``eval_mcq``'s tqdm bar before importing it.
import sgl_eval._vendored.nemo_skills.evaluator.mcq as _mcq_mod

_mcq_mod.tqdm = lambda iterable, **_kwargs: iterable

from sgl_eval._vendored.nemo_skills.evaluator.mcq import eval_mcq  # noqa: E402
from sgl_eval._vendored.nemo_skills.math_metrics import MathMetrics  # noqa: E402
from sgl_eval.evals._prompts import prompt_media_config, render_prompt  # noqa: E402
from sgl_eval.evals._vision import build_user_content  # noqa: E402
from sgl_eval.predictions import PredictionsWriter, sample_to_pred  # noqa: E402
from sgl_eval.runner import SampleFn, ScoreOneFn, run_examples  # noqa: E402
from sgl_eval.sampler import ChatCompletionSampler  # noqa: E402
from sgl_eval.types import Example, ExampleResult, GenConfig, RunResult, Sample  # noqa: E402


def _score_via_eval_mcq(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bridge the file-based evaluator to in-memory prediction dictionaries."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)
    try:
        eval_mcq({"input_file": str(tmp_path)})
        with tmp_path.open() as f:
            return [json.loads(line) for line in f]
    finally:
        tmp_path.unlink(missing_ok=True)


def make_sample_fn(sampler: ChatCompletionSampler, gen: GenConfig, prompt_yaml: Path) -> SampleFn:
    image_position = prompt_media_config(prompt_yaml)["image_position"]

    def sample_fn(ex: Example, _rep_idx: int) -> Sample:
        prompt_text = render_prompt(prompt_yaml, problem=ex.inputs["problem"])
        content = build_user_content(prompt_text, ex.media, image_position=image_position)
        return sampler([{"role": "user", "content": content}], gen)

    return sample_fn


def make_score_one_fn() -> ScoreOneFn:
    def score_one(ex: Example, sample: Sample) -> Tuple[float, Optional[str]]:
        # eval_mcq accepts files; one-row batches allow per-sample progress.
        rows = [{"generation": sample.text, **sample_to_pred(sample, ex)}]
        scored = _score_via_eval_mcq(rows)
        r = scored[0]
        score = 1.0 if r.get("symbolic_correct") else 0.0
        return score, r.get("predicted_answer")

    return score_one


def aggregate_with_math_metrics(results: List[ExampleResult], n_repeats: int) -> Dict[str, float]:
    if not results:
        return {"score": 0.0}
    metrics = MathMetrics()
    for r in results:
        preds = []
        for sample, score, letter in zip(r.samples, r.scores, r.extracted):
            pred = sample_to_pred(sample, r.example)
            pred["predicted_answer"] = letter
            pred["symbolic_correct"] = bool(score)
            preds.append(pred)
        while len(preds) < n_repeats:
            preds.append(dict(preds[-1]))
        metrics.update(preds)
    return _flatten(metrics.get_metrics(), n_repeats)


def _flatten(raw: Dict[str, Any], k: int) -> Dict[str, float]:
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


def run_multichoice_benchmark(
    *,
    name: str,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: Optional[int],
    num_threads: int,
    load_examples: Callable[[Optional[int]], List[Example]],
    prompt_yaml: Path,
    predictions_writer: Optional[PredictionsWriter] = None,
) -> RunResult:
    examples = load_examples(num_examples)
    sample_fn = make_sample_fn(sampler, gen, prompt_yaml)
    score_one_fn = make_score_one_fn()
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
