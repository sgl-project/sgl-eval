"""Stage 2 entry: parallel sample + main-thread score over (example, repeat)
pairs, then assemble. Scoring stays on the main thread so signal-based
grader timeouts (math_verify) work and live accuracy can render on the
progress bars. Internals: ``_executor`` (sample/score loop + assembly),
``_progress`` (tqdm bars).

Protocol:
    sample_fn(example, repeat_idx) -> Sample
    score_one_fn(example, sample)  -> (score, extracted)
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from sgl_eval.runner._executor import _assemble_results, _run_sample_score_phase
from sgl_eval.runner._progress import _build_progress, _start_bar_refresher
from sgl_eval.types import Example, ExampleResult, RunResult, Sample


class WorkerAborted(Exception):
    """Cooperative-cancel signal from ``sample_fn`` (e.g. CLI Ctrl-C closed
    the transport). Runner drops the pair and sets ``RunResult.partial``."""


TickFn = Callable[[int, float], None]
SampleFn = Callable[..., Sample]
ScoreOneFn = Callable[[Example, Sample], Tuple[float, Optional[str]]]
# Streaming-dump hook; fires on whatever thread completed the future, so
# implementations must be thread-safe.
OnSampleScoredFn = Callable[[Example, int, Sample, float, Optional[str]], None]

__all__ = [
    "OnSampleScoredFn",
    "SampleFn",
    "ScoreOneFn",
    "TickFn",
    "WorkerAborted",
    "run_examples",
]


def run_examples(
    name: str,
    examples: List[Example],
    sample_fn: SampleFn,
    score_one_fn: ScoreOneFn,
    num_threads: int = 64,
    n_repeats: int = 1,
    aggregate_fn: Optional[Callable[[List[ExampleResult]], Dict[str, float]]] = None,
    progress: bool = True,
    on_sample_scored: Optional[OnSampleScoredFn] = None,
) -> RunResult:
    debug_serial = os.getenv("SGL_EVAL_DEBUG") == "1"
    total_samples = len(examples) * max(n_repeats, 1)
    workers = 1 if debug_serial else min(num_threads, max(total_samples, 1))

    bars, tick = _build_progress(name, len(examples), n_repeats, enabled=progress)
    stop_refresh, refresh_thread = _start_bar_refresher(bars) if bars else (None, None)

    samples_by_ex: Dict[str, List[Optional[Sample]]] = {
        ex.id: [None] * n_repeats for ex in examples
    }
    scores_by_ex: Dict[str, List[float]] = {ex.id: [0.0] * n_repeats for ex in examples}
    extracted_by_ex: Dict[str, List[Optional[str]]] = {ex.id: [None] * n_repeats for ex in examples}
    ex_by_id = {ex.id: ex for ex in examples}

    tic = time.perf_counter()
    tasks = [(ex, rep) for ex in examples for rep in range(n_repeats)]

    try:
        _run_sample_score_phase(
            tasks=tasks,
            sample_fn=sample_fn,
            score_one_fn=score_one_fn,
            samples_by_ex=samples_by_ex,
            scores_by_ex=scores_by_ex,
            extracted_by_ex=extracted_by_ex,
            ex_by_id=ex_by_id,
            workers=workers,
            tick=tick,
            on_sample_scored=on_sample_scored,
        )
        results = _assemble_results(examples, samples_by_ex, scores_by_ex, extracted_by_ex)
    finally:
        # Join refresher BEFORE closing bars; otherwise the daemon can be
        # mid ``bar.refresh()`` at exit and leak bar text into the shell
        # ("zsh: command not found: aime25").
        if stop_refresh is not None:
            stop_refresh.set()
        if refresh_thread is not None:
            refresh_thread.join(timeout=2.0)
        for bar in bars:
            bar.close()

    latency = time.perf_counter() - tic

    total_completion = sum(s.completion_tokens or 0 for r in results for s in r.samples if s)
    total_prompt = sum(s.prompt_tokens or 0 for r in results for s in r.samples if s)

    aggregate = aggregate_fn(results) if aggregate_fn else _default_aggregate(results)
    for key, value in _finish_reason_rates(results).items():
        aggregate.setdefault(key, value)

    planned_samples = len(examples) * n_repeats
    completed_samples = sum(len(r.samples) for r in results)
    return RunResult(
        name=name,
        per_example=results,
        aggregate=aggregate,
        latency=latency,
        num_examples=len(results),
        n_repeats=n_repeats,
        total_completion_tokens=total_completion,
        total_prompt_tokens=total_prompt,
        partial=completed_samples < planned_samples,
        planned_examples=len(examples),
    )


def _default_aggregate(results: List[ExampleResult]) -> Dict[str, float]:
    if not results:
        return {"score": 0.0}
    per_example_means = [sum(r.scores) / len(r.scores) for r in results if r.scores]
    score = sum(per_example_means) / len(per_example_means) if per_example_means else 0.0
    return {"score": score}


def _finish_reason_rates(results: List[ExampleResult]) -> Dict[str, float]:
    """Per-stop-reason fractions over samples with a known ``finish_reason``
    (None excluded). ``truncated_rate`` flags no-EOS runaways; ``error_rate``
    covers any other reason (sampler sets ``"error"`` on failed requests)."""
    reasons = [s.finish_reason for r in results for s in r.samples if s and s.finish_reason]
    if not reasons:
        return {}
    n = len(reasons)
    return {
        "stop_rate": sum(fr == "stop" for fr in reasons) / n,
        "truncated_rate": sum(fr == "length" for fr in reasons) / n,
        "error_rate": sum(fr not in ("stop", "length") for fr in reasons) / n,
    }
