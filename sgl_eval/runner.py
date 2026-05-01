"""Two-phase runner with per-sample inline scoring.

  Phase 1 (parallel sample + serial score):
    - Worker threads call ``sample_fn(ex, rep)`` -> ``Sample`` over every
      ``(example, repeat_idx)`` pair (max_workers = ``num_threads``).
    - As each future completes, the main thread immediately calls
      ``score_one_fn(ex, sample)`` -> ``(score, extracted)``. Scoring
      stays on the main thread so graders that use signal-based timeouts
      (math_verify) work without ceremony, and the running accuracy can
      be displayed live on the progress bars.
  Phase 2:
    - Assemble ``ExampleResult`` from the already-collected samples /
      scores. No additional grading.
  Phase 3:
    - Aggregate.

Public protocol:

    sample_fn(example, repeat_idx) -> Sample
    score_one_fn(example, sample)  -> (score, extracted)
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from sgl_eval.types import Example, ExampleResult, RunResult, Sample

TickFn = Callable[[int, float], None]
SampleFn = Callable[..., Sample]
ScoreOneFn = Callable[[Example, Sample], Tuple[float, Optional[str]]]
# Called once per (example, repeat) immediately after scoring, on whatever
# thread completed the future. Implementations must be thread-safe.
OnSampleScoredFn = Callable[[Example, int, Sample, float, Optional[str]], None]


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
        # Stop the refresher *and* join it BEFORE closing bars; otherwise
        # the daemon may still be inside a ``bar.refresh()`` call when
        # main exits, leaving stray bar text on the terminal that the
        # shell tries to interpret as commands ("zsh: command not found:
        # aime25", etc.).
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

    return RunResult(
        name=name,
        per_example=results,
        aggregate=aggregate,
        latency=latency,
        num_examples=len(results),
        n_repeats=n_repeats,
        total_completion_tokens=total_completion,
        total_prompt_tokens=total_prompt,
    )


def _run_sample_score_phase(
    *,
    tasks: List[Tuple[Example, int]],
    sample_fn: SampleFn,
    score_one_fn: ScoreOneFn,
    samples_by_ex: Dict[str, List[Optional[Sample]]],
    scores_by_ex: Dict[str, List[float]],
    extracted_by_ex: Dict[str, List[Optional[str]]],
    ex_by_id: Dict[str, Example],
    workers: int,
    tick: TickFn,
    on_sample_scored: Optional[OnSampleScoredFn],
) -> None:
    if workers == 1:
        for ex, rep in tasks:
            sample = sample_fn(ex, rep)
            samples_by_ex[ex.id][rep] = sample
            score, extracted = score_one_fn(ex, sample)
            scores_by_ex[ex.id][rep] = score
            extracted_by_ex[ex.id][rep] = extracted
            if on_sample_scored is not None:
                on_sample_scored(ex, rep, sample, score, extracted)
            tick(rep, score)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sample_fn, ex, rep): (ex.id, rep) for ex, rep in tasks}
        for fut in as_completed(futures):
            ex_id, rep = futures[fut]
            sample = fut.result()
            samples_by_ex[ex_id][rep] = sample
            ex = ex_by_id[ex_id]
            score, extracted = score_one_fn(ex, sample)
            scores_by_ex[ex_id][rep] = score
            extracted_by_ex[ex_id][rep] = extracted
            if on_sample_scored is not None:
                on_sample_scored(ex, rep, sample, score, extracted)
            tick(rep, score)


def _assemble_results(
    examples: List[Example],
    samples_by_ex: Dict[str, List[Optional[Sample]]],
    scores_by_ex: Dict[str, List[float]],
    extracted_by_ex: Dict[str, List[Optional[str]]],
) -> List[ExampleResult]:
    results: List[ExampleResult] = []
    for ex in examples:
        samples = [s for s in samples_by_ex[ex.id] if s is not None]
        results.append(
            ExampleResult(
                example=ex,
                samples=samples,
                scores=list(scores_by_ex[ex.id]),
                extracted=list(extracted_by_ex[ex.id]),
            )
        )
    return results


def _build_progress(
    name: str, num_examples: int, n_repeats: int, *, enabled: bool
) -> Tuple[List[tqdm], TickFn]:
    """Build per-repeat bars + an overall bar; tick updates count and a
    running ``acc`` postfix on the relevant bars."""
    if not enabled:
        return [], lambda _idx, _score: None

    if n_repeats <= 1:
        bar = tqdm(total=num_examples, desc=name, dynamic_ncols=True)
        cum = {"correct": 0.0, "total": 0}

        def tick(_rep_idx: int, score: float) -> None:
            cum["correct"] += float(score)
            cum["total"] += 1
            bar.set_postfix({"acc": f"{cum['correct'] / cum['total']:.2%}"}, refresh=False)
            bar.update(1)

        return [bar], tick

    width = len(str(n_repeats))
    prefix_len = len(f"rep {n_repeats}/{n_repeats}")
    rep_bars = [
        tqdm(
            total=num_examples,
            desc=f"{name} rep {i + 1:>{width}}/{n_repeats}",
            position=i,
            leave=True,
            dynamic_ncols=True,
        )
        for i in range(n_repeats)
    ]
    overall_label = "overall".ljust(prefix_len)
    overall_bar = tqdm(
        total=num_examples * n_repeats,
        desc=f"{name} {overall_label}",
        position=n_repeats,
        leave=True,
        dynamic_ncols=True,
    )

    rep_correct = [0.0] * n_repeats
    rep_total = [0] * n_repeats
    overall_correct = [0.0]
    overall_total = [0]

    def tick(rep_idx: int, score: float) -> None:
        if 0 <= rep_idx < len(rep_bars):
            rep_correct[rep_idx] += float(score)
            rep_total[rep_idx] += 1
            rep_bars[rep_idx].set_postfix(
                {"acc": f"{rep_correct[rep_idx] / rep_total[rep_idx]:.2%}"},
                refresh=False,
            )
            rep_bars[rep_idx].update(1)
        overall_correct[0] += float(score)
        overall_total[0] += 1
        overall_bar.set_postfix(
            {"acc": f"{overall_correct[0] / overall_total[0]:.2%}"},
            refresh=False,
        )
        overall_bar.update(1)

    return rep_bars + [overall_bar], tick


def _start_bar_refresher(
    bars: List[tqdm], interval: float = 0.5
) -> Tuple[threading.Event, threading.Thread]:
    """Periodically call ``bar.refresh()`` on every bar so elapsed/ETA and
    the latest postfix stay live even when that specific bar isn't ticking.
    Returns ``(stop_event, thread)`` so the caller can ``stop.set()`` then
    ``thread.join()`` for clean teardown before closing bars."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            for bar in bars:
                try:
                    bar.refresh()
                except Exception:
                    pass

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread


def _default_aggregate(results: List[ExampleResult]) -> Dict[str, float]:
    if not results:
        return {"score": 0.0}
    per_example_means = [sum(r.scores) / len(r.scores) for r in results if r.scores]
    score = sum(per_example_means) / len(per_example_means) if per_example_means else 0.0
    return {"score": score}
