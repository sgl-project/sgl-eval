"""Stage 2 internals: thread-pool sample/score loop + result assembly.

Private to ``sgl_eval.runner``. The split from ``__init__`` is by pipeline
substage -- this module is the producer of ``per_example`` data; the
public ``run_examples`` orchestrator wraps it with progress UI and final
aggregation. Future replay paths (read predictions JSONL, reconstruct
samples) will reuse ``_assemble_results`` directly without going through
the live executor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from sgl_eval.types import Example, ExampleResult, Sample

# These aliases are duplicated from the public ``__init__`` to avoid an
# import cycle. Keep them in sync if the protocol changes.
TickFn = Callable[[int, float], None]
SampleFn = Callable[..., Sample]
ScoreOneFn = Callable[[Example, Sample], Tuple[float, Optional[str]]]
OnSampleScoredFn = Callable[[Example, int, Sample, float, Optional[str]], None]


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
    # Avoid a circular import: ``WorkerAborted`` lives in the package
    # ``__init__``, which imports this module. Pull it in lazily here.
    from sgl_eval.runner import WorkerAborted

    def record(ex: Example, rep: int, sample: Sample) -> None:
        samples_by_ex[ex.id][rep] = sample
        score, extracted = score_one_fn(ex, sample)
        scores_by_ex[ex.id][rep] = score
        extracted_by_ex[ex.id][rep] = extracted
        if on_sample_scored is not None:
            on_sample_scored(ex, rep, sample, score, extracted)
        tick(rep, score)

    if workers == 1:
        for ex, rep in tasks:
            try:
                sample = sample_fn(ex, rep)
            except WorkerAborted:
                return
            record(ex, rep, sample)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sample_fn, ex, rep): (ex.id, rep) for ex, rep in tasks}
        for fut in as_completed(futures):
            ex_id, rep = futures[fut]
            try:
                sample = fut.result()
            except WorkerAborted:
                # Cooperative cancellation: queued sibling workers will hit
                # this path too -- they short-circuit at the sampler entry.
                continue
            record(ex_by_id[ex_id], rep, sample)


def _assemble_results(
    examples: List[Example],
    samples_by_ex: Dict[str, List[Optional[Sample]]],
    scores_by_ex: Dict[str, List[float]],
    extracted_by_ex: Dict[str, List[Optional[str]]],
) -> List[ExampleResult]:
    """Filter out repeats whose sample never completed (partial-run case
    where the runner was aborted mid-flight). Keeps samples / scores /
    extracted aligned in length so downstream aggregators see consistent
    triples. Examples with zero completed repeats are dropped entirely."""
    results: List[ExampleResult] = []
    for ex in examples:
        samples: List[Sample] = []
        scores: List[float] = []
        extracted: List[Optional[str]] = []
        for s, sc, e in zip(samples_by_ex[ex.id], scores_by_ex[ex.id], extracted_by_ex[ex.id]):
            if s is None:
                continue
            samples.append(s)
            scores.append(sc)
            extracted.append(e)
        if not samples:
            continue
        results.append(
            ExampleResult(example=ex, samples=samples, scores=scores, extracted=extracted)
        )
    return results
