"""Thread-pool sample/score loop + result assembly. Producer of
``per_example`` data; future replay paths reuse ``_assemble_results``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from sgl_eval.types import Example, ExampleResult, Sample

# Duplicated from public ``__init__`` to avoid an import cycle. Keep in sync.
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
    # Lazy import: WorkerAborted lives in __init__, which imports this module.
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
                # Queued siblings hit the same path -- they short-circuit
                # at the sampler entry.
                continue
            record(ex_by_id[ex_id], rep, sample)


def _assemble_results(
    examples: List[Example],
    samples_by_ex: Dict[str, List[Optional[Sample]]],
    scores_by_ex: Dict[str, List[float]],
    extracted_by_ex: Dict[str, List[Optional[str]]],
) -> List[ExampleResult]:
    """Drop repeats with no sample (partial runs); keep samples/scores/
    extracted aligned. Examples with zero reps completed are dropped."""
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
