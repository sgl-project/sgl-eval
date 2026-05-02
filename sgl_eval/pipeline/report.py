"""Stage 3: stdout summary + ``metrics.json`` + footer + exit code.
Owns partial-run reporting (sample-level bounds + example breakdown)."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Dict, Optional

import yaml

from sgl_eval import VENDORED_NS_ROOT
from sgl_eval import __version__ as _SGL_EVAL_VERSION
from sgl_eval.metrics import dump_run, format_summary
from sgl_eval.pipeline.setup import RunContext
from sgl_eval.preset import make_run_meta_block, print_expected_vs_actual
from sgl_eval.types import RunResult


@dataclasses.dataclass(frozen=True)
class _PartialStats:
    """Single-walk snapshot of ``per_example``. ``full / part / dropped``
    partition examples into clean / pad-distorted / absent."""

    completed_samples: int
    planned_samples: int
    unfinished_samples: int
    worst: float
    best: float
    full: int
    part: int
    dropped: int


def render(result: RunResult, ctx: RunContext) -> int:
    """Print summary, dump metrics.json, print footer, return exit code."""
    print(format_summary(result))

    run_meta = _build_run_meta(ctx)
    stats = _partial_stats(result) if result.partial else None
    if stats is not None:
        run_meta["partial"] = True
        run_meta["planned_examples"] = result.planned_examples
        run_meta["completed_samples"] = stats.completed_samples
        # Sample-level bounds: missing samples treated as all-wrong / all-
        # correct. Metric-agnostic; tells the user how much the run is worth.
        run_meta["score_lower_bound"] = stats.worst
        run_meta["score_upper_bound"] = stats.best
        if result.n_repeats > 1:
            run_meta["examples_full"] = stats.full
            run_meta["examples_partial"] = stats.part
            run_meta["examples_dropped"] = stats.dropped
    preset_block = make_run_meta_block(ctx.args, ctx.inputs.preset)
    if preset_block:
        run_meta["preset"] = preset_block

    metrics_path = dump_run(result, ctx.run_dir, run_meta=run_meta)
    print(f"\nMetrics: {metrics_path}")
    if ctx.writer is not None:
        print(f"Predictions: {ctx.run_dir}  ({ctx.inputs.n_repeats} jsonl file(s))")

    if stats is not None:
        print(_format_partial_summary(result, stats), file=sys.stderr)
    else:
        print_expected_vs_actual(result, ctx.inputs.preset)

    # 130 even when result is complete: respect user's Ctrl-C intent.
    return 130 if ctx.sampler.aborted else 0


def _build_run_meta(ctx: RunContext) -> Dict[str, Any]:
    return {
        "timestamp": ctx.stamp,
        "model": ctx.sampler.model,
        "base_url": ctx.inputs.base_url,
        "num_threads": ctx.num_threads,
        "gen": dataclasses.asdict(ctx.inputs.gen),
        "sgl_eval_version": _SGL_EVAL_VERSION,
        "ns_commit_sha": _read_ns_commit_sha(),
    }


def _read_ns_commit_sha() -> Optional[str]:
    """Vendored slice's pinned SHA. ``None`` only when the manifest is
    absent; a malformed manifest is a sync_vendored bug and should surface
    as YAMLError, not be swallowed."""
    manifest = VENDORED_NS_ROOT / "SOURCES.yaml"
    try:
        text = manifest.read_text()
    except FileNotFoundError:
        return None
    return yaml.safe_load(text).get("synced_from_sha")


def _partial_stats(result: RunResult) -> _PartialStats:
    """Single walk of ``per_example`` -> all counts for run_meta + footer."""
    n_repeats = result.n_repeats
    completed = 0
    n_correct = 0.0
    full = part = 0
    for r in result.per_example:
        k = len(r.samples)
        completed += k
        n_correct += sum(r.scores)
        if k == n_repeats:
            full += 1
        elif k > 0:
            part += 1
    planned = result.planned_examples * n_repeats
    worst, best = _score_bounds(n_correct, completed, planned)
    return _PartialStats(
        completed_samples=completed,
        planned_samples=planned,
        unfinished_samples=planned - completed,
        worst=worst,
        best=best,
        full=full,
        part=part,
        dropped=result.planned_examples - len(result.per_example),
    )


def _score_bounds(
    n_correct: float, completed_samples: int, planned_samples: int
) -> tuple[float, float]:
    """``(worst, best)`` sample-level accuracy bounds: missing samples
    treated as all-wrong / all-correct. Metric-agnostic."""
    if planned_samples <= 0:
        return 0.0, 0.0
    n_missing = planned_samples - completed_samples
    return n_correct / planned_samples, (n_correct + n_missing) / planned_samples


def _format_partial_summary(result: RunResult, stats: _PartialStats) -> str:
    """``[partial]`` footer. ``n_repeats > 1`` reports sample-level counts
    + full/partial/dropped breakdown (pad-with-last skews metrics on the
    ``partial`` bucket); otherwise example-level. Score bounds always
    sample-level for comparability across n_repeats."""
    lines = []
    if result.n_repeats > 1:
        lines.append(
            f"[partial] {stats.completed_samples} / {stats.planned_samples} "
            f"samples completed ({stats.unfinished_samples} unfinished, "
            f"n_repeats={result.n_repeats})"
        )
        lines.append(
            f"[partial] examples: {stats.full} full / {stats.part} partial / "
            f"{stats.dropped} dropped ({result.planned_examples} planned)"
        )
    else:
        lines.append(
            f"[partial] {stats.full + stats.part} / {result.planned_examples} "
            f"examples completed ({stats.unfinished_samples} unfinished)"
        )
    lines.append(
        f"[partial] score range: [{stats.worst:.2%}, {stats.best:.2%}] "
        "(missing samples assumed all-wrong / all-correct)"
    )
    lines.append(
        "[partial] expected_vs_actual skipped (partial runs aren't comparable to baselines)."
    )
    return "\n" + "\n".join(lines)
