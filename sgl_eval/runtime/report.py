"""Stage 3: stdout summary + ``metrics.json`` + footer + exit code.

Consumes the ``RunContext`` from Stage 1 and the ``RunResult`` from
Stage 2. Owns partial-run reporting (sample-level accuracy bounds +
example breakdown) since those only matter at result-time.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Dict, Optional

import yaml

from sgl_eval import VENDORED_NS_ROOT
from sgl_eval import __version__ as _SGL_EVAL_VERSION
from sgl_eval.metrics import dump_run, format_summary
from sgl_eval.preset import make_run_meta_block, print_expected_vs_actual
from sgl_eval.runtime.setup import RunContext
from sgl_eval.types import RunResult


@dataclasses.dataclass(frozen=True)
class _PartialStats:
    """Single-walk snapshot of ``per_example`` for partial-run reporting.

    ``full / part / dropped`` partition examples into clean signal
    (every rep done), pad-distorted (some reps done -- aggregator's
    pad-with-last fabricates the rest), and absent (no reps done,
    not in ``per_example`` at all).
    """

    completed_samples: int
    planned_samples: int
    unfinished_samples: int
    worst: float
    best: float
    full: int
    part: int
    dropped: int


def render(result: RunResult, ctx: RunContext) -> int:
    """Print summary, dump metrics.json, print footer, return exit code.

    Exit code 130 when the user pressed Ctrl-C, even if every example
    happened to finish before abort propagated -- their intent was to
    bail, so respect it.
    """
    print(format_summary(result))

    run_meta = _build_run_meta(ctx)
    stats = _partial_stats(result) if result.partial else None
    if stats is not None:
        run_meta["partial"] = True
        run_meta["planned_examples"] = result.planned_examples
        run_meta["completed_samples"] = stats.completed_samples
        # Sample-level accuracy bounds: missing samples treated as all-wrong
        # (worst) or all-correct (best). Metric-agnostic and tight; tells the
        # user how much the partial run is worth.
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
    """Vendored slice's pinned upstream SHA (``SOURCES.yaml``). Returns
    ``None`` only when the manifest is absent (e.g. running from a non-
    standard checkout); a malformed manifest is a sync_vendored bug and
    should surface as a YAMLError, not be swallowed."""
    manifest = VENDORED_NS_ROOT / "SOURCES.yaml"
    try:
        text = manifest.read_text()
    except FileNotFoundError:
        return None
    return yaml.safe_load(text).get("synced_from_sha")


def _partial_stats(result: RunResult) -> _PartialStats:
    """Walk ``result.per_example`` once and bundle every count we need
    to populate ``run_meta`` and the ``[partial]`` footer."""
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
    """Sample-level accuracy bounds for a partial run.

    Lower bound: missing samples are all wrong. Upper bound: missing
    samples are all correct. Metric-agnostic -- whatever the aggregator
    does, accuracy must fall in [worst, best]. Returns (worst, best) in
    [0.0, 1.0].
    """
    if planned_samples <= 0:
        return 0.0, 0.0
    n_missing = planned_samples - completed_samples
    return n_correct / planned_samples, (n_correct + n_missing) / planned_samples


def _format_partial_summary(result: RunResult, stats: _PartialStats) -> str:
    """``[partial]`` footer: how much got done + worst/best score bounds.

    Progress line uses sample-level counts when ``n_repeats > 1`` (per-
    example pad-with-last makes pure example counts misleading there);
    example-level otherwise (equivalent). For ``n_repeats > 1`` we also
    surface the full / partial / dropped example breakdown -- ``partial``
    examples are the ones whose pad-with-last actively skews metrics,
    so the count matters for trusting the score bounds.

    Score bounds are always sample-level so they're comparable across
    n_repeats settings.
    """
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
