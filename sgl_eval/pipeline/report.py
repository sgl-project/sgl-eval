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
from sgl_eval.model_preset import make_model_preset_meta_block
from sgl_eval.pipeline.setup import RunContext
from sgl_eval.preset import make_run_meta_block, print_expected_vs_actual
from sgl_eval.types import RunResult


@dataclasses.dataclass(frozen=True)
class _PartialStats:
    """How much of the planned work actually ran."""

    completed_samples: int
    planned_samples: int


def render(result: RunResult, ctx: RunContext) -> int:
    """Print summary, dump metrics.json, print footer, return exit code."""
    print(format_summary(result))

    run_meta = _build_run_meta(ctx)
    stats = _partial_stats(result) if result.partial else None
    if stats is not None:
        run_meta["partial"] = True
        run_meta["planned_examples"] = result.planned_examples
        run_meta["completed_samples"] = stats.completed_samples
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
    meta: Dict[str, Any] = {
        "timestamp": ctx.stamp,
        "model": ctx.sampler.model,
        "base_url": ctx.inputs.base_url,
        "num_threads": ctx.num_threads,
        "gen": dataclasses.asdict(ctx.inputs.gen),
        "sgl_eval_version": _SGL_EVAL_VERSION,
        "ns_commit_sha": _read_ns_commit_sha(),
    }
    # Benchmarks whose dataset is generated (ruler2) are only identified by
    # these -- without them a metrics.json cannot say which setup it scored.
    if ctx.bench_args:
        meta["bench_args"] = dict(ctx.bench_args)
    model_preset_block = make_model_preset_meta_block(ctx.inputs.model_preset)
    if model_preset_block:
        meta["model_preset"] = model_preset_block
    return meta


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
    return _PartialStats(
        completed_samples=sum(len(r.samples) for r in result.per_example),
        planned_samples=result.planned_examples * result.n_repeats,
    )


def _format_partial_summary(result: RunResult, stats: _PartialStats) -> str:
    """``[partial]`` footer: how much ran, and why no baseline comparison."""
    return (
        f"\n[partial] {stats.completed_samples} / {stats.planned_samples} "
        f"samples completed (of {result.planned_examples} examples x {result.n_repeats})\n"
        "[partial] expected_vs_actual skipped (partial runs aren't comparable to baselines)."
    )
