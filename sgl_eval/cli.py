"""``sgl-eval`` CLI entry point.

Subcommands:
  list                 enumerate registered benchmarks
  ping                 send one chat completion to the endpoint and print it
  run <name>           run a benchmark end-to-end
  preset list/show     manage saved (model, dataset, sampling) presets
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from sgl_eval import VENDORED_NS_ROOT
from sgl_eval import __version__ as _SGL_EVAL_VERSION
from sgl_eval.evals._predictions import PredictionsWriter
from sgl_eval.metrics import dump_run, format_summary
from sgl_eval.preset import (
    add_preset_run_flag,
    make_run_meta_block,
    print_expected_vs_actual,
    register_preset_subcommand,
    resolve_run_inputs,
)
from sgl_eval.registry import get, list_evals
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import GenConfig


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="sgl-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list registered benchmarks")
    p_list.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show per-benchmark defaults (sampling params, n_repeats, thinking)",
    )
    p_list.set_defaults(func=cmd_list)

    p_ping = sub.add_parser("ping", help="send one request to the endpoint")
    _add_endpoint_args(p_ping)
    p_ping.add_argument("--prompt", default="Reply with the single word: pong.")
    p_ping.add_argument("--max-tokens", type=int, default=64)
    p_ping.add_argument("--temperature", type=float, default=0.0)
    p_ping.set_defaults(func=cmd_ping)

    p_run = sub.add_parser("run", help="run a benchmark")
    p_run.add_argument(
        "name",
        nargs="?",
        default=None,
        help="benchmark name (see `sgl-eval list`); optional if --preset provides one",
    )
    add_preset_run_flag(p_run)
    _add_endpoint_args(p_run, base_url_required=False)
    p_run.add_argument("--num-examples", type=int, default=None)
    p_run.add_argument("--num-threads", type=int, default=64)
    p_run.add_argument("--n-repeats", type=int, default=None)
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--top-p", type=float, default=None)
    p_run.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override chat_template_kwargs.thinking (per-benchmark default applies otherwise)",
    )
    p_run.add_argument(
        "--out-dir",
        default="~/.sgl_eval",
        help="parent dir for run folders; each run writes into "
        "<out-dir>/sgl_eval_<name>_<stamp>/ (default: ~/.sgl_eval)",
    )
    p_run.add_argument(
        "--no-dump-predictions",
        dest="dump_predictions",
        action="store_false",
        help="skip streaming per-sample prediction JSONL (output-rs*.jsonl)",
    )
    p_run.set_defaults(func=cmd_run, dump_predictions=True)

    register_preset_subcommand(sub)

    args = parser.parse_args(argv)
    return args.func(args)


def _add_endpoint_args(p: argparse.ArgumentParser, *, base_url_required: bool = True) -> None:
    p.add_argument(
        "--base-url",
        required=base_url_required,
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://host:30000/v1",
    )
    p.add_argument("--model", default=None, help="model id (defaults to first /v1/models entry)")
    p.add_argument("--api-key", default="EMPTY")


def cmd_list(args: argparse.Namespace) -> int:
    specs = list_evals()
    if not specs:
        print("(no benchmarks registered yet)")
        return 0
    if args.verbose:
        for s in specs:
            gen = s.default_gen
            ctk = gen.chat_template_kwargs or {}
            thinking = bool(ctk.get("thinking"))
            print(f"\n{s.name}  [{s.category}]")
            print(f"  description : {s.description}")
            print(f"  n_repeats   : {s.default_n_repeats}")
            print(f"  thinking    : {thinking}")
            print(f"  temperature : {gen.temperature}")
            print(f"  top_p       : {gen.top_p}")
            print(f"  max_tokens  : {gen.max_tokens}")
        return 0
    width = max(len(s.name) for s in specs)
    for s in specs:
        print(f"  {s.name:<{width}s}  [{s.category}]  {s.description}")
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    sampler = ChatCompletionSampler(base_url=args.base_url, model=args.model, api_key=args.api_key)
    gen = GenConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    sample = sampler([{"role": "user", "content": args.prompt}], gen)
    print(f"model            : {sampler.model}")
    print(f"finish_reason    : {sample.finish_reason}")
    print(f"completion_tokens: {sample.completion_tokens}")
    print(f"prompt_tokens    : {sample.prompt_tokens}")
    print("--- response ---")
    print(sample.text)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    inputs = resolve_run_inputs(args, get)
    spec = get(inputs.benchmark)

    sampler = ChatCompletionSampler(
        base_url=inputs.base_url, model=inputs.model, api_key=args.api_key
    )
    _warn_if_greedy_repeats(inputs.n_repeats, inputs.gen)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir).expanduser() / f"sgl_eval_{spec.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    writer = PredictionsWriter(run_dir, inputs.n_repeats) if args.dump_predictions else None
    prev_sigint = signal.signal(signal.SIGINT, _make_sigint_handler(sampler))
    try:
        result = spec.run(
            sampler=sampler,
            gen=inputs.gen,
            n_repeats=inputs.n_repeats,
            num_examples=inputs.num_examples,
            num_threads=args.num_threads,
            predictions_writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()
        signal.signal(signal.SIGINT, prev_sigint)

    print(format_summary(result))
    run_meta = _build_run_meta(
        args=args, sampler=sampler, gen=inputs.gen, stamp=stamp, base_url=inputs.base_url
    )
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
    preset_block = make_run_meta_block(args, inputs.preset)
    if preset_block:
        run_meta["preset"] = preset_block
    metrics_path = dump_run(result, run_dir, run_meta=run_meta)
    print(f"\nMetrics: {metrics_path}")
    if writer is not None:
        print(f"Predictions: {run_dir}  ({inputs.n_repeats} jsonl file(s))")
    if stats is not None:
        print(_format_partial_summary(result, stats), file=sys.stderr)
    else:
        print_expected_vs_actual(result, inputs.preset)
    # Exit code 130 if the user pressed Ctrl-C, even when every example
    # happened to finish before the abort propagated -- their intent was
    # to bail, so respect it.
    return 130 if sampler.aborted else 0


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


def _partial_stats(result: Any) -> _PartialStats:
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


def _format_partial_summary(result: Any, stats: _PartialStats) -> str:
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


def _make_sigint_handler(sampler: ChatCompletionSampler) -> Any:
    """First Ctrl-C: kill in-flight requests and flag for partial dump.
    Second Ctrl-C: hard-exit (escape hatch if partial cleanup hangs)."""
    count = 0

    def _handler(_signum: int, _frame: Any) -> None:
        nonlocal count
        count += 1
        if count >= 2:
            print("\nSecond Ctrl-C; exiting hard.", file=sys.stderr)
            os._exit(130)
        print(
            "\nAborting; killing in-flight requests, dumping partial results "
            "(press Ctrl-C again to force-exit)...",
            file=sys.stderr,
        )
        sampler.abort()

    return _handler


def _build_run_meta(
    *,
    args: argparse.Namespace,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    stamp: str,
    base_url: str,
) -> Dict[str, Any]:
    return {
        "timestamp": stamp,
        "model": sampler.model,
        "base_url": base_url,
        "num_threads": args.num_threads,
        "gen": dataclasses.asdict(gen),
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


def _warn_if_greedy_repeats(n_repeats: int, gen: GenConfig) -> None:
    if n_repeats > 1 and gen.temperature == 0.0:
        print(
            f"WARNING: n_repeats={n_repeats} but temperature=0.0 (greedy). "
            f"All {n_repeats} samples per example will be identical -- pass "
            "--temperature N (e.g. 1.0 for DSv3.2/V4, 0.6 for R1) for stochastic sampling.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
