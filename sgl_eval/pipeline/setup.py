"""Stage 1: resolve args, build sampler, mkdir, install sigint. Produces
the ``RunContext`` consumed by Stage 2 / Stage 3."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from sgl_eval.evals._loader import load_from_path
from sgl_eval.predictions import PredictionsWriter
from sgl_eval.preset import ResolvedRunInputs, resolve_run_inputs
from sgl_eval.registry import EvalSpec, get
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import Example, GenConfig


@dataclass
class RunContext:
    """Stage 1 -> Stage 2/3 handoff bag."""

    inputs: ResolvedRunInputs
    sampler: ChatCompletionSampler
    spec: EvalSpec
    run_dir: Path
    writer: Optional[PredictionsWriter]
    stamp: str
    num_threads: int
    args: argparse.Namespace
    load_examples: Optional[Callable[[Optional[int]], List[Example]]]
    _prev_sigint_handler: Any


def prepare_run(args: argparse.Namespace) -> RunContext:
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
    load_examples = load_from_path(args.from_dataset) if args.from_dataset else None

    return RunContext(
        inputs=inputs,
        sampler=sampler,
        spec=spec,
        run_dir=run_dir,
        writer=writer,
        stamp=stamp,
        num_threads=args.num_threads,
        args=args,
        load_examples=load_examples,
        _prev_sigint_handler=prev_sigint,
    )


def teardown(ctx: RunContext) -> None:
    if ctx.writer is not None:
        ctx.writer.close()
    signal.signal(signal.SIGINT, ctx._prev_sigint_handler)


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


def _warn_if_greedy_repeats(n_repeats: int, gen: GenConfig) -> None:
    if n_repeats > 1 and gen.temperature == 0.0:
        print(
            f"WARNING: n_repeats={n_repeats} but temperature=0.0 (greedy). "
            f"All {n_repeats} samples per example will be identical -- pass "
            "--temperature N (e.g. 1.0 for DSv3.2/V4, 0.6 for R1) for stochastic sampling.",
            file=sys.stderr,
        )
