"""Prepare run inputs and resources, including prediction files and SIGINT handling."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sgl_eval import __version__ as _SGL_EVAL_VERSION
from sgl_eval.evals._loader import load_from_path
from sgl_eval.evals._prompts import resolve_prompt
from sgl_eval.predictions import PredictionsWriter
from sgl_eval.preset import ResolvedRunInputs, resolve_run_inputs
from sgl_eval.registry import EvalSpec, collect_bench_args, get
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import Example, GenConfig

RUN_CONFIG_FILENAME = "run_config.json"
# The helper moved to registry so preset resolution can use it; keep the old name importable.
_collect_bench_args = collect_bench_args


@dataclass
class RunContext:
    """Stage 1 -> Stage 2/3 handoff bag."""

    inputs: ResolvedRunInputs
    # None when the benchmark run needs no endpoint (e.g. an oracle mode).
    sampler: Optional[ChatCompletionSampler]
    spec: EvalSpec
    run_dir: Path
    writer: Optional[PredictionsWriter]
    stamp: str
    num_threads: int
    args: argparse.Namespace
    load_examples: Optional[Callable[[Optional[int]], List[Example]]]
    bench_args: Dict[str, Any]
    # None means "the benchmark's registered prompt"; set only by --prompt.
    prompt_yaml: Optional[Path]
    # True when --run-dir pointed at a compatible earlier run of a resumable benchmark.
    resume: bool
    # Set by the first Ctrl-C; benchmarks poll it to stop their own work.
    cancel_event: threading.Event
    _prev_sigint_handler: Any


def prepare_run(args: argparse.Namespace) -> RunContext:
    inputs = resolve_run_inputs(args, get)
    spec = get(inputs.benchmark)
    bench_args = collect_bench_args(args, spec.name)

    needs_endpoint = spec.requires_endpoint(bench_args) if spec.requires_endpoint else True
    sampler = (
        ChatCompletionSampler(base_url=inputs.base_url, model=inputs.model, api_key=args.api_key)
        if needs_endpoint
        else None
    )
    _warn_if_greedy_repeats(inputs.n_repeats, inputs.gen)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    num_threads = args.num_threads if args.num_threads is not None else spec.default_num_threads
    prompt_yaml = _resolve_prompt_override(getattr(args, "prompt", None))
    fingerprint = _run_config_payload(
        spec, inputs, sampler.model if sampler else inputs.model, bench_args, prompt_yaml
    )
    run_dir, resume = _resolve_run_dir(args, spec, stamp, fingerprint)

    writer = (
        PredictionsWriter(run_dir, inputs.n_repeats, spec.pred_schema)
        if args.dump_predictions
        else None
    )
    cancel_event = threading.Event()
    prev_sigint = signal.signal(signal.SIGINT, _make_sigint_handler(sampler, cancel_event))
    load_examples = load_from_path(args.from_dataset) if args.from_dataset else None

    return RunContext(
        inputs=inputs,
        sampler=sampler,
        spec=spec,
        run_dir=run_dir,
        writer=writer,
        stamp=stamp,
        num_threads=num_threads,
        args=args,
        load_examples=load_examples,
        bench_args=bench_args,
        prompt_yaml=prompt_yaml,
        resume=resume,
        cancel_event=cancel_event,
        _prev_sigint_handler=prev_sigint,
    )


def _resolve_prompt_override(spec: Optional[str]) -> Optional[Path]:
    """Resolve up front so a typo fails before the server is loaded, not at first render."""
    if spec is None:
        return None
    path = resolve_prompt(spec)
    if not path.exists():
        raise FileNotFoundError(f"--prompt {spec!r}: no prompt yaml at {path}")
    return path


def _run_config_payload(
    spec: EvalSpec,
    inputs: ResolvedRunInputs,
    model: Optional[str],
    bench_args: Dict[str, Any],
    prompt_yaml: Optional[Path],
) -> Dict[str, Any]:
    """What a run evaluates. Two runs with the same payload may share a run dir."""
    return {
        "benchmark": spec.name,
        "model": model,
        "gen": dataclasses.asdict(inputs.gen),
        "n_repeats": inputs.n_repeats,
        "num_examples": inputs.num_examples,
        "bench_args": {
            key: value for key, value in bench_args.items() if key not in spec.fingerprint_exclude
        },
        "prompt": prompt_yaml.read_text() if prompt_yaml is not None else None,
        "sgl_eval_version": _SGL_EVAL_VERSION,
    }


def _resolve_run_dir(
    args: argparse.Namespace, spec: EvalSpec, stamp: str, fingerprint: Dict[str, Any]
) -> Tuple[Path, bool]:
    """Pick the run directory; with --run-dir, decide whether this is a resume."""
    run_dir_arg = getattr(args, "run_dir", None)
    if run_dir_arg is None:
        run_dir = Path(args.out_dir).expanduser() / f"sgl_eval_{spec.name}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_config(run_dir, fingerprint)
        print(f"Run directory: {run_dir}")
        return run_dir, False

    run_dir = Path(run_dir_arg).expanduser()
    config_path = run_dir / RUN_CONFIG_FILENAME
    if config_path.exists():
        if not spec.resumable:
            sys.exit(
                f"error: --run-dir {run_dir} already holds a run and {spec.name} cannot "
                "resume one; pick a new directory."
            )
        previous = json.loads(config_path.read_text())
        changed = sorted(
            key
            for key in set(previous) | set(fingerprint)
            if previous.get(key) != fingerprint.get(key)
        )
        if changed:
            sys.exit(
                f"error: --run-dir {run_dir} was created with a different configuration "
                f"({', '.join(changed)} differ). Resume needs the same evaluation settings; "
                "use a new directory for a different run."
            )
        print(f"Run directory: {run_dir} (resuming)")
        return run_dir, True

    if run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(
            f"error: --run-dir {run_dir} is not empty and has no {RUN_CONFIG_FILENAME}; "
            "refusing to mix outputs into it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(run_dir, fingerprint)
    print(f"Run directory: {run_dir}")
    return run_dir, False


def _write_run_config(run_dir: Path, fingerprint: Dict[str, Any]) -> None:
    (run_dir / RUN_CONFIG_FILENAME).write_text(json.dumps(fingerprint, indent=2, sort_keys=True))


def teardown(ctx: RunContext) -> None:
    if ctx.writer is not None:
        ctx.writer.close()
    signal.signal(signal.SIGINT, ctx._prev_sigint_handler)


def _make_sigint_handler(
    sampler: Optional[ChatCompletionSampler], cancel_event: threading.Event
) -> Any:
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
        cancel_event.set()
        if sampler is not None:
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
