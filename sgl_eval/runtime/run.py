"""``cmd_run`` orchestrator: walks Stage 1 (setup) -> Stage 2 (spec.run)
-> Stage 3 (report.render). Thin glue; each stage's heavy lifting lives
in its own module."""

from __future__ import annotations

import argparse

from sgl_eval.runtime import report, setup


def cmd_run(args: argparse.Namespace) -> int:
    ctx = setup.prepare_run(args)
    try:
        result = ctx.spec.run(
            sampler=ctx.sampler,
            gen=ctx.inputs.gen,
            n_repeats=ctx.inputs.n_repeats,
            num_examples=ctx.inputs.num_examples,
            num_threads=ctx.num_threads,
            predictions_writer=ctx.writer,
        )
    finally:
        setup.teardown(ctx)
    return report.render(result, ctx)
