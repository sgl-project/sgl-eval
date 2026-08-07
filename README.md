# sgl-eval

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

One-click accuracy evaluation harness for [SGLang](https://github.com/sgl-project/sglang).

Point at any OpenAI-compatible endpoint. Scoring logic (graders, evaluators,
prompts, dataset configs) is vendored from [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills);
sgl-eval contributes the transport, runner, and benchmark wiring.

---

## Quick start

```bash
pip install git+https://github.com/sgl-project/sgl-eval

sgl-eval ping --base-url http://localhost:30000/v1
sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-examples 50
```

Four subcommands: `run`, `list`, `ping`, `preset`. `sgl-eval run --help` is
the full flag reference -- endpoint, sampling overrides (`--temperature`,
`--seed`, `--thinking`, ...), and any flags the benchmark itself adds.

---

## Reading a run

Each run prints the headline metric first -- single-shot accuracy, averaged
across the `k` repeats when `k > 1` -- and writes the same payload plus
provenance (model, endpoint, sampling config, vendored NS commit) as
`metrics.json` under `--out-dir`.

```
== aime25 ==
30 examples x 16 repeats  |  823.7s  |  4293 tok/s  |  3.5M tokens

* pass@1[avg-of-16]  =  78.96% +/- 1.21% (SEM 0.30%)
  pass@16            =  93.33%
  majority@16        =  93.33%
  no_answer          =  20.00%  [warn: consider --max-tokens]
```

While the run is going, the progress bar carries a live accuracy. For a
sanity check that is usually the whole point: watch it, decide, stop.

```
gsm8k:  34%|###4      | 452/1319 [02:11<04:12, 3.4it/s, acc=81.42%]
```

Every scored sample is streamed to
`<out-dir>/sgl_eval_<name>_<stamp>/output-rs*.jsonl` as it lands (disable
with `--no-dump-predictions`), so the per-sample record survives however the
run ends.

---

## Running less than the whole thing

- **`--num-examples N`** -- only the first `N` examples.
- **`Ctrl-C`** -- kills in-flight requests, keeps everything already scored,
  and writes `metrics.json` flagged `partial: true` with how much ran, so a
  half-run can't later be mistaken for a full one. Exits `130`; a second
  `Ctrl-C` hard-exits if cleanup hangs. The preset `expected_vs_actual`
  comparison is skipped -- a half-run isn't comparable to a baseline.
- **`--from-dataset <path>`** -- swap in your own NS-shape JSONL
  (`{id?, problem, expected_answer}`) for one run. Only the questions
  change; scoring still goes through the vendored grader.

---

## Benchmarks

`sgl-eval list` for the registered set, `sgl-eval list -v` for each one's
defaults. See [`benchmarks.md`](benchmarks.md) for the ones that need more
than an endpoint (today: `ruler2`), and for how to match a NeMo-Skills run.

## Presets

Save a `(benchmark, endpoint, sampling, n_repeats, expected)` bundle to
`~/.sgl_eval/presets/<name>.yaml` and replay with `sgl-eval run --preset
<name>`. See [`preset.md`](preset.md) for schema, example, usage, and
override priority.

---

## Architecture

**Anything that decides a score is vendored verbatim from NeMo-Skills.**
sgl-eval contributes only transport: an OpenAI client, a threadpool runner,
a CLI, and the thin glue that wires upstream pieces into one command.

```
+----------------------------------------------------+
|  sgl-eval                                          |
|    cli, sampler, runner, registry, metrics         |
|    evals/                                          |
+----------------------------------------------------+
|  vendored from NeMo-Skills                         |
|    math_grader, evaluator/, metrics/,              |
|    dataset/<bench>/, prompts/*.yaml                |
+----------------------------------------------------+
```

The slice is pinned at a specific commit in
`sgl_eval/_vendored/nemo_skills/SOURCES.yaml`. To upgrade, bump
`synced_from_sha` there and run:

```bash
python scripts/sync_vendored.py    # re-fetch all vendored files
pytest                             # upstream's own tests run against the
                                   # new slice -- catches behavior drift
```

Adding a benchmark inside an existing category (`math`, `multichoice`) is one
row in `_registry.py:_TABLE`. A new category needs a runner alongside it --
graders are usually already in NeMo-Skills.

---

## Scope

The goal is to be the single accuracy-eval client SGLang's CI calls, in place
of `sglang.test.run_eval` and the assorted per-test harnesses.

Not in scope: performance benchmarking (latency / throughput / scheduling --
that is SGLang's `bench_serving.py`; sgl-eval records them only as side
metrics, never as the headline), training or fine-tuning, multi-server
orchestration (one endpoint per invocation), and OS-level agent loops.

---

## License

Apache-2.0. See `LICENSE`. Vendored NeMo-Skills sources are also Apache-2.0;
see `NOTICE` for attribution and the list of vendored files.
