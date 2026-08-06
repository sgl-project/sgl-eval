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

---

## Usage

Three subcommands: `list`, `ping`, `run <name>`. See `sgl-eval --help` for
flags.

Each run prints a summary with the headline metric on top -- single-shot
accuracy, averaged across the `k` repeats when `k > 1` -- and writes the
full payload as JSON under `--out-dir`. For example:

```
== aime25 ==
30 examples x 16 repeats  |  823.7s  |  4293 tok/s  |  3.5M tokens

* pass@1[avg-of-16]  =  78.96% +/- 1.21% (SEM 0.30%)
  pass@16            =  93.33%
  majority@16        =  93.33%
  no_answer          =  20.00%  [warn: consider --max-tokens]
```

---

## Partial runs & subsetting

A run doesn't have to cover the full dataset, finish to completion, or even
use the vendored questions. Four mechanisms:

### Subsetting -- `--num-examples N`

Run only the first `N` examples (smoke tests, quick sanity checks). Omit it
(or set `num_examples: null` in a preset) to run the full set.

```bash
sgl-eval run aime25 --base-url http://localhost:30000/v1 --num-examples 5
```

### Early stop -- `Ctrl-C`

A run can be stopped early without losing scored work:

- **First `Ctrl-C`** -- kills in-flight requests, dumps everything scored
  so far, writes `metrics.json` flagged `partial: true`, and exits `130`.
- **Second `Ctrl-C`** -- hard-exit (escape hatch if cleanup hangs).

A partial run reports a **sample-level score range** instead of a single
number: missing samples are counted once as all-wrong (lower bound) and
once as all-correct (upper bound), so the true score is guaranteed to lie
inside. The preset `expected_vs_actual` comparison is skipped (a half-run
isn't comparable to a baseline).

```
[partial] 240 / 480 samples completed (240 unfinished, n_repeats=16)
[partial] examples: 12 full / 6 partial / 12 dropped (30 planned)
[partial] score range: [39.17%, 89.17%] (missing samples assumed all-wrong / all-correct)
```

### Custom dataset -- `--from-dataset <path>`

Replace the vendored dataset for one run with your own NS-shape JSONL
(`{id?, problem, expected_answer}`, one object per line). Only the
questions change -- scoring still goes through the vendored grader.

```bash
sgl-eval run aime25 --base-url http://localhost:30000/v1 --from-dataset ./my_problems.jsonl
```

### Offline recompute -- `refresh`

Every run streams per-sample predictions to
`<out-dir>/sgl_eval_<name>_<stamp>/output-rs*.jsonl` (disable with
`--no-dump-predictions`). `refresh` rebuilds `metrics.json` from those
files -- re-aggregating (pass@k / majority@k / token tally / partial
counts / score bounds) without re-sampling, and preserving provenance
(`model` / `base_url` / `ns_commit_sha` / `preset` / ...). It makes no
requests; a partial run refreshes into the same score range.

```bash
sgl-eval refresh ~/.sgl_eval/sgl_eval_aime25_<stamp>/
```

---

## Presets

Save a `(benchmark, endpoint, sampling, n_repeats, expected)` bundle to
`~/.sgl_eval/presets/<name>.yaml` and replay with `sgl-eval run --preset
<name>`. See [`preset.md`](preset.md) for schema, example, usage, and
override priority.

---

## Supported benchmarks

`sgl-eval list` for the registered set; `sgl-eval list -v` for per-benchmark
defaults (`n_repeats`, `thinking`, sampling params). All scoring behavior
(prompt, answer extraction, grading, pass@k / majority@k aggregation) comes
from the vendored NeMo-Skills slice.

### RULER2 -- long context

`ruler2` is a **group** of 12 synthetic subtasks (`mk_niah_*`, `mv_niah_*`,
`qa_*`), averaged into one headline by upstream's own
`ruler2_score.compute_score`. It differs from every other benchmark in three
ways worth knowing before the first run:

```bash
pip install 'sgl-eval[longcontext]'      # transformers, nltk, wonderwords, inflect

sgl-eval run ruler2 --base-url http://localhost:30000/v1 \
  --ruler2-seq-len 131072
```

**1. The dataset is generated, not downloaded.** It is bound to a tokenizer and
a target length, so the cache key is `(tokenizer, seq_len, dataset_size)` under
`~/.cache/sgl_eval/ruler2/<setup>/`. First build of a 128k setup takes tens of
minutes and lands ~600 MB; later runs reuse it.

| flag | default | meaning |
|---|---|---|
| `--ruler2-seq-len` | **required** | target context length; no default because the data is per-length |
| `--ruler2-tokenizer` | the served model id | HF repo id or local path used to size samples |
| `--ruler2-tokenizer-type` | `hf` | `hf` or `openai` (tiktoken) |
| `--ruler2-dataset-size` | `100` | samples per subtask |
| `--ruler2-tasks` | all 12 | space-separated subset; scores a flagged partial average |

`sgl-eval run --help` lists them under `ruler2 options`. They go straight to the
vendored generator, whose own `random_seed` is fixed at 42 -- so a given config
reproduces NeMo-Skills' dataset byte for byte. There is deliberately **no knob
that shrinks `seq_len`**: it defines the dataset and therefore the score, which
puts it under the vendoring rule. The chosen values are recorded in
`metrics.json` so a result always names its setup.

**2. The window must hold prompt *plus* answer.** Prompts are sized with the
raw tokenizer, but requests go through `/v1/chat/completions`, where the server
prepends a chat template. And with `max_tokens` unset, a window exactly equal to
`seq_len` leaves zero room to generate. Both failures are silent -- the sampler
turns them into empty samples scoring 0, so the run finishes with all-zero
metrics and a successful exit code. sgl-eval therefore queries the endpoint's
context length up front and **refuses to start** unless it is at least
`seq_len + max_tokens` (or `seq_len + 512` when `max_tokens` is unset). RULER2 is
meant to be swept over lengths *below* the window, so pick `seq_len` accordingly
rather than matching it to the window.

**3. Concurrency defaults to 4, not 64.** The runner limits in-flight
*requests*, not tokens; 64 concurrent 128k prompts would put ~8M tokens in
flight. Override with `--num-threads` if the server can take it. Note that batch
composition affects generation numerics, so `num_threads` is part of what a
result is bound to -- it is recorded in `metrics.json`.

### Matching a NeMo-Skills run

Same vendored generator, prompt (`generic/default` is a bare `{question}`
passthrough), endpoint type (`chat`), grader, and aggregator. Sampling lines up
field for field with NS's `InferenceConfig` -- including `min_p=0.0` and
`repetition_penalty=1.0`, which are sent explicitly rather than left to the
served model's `generation_config.json`. One knob differs: NS sends `seed=0`,
sgl-eval sends none unless asked, so add `--seed 0` for an exact match. It has
no effect at `temperature=0`.

Three values are yours to align, because they define the dataset rather than
the request: pass the same `--ruler2-tokenizer` NS was given (sgl-eval
otherwise defaults to the served model id), the same `seq_len` (the official
RULER2 sweep picks it explicitly, e.g. `1048576 - 768` to reserve room to
answer), and the same `dataset_size`. `--num-examples` is *not* a substitute for
`dataset_size`: it slices the generated file (NS's `++max_samples`), it does not
change what gets generated.

Output shows the headline plus a per-subtask breakdown, which is what tells you
*which* capability degraded:

```
* score        =  62.41%
    mk_niah_basic  =  91.00%
    qa_hard        =  28.50%
    ...
```

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

---

## Roadmap

- **Replace the accuracy-eval surface in `sgl-project/sglang`.** Today
  `sglang.test.run_eval` + assorted per-test ad-hoc harnesses do this
  job. sgl-eval aims to be the single client SGLang's CI calls.
- **More benchmarks within `math` and `multichoice`** (MATH-500, AIME26,
  MMLU-Pro, GPQA-extended, ...). Each is one row in `_registry.py:_TABLE`.
- **New metrics types** (require a new runner per category, but graders
  are usually already in NeMo-Skills):
  - `long_context`: LongBench V2, MRCR (RULER2 is in -- see below)
  - `code`: HumanEval, MBPP, LiveCodeBench (with execution sandbox)
  - `instruction_following`: IFEval, IFBench
  - `multimodal` (VLM): MMMU, MathVista (needs an image-aware sampler)
  - `agentic` / tool use: BFCL, Tau-Bench
- **More vendor sources** beyond NeMo-Skills, when their slice is the
  best canonical implementation: `lm-evaluation-harness`, `lmms-eval`,
  `openai/simple-evals`. Same `_vendored/<source>/` + `SOURCES.yaml`
  pattern.
- **LLM-as-judge benchmarks** (Arena-Hard, MTBench). Needs a second
  judge endpoint and prompt-pair handling -- a real architectural
  addition, not just a benchmark row.
- **Regression CI infra**: publish per-run metrics to a `sgl-eval-data`
  repo, compare against rolling baselines, fail PRs on regression.

---

## Out of scope

- **Performance benchmarking.** Latency, throughput, scheduling. Lives
  in SGLang's `bench_serving.py`. sgl-eval records `latency` /
  `output_throughput` only as side metrics, never as the headline.
- **Model training or fine-tuning.**
- **Multi-server orchestration.** Each invocation targets one
  OpenAI-compatible endpoint.
- **Browser / OS-level agent loops** (full BrowseComp-style sandboxing).

---

## License

Apache-2.0. See `LICENSE`. Vendored NeMo-Skills sources are also Apache-2.0;
see `NOTICE` for attribution and the list of vendored files.
