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

## Presets

Save a `(benchmark, endpoint, sampling, n_repeats, expected)` bundle to
`~/.sgl_eval/presets/<name>.yaml` and replay it with `--preset <name>`.
CLI flags always override preset values, so a preset is a starting point,
not a lock.

### Schema

```yaml
benchmark: aime24                  # required: a registered benchmark name

endpoint:
  base_url: http://host:30000/v1   # null/omit -> CLI must provide --base-url
  model: dsv3.2                    # null/omit -> autodiscover from /v1/models

n_repeats: 16
num_examples: null                 # null = full dataset; int = first N

sampling:                          # all sub-fields optional
  temperature: 1.0
  top_p: 0.95
  max_tokens: null
  thinking: true                   # mapped to chat_template_kwargs.thinking

expected:                          # optional, informational only
  score: 0.85                      # 0..1, headline metric (pass@1 for k>1,
                                   # plain score for k==1). Printed
                                   # alongside actual after each run --
                                   # never gates exit code.
```

Unknown / typo'd fields raise `ValueError` (a misspelled key like `top_pp`
fails loudly instead of being silently dropped).

### Example

`~/.sgl_eval/presets/dsv32-aime24.yaml`:

```yaml
benchmark: aime24
endpoint:
  base_url: http://host:30000/v1
  model: dsv3.2
n_repeats: 16
sampling:
  temperature: 1.0
  thinking: true
expected:
  score: 0.85
```

### Usage

```bash
# Run as-is
sgl-eval run --preset dsv32-aime24

# CLI override -- a single field
sgl-eval run --preset dsv32-aime24 --temperature 0.6

# CLI override -- the benchmark itself
sgl-eval run aime25 --preset dsv32-aime24

# Path instead of a name (lets you keep a preset next to a project)
sgl-eval run --preset ./local-config.yaml

# Inspect
sgl-eval preset list
sgl-eval preset show dsv32-aime24
```

### Priority

`CLI flag (if not None) > preset field (if not None) > benchmark default`

`0.0` / `False` count as set -- only `None` falls through. So
`temperature: 0.0` (greedy) in a preset is honored, not skipped.

### Provenance

Every run writes a `preset` block into `metrics.json` recording the
spec / resolved path / benchmark / expected_score, so a recorded run is
fully traceable back to the preset that drove it.

---

## Supported benchmarks

`sgl-eval list` for the registered set; `sgl-eval list -v` for per-benchmark
defaults (`n_repeats`, `thinking`, sampling params). All scoring behavior
(prompt, answer extraction, grading, pass@k / majority@k aggregation) comes
from the vendored NeMo-Skills slice.

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
  - `long_context`: LongBench V2, RULER, MRCR
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
