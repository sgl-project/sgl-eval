# Benchmarks

`sgl-eval list` prints the registered set; `sgl-eval list -v` prints each
one's defaults (`n_repeats`, `thinking`, sampling params). Those are the
source of truth -- this file does not repeat them.

What follows is only what a benchmark needs **beyond** pointing sgl-eval at
an endpoint. Most need nothing.

| benchmark | category | notes |
|---|---|---|
| `gsm8k` | math | -- |
| `aime24/25/26` | math | one per contest year, same shape |
| `mmlu` | multichoice | -- |
| `gpqa` | multichoice | Diamond split |
| `mmmu_pro` | multichoice | vision-dependent; the endpoint must serve a VLM |
| `ruler2` | ruler2 | extra install, a required flag, generated data -- [see below](#ruler2) |

All scoring behavior (prompt, answer extraction, grading, pass@k /
majority@k aggregation) comes from the vendored NeMo-Skills slice, whatever
the benchmark.

---

## Matching a NeMo-Skills run

Sampling lines up with NS's `InferenceConfig` field for field, including
`min_p=0.0` and `repetition_penalty=1.0` -- sent explicitly rather than left
to the served model's `generation_config.json`, which would otherwise decide
them.

One knob differs: NS sends `seed=0`, sgl-eval sends none unless asked. Add
`--seed 0` to match. It has no effect at `temperature=0`.

---

## ruler2

A **group** of 12 synthetic long-context subtasks (`mk_niah_*`, `mv_niah_*`,
`qa_*`), averaged into one headline by upstream's own
`ruler2_score.compute_score`.

```bash
pip install 'sgl-eval[longcontext]'      # transformers, nltk, wonderwords, inflect

sgl-eval run ruler2 --base-url http://localhost:30000/v1 \
  --ruler2-seq-len 131072
```

Output is the headline plus a per-subtask breakdown, which is what tells you
*which* capability degraded:

```
* score        =  62.41%
    mk_niah_basic  =  91.00%
    qa_hard        =  28.50%
    ...
```

`sgl-eval run --help` lists the `--ruler2-*` flags under `ruler2 options`.
Three things about them are not obvious from the help text:

**The dataset is generated, not downloaded.** It is bound to a tokenizer and
a target length, so the cache key is `(tokenizer, seq_len, dataset_size)`
under `~/.cache/sgl_eval/ruler2/<setup>/`. First build of a 128k setup takes
tens of minutes and lands ~600 MB; later runs reuse it. Generation runs the
vendored scripts with upstream's fixed `random_seed=42`, so a given config
reproduces NeMo-Skills' dataset byte for byte. There is deliberately **no
knob that shrinks `seq_len`**: it defines the dataset and therefore the
score, which puts it under the vendoring rule.

**The window must hold prompt *plus* answer.** Prompts are sized with the raw
tokenizer, but requests go through `/v1/chat/completions`, where the server
prepends a chat template; and with `max_tokens` unset, a window exactly equal
to `seq_len` leaves zero room to generate. Both failures are silent -- empty
samples score 0, so the run would finish with all-zero metrics and a
successful exit code. sgl-eval therefore reads the endpoint's context length
up front and **refuses to start** below `seq_len + max_tokens` (or
`seq_len + 512` when `max_tokens` is unset). RULER2 is meant to be swept over
lengths *below* the window, so pick `seq_len` accordingly rather than
matching it to the window.

**Concurrency defaults to 4, not 64.** The runner limits in-flight
*requests*, not tokens; 64 concurrent 128k prompts would put ~8M tokens in
flight. Raise it with `--num-threads` if the server can take it -- batch
composition affects generation numerics, so the value is recorded in
`metrics.json` as part of what the result is bound to.

### Matching an NS ruler2 run specifically

Three values define the dataset rather than the request, so they are yours to
align: the same `--ruler2-tokenizer` (sgl-eval otherwise defaults to the
served model id), the same `--ruler2-seq-len` (the official sweep picks it
explicitly, e.g. `1048576 - 768` to reserve room to answer), and the same
`--ruler2-dataset-size`.

`--num-examples` is *not* a substitute for `--ruler2-dataset-size`: it slices
the generated file (NS's `++max_samples`), it does not change what gets
generated.
