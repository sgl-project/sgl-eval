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
| `mmlu_pro` | multichoice | text 10-choice; **not** `mmmu_pro` -- [see below](#mmlu-pro-vs-mmmu-pro) |
| `gpqa` | multichoice | Diamond split |
| `mmmu_pro` | multichoice | VLM endpoint; MMMU-Pro `standard (10 options)` -- [see below](#mmmu-pro-variants) |
| `mmmu_pro_vision` | multichoice | VLM endpoint; MMMU-Pro `vision` -- [see below](#mmmu-pro-variants) |
| `mindcube` | multichoice | VLM endpoint; MindCube tinybench, raw-QA setting -- [see below](#mindcube) |
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

Three knobs differ.

**`seed`**: NS sends `seed=0`, sgl-eval sends none unless asked. Add `--seed 0`
to match. It has no effect at `temperature=0`.

**`prompt_config`**: each benchmark uses upstream's default -- the one its
vendored `dataset/<name>/__init__.py` names in `GENERATION_ARGS`. An NS run
that passed `++prompt_config=<other>` asked a different question, so its score
will not reproduce until you pass the same prompt. `--prompt <name-or-path>`
does that: a bare name resolves against the vendored NS prompts and then
sgl-eval's own `evals/prompts/`, and anything with a `/` or a `.yaml` suffix
is read as a file, so a prompt this repo does not ship still works.

The case that actually comes up is AIME. Published AIME numbers are frequently
produced with `eval/matharena/aime`, which states that the answer is an integer
between 0 and 999; `generic/math` -- the default -- does not. Vendored here as
`matharena-aime`:

```bash
sgl-eval run aime25 --base-url http://localhost:30000/v1 \
  --model <model> --n-repeats 32 --temperature 1.0 --top-p 0.95 \
  --prompt matharena-aime
```

An overridden prompt is recorded under `prompt` in `metrics.json`; a score
carrying that key is not comparable with one that does not. `ruler2` rejects
the flag -- its prompt is a pure passthrough and the context is assembled by
the prepare scripts.

**`temperature`, if the NS run went through `ns eval`**: that pipeline does
*not* use `InferenceConfig`'s `temperature=0.0`. The repeat suffix on
`--benchmarks` decides it -- `<bench>` and `<bench>:0` mean greedy, but
`<bench>:1` and above default to **`temperature=0.7`**. So a harness building
its spec as `f"{bench}:{repeats}"` gets 0.7 even when it means "run once",
and sgl-eval's greedy default will not reproduce it. Pass
`--temperature 0.7 --seed 0` to match such a run, or re-baseline against
greedy.

---

## MMLU-Pro vs MMMU-Pro

Two unrelated benchmarks one letter apart, adjacent in `sgl-eval list`.

| | `mmlu_pro` | `mmmu_pro` |
|---|---|---|
| dataset | TIGER-Lab/MMLU-Pro, 12032 text questions | MMMU-Pro, multimodal |
| options | exactly 10, A-J | 4-10, varies per question |
| endpoint | any chat endpoint | needs a VLM |
| prompt | vendored `mcq-10choices`, enumerates A-J | sgl-eval's `mmmu-pro-cot`, `$LETTER` + CoT |

The prompts are deliberately not shared: `resolve_prompt` prefers the vendored
copy, so giving them one basename would silently hand MMMU-Pro a prompt that
enumerates ten letters for a question that may have four.

`mmlu_pro`'s split is ordered by subject, so it carries `sample_seed` -- a small
`--num-examples` samples across subjects instead of scoring one of them.

---

## MMMU-Pro variants

MMMU-Pro ships several HuggingFace configs. Two are registered, and they are
**different tasks -- their scores are not comparable**:

| | `mmmu_pro` | `mmmu_pro_vision` |
|---|---|---|
| HF config | `standard (10 options)` | `vision` |
| question text | in the prompt | rendered into the screenshot |
| options | up to 10, as text | in the screenshot (and echoed as text) |
| images per question | `<image 1..7>`, placed inline | one screenshot, placed first |
| prompt | sgl-eval's `mmmu-pro-cot`, asks for CoT | vendored `vlm/mmmu-pro`, no CoT |

`mmmu_pro_vision` is the one upstream NeMo-Skills ships, so it is the row to
use when reproducing an NS number or an `ns eval --benchmarks=mmmu-pro` run.
Upstream has no `standard (10 options)` module, which is why `mmmu_pro` keeps
an sgl-eval-own loader and prompt.

Picking the row is not sufficient on its own -- sgl-eval's defaults are
greedy and let the endpoint pick `max_tokens`, so both have to be passed
explicitly to match an `ns eval` run (see the `temperature` note
[above](#matching-a-nemo-skills-run) for why 0.7 rather than 0.0):

```bash
sgl-eval run mmmu_pro_vision --base-url ... \
    --max-tokens 32768 --temperature 0.7 --seed 0 --num-threads 512
```

`--num-threads` only affects wall-clock, not the expected score -- the
registered default is 64, NS's own runs use 512. Add `--num-examples N` if the
run being matched capped its sample count.

Verified equivalent to `ns eval --benchmarks=mmmu-pro`: same prepared
`test.jsonl` (identical md5), byte-identical rendered messages, and with
concurrency pinned to 1 on both sides, byte-identical generations and the same
score question by question.

> Concurrency should be score-neutral but is not: two back-to-back greedy runs
> at `--num-threads 64` differed on ~half the raw generations and landed 7
> points apart on 100 questions, because batch composition shifts kernel
> selection and flips argmax wherever the model is unsure. That spread dwarfs
> any harness difference -- treat a single concurrent run as a noisy estimate,
> and pin concurrency to 1 when a number has to be reproducible.

One difference has no sgl-eval equivalent: NS can mark an answer incorrect
when its token count exceeds a `max_seq_len` threshold. Nothing sets it in
the sglang harness, so it does not affect a comparison today.

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

---

## MindCube

[MindCube](https://github.com/mll-lab-nu/MindCube) (Yin et al., 2025) asks
whether a VLM can build a spatial mental model from a few views: 2-4 images of
a scene plus a multiple-choice question about positions, orientations, or
"what-if" movements. `mindcube` runs the paper's evaluation split
(`MindCube_tinybench.jsonl`, 1,050 questions: 600 `among`, 250 `around`,
200 `rotation`) in the **raw QA** setting -- views + question, answer directly.
The cognitive-map settings (`aug_cgmap_*`, `plain_cgmap_*`, `ff_rsn`) need
scaffold data and a map-parsing evaluator and are not registered.

sgl-eval-own, like `mmmu_pro`: NeMo-Skills has no MindCube module. What is
sgl-eval's and what is the official protocol's:

| | source |
|---|---|
| prompt | official `RAW_QA` template (`[Task] ... [Answer Instruction] ... [Question]`), verbatim |
| image placement | views before the text, in dataset order (image 1..N as the question refers to them), original bytes |
| answer extraction | official `extract_answer` regex cascade, ported verbatim |
| correctness | extracted letter == `gt_answer` |
| overall | official filtered accuracy (`translation` tracked under `task.translation` but excluded from `score`; tinybench has none) |
| per-setting | `task.among` / `task.around` / `task.rotation` |
| `no_answer` | share of responses the official extractor could not parse |
| `--n-repeats k` | `pass@1[avg-of-k]` / `pass@k` / `majority@k` via the vendored `MathMetrics` |

Data: `MLL-Lab/MindCube` `data.zip` (~640 MB) is downloaded from the Hub and
extracted once into `~/.cache/sgl_eval/mindcube/`. Set `SGL_EVAL_MINDCUBE_DIR`
to a directory that already contains the extracted `data/` to skip that.

The official prompt asks for the answer only, so the default is
`thinking=False`; pass `--thinking` for reasoning models. The extractor reads
the model's final `content`, not its reasoning stream. `--num-examples N`
takes a setting-balanced sample. `--from-dataset` is not supported (image
inputs are required).
