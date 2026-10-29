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
| `ruler2` | ruler2 | extra install, a required flag, generated data -- [see below](#ruler2) |
| `deepswe` | harbor | Docker + Compose on this host, a tool-calling endpoint, hours per trial -- [see below](#deepswe) |

All scoring behavior (prompt, answer extraction, grading, pass@k /
majority@k aggregation) comes from the vendored NeMo-Skills slice for the
math and multichoice benchmarks, and from the vendored pier trial runtime
plus mini-swe-agent for `deepswe`.

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

An overridden prompt, including its YAML contents, is recorded under `prompt`
in `metrics.json`; a score carrying that key is not comparable with one that
does not. `ruler2` rejects
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

### Custom math answer formats

For math benchmarks, a custom prompt YAML can also set `system` and
`evaluator_config`. The system template overrides `GenConfig.system_message`
for that run; omitting it preserves the caller's system message, and an empty
string clears it. The `user` template is always rendered. Use doubled braces
for literal braces in either template.

For example, save the following as `aime-exact-answer.yaml` to request the
Explanation / Exact Answer / Confidence format discussed in
[#22](https://github.com/sgl-project/sgl-eval/pull/22):

```yaml
system: |-
  Your response should be in the following format:
  Explanation: {{your explanation for your final answer}}
  Exact Answer: {{your succinct, final answer}}
  Confidence: {{your confidence score between 0% and 100% for your answer}}
user: "{problem}"
evaluator_config:
  relaxed_extraction: true
  extract_regex: '(?m)^[ \t]*(?:\*\*)?Exact Answer:(?:\*\*)?[ \t]*(?:\*\*)?(\S(?:[^\r\n]*?\S)?)(?:\*\*)?[ \t]*\r?$'
```

```bash
sgl-eval run aime26 --base-url http://localhost:30000/v1 \
  --prompt ./aime-exact-answer.yaml \
  --max-tokens 163840 --temperature 1.0 --top-p 0.95 --num-threads 64
```

`evaluator_config` is passed to the vendored NeMo-Skills `MathEvaluator`.
The example extracts a nonempty answer from its own line, accepts optional
bold Markdown, preserves multiplication such as `6*7`, and falls back to
`\boxed{...}` when no answer line matches. These fields are supported only
for math benchmarks. The default AIME prompt and boxed extraction stay
unchanged; this custom protocol needs its own baseline.

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

## deepswe

[DeepSWE](https://github.com/datacurve-ai/deep-swe): 113 long-horizon
coding-agent tasks from active open-source repositories, in the Harbor task
format, each with a prebuilt Docker image and hidden tests. The model works
through [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
(v2.4.6, its stock `mini.yaml` prompt, OpenAI function calling with one
`bash` tool). The trial lifecycle, patch collection, verifier and reward
aggregation are the vendored [pier](https://github.com/datacurve-ai/pier)
runtime at v0.3.1, so a score here follows the same rules as a `pier run`.

The agent loop runs in the sgl-eval process. The task container only executes
its shell commands, so tasks keep their `no-network` policy and the model
endpoint is reached from the host, not from inside a sandbox.

### Prerequisites

- **Docker with the Compose v2 plugin** on the machine running sgl-eval
  (`docker info` and `docker compose version` must work for this user).
  Budget well over 100 GB of disk for the task images.
- **A tool-calling endpoint.** The server must return `tool_calls` for
  requests that carry `tools` (sglang: `--tool-call-parser <name>`, plus
  `--reasoning-parser` for reasoning models) and hold long contexts: prompts
  routinely pass 200K tokens, so 393216 is the recommended floor and below
  131072 sgl-eval refuses to start. Both are checked before the first trial.
- **Concurrency is trials, not requests.** Every trial owns a task container
  (2 CPUs / 8 GB by its `task.toml`) and, at the end, a verifier container.
  The default is `--num-threads 4`; each trial takes one to a few hours.

### Three steps

```bash
# 1. Does Docker work here? Runs the reference solution, no model involved.
sgl-eval run deepswe --deepswe-oracle --deepswe-task abs-module-cache-flags

# 2. One real trial against your endpoint.
sgl-eval run deepswe --base-url http://localhost:30000/v1 --model <model> \
  --temperature 1.0 --top-p 0.95 --deepswe-task abs-module-cache-flags

# 3. The whole set, resumable.
sgl-eval run deepswe --base-url http://localhost:30000/v1 --model <model> \
  --temperature 1.0 --top-p 0.95 --num-threads 4 --run-dir ~/runs/deepswe-<model>
```

The first run pulls each task's image on demand; a fresh host spends its
first half hour mostly pulling. Sampling is model-dependent as everywhere in
sgl-eval; the DeepSeek V4 presets (`--load-preset-from-model-id`) apply here
too.

### Flags

`sgl-eval run --help` lists them under `deepswe options`.

- `--deepswe-task ID` (repeatable) selects tasks by directory name;
  `--num-examples N` takes the first `N` alphabetically.
- `--deepswe-oracle` replays `solution/solve.sh` instead of a model.
- `--deepswe-agent-timeout-multiplier X` (default 2.0) scales each task's
  `agent.timeout_sec` (10800 s for DeepSWE). A trial whose agent hits the
  deadline is still collected and verified.
- `--deepswe-max-retries N` (default 1) re-runs a trial whose attempt ended
  in an exception, keeping only the last attempt. pier's exclusions apply:
  agent and verifier timeouts and reward-file errors are never retried.

### Resuming with `--run-dir`

`--run-dir DIR` makes the run directory explicit. Point a second invocation
at the same directory with the same command and finished trials are kept
while the rest run; the header says how many were replayed. The evaluation
settings (model, sampling, task selection, repeats, timeouts, retry policy)
are recorded in `run_config.json` and a different configuration is refused.
`--deepswe-retry-errored` additionally schedules finished-but-errored trials
again; it is the one flag a resume may change.

`Ctrl-C` cancels the running trials (their containers are stopped), keeps
what finished, and writes a `partial` `metrics.json`; re-running with the
same `--run-dir` continues. A second `Ctrl-C` exits immediately and may leave
containers behind.

### Reading the score

The headline `score` is pier's reward mean over finished trials. A trial's
reward comes from the verifier's `reward.txt` / `reward.json`; a trial that
finished without a reward (the agent or verifier raised) counts as 0 in the
denominator, exactly as pier's `Mean` does. The summary also prints:

- `resolved` (reward 1) and `failed` out of the finished trials;
- `errored`, how many finished trials carried an exception (they are already
  inside `failed` or, rarely, `resolved`: an agent that timed out after
  committing a correct fix still scores 1);
- `not_run`, planned trials that did not finish (Ctrl-C or a crash); when it
  is non-zero the run is `partial` and the score is not comparable;
- `f2p_mean` / `p2p_mean`, the fail-to-pass and pass-to-pass fractions the
  verifiers report.

Token lines are split: `avg_*_tokens/trial` sum a whole trajectory, and
`avg_*_tokens/response` average over model responses (format-error retries
included).

Per trial, `<run-dir>/trials/<task>__rs<n>/` holds pier's canonical
`result.json`, `agent/mini-swe-agent.trajectory.json`, the collected
`artifacts/model.patch`, and `verifier/` with `reward.json`, `ctrf.json` and
the raw test output.

### Matching a pier run

`metrics.json` records the dataset revision, the vendored pier and
mini-swe-agent commits, every task's image and checksum, the timeouts and the
retry policy. Two things differ from an agent installed inside the container
by pier and are worth knowing when comparing numbers:

- sgl-eval sends its sampling fields explicitly (`temperature`, `top_p`,
  `min_p`, `repetition_penalty`, `reasoning_effort` when set), where litellm
  dropped `reasoning_effort` for OpenAI-compatible endpoints. Launch the server
  with the same defaults on both sides.
- The task container is the pristine task image. pier's installed-agent image
  additionally installs `curl`, `build-essential`, `git` and a Python for
  mini-swe-agent, and its shell commands can see `OPENAI_*` variables.
