# sgl-eval — project context for Claude Code

One-click accuracy evaluation harness for SGLang. Point at any
OpenAI-compatible endpoint, get reproducible numbers.

## Core architectural principle

**Anything that decides a score is vendored verbatim from its upstream.**
Math and multichoice benchmarks vendor
[NVIDIA/NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills); the `deepswe`
agentic benchmark vendors the [pier](https://github.com/datacurve-ai/pier)
trial runtime and the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
loop. sgl-eval contributes only transport (OpenAI client, threadpool runner,
CLI, host-side adapters) and the thin glue that wires upstream pieces into
one command.

Enforced by:

- `sgl_eval/_vendored/<pkg>/` holds each vendored slice
  (`nemo_skills`, `pier`, `mini_swe_agent`).
- `sgl_eval/_vendored/<pkg>/SOURCES.yaml` pins the upstream commit, records
  every file's source path and every transform applied to it.
- `python scripts/sync_vendored.py --check` regenerates every package from
  its manifest and fails on any drift (CI runs it).
- `scripts/audit_vendored.py` (run by `tests/test_vendor_audit.py`) fails
  if any vendored file still has unrewritten upstream imports or imports
  sgl-eval code.
- NS's own `tests/test_math_equal.py` + `tests/test_base_metrics.py` are
  vendored alongside and run on every `pytest`. Drift surfaces
  immediately.

## File layout

```
sgl_eval/
├── cli.py / sampler.py / runner/ / registry.py / metrics.py / types.py
│   sgl-eval's own code: transport + plumbing.
├── pipeline/                     # run setup (run dir, resume fingerprint), report
├── evals/
│   per-benchmark glue (math runner, mcq runner, harbor runner, registry
│   table, prompt render, prediction dict adapter, dataset loader).
├── agents/mini_swe_agent.py      # host-side agent: transport + sandbox hooks
├── sandbox/docker.py             # per-action exec in a pier-managed container
└── _vendored/                    # DO NOT hand-edit.
    ├── nemo_skills/              # math grader, metrics, evaluators, datasets, prompts
    ├── pier/                     # Trial / TrialQueue / Verifier / DockerEnvironment / Mean
    └── mini_swe_agent/           # DefaultAgent, LitellmModel behavior, LocalEnvironment, mini.yaml
```

## Editing rules

- **Never hand-edit anything under `sgl_eval/_vendored/`.** To change
  vendored content, edit that package's `SOURCES.yaml` and run
  `python scripts/sync_vendored.py <pkg>`. Transforms (`replace`,
  `drop_statements`, `drop_functions`, `drop_imports`) carry an expected
  match count and fail loudly when upstream moves.
- New functionality that touches scoring (grader, aggregator, prompt,
  dataset prep, trial lifecycle) goes through vendoring. New transport /
  runner / CLI features are SE code, fine to add directly.
- Sampling / generation defaults belong in
  `sgl_eval/evals/_registry.py:_TABLE`, not in vendored code.

## Defaults

Sampling params (`temperature=0.0`, `top_p=0.95`, `max_tokens=None`)
match upstream NeMo-Skills' `InferenceConfig`. They are
**model-dependent**; users override via CLI per-model
(`--temperature 1.0` for DSv3.2/V4, `0.6` for R1, etc.). The runner
warns when `n_repeats > 1` and `temperature == 0.0`.

Per-benchmark `default_n_repeats` and `thinking` are sgl-eval's choice
(NS leaves both to CLI). They live in `_registry.py:_TABLE`.

## Testing

```bash
pytest                                # ours + vendored NS corner cases
python scripts/audit_vendored.py      # vendor import sanity
python scripts/sync_vendored.py --check   # vendored trees match their manifests
pre-commit run --all-files            # lint + format + codespell
```

`deepswe` needs Docker; its acceptance is manual (`benchmarks.md`).

## Available skills

- `/add-benchmark <name>` — vendor a new dataset module + register it.
- `/vendor-update` — bump the synced NeMo-Skills SHA + verify.
- `/review-vendor-coverage` — audit whether SE code creeps into scoring.
