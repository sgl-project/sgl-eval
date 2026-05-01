# Presets

Save a `(benchmark, endpoint, sampling, n_repeats, expected)` bundle to
`~/.sgl_eval/presets/<name>.yaml` and replay it with `--preset <name>`.
CLI flags always override preset values, so a preset is a starting point,
not a lock.

---

## Schema

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

---

## Example

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

---

## Usage

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

---

## Priority

`CLI flag (if not None) > preset field (if not None) > benchmark default`

`0.0` / `False` count as set -- only `None` falls through. So
`temperature: 0.0` (greedy) in a preset is honored, not skipped.

---

## Provenance

Every run writes a `preset` block into `metrics.json` recording the
spec / resolved path / benchmark / expected_score, so a recorded run is
fully traceable back to the preset that drove it.
