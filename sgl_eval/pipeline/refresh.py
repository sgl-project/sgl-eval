"""``sgl-eval refresh <run_dir>``: rebuild ``metrics.json`` locally from
existing ``output-rs*.jsonl``. Recomputes derivable fields (aggregate,
token tally, partial counts, score bounds); preserves provenance
(``model`` / ``base_url`` / ``latency_seconds`` / ``total_prompt_tokens``
/ ``ns_commit_sha`` / ``preset`` ...). No requests."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sgl_eval.metrics import dump_run, format_summary
from sgl_eval.pipeline.report import _format_partial_summary, _partial_stats
from sgl_eval.predictions import PredictionsReader
from sgl_eval.registry import get
from sgl_eval.types import Example, ExampleResult, RunResult, Sample

# dump_run sets these from ``RunResult``; must not appear in run_meta
# (overlap guard would raise).
_CORE_FIELDS = frozenset(
    {
        "name",
        "num_examples",
        "n_repeats",
        "latency_seconds",
        "output_throughput_tps",
        "total_completion_tokens",
        "total_prompt_tokens",
        "aggregate",
    }
)

# Cleared before re-emit so a run that's now complete loses stale
# ``partial=True`` / bounds from the previous (partial) metrics.json.
_REFRESH_FIELDS = frozenset(
    {
        "partial",
        "planned_examples",
        "completed_samples",
        "score_lower_bound",
        "score_upper_bound",
        "examples_full",
        "examples_partial",
        "examples_dropped",
    }
)

_RUN_DIR_RE = re.compile(r"^sgl_eval_(?P<name>.+)_\d{8}-\d{6}$")


def cmd_refresh(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        sys.exit(f"error: {run_dir} is not a directory")

    old_payload = _load_old_metrics(run_dir)
    name, n_repeats_hint = _resolve_name_and_n_repeats(run_dir, old_payload)
    spec = get(name)

    reader = PredictionsReader(run_dir, n_repeats=n_repeats_hint)
    if reader.n_repeats == 0:
        sys.exit(f"error: no output-rs*.jsonl found in {run_dir}")

    per_example = _build_per_example(reader)
    n_repeats = reader.n_repeats
    aggregate = _aggregate(spec.category, per_example, n_repeats)

    completed_samples = sum(len(r.samples) for r in per_example)
    planned_examples = _resolve_planned_examples(old_payload, per_example)
    planned_samples = planned_examples * n_repeats

    result = RunResult(
        name=name,
        per_example=per_example,
        aggregate=aggregate,
        latency=(old_payload or {}).get("latency_seconds", 0.0),
        num_examples=len(per_example),
        n_repeats=n_repeats,
        total_completion_tokens=sum(
            (s.completion_tokens or 0) for r in per_example for s in r.samples
        ),
        # prompt_tokens isn't stored in jsonl; preserve old value or 0.
        total_prompt_tokens=(old_payload or {}).get("total_prompt_tokens", 0),
        partial=completed_samples < planned_samples,
        planned_examples=planned_examples,
    )

    run_meta = _strip_managed(dict(old_payload or {}))
    if result.partial:
        stats = _partial_stats(result)
        run_meta["partial"] = True
        run_meta["planned_examples"] = result.planned_examples
        run_meta["completed_samples"] = stats.completed_samples
        run_meta["score_lower_bound"] = stats.worst
        run_meta["score_upper_bound"] = stats.best
        if n_repeats > 1:
            run_meta["examples_full"] = stats.full
            run_meta["examples_partial"] = stats.part
            run_meta["examples_dropped"] = stats.dropped

    print(format_summary(result))
    metrics_path = dump_run(result, run_dir, run_meta=run_meta)
    print(f"\nMetrics: {metrics_path}")
    if result.partial:
        print(_format_partial_summary(result, _partial_stats(result)), file=sys.stderr)

    return 0


def _load_old_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_name_and_n_repeats(
    run_dir: Path, old_payload: Optional[Dict[str, Any]]
) -> Tuple[str, Optional[int]]:
    if old_payload:
        return old_payload["name"], old_payload.get("n_repeats")
    m = _RUN_DIR_RE.match(run_dir.name)
    if not m:
        sys.exit(
            f"error: cannot infer benchmark from {run_dir.name!r}; "
            "expected ``sgl_eval_<bench>_<stamp>`` or a metrics.json with ``name``"
        )
    return m.group("name"), None


def _resolve_planned_examples(
    old_payload: Optional[Dict[str, Any]], per_example: List[ExampleResult]
) -> int:
    """Prefer old metrics (the live run knew the dataset slice). Fallback
    to len(per_example) forces ``partial=False`` -- from-scratch refresh
    has no signal of dropped examples."""
    if old_payload:
        for k in ("planned_examples", "num_examples"):
            if k in old_payload:
                return int(old_payload[k])
    return len(per_example)


def _build_per_example(reader: PredictionsReader) -> List[ExampleResult]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for _rep, row in reader.iter_all():
        ex_id = row["id"]
        slot = by_id.setdefault(
            ex_id,
            {
                "example": Example(
                    id=ex_id,
                    inputs={"problem": row.get("problem", "")},
                    target=row.get("expected_answer", ""),
                ),
                "samples": [],
                "scores": [],
                "extracted": [],
            },
        )
        slot["samples"].append(
            Sample(
                text=row.get("generation", ""),
                completion_tokens=row.get("num_generated_tokens"),
                reasoning_tokens=row.get("num_reasoning_tokens"),
                generation_start_time=row.get("generation_start_time"),
                generation_end_time=row.get("generation_end_time"),
            )
        )
        slot["scores"].append(1.0 if row.get("symbolic_correct") else 0.0)
        slot["extracted"].append(row.get("predicted_answer"))
    return [
        ExampleResult(
            example=v["example"],
            samples=v["samples"],
            scores=v["scores"],
            extracted=v["extracted"],
        )
        for v in by_id.values()
    ]


def _aggregate(category: str, per_example: List[ExampleResult], n_repeats: int) -> Dict[str, float]:
    if n_repeats > 1:
        if category == "math":
            from sgl_eval.evals._math import aggregate_with_math_metrics

            return aggregate_with_math_metrics(per_example, n_repeats)
        if category == "multichoice":
            from sgl_eval.evals._multichoice import aggregate_with_math_metrics

            return aggregate_with_math_metrics(per_example, n_repeats)
    if not per_example:
        return {"score": 0.0}
    means = [sum(r.scores) / len(r.scores) for r in per_example if r.scores]
    return {"score": sum(means) / len(means) if means else 0.0}


def _strip_managed(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in _CORE_FIELDS | _REFRESH_FIELDS}
