"""Metric output. Writes ``metrics.json`` per run plus a stdout summary.

A run lives in its own directory (``<out>/sgl_eval_<name>_<stamp>/``) which
also holds the streaming ``output-rs{i}.jsonl`` prediction files written by
``PredictionsWriter``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sgl_eval.types import RunResult


def dump_run(
    result: RunResult,
    out_dir: str | os.PathLike,
    *,
    run_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``metrics.json`` into ``out_dir`` (the per-run folder).

    ``run_meta`` is merged into the top-level payload alongside the
    aggregate -- intended for endpoint / model / sampling config /
    sgl-eval + NS provenance, anything that helps a future reader
    reproduce the run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    payload: Dict[str, Any] = {
        "name": result.name,
        "num_examples": result.num_examples,
        "n_repeats": result.n_repeats,
        "latency_seconds": result.latency,
        "output_throughput_tps": result.output_throughput,
        "total_completion_tokens": result.total_completion_tokens,
        "total_prompt_tokens": result.total_prompt_tokens,
        "aggregate": result.aggregate,
    }
    if run_meta:
        # Reject overlap with core fields so a future caller can't silently
        # clobber ``aggregate`` / ``name`` / etc. by reusing those keys.
        overlap = run_meta.keys() & payload.keys()
        if overlap:
            raise ValueError(f"run_meta overlaps reserved metrics fields: {sorted(overlap)}")
        payload.update(run_meta)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return path


def format_summary(result: RunResult) -> str:
    """Compact stdout summary. Headline metric (``pass@1[avg-of-k]`` when
    ``k > 1``, plain ``score`` when ``k == 1``) is prefixed with ``*``;
    auxiliary metrics indented two spaces."""
    k = result.n_repeats
    agg = result.aggregate

    if k == 1:
        spec = f"{result.num_examples} examples (single-shot)"
    else:
        spec = f"{result.num_examples} examples x {k} repeats"
    meta = (
        f"{spec}  |  {result.latency:.1f}s"
        f"  |  {result.output_throughput:.0f} tok/s"
        f"  |  {_fmt_tokens(result.total_completion_tokens)} tokens"
    )

    rows = _build_rows(agg, k)
    label_w = max(len(label) for _, label, _, _ in rows) if rows else 0

    lines = [f"== {result.name} ==", meta, ""]
    for is_headline, label, value, note in rows:
        marker = "*" if is_headline else " "
        note_str = f"  [{note}]" if note else ""
        lines.append(f"{marker} {label:<{label_w}}  =  {value}{note_str}")
    return "\n".join(lines)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _build_rows(agg: Dict[str, float], k: int) -> List[Tuple[bool, str, str, Optional[str]]]:
    """Return ``[(is_headline, label, value_str, note_or_None), ...]``."""
    rows: List[Tuple[bool, str, str, Optional[str]]] = []
    if k > 1 and "pass@1" in agg:
        score = agg["pass@1"]
        std = agg.get("pass@1_std", 0.0)
        sem = agg.get("pass@1_sem", 0.0)
        if std > 0:
            value = f"{score * 100:.2f}% +/- {std * 100:.2f}% (SEM {sem * 100:.2f}%)"
        else:
            value = f"{score * 100:.2f}%"
        rows.append((True, f"pass@1[avg-of-{k}]", value, None))
        if f"pass@{k}" in agg:
            rows.append((False, f"pass@{k}", f"{agg[f'pass@{k}'] * 100:.2f}%", None))
        if f"majority@{k}" in agg:
            rows.append((False, f"majority@{k}", f"{agg[f'majority@{k}'] * 100:.2f}%", None))
    else:
        rows.append((True, "score", f"{agg.get('score', 0.0) * 100:.2f}%", None))

    no_answer = agg.get("no_answer")
    if no_answer is not None:
        note = "warn: consider --max-tokens" if no_answer >= 0.05 else None
        rows.append((False, "no_answer", f"{no_answer * 100:.2f}%", note))
    return rows
