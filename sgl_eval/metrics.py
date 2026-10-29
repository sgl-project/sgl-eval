"""Write metrics.json and format the stdout summary for a run directory."""

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
    """Write metrics.json with optional provenance fields.

    run_meta may add fields but must not replace core result fields.
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
        "token_usage": _token_usage(result),
        "aggregate": result.aggregate,
    }
    if result.response_usage:
        payload["response_usage"] = result.response_usage
    if result.metadata:
        payload["metadata"] = result.metadata
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
    """Mark the headline with *: pass@1[avg-of-k] for repeats, score otherwise."""
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
    for label, value in result.metadata.get("summary_rows", []):
        rows.append((False, str(label), str(value), None))
    token_usage = _token_usage(result)
    num_samples = sum(len(r.samples) for r in result.per_example)
    # A trajectory benchmark's sample is a whole trial, so its totals are per
    # trial; the per-response means come from ``response_usage``.
    per_sample_suffix = "/trial" if result.response_usage else ""
    for field, label in (
        ("prompt_tokens", "avg_input_tokens"),
        ("completion_tokens", "avg_output_tokens"),
        ("reasoning_tokens", "avg_thinking_tokens"),
    ):
        stats = token_usage[field]
        mean = stats["mean"]
        value = f"{mean:,.1f} tokens" if mean is not None else "N/A"
        count = stats["count"]
        note = (
            f"usage reported for {count}/{num_samples} samples" if 0 < count < num_samples else None
        )
        rows.append((False, label + per_sample_suffix, value, note))
    if result.response_usage:
        usage = result.response_usage
        reported, total = usage.get("usage_reported", 0), usage.get("n_responses", 0)
        note = f"usage reported for {reported}/{total} responses" if reported < total else None
        for key, label in (
            ("mean_prompt_tokens", "avg_input_tokens/response"),
            ("mean_completion_tokens", "avg_output_tokens/response"),
            ("mean_reasoning_tokens", "avg_thinking_tokens/response"),
        ):
            mean = usage.get(key)
            value = f"{mean:,.1f} tokens" if mean is not None else "N/A"
            rows.append((False, label, value, note))
    label_w = max(len(label) for _, label, _, _ in rows) if rows else 0

    lines = [f"== {result.name} ==", meta, ""]
    for is_headline, label, value, note in rows:
        marker = "*" if is_headline else " "
        note_str = f"  [{note}]" if note else ""
        lines.append(f"{marker} {label:<{label_w}}  =  {value}{note_str}")
    return "\n".join(lines)


def _token_usage(result: RunResult) -> Dict[str, Dict[str, Any]]:
    """Average each usage field over reported values, retaining explicit zeros."""
    samples = [sample for r in result.per_example for sample in r.samples]
    usage = {}
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        values = [getattr(sample, field) for sample in samples]
        reported = [value for value in values if value is not None]
        usage[field] = {
            "mean": sum(reported) / len(reported) if reported else None,
            "count": len(reported),
        }
    return usage


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

    stop_rate = agg.get("stop_rate")
    if stop_rate is not None:
        rows.append((False, "stop_rate", f"{stop_rate * 100:.2f}%", None))
    truncated_rate = agg.get("truncated_rate")
    if truncated_rate is not None:
        # Nonzero => generations hit the token cap (no-EOS runaway).
        note = "warn: hitting max_tokens" if truncated_rate >= 0.01 else None
        rows.append((False, "truncated_rate", f"{truncated_rate * 100:.2f}%", note))
    error_rate = agg.get("error_rate")
    if error_rate is not None:
        note = "warn: request errors" if error_rate >= 0.01 else None
        rows.append((False, "error_rate", f"{error_rate * 100:.2f}%", note))

    # Group benchmarks (ruler2) publish per-subtask scores as ``task.<name>``;
    # the headline averages them, so the breakdown says which one moved.
    task_keys = sorted(key for key in agg if key.startswith("task."))
    if task_keys:
        if agg.get("task_subset"):
            rows.append((False, "group", "SUBSET (not the full 12-task group)", None))
        for key in task_keys:
            rows.append((False, f"  {key[len('task.'):]}", f"{agg[key] * 100:.2f}%", None))
    return rows
