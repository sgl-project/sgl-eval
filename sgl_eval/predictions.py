"""``output-rs{rep}.jsonl`` is the canonical per-sample record. ``Writer``
streams during a live run; ``Reader`` feeds replay (re-aggregate without
re-sampling) and offline analysis (flaky / timeout). ``sample_to_pred``
mints the NS wire shape consumed by ``MathEvaluator.eval_single`` at
sample time and ``MathMetrics.update`` at aggregate time."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from sgl_eval.types import Example, Sample


def sample_to_pred(sample: Sample, example: Example) -> Dict[str, Any]:
    completion = sample.completion_tokens or 0
    reasoning = sample.reasoning_tokens or 0
    # ``id`` mirrors NS native row shape and is the per-problem key for
    # partial-resume / sub-sampling on top of the JSONL dumps.
    pred: Dict[str, Any] = {
        "id": example.id,
        "expected_answer": str(example.target),
        "num_generated_tokens": completion,
        "num_reasoning_tokens": reasoning,
        "num_answer_tokens": max(completion - reasoning, 0),
        # Generation stop reason ("stop" / "length" / "error").
        "finish_reason": sample.finish_reason,
        "problem": example.inputs.get("problem", ""),
    }
    # NS ``BaseMetrics.update`` skips entries missing these keys; only set
    # when valid so min(...) doesn't collapse to 0.
    if sample.generation_start_time is not None:
        pred["generation_start_time"] = sample.generation_start_time
    if sample.generation_end_time is not None:
        pred["generation_end_time"] = sample.generation_end_time
    return pred


class PredictionsWriter:
    """Streaming JSONL writer. One file per repeat; flushes every line so
    a Ctrl-C / crash leaves all already-scored samples on disk. Thread-safe
    via a single lock (negligible contention vs LLM RTT)."""

    def __init__(self, out_dir: Path, n_repeats: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._files = [
            (out_dir / f"output-rs{i}.jsonl").open("w", encoding="utf-8") for i in range(n_repeats)
        ]
        self._lock = threading.Lock()

    def __call__(
        self,
        ex: Example,
        rep: int,
        sample: Sample,
        score: float,
        extracted: Optional[str],
    ) -> None:
        pred: Dict[str, Any] = {
            "generation": sample.text,
            **sample_to_pred(sample, ex),
            "predicted_answer": extracted,
            "symbolic_correct": bool(score),
        }
        line = json.dumps(pred, ensure_ascii=False) + "\n"
        with self._lock:
            f = self._files[rep]
            f.write(line)
            f.flush()

    def close(self) -> None:
        for f in self._files:
            if not f.closed:
                f.close()

    def __enter__(self) -> "PredictionsWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class PredictionsReader:
    """Read back ``output-rs{rep}.jsonl`` from a run directory. Symmetric
    to ``PredictionsWriter``; consumed by replay and offline analysis.
    ``n_repeats`` is locked at construction time."""

    def __init__(self, run_dir: Path, n_repeats: Optional[int] = None) -> None:
        self.run_dir = Path(run_dir)
        self.n_repeats = n_repeats if n_repeats is not None else self._detect_n_repeats()

    def _detect_n_repeats(self) -> int:
        # Max-index + 1, not file count: a partial run can be missing
        # interior rs files, which would make a count under-report.
        max_idx = -1
        for path in self.run_dir.glob("output-rs*.jsonl"):
            try:
                idx = int(path.stem.removeprefix("output-rs"))
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1

    def iter_rep(self, rep: int) -> Iterator[Dict[str, Any]]:
        path = self.run_dir / f"output-rs{rep}.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def iter_all(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """``(rep, pred)`` over every repeat, in rep-major order."""
        for rep in range(self.n_repeats):
            for pred in self.iter_rep(rep):
                yield rep, pred
