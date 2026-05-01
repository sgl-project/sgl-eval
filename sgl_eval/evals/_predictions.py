"""Build prediction dicts that match upstream NeMo-Skills' wire shape.

Both ``MathEvaluator.eval_single`` (during sampling) and ``MathMetrics.update``
(during aggregation) consume dicts with the same field set. We mint that dict
once here and reuse it on both sides so the data we feed into vendored code
matches the contract upstream's own ``inference/generate.py`` would produce.

``PredictionsWriter`` is the streaming on-disk counterpart: it sits behind
``runner.run_examples``' ``on_sample_scored`` hook and appends one NS-shape
JSON line per scored sample to ``output-rs{rep}.jsonl``, flushing after every
write so a Ctrl-C / crash leaves all already-scored samples on disk.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from sgl_eval.types import Example, Sample


def sample_to_pred(sample: Sample, example: Example) -> Dict[str, Any]:
    completion = sample.completion_tokens or 0
    reasoning = sample.reasoning_tokens or 0
    # ``id`` mirrors NS upstream's native row shape (e.g. aime24/test.txt
    # ships ``{"id": "aime24-0", ...}``) and keeps a stable per-problem
    # key for partial-resume / sub-sampling on top of the JSONL dumps.
    pred: Dict[str, Any] = {
        "id": example.id,
        "expected_answer": str(example.target),
        "num_generated_tokens": completion,
        "num_reasoning_tokens": reasoning,
        "num_answer_tokens": max(completion - reasoning, 0),
        "problem": example.inputs.get("problem", ""),
    }
    # Upstream's ``BaseMetrics.update`` skips entries that omit the timestamp
    # keys, so include them only when valid (avoids min(...) collapsing to 0).
    if sample.generation_start_time is not None:
        pred["generation_start_time"] = sample.generation_start_time
    if sample.generation_end_time is not None:
        pred["generation_end_time"] = sample.generation_end_time
    return pred


class PredictionsWriter:
    """Streaming JSONL writer for NS-shape prediction records.

    Opens ``output-rs{i}.jsonl`` for each repeat under ``out_dir``, appends
    one JSON line per scored sample (in thread-pool completion order, not
    dataset order), and flushes after every line. Thread-safe across the
    runner's parallel workers via a single lock -- contention is negligible
    because writing a few hundred bytes is cheap relative to LLM round-trip.
    """

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
