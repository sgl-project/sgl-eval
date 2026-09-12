"""``output-rs{rep}.jsonl`` is the canonical per-sample record, streamed
during a live run: the faithful log of what the model actually said, kept
for offline analysis (flaky / timeout / eyeballing). ``sample_to_pred``
mints the NS wire shape consumed by ``MathEvaluator.eval_single`` at
sample time and ``MathMetrics.update`` at aggregate time."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from sgl_eval.types import Example, Sample


@dataclass(frozen=True)
class PredSchema:
    """How a scored sample becomes an NS prediction dict.

    Defaults are the math/mcq shape. RULER2 flips every field: its grader
    iterates ``expected_answer`` as a list (stringified, it walks the repr
    character by character and scores almost everything correct), its score is a
    float, and echoing the ~500KB prompt would dwarf the run.
    """

    stringify_target: bool = True
    include_prompt: bool = True
    score_field: str = "symbolic_correct"
    binary_score: bool = True


_DEFAULT_SCHEMA = PredSchema()


def sample_to_pred(
    sample: Sample, example: Example, schema: PredSchema = _DEFAULT_SCHEMA
) -> Dict[str, Any]:
    completion = sample.completion_tokens or 0
    reasoning = sample.reasoning_tokens or 0
    # ``id`` mirrors NS native row shape and is the per-problem key for
    # partial-resume / sub-sampling on top of the JSONL dumps.
    pred: Dict[str, Any] = {
        "id": example.id,
        "expected_answer": str(example.target) if schema.stringify_target else example.target,
        "num_generated_tokens": completion,
        "num_reasoning_tokens": reasoning,
        "num_answer_tokens": max(completion - reasoning, 0),
        # Generation stop reason ("stop" / "length" / "error").
        "finish_reason": sample.finish_reason,
        "problem": example.inputs.get("problem", "") if schema.include_prompt else "",
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

    def __init__(self, out_dir: Path, n_repeats: int, schema: PredSchema = _DEFAULT_SCHEMA) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._files = [
            (out_dir / f"output-rs{i}.jsonl").open("w", encoding="utf-8") for i in range(n_repeats)
        ]
        self._lock = threading.Lock()
        self._schema = schema

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
            "reasoning_content": sample.reasoning_content,
            **sample_to_pred(sample, ex, self._schema),
            "predicted_answer": extracted,
            self._schema.score_field: bool(score) if self._schema.binary_score else score,
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
