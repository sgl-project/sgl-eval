"""On-disk per-sample predictions: NS-shape dict mint + streaming JSONL R/W.

The ``output-rs{rep}.jsonl`` files are the canonical record of every
scored sample in a run. They're a first-class data layer:

  - written live during a run via ``PredictionsWriter`` (behind the
    runner's ``on_sample_scored`` hook),
  - read back later via ``PredictionsReader`` for replay (re-aggregate
    metrics without re-sampling) and offline analysis (flaky / timeout
    breakdowns).

``sample_to_pred`` mints the per-sample dict in NS wire shape, used both
at sampling time (fed into ``MathEvaluator.eval_single``) and at
aggregation time (fed into ``MathMetrics.update``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

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


class PredictionsReader:
    """Read back ``output-rs{rep}.jsonl`` from a run directory.

    Symmetric counterpart to ``PredictionsWriter``: same files, same
    NS wire shape, just inverted. Used by ``sgl-eval replay`` to
    re-aggregate metrics without re-sampling and by future analysis
    paths (flaky-rate, timeout breakdown, etc.) that consume scored
    predictions but don't touch the model.

    Auto-detects ``n_repeats`` by counting ``output-rs*.jsonl`` files
    in ``run_dir`` unless an explicit value is passed; the number is
    locked at construction time so subsequent file changes don't surprise
    in-flight iteration.
    """

    def __init__(self, run_dir: Path, n_repeats: Optional[int] = None) -> None:
        self.run_dir = Path(run_dir)
        self.n_repeats = n_repeats if n_repeats is not None else self._detect_n_repeats()

    def _detect_n_repeats(self) -> int:
        # Highest existing ``output-rs<N>.jsonl`` index + 1. Counting via
        # glob would over-report if some rs files are missing (e.g. partial
        # runs that aborted before any rep N sample completed).
        max_idx = -1
        for path in self.run_dir.glob("output-rs*.jsonl"):
            stem = path.stem  # output-rs<N>
            try:
                idx = int(stem.removeprefix("output-rs"))
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1

    def iter_rep(self, rep: int) -> Iterator[Dict[str, Any]]:
        """Yield prediction dicts for one repeat, in on-disk order."""
        path = self.run_dir / f"output-rs{rep}.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def iter_all(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Yield ``(rep, pred)`` over every repeat. Order is by rep then
        on-disk; useful for analyses that don't care about cross-rep
        grouping."""
        for rep in range(self.n_repeats):
            for pred in self.iter_rep(rep):
                yield rep, pred
