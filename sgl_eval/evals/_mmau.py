"""MMAU audio-understanding benchmark (sgl-eval-own; NeMo-Skills has no MMAU).

Dataset: ``gamma-lab-umd/MMAU-test-mini`` -- 1000 audio multiple-choice
questions across sound / music / speech. Each clip is decoded and re-encoded
once at load time to 16 kHz mono WAV, then attached as a ``MediaItem`` so the
sample path is the same ``build_user_content`` transport the vision
benchmarks use.

Scoring follows the official MMAU protocol (v05.15.25): a prediction is
correct when its tokens contain the gold answer's tokens while staying
disjoint from the tokens unique to the wrong choices. Aggregation reports
the overall accuracy plus a per-task breakdown via the standard ``task.*``
metric keys.

Heavy deps (``librosa`` / ``soundfile`` / ``pyarrow`` / ``huggingface_hub``)
are imported lazily inside functions: ``registry._autoload`` swallows import
errors, so a module-level import failure would silently drop the benchmark.
Install them via ``pip install sgl-eval[audio]``.
"""

from __future__ import annotations

import io
import json
import re
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from sgl_eval.evals._vision import build_user_content
from sgl_eval.predictions import PredictionsWriter
from sgl_eval.registry import EvalSpec, register
from sgl_eval.runner import SampleFn, ScoreOneFn, run_examples
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import Example, ExampleResult, GenConfig, MediaItem, RunResult, Sample

SAMPLE_RATE = 16000
DATASET_REPO = "gamma-lab-umd/MMAU-test-mini"
DATASET_FILE = "test_mini.parquet"

_AUDIO_DEP_HINT = (
    "error: mmau requires audio dependencies; install them with `pip install sgl-eval[audio]`"
)


# ---------- scoring ----------


def _tokens(text: str) -> set:
    return set(re.findall(r"\b\w+\b", text.lower()))


def mmau_string_match(answer: str, prediction: str, choices: List[str]) -> bool:
    """Official MMAU correctness rule: the prediction must contain the
    gold answer's tokens and none of the tokens unique to wrong choices."""
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(answer)
    if not pred_tokens:
        return False
    wrong_tokens: set = set()
    for choice in choices:
        choice_tokens = _tokens(choice)
        if choice_tokens != gold_tokens:
            wrong_tokens |= choice_tokens - gold_tokens
    return gold_tokens <= pred_tokens and not (pred_tokens & wrong_tokens)


def _strip_choice_letter(text: str) -> str:
    """Drop a leading ``"(A) "``-style letter prefix from a choice/answer."""
    if len(text) >= 4 and text[0] == "(" and text[2] == ")" and text[3] == " ":
        if text[1].isalpha():
            return text[4:]
    return text


def _format_question(question: str, choices: List[str]) -> str:
    """Render the MMAU multiple-choice instruction for one sample."""
    lines = [question, "", "Choice: ", *choices]
    lines.append(
        f"Choose a choices from the given {len(choices)} choices. Do not provide any "
        "additional explanations or content. Output must match exactly one of the listed choices."
    )
    return "\n".join(lines)


# ---------- audio preprocessing ----------


def _encode_mono_16k_wav(raw_bytes: bytes) -> bytes:
    """Decode an audio clip and re-encode it as 16 kHz mono PCM16 WAV."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise SystemExit(f"{_AUDIO_DEP_HINT} ({e})")

    audio_array, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        try:
            import librosa
        except ImportError as e:
            raise SystemExit(f"{_AUDIO_DEP_HINT} ({e})")
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    buf = io.BytesIO()
    sf.write(
        buf, np.asarray(audio_array, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16"
    )
    return buf.getvalue()


# ---------- dataset ----------


def _fetch_dataset_rows() -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(f"{_AUDIO_DEP_HINT} ({e})")

    parquet_path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    return pq.ParquetFile(parquet_path).read().to_pylist()


def _task_of(row: Dict[str, Any]) -> str:
    return json.loads(row["other_attributes"])["task"]


def _interleave_rows_by_task(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Cap the row count while keeping tasks balanced: the parquet is grouped
    by task, so a plain head-slice would smoke-test only one task. Selection
    happens before the (expensive) audio re-encode."""
    queues: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        queues.setdefault(_task_of(row), []).append(row)
    picked: List[Dict[str, Any]] = []
    depth = 0
    task_queues = list(queues.values())
    while len(picked) < limit and any(depth < len(q) for q in task_queues):
        for q in task_queues:
            if depth < len(q):
                picked.append(q[depth])
                if len(picked) >= limit:
                    break
        depth += 1
    return picked


def _build_example(row: Dict[str, Any], idx: int) -> Example:
    choices = [_strip_choice_letter(c) for c in row["choices"]]
    wav_bytes = _encode_mono_16k_wav(row["context"]["bytes"])
    return Example(
        id=str(row.get("id") or f"mmau-{idx}"),
        inputs={"problem": _format_question(row["instruction"], choices)},
        target=_strip_choice_letter(row["answer"]),
        meta={"task": _task_of(row), "choices": choices},
        media=[MediaItem(kind="audio", data=wav_bytes, mime="audio/wav")],
    )


def load_mmau_examples(num_examples: Optional[int]) -> List[Example]:
    rows = _fetch_dataset_rows()
    if num_examples is not None:
        rows = _interleave_rows_by_task(rows, num_examples)
    examples: List[Example] = []
    for i, row in enumerate(rows):
        try:
            examples.append(_build_example(row, i))
        except SystemExit:
            raise
        except Exception as e:
            # One undecodable clip must not abort the whole load.
            warnings.warn(f"skipping MMAU row {row.get('id') or i}: {e}")
    return examples


# ---------- sample / score ----------


def make_sample_fn(sampler: ChatCompletionSampler, gen: GenConfig) -> SampleFn:
    def sample_fn(ex: Example, _rep_idx: int) -> Sample:
        content = build_user_content(ex.inputs["problem"], ex.media)
        return sampler([{"role": "user", "content": content}], gen)

    return sample_fn


def make_score_one_fn() -> ScoreOneFn:
    def score_one(ex: Example, sample: Sample) -> Tuple[float, Optional[str]]:
        correct = mmau_string_match(str(ex.target), sample.text, ex.meta["choices"])
        return (1.0 if correct else 0.0), None

    return score_one


# ---------- aggregate ----------


def _per_task_scores(results: List[ExampleResult]) -> Dict[str, float]:
    """Per-task means, published under the standard ``task.<name>`` keys the
    summary renderer already knows how to print (same idiom as ruler2)."""
    by_task: Dict[str, List[float]] = {}
    for r in results:
        task = r.example.meta.get("task")
        if task is None or not r.scores:
            continue
        by_task.setdefault(task, []).append(sum(r.scores) / len(r.scores))
    return {f"task.{task}": sum(vals) / len(vals) for task, vals in sorted(by_task.items())}


def aggregate_mmau(results: List[ExampleResult], n_repeats: int) -> Dict[str, float]:
    if not results:
        return {"score": 0.0}
    if n_repeats > 1:
        # Overall pass@1[avg-of-k] (+ std/SEM) via the vendored MathMetrics,
        # same as math/multichoice; per-task keys are plain means.
        from sgl_eval.evals._multichoice import aggregate_with_math_metrics

        flat = aggregate_with_math_metrics(results, n_repeats)
    else:
        means = [sum(r.scores) / len(r.scores) for r in results if r.scores]
        flat = {"score": sum(means) / len(means) if means else 0.0}
    flat.update(_per_task_scores(results))
    return flat


# ---------- registration ----------


def run_mmau_benchmark(
    *,
    name: str,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: Optional[int],
    num_threads: int,
    load_examples: Optional[Callable[[Optional[int]], List[Example]]] = None,
    predictions_writer: Optional[PredictionsWriter] = None,
) -> RunResult:
    examples = load_mmau_examples(num_examples)
    return run_examples(
        name=name,
        examples=examples,
        sample_fn=make_sample_fn(sampler, gen),
        score_one_fn=make_score_one_fn(),
        num_threads=num_threads,
        n_repeats=n_repeats,
        aggregate_fn=lambda results: aggregate_mmau(results, n_repeats),
        on_sample_scored=predictions_writer,
    )


def _run(
    *,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: Optional[int],
    num_threads: int,
    predictions_writer: Optional[PredictionsWriter] = None,
    load_examples: Optional[Callable[[Optional[int]], List[Example]]] = None,
    bench_args: Optional[Any] = None,
    prompt_yaml: Optional[Any] = None,
) -> RunResult:
    if prompt_yaml is not None:
        raise ValueError(
            "mmau does not take a prompt override: its instruction is fixed by "
            "the official MMAU eval protocol."
        )
    if load_examples is not None:
        sys.exit("error: --from-dataset is not supported for mmau (audio inputs required)")
    return run_mmau_benchmark(
        name="mmau",
        sampler=sampler,
        gen=gen,
        n_repeats=n_repeats,
        num_examples=num_examples,
        num_threads=num_threads,
        load_examples=load_examples,
        predictions_writer=predictions_writer,
    )


register(
    EvalSpec(
        name="mmau",
        category="audio",
        description="MMAU v05.15.25 test-mini (1000 audio MCQs: sound/music/speech).",
        default_gen=GenConfig(),
        default_n_repeats=1,
        run=_run,
    )
)
