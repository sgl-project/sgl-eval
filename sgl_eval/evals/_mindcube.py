"""MindCube spatial-reasoning benchmark (sgl-eval-own; NeMo-Skills has no MindCube).

Dataset: ``MLL-Lab/MindCube`` -- ``data.zip`` on the HuggingFace Hub, holding
``data/raw/MindCube_tinybench.jsonl`` (1,050 questions, the paper's evaluation
split) and the referenced images (2-4 views per question, ``among`` /
``around`` / ``rotation`` settings). The zip is downloaded and extracted once
into ``~/.cache/sgl_eval/mindcube/`` (or ``$SGL_EVAL_MINDCUBE_DIR``), then each
question's views are attached as ``MediaItem``\\s so the sample path is the same
``build_user_content`` transport the other vision benchmarks use.

This registers the paper's **raw QA** setting only: the frozen-VLM protocol
where the model sees the views and the question and answers directly. The
cognitive-map settings (``aug_cgmap_*`` / ``plain_cgmap_*`` / ``ff_rsn``) need
scaffold data and a map-parsing evaluator and are out of scope here.

Scoring follows the official evaluator (``src/evaluation``): the answer letter
is pulled from the response by the official ``extract_answer`` regex cascade
(ported verbatim) and compared to ``gt_answer``; the headline accuracy is the
official filtered overall (``translation`` questions are tracked but excluded
from the overall, exactly as upstream's ``base_metrics`` does), plus a
per-setting breakdown via the standard ``task.*`` metric keys.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sgl_eval.evals._vision import build_user_content
from sgl_eval.predictions import PredictionsWriter
from sgl_eval.registry import EvalSpec, register
from sgl_eval.runner import SampleFn, ScoreOneFn, run_examples
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import Example, ExampleResult, GenConfig, MediaItem, RunResult, Sample

DATASET_REPO = "MLL-Lab/MindCube"
DATASET_FILE = "data.zip"
BENCH_JSONL = "data/raw/MindCube_tinybench.jsonl"
_CACHE_DIR = Path.home() / ".cache" / "sgl_eval" / "mindcube"
_ENV_DIR = "SGL_EVAL_MINDCUBE_DIR"

# Official raw-QA prompt, verbatim from MindCube
# ``src/prompt_generation/templates.py`` (RAW_QA_BACKGROUND_INSTRUCTION,
# QUESTION_HEADER, RawQATemplate.generate_prompt).
RAW_QA_INSTRUCTION = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
"""
QUESTION_HEADER = "[Question]\n"

# Official aggregation: which settings count toward the overall accuracy
# (``src/evaluation/core/base_metrics.py``). ``translation`` is tracked but
# excluded from the overall; the tinybench split ships no translation rows.
SETTINGS = ("around", "rotation", "translation", "among", "other")
INCLUDE_IN_OVERALL = {
    "around": True,
    "rotation": True,
    "translation": False,
    "among": True,
    "other": True,
}

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


# ---------- official scoring (ported verbatim from src/evaluation/core/extractors.py) ----------


def extract_answer(text: str) -> str | None:
    """Official MindCube answer extractor: the last letter A-E found by the
    highest-priority matching pattern, or ``None``."""
    if not text:
        return None

    simple_pattern_matches = list(re.finditer(r"([A-E])\.", text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)

    answer_section_match = re.search(r"<Answer>(.*?)(?:<|$)", text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        for pattern in [
            r"[Mm]y answer is ([A-E])",
            r"[Mm]y answer is ([A-E])\.",
            r"[Tt]he answer is ([A-E])",
            r"(?:Answer: )?([A-E])\.",
            r"\b([A-E])\b",
        ]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)

    patterns = [
        r"(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'\",]+(?=(?:\n|$|\.|\"))",
        r"(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'\"]+",
        r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)",
        r"[\*\"]([A-E])[\*\"]",
        r"\bAnswer:?\s*([A-E])\b",
        r"[Mm]y answer is ([A-E])",
        r"[Mm]y answer is ([A-E])\.",
        r"answer is ([A-E])",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].group(1)

    lines = text.split("\n")
    line_matches = []
    for i, line in enumerate(lines):
        match = re.search(r"([A-E])\. [A-Za-z0-9 \-\(\)\'\",]+", line)
        if match:
            line_matches.append((i, match.group(1)))
    if line_matches:
        return line_matches[-1][1]

    for i in reversed(range(len(lines))):
        match = re.search(r"\b([A-E])\b", lines[i])
        if match:
            return match.group(1)

    return None


def setting_of(item_id: str) -> str:
    """Official ``get_setting_from_id``: the setting is encoded in the id."""
    if not item_id:
        return "other"
    lowered = item_id.lower()
    for setting in ("around", "rotation", "translation", "among"):
        if setting in lowered:
            return setting
    return "other"


def build_prompt(question: str) -> str:
    """Official ``RawQATemplate.generate_prompt``: instruction + question."""
    return "\n".join([RAW_QA_INSTRUCTION, QUESTION_HEADER + question])


# ---------- dataset ----------


def _data_root() -> Path:
    """Directory holding ``data/raw/...`` and ``data/other_all_image/...``.

    ``$SGL_EVAL_MINDCUBE_DIR`` may point at an already-extracted copy (the
    directory that *contains* ``data/``); otherwise the zip is fetched from the
    Hub and extracted once into the sgl-eval cache.
    """
    override = os.environ.get(_ENV_DIR)
    if override:
        root = Path(override).expanduser()
        if not (root / BENCH_JSONL).exists():
            sys.exit(f"error: {_ENV_DIR}={override} does not contain {BENCH_JSONL}")
        return root
    if (_CACHE_DIR / BENCH_JSONL).exists():
        return _CACHE_DIR
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - datasets pulls it in
        raise SystemExit(f"error: mindcube needs huggingface_hub ({e})")
    zip_path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_CACHE_DIR)
    if not (_CACHE_DIR / BENCH_JSONL).exists():
        sys.exit(f"error: extracted {DATASET_FILE} but {BENCH_JSONL} is missing")
    return _CACHE_DIR


def _read_rows(root: Path) -> list[dict[str, Any]]:
    with (root / BENCH_JSONL).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _interleave_rows_by_setting(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Cap the row count while keeping settings balanced: the jsonl is grouped
    by setting, so a plain head-slice would smoke-test only one of them."""
    queues: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        queues.setdefault(setting_of(row.get("id", "")), []).append(row)
    picked: list[dict[str, Any]] = []
    depth = 0
    setting_queues = list(queues.values())
    while len(picked) < limit and any(depth < len(q) for q in setting_queues):
        for q in setting_queues:
            if depth < len(q):
                picked.append(q[depth])
                if len(picked) >= limit:
                    break
        depth += 1
    return picked


def _image_media(root: Path, rel_paths: list[str]) -> list[MediaItem]:
    """Attach the views in dataset order (image 1..N as the question refers to
    them), as their original bytes -- no re-encode, no resize."""
    media: list[MediaItem] = []
    for rel in rel_paths:
        path = root / "data" / rel
        if not path.exists():
            raise ValueError(f"missing image {rel}")
        mime = _MIME.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f"unsupported image type {path.suffix!r} for {rel}")
        media.append(MediaItem(kind="image", data=path.read_bytes(), mime=mime))
    if not media:
        raise ValueError("no images")
    return media


def _build_example(root: Path, row: dict[str, Any], idx: int) -> Example:
    item_id = str(row.get("id") or f"mindcube-{idx}")
    answer = str(row["gt_answer"]).strip().upper()
    if not answer:
        raise ValueError("empty gt_answer")
    return Example(
        id=item_id,
        inputs={"problem": build_prompt(row["question"])},
        target=answer,
        meta={
            "setting": setting_of(item_id),
            "type": str(row.get("type")),
            "category": row.get("category"),
        },
        media=_image_media(root, list(row.get("images") or [])),
    )


def load_mindcube_examples(num_examples: int | None) -> list[Example]:
    root = _data_root()
    rows = _read_rows(root)
    if num_examples is not None:
        rows = _interleave_rows_by_setting(rows, num_examples)
    examples: list[Example] = []
    for i, row in enumerate(rows):
        try:
            examples.append(_build_example(root, row, i))
        except (ValueError, KeyError) as e:
            # One bad row must not abort the whole load, but it must be visible.
            warnings.warn(f"skipping MindCube row {row.get('id') or i}: {e}")
    return examples


# ---------- sample / score ----------


def make_sample_fn(sampler: ChatCompletionSampler, gen: GenConfig) -> SampleFn:
    def sample_fn(ex: Example, _rep_idx: int) -> Sample:
        # Official inference puts the views before the text prompt.
        content = build_user_content(ex.inputs["problem"], ex.media, image_position="before")
        return sampler([{"role": "user", "content": content}], gen)

    return sample_fn


def make_score_one_fn() -> ScoreOneFn:
    def score_one(ex: Example, sample: Sample) -> tuple[float, str | None]:
        letter = extract_answer(sample.text)
        correct = letter is not None and letter == str(ex.target)
        return (1.0 if correct else 0.0), letter

    return score_one


# ---------- aggregate ----------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_mindcube(results: list[ExampleResult], n_repeats: int) -> dict[str, float]:
    """Official filtered overall (settings with ``include_in_overall``) as
    ``score``/``pass@1``, per-setting means as ``task.<setting>``, and the
    no-answer rate (responses the official extractor could not parse)."""
    if not results:
        return {"score": 0.0}
    counted = [r for r in results if INCLUDE_IN_OVERALL.get(r.example.meta.get("setting"), True)]
    if n_repeats > 1:
        # pass@1[avg-of-k] (+ std/SEM), pass@k and majority@k via the vendored
        # MathMetrics, over the official filtered set.
        from sgl_eval.evals._multichoice import aggregate_with_math_metrics

        flat = aggregate_with_math_metrics(counted, n_repeats) if counted else {"score": 0.0}
    else:
        flat = {"score": _mean([_mean(r.scores) for r in counted if r.scores])}
    by_setting: dict[str, list[float]] = {}
    for r in results:
        if r.scores:
            by_setting.setdefault(r.example.meta.get("setting", "other"), []).append(
                _mean(r.scores)
            )
    for setting, vals in sorted(by_setting.items()):
        flat[f"task.{setting}"] = _mean(vals)
    total = sum(len(r.extracted) for r in results)
    missing = sum(1 for r in results for letter in r.extracted if letter is None)
    flat["no_answer"] = missing / total if total else 0.0
    return flat


# ---------- registration ----------


def run_mindcube_benchmark(
    *,
    name: str,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: int | None,
    num_threads: int,
    predictions_writer: PredictionsWriter | None = None,
) -> RunResult:
    examples = load_mindcube_examples(num_examples)
    return run_examples(
        name=name,
        examples=examples,
        sample_fn=make_sample_fn(sampler, gen),
        score_one_fn=make_score_one_fn(),
        num_threads=num_threads,
        n_repeats=n_repeats,
        aggregate_fn=lambda results: aggregate_mindcube(results, n_repeats),
        on_sample_scored=predictions_writer,
    )


def _run(
    *,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: int | None,
    num_threads: int,
    predictions_writer: PredictionsWriter | None = None,
    load_examples: Callable[[int | None], list[Example]] | None = None,
    bench_args: Any | None = None,
    **_unused: Any,
) -> RunResult:
    if load_examples is not None:
        sys.exit("error: --from-dataset is not supported for mindcube (image inputs required)")
    return run_mindcube_benchmark(
        name="mindcube",
        sampler=sampler,
        gen=gen,
        n_repeats=n_repeats,
        num_examples=num_examples,
        num_threads=num_threads,
        predictions_writer=predictions_writer,
    )


register(
    EvalSpec(
        name="mindcube",
        category="multichoice",
        description="MindCube tinybench (1050 multi-view spatial MCQs; raw-QA setting, official scorer).",
        default_gen=GenConfig(),
        default_n_repeats=1,
        run=_run,
    )
)
