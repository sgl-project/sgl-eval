"""Adapt vendored RULER2 generation, graders, and metrics to the runner.

Datasets depend on the tokenizer and sequence length and contain assembled
prompts. Each task selects its grader through vendored prepare_task_for_ns;
Ruler2Metrics feeds the full-group compute_score aggregation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Silence the vendored evaluators' tqdm bars before importing them (they run
# once per sample, so the bar would be pure noise).
import sgl_eval._vendored.nemo_skills.evaluator.mcq as _mcq_mod
import sgl_eval._vendored.nemo_skills.evaluator.ruler as _ruler_mod

_mcq_mod.tqdm = lambda iterable, **_kwargs: iterable
_ruler_mod.tqdm = lambda iterable, **_kwargs: iterable

from sgl_eval._vendored.nemo_skills.dataset.ruler2 import prepare as _prepare  # noqa: E402
from sgl_eval._vendored.nemo_skills.dataset.ruler2.ruler2_score import (  # noqa: E402
    compute_score,
)
from sgl_eval._vendored.nemo_skills.evaluator.mcq import eval_mcq  # noqa: E402
from sgl_eval._vendored.nemo_skills.evaluator.ruler import eval_ruler2  # noqa: E402
from sgl_eval._vendored.nemo_skills.ruler2_metrics import Ruler2Metrics  # noqa: E402
from sgl_eval.evals._prompts import render_prompt, vendored_prompt  # noqa: E402
from sgl_eval.predictions import PredictionsWriter, PredSchema, sample_to_pred  # noqa: E402
from sgl_eval.runner import _finish_reason_rates, run_examples  # noqa: E402
from sgl_eval.sampler import ChatCompletionSampler  # noqa: E402
from sgl_eval.types import (  # noqa: E402
    Example,
    ExampleResult,
    GenConfig,
    RunResult,
    Sample,
)

_CACHE_ROOT = Path.home() / ".cache" / "sgl_eval" / "ruler2"

# Membership must match vendored compute_score and the prepare_<task> functions.
ALL_TASKS: Tuple[str, ...] = (
    "mk_niah_basic",
    "mk_niah_easy",
    "mk_niah_medium",
    "mk_niah_hard",
    "mv_niah_basic",
    "mv_niah_easy",
    "mv_niah_medium",
    "mv_niah_hard",
    "qa_basic",
    "qa_easy",
    "qa_medium",
    "qa_hard",
)

# RULER2's prompts hold ~500KB of context each; ``PredSchema`` defaults would
# echo them into output-rs*.jsonl and stringify the list-valued answer.
PRED_SCHEMA = PredSchema(
    stringify_target=False,
    include_prompt=False,
    score_field="is_correct",
    binary_score=False,
)

_MISSING_DEPS_HINT = (
    "RULER2 dataset generation needs the `longcontext` extra:\n"
    "    pip install 'sgl-eval[longcontext]'\n"
    "(transformers, nltk, wonderwords, inflect). Scoring an already-generated "
    "dataset does not need them."
)


@dataclass(frozen=True)
class Ruler2Config:
    """Inputs passed unchanged to vendored dataset generation.

    Sequence length defines the benchmark; endpoint capacity is checked
    separately and must not shrink the generated context.
    """

    max_seq_length: int
    tokenizer_path: str
    tokenizer_type: str = "hf"
    dataset_size: int = 100
    tasks: Tuple[str, ...] = ALL_TASKS

    @property
    def setup_slug(self) -> str:
        tok = re.sub(r"[^A-Za-z0-9._-]+", "-", self.tokenizer_path).strip("-")
        return f"{tok}_{self.max_seq_length}_n{self.dataset_size}"

    @property
    def cache_dir(self) -> Path:
        return _CACHE_ROOT / self.setup_slug

    @classmethod
    def from_bench_args(cls, bench_args: Optional[Dict[str, Any]], *, model: str) -> "Ruler2Config":
        """CLI parsing validates values; sequence length is required only for RULER2."""
        args = dict(bench_args or {})
        if args.get("seq_len") is None:
            sys.exit(
                "error: ruler2 requires a target sequence length, e.g.\n"
                "    --ruler2-seq-len 131072\n"
                "It has no default: the dataset is generated per length."
            )
        return cls(
            max_seq_length=args["seq_len"],
            # The served model id is a HF repo id for most SGLang deployments;
            # override for local paths or gated repos.
            tokenizer_path=args.get("tokenizer") or model,
            tokenizer_type=args.get("tokenizer_type") or "hf",
            dataset_size=args.get("dataset_size") or 100,
            tasks=tuple(args["tasks"]) if args.get("tasks") else ALL_TASKS,
        )


def _positive_int(raw: str) -> int:
    """argparse ``type``: rejects 0 and negatives at parse time."""
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def add_arguments(group: Any) -> None:
    group.add_argument(
        "--ruler2-seq-len",
        type=_positive_int,
        metavar="N",
        help="target context length; required, as the dataset is generated per length",
    )
    group.add_argument(
        "--ruler2-tokenizer",
        metavar="HF_ID_OR_PATH",
        help="tokenizer used to size samples (default: the served model id)",
    )
    group.add_argument(
        "--ruler2-tokenizer-type",
        choices=("hf", "openai"),
        default="hf",
        help="the vendored gemini tokenizer is dropped, so it is not offered",
    )
    group.add_argument(
        "--ruler2-dataset-size",
        type=_positive_int,
        default=100,
        metavar="N",
        help="samples per subtask (default: 100)",
    )
    group.add_argument(
        "--ruler2-tasks",
        nargs="+",
        choices=ALL_TASKS,
        metavar="TASK",
        help=f"subset of the 12 subtasks, headline is then flagged task_subset ({', '.join(ALL_TASKS)})",
    )


def _grader_for(cfg: Ruler2Config, task: str) -> Tuple[str, str]:
    """Read the grader mapping from vendored prepare_task_for_ns output."""
    _prepare.prepare_task_for_ns(str(cfg.cache_dir), task)
    init_py = cfg.cache_dir / task / "__init__.py"
    namespace: Dict[str, Any] = {}
    exec(compile(init_py.read_text(), str(init_py), "exec"), namespace)  # noqa: S102
    gen_args: str = namespace["GENERATION_ARGS"]
    parsed = dict(
        tok[2:].split("=", 1) for tok in gen_args.split() if tok.startswith("++") and "=" in tok
    )
    eval_type = parsed.get("eval_type", "")
    if eval_type not in ("ruler2", "multichoice"):
        raise RuntimeError(f"vendored ruler2 task {task!r}: unexpected ++eval_type={eval_type!r}")
    return eval_type, parsed.get("eval_config.match_type", "")


def _ensure_task_data(cfg: Ruler2Config, task: str) -> Path:
    """Call per-task generators to avoid prepare_dataset writing into _vendored."""
    task_dir = cfg.cache_dir / task
    out_path = task_dir / "test.jsonl"
    if out_path.exists():
        return out_path
    # Generate into a staging dir and rename, so a kill mid-write cannot leave a
    # truncated test.jsonl that later runs would accept as a complete dataset.
    staging = cfg.cache_dir / f"{task}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    prepare_fn: Callable[..., None] = getattr(_prepare, f"prepare_{task}")
    print(
        f"  generating ruler2/{task} at {cfg.max_seq_length} tokens "
        f"({cfg.dataset_size} samples)...",
        flush=True,
    )
    try:
        prepare_fn(
            str(staging),
            cfg.tokenizer_type,
            cfg.tokenizer_path,
            cfg.max_seq_length,
            cfg.dataset_size,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"error: ruler2 {task} generation failed (exit {e.returncode}).\n{_MISSING_DEPS_HINT}"
        )
    staged = staging / "test.jsonl"
    if not staged.exists():
        sys.exit(f"error: ruler2 {task} generation produced no {staged}")
    task_dir.mkdir(parents=True, exist_ok=True)
    staged.replace(out_path)
    shutil.rmtree(staging, ignore_errors=True)
    return out_path


def _load_task(path: Path, task: str, num_examples: Optional[int]) -> List[Example]:
    """Rows carry ``question`` (the assembled long prompt) and
    ``expected_answer`` (a list for ruler2 graders, a letter for multichoice)."""
    examples: List[Example] = []
    with path.open("rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    id=f"{task}-{row.get('index', i)}",
                    inputs={"question": row["question"]},
                    target=row["expected_answer"],
                    meta={"task": task, "length": row.get("length")},
                )
            )
            if num_examples and len(examples) >= num_examples:
                break
    return examples


def _make_sample_fn(sampler: ChatCompletionSampler, gen: GenConfig, prompt_yaml: Path):
    def sample_fn(ex: Example, _rep_idx: int) -> Sample:
        text = render_prompt(prompt_yaml, problem="", question=ex.inputs["question"])
        return sampler([{"role": "user", "content": text}], gen)

    return sample_fn


def _score_via(evaluator: Callable[[Dict[str, Any]], None], row: Dict[str, Any]) -> Dict[str, Any]:
    """Both vendored evaluators are file-batch only; feed a 1-row jsonl."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)
    try:
        evaluator({"input_file": str(tmp_path)})
        with tmp_path.open() as f:
            return json.loads(f.readline())
    finally:
        tmp_path.unlink(missing_ok=True)


def _make_score_one_fn(eval_type: str, match_type: str):
    def score_one(ex: Example, sample: Sample) -> Tuple[float, Optional[str]]:
        row = {"generation": sample.text, **sample_to_pred(sample, ex, PRED_SCHEMA)}
        if eval_type == "multichoice":
            scored = _score_via(eval_mcq, row)
            return (1.0 if scored.get("symbolic_correct") else 0.0), scored.get("predicted_answer")
        scored = _score_via(lambda cfg: eval_ruler2({**cfg, "match_type": match_type}), row)
        return float(scored.get("is_correct") or 0.0), scored.get("predicted_answer")

    return score_one


def _task_metrics(results: List[ExampleResult], n_repeats: int, eval_type: str) -> Dict[str, Any]:
    """Ruler2Metrics expects is_correct floats or symbolic_correct booleans."""
    metrics = Ruler2Metrics()
    for r in results:
        preds = []
        for sample, score, extracted in zip(r.samples, r.scores, r.extracted):
            pred = sample_to_pred(sample, r.example, PRED_SCHEMA)
            pred["predicted_answer"] = extracted
            if eval_type == "multichoice":
                pred["symbolic_correct"] = bool(score)
            else:
                pred["is_correct"] = float(score)
            preds.append(pred)
        while len(preds) < n_repeats:
            preds.append(dict(preds[-1]))
        metrics.update(preds)
    return metrics.get_metrics()


def _headline(per_task: Dict[str, Dict[str, Any]], k: int, *, namespace: str) -> Dict[str, float]:
    """Use vendored compute_score for a complete group; flag locally averaged subsets.

    Completeness depends on finished tasks, including when a full run is aborted.
    """
    agg_key = "pass@1" if k == 1 else f"pass@1[avg-of-{k}]"
    flat: Dict[str, float] = {}
    for task, raw in per_task.items():
        flat[f"task.{task}"] = raw.get(agg_key, {}).get("accuracy", 0.0) / 100.0

    if set(per_task) == set(ALL_TASKS):
        namespaced = {f"{namespace}.{task}": raw for task, raw in per_task.items()}
        scored = compute_score(namespaced)
        flat["score"] = scored[namespace][agg_key]["accuracy"] / 100.0
    else:
        scores = [flat[f"task.{task}"] for task in per_task]
        flat["score"] = sum(scores) / len(scores) if scores else 0.0
        flat["task_subset"] = 1.0

    if k > 1:
        flat["pass@1"] = flat["score"]
    return flat


# Arbitrary answer headroom for preflight when max_tokens is unset;
# this threshold does not change the generated dataset.
_MIN_GEN_BUDGET = 512


def _preflight_context_length(
    sampler: ChatCompletionSampler, cfg: Ruler2Config, gen: GenConfig
) -> None:
    """Require room for the generated prompt and answer when the endpoint reports a limit.

    An unreadable limit warns; a known insufficient limit fails before sampling.
    """
    import httpx

    gen_budget = gen.max_tokens or _MIN_GEN_BUDGET
    needed = cfg.max_seq_length + gen_budget

    base = str(sampler.client.base_url).rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        resp = httpx.get(f"{root}/get_model_info", timeout=10)
        info = resp.json() if resp.status_code == 200 else {}
    except Exception:
        info = {}

    for key in ("max_context_length", "context_length", "max_model_len"):
        raw = info.get(key)
        if raw is None:
            continue
        limit = int(raw)
        if limit < needed:
            sys.exit(
                f"error: endpoint reports {key}={limit}, but ruler2 at "
                f"seq_len={cfg.max_seq_length} needs {needed} "
                f"(prompt + {gen_budget} to generate).\n"
                f"Every request would 400 or have no room to answer, scoring 0 "
                f"with a successful exit code.\n"
                f"Fix: serve with a larger --context-length, or lower "
                f"--ruler2-seq-len (RULER2 is meant to be swept over "
                f"lengths below the window)."
            )
        print(f"Preflight: endpoint {key}={limit} >= {needed} (seq_len + gen budget)")
        return

    print(
        f"WARNING: could not read the endpoint's context length; ruler2 at "
        f"seq_len={cfg.max_seq_length} needs {needed} tokens including room to "
        f"answer. Too small shows up as error_rate near 100%.",
        file=sys.stderr,
    )


def run_ruler2_benchmark(
    *,
    name: str,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: Optional[int],
    num_threads: int,
    predictions_writer: Optional[PredictionsWriter] = None,
    load_examples: Optional[Callable[[Optional[int]], List[Example]]] = None,
    bench_args: Optional[Dict[str, Any]] = None,
) -> RunResult:
    if load_examples is not None:
        sys.exit(
            "error: --from-dataset is not supported for ruler2; it is a group of "
            "12 generated subtasks. Point --ruler2-tokenizer / --ruler2-seq-len instead."
        )
    cfg = Ruler2Config.from_bench_args(bench_args, model=sampler.model)
    _preflight_context_length(sampler, cfg, gen)

    prompt_yaml = vendored_prompt("default")
    sample_fn = _make_sample_fn(sampler, gen, prompt_yaml)

    per_task_raw: Dict[str, Dict[str, Any]] = {}
    merged: List[ExampleResult] = []
    planned = 0
    partial = False
    start = time.time()

    for task in cfg.tasks:
        data_path = _ensure_task_data(cfg, task)
        examples = _load_task(data_path, task, num_examples)
        eval_type, match_type = _grader_for(cfg, task)
        result = run_examples(
            name=f"{name}.{task}",
            examples=examples,
            sample_fn=sample_fn,
            score_one_fn=_make_score_one_fn(eval_type, match_type),
            num_threads=num_threads,
            n_repeats=n_repeats,
            aggregate_fn=None,
            on_sample_scored=predictions_writer,
        )
        per_task_raw[task] = _task_metrics(result.per_example, n_repeats, eval_type)
        merged.extend(result.per_example)
        planned += result.planned_examples
        partial = partial or result.partial
        if sampler.aborted:
            break

    aggregate = _headline(per_task_raw, n_repeats, namespace=cfg.setup_slug)
    for key, value in _finish_reason_rates(merged).items():
        aggregate.setdefault(key, value)

    return RunResult(
        name=name,
        per_example=merged,
        aggregate=aggregate,
        latency=time.time() - start,
        num_examples=len(merged),
        n_repeats=n_repeats,
        total_completion_tokens=sum(s.completion_tokens or 0 for r in merged for s in r.samples),
        total_prompt_tokens=sum(s.prompt_tokens or 0 for r in merged for s in r.samples),
        partial=partial,
        planned_examples=planned,
    )
