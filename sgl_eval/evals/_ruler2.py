"""Glue between sgl-eval's sampler/runner and the vendored NeMo-Skills RULER2
pieces (``eval_ruler2`` / ``eval_mcq``, ``Ruler2Metrics``, ``compute_score``).

RULER2 differs from the math/mcq benchmarks in three ways that shape this file:

  - **The dataset is generated, not downloaded**, and is bound to a specific
    (tokenizer, sequence length) pair -- so both are part of the cache key.
  - **It is a group of 12 subtasks** whose graders differ. Which grader a
    subtask uses is read back from the vendored ``prepare_task_for_ns`` rather
    than duplicated here.
  - **The prompt is pre-assembled** by the prepare scripts, so the prompt
    config (``generic/default``) is a bare ``{question}`` passthrough.

Pipeline mirror:
  - Stage 1 (dataset): vendored ``prepare_<task>`` -> ``~/.cache/sgl_eval/ruler2/<setup>/``
  - Stage 2a (prompt render): vendored ``prompts/default.yaml``
  - Stage 2c (extract + score): vendored ``eval_ruler2`` or ``eval_mcq``
  - Stage 4 (aggregate): vendored ``Ruler2Metrics`` per subtask, then vendored
    ``ruler2_score.compute_score`` for the 12-task headline.
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

# Order and membership must match the vendored ``compute_score``, which
# KeyErrors on a missing task. Each name also has to resolve to a
# ``prepare_<task>`` in the vendored prepare module -- both are asserted by
# tests/test_ruler2.py rather than trusted.
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
    """Generation-side knobs. Every field is passed straight through to the
    vendored ``prepare_<task>``, so a given config reproduces NeMo-Skills'
    dataset byte for byte (its ``random_seed`` is fixed at 42 upstream).

    sgl-eval deliberately adds no knob that changes the produced data:
    ``max_seq_length`` decides the dataset and therefore the score, which puts
    it under the vendoring rule. Fitting the prompts into a server is the
    preflight check's job, not the dataset's.
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
        """Values arrive already typed and range-checked by ``add_arguments``.
        The one rule argparse cannot express is "required, but only when this
        benchmark is the one being run"."""
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
    """ruler2's own CLI surface. Wired via ``EvalSpec.add_arguments``, so
    ``sgl-eval run --help`` documents these and argparse rejects a bad value
    before a run starts."""
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
        help=f"subset of the 12 subtasks, scored as a flagged partial average ({', '.join(ALL_TASKS)})",
    )


def _grader_for(cfg: Ruler2Config, task: str) -> Tuple[str, str]:
    """``(eval_type, match_type)`` for one subtask, read back from the vendored
    ``prepare_task_for_ns`` so the grader mapping is never duplicated here."""
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
    """Generate ``<cache>/<task>/test.jsonl`` once. Calls the per-task
    ``prepare_<task>`` directly: ``prepare_dataset`` hardcodes its output to
    ``Path(__file__).parent / setup``, i.e. inside ``_vendored``."""
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
    """Vendored ``Ruler2Metrics`` over one subtask. It reads ``is_correct``
    (float) or ``symbolic_correct`` (bool) depending on the grader, so the
    score field has to match what that subtask's evaluator produced."""
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
    """Flatten to the ``Dict[str, float]`` shape ``format_summary`` renders.

    The full-group average comes from vendored ``compute_score``, which raises on
    a missing task; anything short of 12 is averaged here and flagged
    ``partial_group``. Gate on what completed, not what was requested -- a run
    aborted at task 5 needs the same fallback as ``tasks=a,b``.
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
        flat["partial_group"] = 1.0

    if k > 1:
        flat["pass@1"] = flat["score"]
    return flat


def aggregate_from_predictions(
    per_example: List[ExampleResult], n_repeats: int
) -> Dict[str, float]:
    """Rebuild the group headline from dumped predictions (``sgl-eval refresh``).

    Subtask comes back from the id (``<task>-<index>``); no task name contains
    ``-``. Every row carries the float ``is_correct`` whichever grader ran, so
    all 12 re-aggregate through the ``ruler2`` branch -- for the multichoice
    pair that is the same number, under the other key ``Ruler2Metrics`` takes.
    """
    by_task: Dict[str, List[ExampleResult]] = {}
    for result in per_example:
        by_task.setdefault(result.example.id.rsplit("-", 1)[0], []).append(result)

    unknown = sorted(set(by_task) - set(ALL_TASKS))
    if unknown:
        sys.exit(
            f"error: ruler2 predictions contain unrecognized subtask(s) {unknown}. "
            "Expected ids shaped <task>-<index>; refusing to guess a headline."
        )
    per_task = {task: _task_metrics(rs, n_repeats, "ruler2") for task, rs in by_task.items()}
    return _headline(per_task, n_repeats, namespace="ruler2")


# Minimum room the endpoint must leave for the answer when ``max_tokens`` is
# unset. This is a PREFLIGHT THRESHOLD, not a dataset knob: it never changes
# what gets generated, it only decides whether we refuse to start. RULER2
# answers are short (a needle, a letter, a span), so this is generous.
_MIN_GEN_BUDGET = 512


def _preflight_context_length(
    sampler: ChatCompletionSampler, cfg: Ruler2Config, gen: GenConfig
) -> None:
    """Refuse to spend hours on a server that cannot hold prompt + answer.

    Two ways this bites, both silent: a window below ``seq_len`` makes every
    request 400, and a window equal to ``seq_len`` leaves zero room to generate
    when ``max_tokens`` is None. The sampler turns both into empty samples
    scoring 0, so the run completes with all-zero metrics and exit code 0.

    Warn (do not fail) when the endpoint does not expose its limit -- refusing
    to run against a server we cannot introspect would be worse.
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
