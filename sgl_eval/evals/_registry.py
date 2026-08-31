"""Benchmark registration table.

One row per benchmark. Adding a benchmark = appending a row -- no new file,
no new module. Each entry encodes only the things that genuinely differ:

- ``loader``: ``"bundled"`` (read vendored test.txt) or ``"prepare"``
  (run vendored ``prepare.py:save_data``).
- ``save_args`` / ``save_kwargs``: signature for ``save_data`` when
  ``loader == "prepare"``.
- ``argparse_main``: upstream's prepare.py exposes an argparse ``main(args)``
  rather than ``save_data``; ``save_args[0]`` becomes ``args.split``.
- ``archive_url`` / ``archive_sha256``: prefetch and digest-check the prepare
  module's archive here instead of letting the vendored script fetch it.
- ``media_dir`` / ``media_field``: multimodal ``prepare`` benchmarks only --
  the sidecar directory ``save_data`` writes beside its jsonl, and the jsonl
  column holding each row's path into it.
- ``sample_seed``: turns ``--num-examples N`` into a seeded sample of the whole
  split, for benchmarks whose jsonl is grouped rather than shuffled.
- ``thinking``: whether to set ``chat_template_kwargs={"thinking": True}``
  by default. True for reasoning benchmarks (aime / gpqa); off otherwise.
- ``default_n_repeats``: per-example repeat count (sgl-eval choice; NS
  also leaves this to CLI via ``--benchmarks=name:N``).
- ``default_num_threads``: concurrency ceiling, defaulting to 64. Long-context
  benchmarks must lower it -- the runner limits requests, not tokens.
- ``description``: human-readable one-liner.

Sampling params (``temperature`` / ``top_p`` / ``max_tokens``) are **not**
pinned per benchmark. They come from the global NS-aligned default in
``GenConfig`` (``temperature=0.0``, ``top_p=0.95``, ``max_tokens=None``).
Users override per run via CLI flags. ``temperature`` in particular is a
model property -- DSv3.2/V4 want 1.0, R1 wants 0.6, etc. -- and pinning a
single value here would encode a model-specific assumption.

``metrics_type`` and the prompt yaml basename are derived at registration
time from the vendored ``dataset/<name>/__init__.py`` (``METRICS_TYPE`` +
``GENERATION_ARGS``), so we never hand-mirror upstream's per-benchmark
choices. Two rows opt out by declaring both explicitly: ``mmmu_pro``
(upstream ships MMMU-Pro's ``vision`` config only, not the
``standard (10 options)`` one this row evaluates) and ``ruler2`` (upstream
ships no dataset metadata -- its subtasks only exist once generated).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Tuple

from sgl_eval.evals._loader import load_bundled, load_via_prepare
from sgl_eval.evals._math import run_math_benchmark
from sgl_eval.evals._mmmu_pro import load_mmmu_pro
from sgl_eval.evals._multichoice import run_multichoice_benchmark
from sgl_eval.evals._prompts import resolve_prompt
from sgl_eval.evals._ruler2 import PRED_SCHEMA as _RULER2_PRED_SCHEMA
from sgl_eval.evals._ruler2 import add_arguments as _add_ruler2_arguments
from sgl_eval.evals._ruler2 import run_ruler2_benchmark
from sgl_eval.predictions import PredSchema
from sgl_eval.registry import EvalSpec, register
from sgl_eval.types import GenConfig

_MMLU_ARCHIVE_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
# Git LFS object digest for data.tar at the pinned Hugging Face revision.
_MMLU_ARCHIVE_SHA256 = "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b"

_TABLE = [
    {
        "name": "gsm8k",
        "loader": "prepare",
        "save_args": ("test",),
        "thinking": False,
        "default_n_repeats": 1,
        "description": "GSM8K grade-school math (single-shot, mean accuracy).",
    },
    {
        "name": "aime24",
        "loader": "bundled",
        "thinking": True,
        "default_n_repeats": 16,
        "description": "AIME 2024 (30 problems, integer answers).",
    },
    {
        "name": "aime25",
        "loader": "bundled",
        "thinking": True,
        "default_n_repeats": 16,
        "description": "AIME 2025 (30 problems, integer answers).",
    },
    {
        "name": "aime26",
        "loader": "bundled",
        "thinking": True,
        "default_n_repeats": 16,
        "description": "AIME 2026 (30 problems, integer answers).",
    },
    {
        # One CSV per subject in the Hendrycks tar, so the first 60 of 14042
        # rows are all one category -- `--num-examples` has to sample.
        "name": "mmlu",
        "loader": "prepare",
        "save_args": ("test",),
        "sample_seed": 0,
        # `cais` is the Center for AI Safety, the MMLU authors' own org -- this
        # is the same archive served first-party, not a third-party mirror of it.
        "archive_url": (
            f"https://huggingface.co/datasets/cais/mmlu/resolve/{_MMLU_ARCHIVE_REVISION}/data.tar"
        ),
        "archive_sha256": _MMLU_ARCHIVE_SHA256,
        "thinking": False,
        "default_n_repeats": 1,
        "description": "MMLU all-subjects multichoice (mean accuracy).",
    },
    {
        "name": "gpqa",
        "loader": "prepare",
        "save_args": ("diamond",),
        "save_kwargs": {"random_seed": 42},
        "thinking": True,
        "default_n_repeats": 8,
        "description": "GPQA Diamond (graduate-level QA, pass@k + majority@k).",
    },
    {
        # MMLU-Pro, not to be confused with `mmmu_pro` below -- one letter
        # apart and adjacent in `sgl-eval list`, but a text 10-choice exam.
        "name": "mmlu_pro",
        "loader": "prepare",
        "save_args": ("test",),
        "argparse_main": True,
        # 12032 rows ordered by category, so a small --num-examples would
        # otherwise score one subject only.
        "sample_seed": 0,
        "thinking": False,
        "default_n_repeats": 1,
        "description": "MMLU-Pro (12032 questions, 10-choice, reasoning-heavy).",
    },
    {
        # SE-own: upstream ships MMMU-Pro's `vision` config only (registered
        # below as `mmmu_pro_vision`), not `standard (10 options)`. So
        # metrics_type + prompt + loader_fn bypass the vendored
        # dataset/__init__.py + prepare path.
        "name": "mmmu_pro",
        "metrics_type": "multichoice",
        "prompt": "mmmu-pro-cot",
        "loader_fn": lambda num_examples: load_mmmu_pro("test", num_examples),
        "thinking": False,
        "default_n_repeats": 1,
        "description": "MMMU-Pro (multimodal, 10-choice, vision-dependent).",
    },
    {
        # MMMU-Pro `vision`: question and options are rendered into one
        # screenshot, so `problem` carries just the option list. A different
        # task from `mmmu_pro` above, not a different prompt for the same one
        # -- scores are not comparable between the two rows.
        "name": "mmmu_pro_vision",
        "loader": "prepare",
        "save_args": ("test",),
        "media_dir": "images",
        "media_field": "image_path",
        "thinking": False,
        "default_n_repeats": 1,
        "description": "MMMU-Pro vision config (whole question rendered as one screenshot).",
    },
    {
        # Group benchmark: 12 subtasks scored separately, then averaged by
        # vendored ruler2_score.compute_score. Its dataset is generated per
        # (tokenizer, seq length), so it needs --ruler2-seq-len N and has
        # no upstream dataset/__init__.py metadata to derive from.
        "name": "ruler2",
        "metrics_type": "ruler2",
        "prompt": "default",
        "loader_fn": None,
        "thinking": False,
        "default_n_repeats": 1,
        # 64 concurrent 128k prompts is ~8M tokens in flight; the runner caps
        # request count, not tokens.
        "default_num_threads": 4,
        "description": "RULER2 synthetic long-context, 12 subtasks (needs --ruler2-seq-len N).",
    },
]


def _parse_generation_args(gen_args: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for tok in gen_args.split():
        if tok.startswith("++") and "=" in tok:
            key, value = tok[2:].split("=", 1)
            parsed[key] = value
    return parsed


def _resolve_upstream_metadata(name: str) -> Tuple[str, str]:
    mod = importlib.import_module(f"sgl_eval._vendored.nemo_skills.dataset.{name}")
    metrics_type = mod.METRICS_TYPE
    prompt_config = _parse_generation_args(mod.GENERATION_ARGS).get("prompt_config", "")
    if not prompt_config:
        raise RuntimeError(f"upstream {name}/__init__.py: GENERATION_ARGS missing ++prompt_config")
    return metrics_type, prompt_config.split("/")[-1]


def _build_default_gen(thinking: bool) -> GenConfig:
    """All NS-aligned defaults (``temperature=0.0``, ``top_p=0.95``,
    ``max_tokens=None``); only ``chat_template_kwargs.thinking`` varies."""
    return GenConfig(
        chat_template_kwargs={"thinking": True} if thinking else None,
    )


def _build_loader(entry: dict):
    if "loader_fn" in entry:
        return entry["loader_fn"]
    kind = entry["loader"]
    if kind == "bundled":
        return load_bundled(entry["name"])
    if kind == "prepare":
        return load_via_prepare(
            entry["name"],
            list(entry["save_args"]),
            entry.get("save_kwargs", {}),
            media_dir=entry.get("media_dir"),
            media_field=entry.get("media_field"),
            sample_seed=entry.get("sample_seed"),
            argparse_main=entry.get("argparse_main", False),
            archive_url=entry.get("archive_url"),
            archive_sha256=entry.get("archive_sha256"),
        )
    raise ValueError(f"unknown loader kind: {kind!r}")


# Per-category behavior, in one place. Every factory takes the same arguments so
# the registration loop below stays free of benchmark names; a new category is a
# new row here plus its runner module.
def _math_run(name: str, prompt_basename: str, loader: Callable):
    default_prompt_yaml = resolve_prompt(prompt_basename)

    def run(
        *,
        sampler,
        gen,
        n_repeats,
        num_examples,
        num_threads,
        predictions_writer=None,
        load_examples=None,
        bench_args=None,
        prompt_yaml=None,
    ):
        return run_math_benchmark(
            name=name,
            sampler=sampler,
            gen=gen,
            n_repeats=n_repeats,
            num_examples=num_examples,
            num_threads=num_threads,
            load_examples=load_examples or loader,
            prompt_yaml=prompt_yaml or default_prompt_yaml,
            predictions_writer=predictions_writer,
        )

    return run


def _mcq_run(name: str, prompt_basename: str, loader: Callable):
    default_prompt_yaml = resolve_prompt(prompt_basename)

    def run(
        *,
        sampler,
        gen,
        n_repeats,
        num_examples,
        num_threads,
        predictions_writer=None,
        load_examples=None,
        bench_args=None,
        prompt_yaml=None,
    ):
        return run_multichoice_benchmark(
            name=name,
            sampler=sampler,
            gen=gen,
            n_repeats=n_repeats,
            num_examples=num_examples,
            num_threads=num_threads,
            load_examples=load_examples or loader,
            prompt_yaml=prompt_yaml or default_prompt_yaml,
            predictions_writer=predictions_writer,
        )

    return run


def _ruler2_run(name: str, _prompt_basename: str, _loader: Callable):
    def run(
        *,
        sampler,
        gen,
        n_repeats,
        num_examples,
        num_threads,
        predictions_writer=None,
        load_examples=None,
        bench_args=None,
        prompt_yaml=None,
    ):
        if prompt_yaml is not None:
            raise ValueError(
                f"{name} does not take a prompt override: its prompt is a pure "
                'passthrough (`user: "{question}"`) and the whole context is '
                "assembled by the prepare scripts."
            )
        return run_ruler2_benchmark(
            name=name,
            sampler=sampler,
            gen=gen,
            n_repeats=n_repeats,
            num_examples=num_examples,
            num_threads=num_threads,
            predictions_writer=predictions_writer,
            load_examples=load_examples,
            bench_args=bench_args,
        )

    return run


_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "math": {"make_run": _math_run},
    "multichoice": {"make_run": _mcq_run},
    "ruler2": {
        "make_run": _ruler2_run,
        "pred_schema": _RULER2_PRED_SCHEMA,
        "add_arguments": _add_ruler2_arguments,
    },
}


for _entry in _TABLE:
    _name = _entry["name"]
    if "metrics_type" in _entry:
        _metrics_type = _entry["metrics_type"]
        _prompt_basename = _entry["prompt"]
    else:
        _metrics_type, _prompt_basename = _resolve_upstream_metadata(_name)
    if _metrics_type not in _CATEGORIES:
        raise ValueError(f"unsupported metrics_type: {_metrics_type!r}")
    _category = _CATEGORIES[_metrics_type]
    register(
        EvalSpec(
            name=_name,
            category=_metrics_type,
            description=_entry["description"],
            default_gen=_build_default_gen(_entry["thinking"]),
            default_n_repeats=_entry["default_n_repeats"],
            run=_category["make_run"](_name, _prompt_basename, _build_loader(_entry)),
            default_num_threads=_entry.get("default_num_threads", 64),
            pred_schema=_category.get("pred_schema") or PredSchema(),
            add_arguments=_category.get("add_arguments"),
        )
    )
