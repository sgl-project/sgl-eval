"""Benchmark registration table.

One row per benchmark. Adding a benchmark = appending a row -- no new file,
no new module. Each entry encodes only the things that genuinely differ:

- ``loader``: ``"bundled"`` (read vendored test.txt) or ``"prepare"``
  (run vendored ``prepare.py:save_data``).
- ``save_args`` / ``save_kwargs``: signature for ``save_data`` when
  ``loader == "prepare"``.
- ``thinking``: whether to set ``chat_template_kwargs={"thinking": True}``
  by default. True for reasoning benchmarks (aime / gpqa); off otherwise.
- ``default_n_repeats``: per-example repeat count (sgl-eval choice; NS
  also leaves this to CLI via ``--benchmarks=name:N``).
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
choices.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Tuple

from sgl_eval.evals._loader import load_bundled, load_via_prepare
from sgl_eval.evals._math import run_math_benchmark
from sgl_eval.evals._multichoice import run_multichoice_benchmark
from sgl_eval.evals._prompts import vendored_prompt
from sgl_eval.registry import EvalSpec, register
from sgl_eval.types import GenConfig

_AIME_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

_AIME_EVALUATOR_CONFIG = {
    "relaxed_extraction": True,
    "extract_regex": r"Exact Answer:\**\s*\**\s*([^\n*]+)",
}

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
        "system_message": _AIME_SYSTEM_PROMPT,
        "evaluator_config": _AIME_EVALUATOR_CONFIG,
        "description": "AIME 2024 (30 problems, integer answers).",
    },
    {
        "name": "aime25",
        "loader": "bundled",
        "thinking": True,
        "default_n_repeats": 16,
        "system_message": _AIME_SYSTEM_PROMPT,
        "evaluator_config": _AIME_EVALUATOR_CONFIG,
        "description": "AIME 2025 (30 problems, integer answers).",
    },
    {
        "name": "aime26",
        "loader": "bundled",
        "thinking": True,
        "default_n_repeats": 16,
        "system_message": _AIME_SYSTEM_PROMPT,
        "evaluator_config": _AIME_EVALUATOR_CONFIG,
        "description": "AIME 2026 (30 problems, integer answers).",
    },
    {
        "name": "mmlu",
        "loader": "prepare",
        "save_args": ("test",),
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


def _build_default_gen(thinking: bool, system_message: str | None = None) -> GenConfig:
    """All NS-aligned defaults (``temperature=0.0``, ``top_p=0.95``,
    ``max_tokens=None``); ``chat_template_kwargs.thinking`` and
    ``system_message`` vary per benchmark."""
    return GenConfig(
        chat_template_kwargs={"thinking": True} if thinking else None,
        system_message=system_message,
    )


def _build_loader(entry: dict):
    kind = entry["loader"]
    if kind == "bundled":
        return load_bundled(entry["name"])
    if kind == "prepare":
        return load_via_prepare(
            entry["name"],
            list(entry["save_args"]),
            entry.get("save_kwargs", {}),
        )
    raise ValueError(f"unknown loader kind: {kind!r}")


def _build_run(
    name: str,
    metrics_type: str,
    prompt_basename: str,
    loader: Callable,
    evaluator_config: dict | None = None,
):
    if metrics_type == "math":

        def run(
            *,
            sampler,
            gen,
            n_repeats,
            num_examples,
            num_threads,
            predictions_writer=None,
            load_examples=None,
        ):
            return run_math_benchmark(
                name=name,
                sampler=sampler,
                gen=gen,
                n_repeats=n_repeats,
                num_examples=num_examples,
                num_threads=num_threads,
                load_examples=load_examples or loader,
                evaluator_config=evaluator_config,
                predictions_writer=predictions_writer,
            )

        return run
    if metrics_type == "multichoice":
        prompt_yaml = vendored_prompt(prompt_basename)

        def run(
            *,
            sampler,
            gen,
            n_repeats,
            num_examples,
            num_threads,
            predictions_writer=None,
            load_examples=None,
        ):
            return run_multichoice_benchmark(
                name=name,
                sampler=sampler,
                gen=gen,
                n_repeats=n_repeats,
                num_examples=num_examples,
                num_threads=num_threads,
                load_examples=load_examples or loader,
                prompt_yaml=prompt_yaml,
                predictions_writer=predictions_writer,
            )

        return run
    raise ValueError(f"unsupported upstream metrics_type: {metrics_type!r}")


for _entry in _TABLE:
    _name = _entry["name"]
    _metrics_type, _prompt_basename = _resolve_upstream_metadata(_name)
    _loader = _build_loader(_entry)
    _run = _build_run(
        _name, _metrics_type, _prompt_basename, _loader, _entry.get("evaluator_config")
    )
    register(
        EvalSpec(
            name=_name,
            category=_metrics_type,
            description=_entry["description"],
            default_gen=_build_default_gen(_entry["thinking"], _entry.get("system_message")),
            default_n_repeats=_entry["default_n_repeats"],
            run=_run,
        )
    )
