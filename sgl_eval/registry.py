"""Benchmark registry. Each benchmark module registers an ``EvalSpec``
factory under a stable name; the CLI looks them up here."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sgl_eval.predictions import PredSchema
from sgl_eval.types import ExampleResult, GenConfig, RunResult

EvalRunFn = Callable[..., RunResult]
# ``sgl-eval refresh`` hook: rebuild the aggregate from dumped predictions.
# Returning ``None`` means the sample-level mean is already correct.
AggregateFromPredictions = Callable[[List[ExampleResult], int], Optional[Dict[str, float]]]


@dataclass
class EvalSpec:
    name: str
    category: str
    description: str
    default_gen: GenConfig
    default_n_repeats: int
    run: EvalRunFn
    # Concurrency the benchmark is safe at by default. The runner limits by
    # request count, not token budget, so a long-context benchmark has to ask
    # for a lower ceiling than the 64 that suits short prompts.
    default_num_threads: int = 64
    # How a scored sample is written to output-rs*.jsonl.
    pred_schema: PredSchema = field(default_factory=PredSchema)
    # Lets a benchmark add its own options to ``sgl-eval run``, so argparse owns
    # their types, choices and --help instead of a stringly-typed side channel.
    # Names must be prefixed ``--<benchmark>-*``; ``prepare_run`` collects them.
    add_arguments: Optional[Callable[[Any], None]] = None
    # Lets refresh rebuild pass@k / group headlines without knowing which
    # benchmark it is looking at.
    aggregate_predictions: Optional[AggregateFromPredictions] = None


_REGISTRY: Dict[str, EvalSpec] = {}


def register(spec: EvalSpec) -> EvalSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Eval `{spec.name}` already registered.")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> EvalSpec:
    _autoload()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Eval `{name}` not found. Available: {available}")
    return _REGISTRY[name]


def list_evals() -> List[EvalSpec]:
    _autoload()
    return sorted(_REGISTRY.values(), key=lambda s: (s.category, s.name))


def _autoload() -> None:
    """Import every module under ``sgl_eval.evals`` so registrations fire."""
    try:
        package = importlib.import_module("sgl_eval.evals")
    except ModuleNotFoundError:
        return
    for finder, mod_name, is_pkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        if is_pkg:
            continue
        try:
            importlib.import_module(mod_name)
        except Exception:
            continue
