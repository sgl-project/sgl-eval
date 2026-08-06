"""Benchmark registry. Each benchmark module registers an ``EvalSpec``
factory under a stable name; the CLI looks them up here."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from sgl_eval.predictions import PredSchema
from sgl_eval.types import GenConfig, RunResult

EvalRunFn = Callable[..., RunResult]


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
