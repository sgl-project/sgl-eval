"""Register EvalSpec instances under stable benchmark names for CLI lookup."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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
    # Limits requests, not tokens; long-context benchmarks need lower defaults.
    default_num_threads: int = 64
    # How a scored sample is written to output-rs*.jsonl.
    pred_schema: PredSchema = field(default_factory=PredSchema)
    # Names must use --<benchmark>-* so prepare_run can collect their values.
    add_arguments: Optional[Callable[[Any], None]] = None


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
