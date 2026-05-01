"""Saved-config presets.

A preset bundles ``(benchmark, endpoint, sampling, n_repeats, num_examples,
expected score)`` into one YAML file under ``~/.sgl_eval/presets/<name>.yaml``
so a recurring run is one ``--preset <name>`` away. CLI flags always take
precedence -- a preset is a starting point, not a lock.

The schema is intentionally narrow: anything that doesn't change the
identity of "the run we're trying to reproduce" stays out (api_key,
num_threads, out_dir).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PRESET_ROOT = Path.home() / ".sgl_eval" / "presets"


@dataclass
class Endpoint:
    base_url: Optional[str] = None
    model: Optional[str] = None


@dataclass
class Sampling:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Mapped to ``GenConfig.chat_template_kwargs.thinking`` at apply time;
    # exposed flat here so preset YAML stays human-readable.
    thinking: Optional[bool] = None


@dataclass
class Expected:
    # Headline metric (``pass@1`` for k>1, plain ``score`` for k==1) on
    # [0, 1]. Informational only -- printed alongside actual at run end,
    # never gates exit code.
    score: Optional[float] = None


@dataclass
class Preset:
    benchmark: str
    endpoint: Endpoint = field(default_factory=Endpoint)
    n_repeats: Optional[int] = None
    num_examples: Optional[int] = None
    sampling: Sampling = field(default_factory=Sampling)
    expected: Optional[Expected] = None

    @classmethod
    def from_dict(cls, raw: Any, *, source: str) -> "Preset":
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: top-level must be a mapping, got {type(raw).__name__}")
        if "benchmark" not in raw:
            raise ValueError(f"{source}: missing required field 'benchmark'")
        _check_unknown(cls, raw, source)
        return cls(
            benchmark=raw["benchmark"],
            endpoint=_load_section(Endpoint, raw.get("endpoint") or {}, f"{source}.endpoint"),
            n_repeats=raw.get("n_repeats"),
            num_examples=raw.get("num_examples"),
            sampling=_load_section(Sampling, raw.get("sampling") or {}, f"{source}.sampling"),
            expected=(
                _load_section(Expected, raw["expected"], f"{source}.expected")
                if raw.get("expected") is not None
                else None
            ),
        )


def _load_section(cls: type, raw: Any, source: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: must be a mapping, got {type(raw).__name__}")
    _check_unknown(cls, raw, source)
    return cls(**raw)


def _check_unknown(cls: type, raw: Dict[str, Any], source: str) -> None:
    """Strict schema -- typos in field names should fail loudly, not be
    silently ignored. ``Preset.benchmark`` is required so callers must
    spell it correctly to even get here."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{source}: unknown fields {sorted(unknown)}")


def resolve_preset_path(spec: str) -> Path:
    """``spec`` is either a name (resolved under ``PRESET_ROOT``) or an
    explicit path. We treat anything containing ``/`` or ending in
    ``.yaml`` / ``.yml`` as a path so users can keep ad-hoc presets next
    to a project."""
    if "/" in spec or spec.endswith((".yaml", ".yml")):
        return Path(spec).expanduser()
    return PRESET_ROOT / f"{spec}.yaml"


def load_preset(spec: str) -> Preset:
    path = resolve_preset_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"preset not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Preset.from_dict(raw, source=str(path))


def list_presets() -> List[Path]:
    if not PRESET_ROOT.exists():
        return []
    return sorted([*PRESET_ROOT.glob("*.yaml"), *PRESET_ROOT.glob("*.yml")])
