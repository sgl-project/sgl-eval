"""Saved-config presets.

A preset bundles ``(benchmark, endpoint, sampling, n_repeats, num_examples,
expected score)`` into one YAML file under ``~/.sgl_eval/presets/<name>.yaml``
so a recurring run is one ``--preset <name>`` away. CLI flags always take
precedence -- a preset is a starting point, not a lock.

The schema is intentionally narrow: anything that doesn't change the
identity of "the run we're trying to reproduce" stays out (api_key,
num_threads, out_dir).

This module is the single source of truth for everything preset-related:

  - the dataclass schema + strict YAML loader
  - the ``--preset`` CLI flag and the ``preset list/show`` subcommand
  - the ``CLI > preset > spec default`` resolution chain
  - the ``preset`` provenance block written into ``metrics.json``
  - the post-run "Expected: X% Got: Y%" comparison print

``cli.py`` only wires it via ``add_preset_run_flag`` /
``register_preset_subcommand`` / ``resolve_run_inputs`` / ``make_run_meta_block``
/ ``print_expected_vs_actual``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from sgl_eval.types import GenConfig

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


# ---------- CLI integration ----------
#
# Kept inside this module (not in cli.py) so the preset feature is a single
# self-contained unit: schema + load + override-priority + CLI surface.


def add_preset_run_flag(p_run: argparse.ArgumentParser) -> None:
    """Add ``--preset`` to ``sgl-eval run``."""
    p_run.add_argument(
        "--preset",
        default=None,
        help="preset name (under ~/.sgl_eval/presets/) or path to a preset .yaml; "
        "CLI flags always override preset values",
    )


def register_preset_subcommand(sub: Any) -> None:
    """Register the top-level ``preset`` subcommand and its ``list``/``show``
    children on the main argparse subparsers."""
    p_preset = sub.add_parser("preset", help="manage saved presets")
    preset_sub = p_preset.add_subparsers(dest="preset_cmd", required=True)
    p_list = preset_sub.add_parser("list", help=f"list presets in {PRESET_ROOT}")
    p_list.set_defaults(func=_cmd_preset_list)
    p_show = preset_sub.add_parser("show", help="print a preset's content")
    p_show.add_argument("name", help="preset name (under PRESET_ROOT) or path")
    p_show.set_defaults(func=_cmd_preset_show)


def _cmd_preset_list(args: argparse.Namespace) -> int:
    paths = list_presets()
    if not paths:
        print(f"(no presets in {PRESET_ROOT})")
        return 0
    width = max(len(p.stem) for p in paths)
    for p in paths:
        print(f"  {p.stem:<{width}s}  ({p})")
    return 0


def _cmd_preset_show(args: argparse.Namespace) -> int:
    path = resolve_preset_path(args.name)
    if not path.exists():
        print(f"preset not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


# ---------- override priority resolution ----------


def pick(*candidates: Any) -> Any:
    """First non-``None`` candidate wins. Used for the
    ``CLI > preset > default`` resolution chain so that ``0`` /
    ``0.0`` / ``False`` aren't mistaken for "unset" the way ``or`` would."""
    for c in candidates:
        if c is not None:
            return c
    return None


def apply_to_gen(
    default: GenConfig, preset: Optional["Preset"], args: argparse.Namespace
) -> GenConfig:
    """Resolve a final ``GenConfig`` honoring CLI > preset > default.

    ``args`` must expose ``temperature``, ``top_p``, ``max_tokens``,
    ``thinking`` (any of which may be ``None`` for "unset").
    """
    p = preset.sampling if preset else None
    chat_template_kwargs = dict(default.chat_template_kwargs or {})
    thinking = pick(args.thinking, p.thinking if p else None)
    if thinking is not None:
        chat_template_kwargs["thinking"] = thinking
    return GenConfig(
        temperature=pick(args.temperature, p.temperature if p else None, default.temperature),
        top_p=pick(args.top_p, p.top_p if p else None, default.top_p),
        max_tokens=pick(args.max_tokens, p.max_tokens if p else None, default.max_tokens),
        reasoning_effort=default.reasoning_effort,
        chat_template_kwargs=chat_template_kwargs or None,
        extra_body=default.extra_body,
        seed=default.seed,
        system_message=default.system_message,
    )


@dataclass
class ResolvedRunInputs:
    """One-shot view of every value ``cmd_run`` needs after CLI > preset >
    default resolution. Avoids spreading ``pick(...)`` calls through the
    CLI body."""

    benchmark: str
    base_url: str
    model: Optional[str]
    n_repeats: int
    num_examples: Optional[int]
    gen: GenConfig
    preset: Optional["Preset"]


def resolve_run_inputs(
    args: argparse.Namespace,
    spec_lookup: Callable[[str], Any],
) -> ResolvedRunInputs:
    """Apply CLI > preset > spec default to every run-level setting.

    ``spec_lookup`` is ``registry.get`` (or compatible); injected so this
    module stays free of ``registry`` import. Exits the process via
    ``sys.exit`` on missing benchmark or base_url -- same UX the CLI had
    before this refactor.
    """
    preset = load_preset(args.preset) if args.preset else None
    benchmark = pick(args.name, preset.benchmark if preset else None)
    if not benchmark:
        sys.exit("error: benchmark name required (positional arg or --preset)")
    spec = spec_lookup(benchmark)

    base_url = pick(args.base_url, preset.endpoint.base_url if preset else None)
    if not base_url:
        sys.exit("error: --base-url required (or set in preset endpoint.base_url)")

    return ResolvedRunInputs(
        benchmark=spec.name,
        base_url=base_url,
        model=pick(args.model, preset.endpoint.model if preset else None),
        n_repeats=pick(
            args.n_repeats,
            preset.n_repeats if preset else None,
            spec.default_n_repeats,
        ),
        num_examples=pick(args.num_examples, preset.num_examples if preset else None),
        gen=apply_to_gen(spec.default_gen, preset, args),
        preset=preset,
    )


# ---------- run-time integration ----------


def make_run_meta_block(
    args: argparse.Namespace, preset: Optional["Preset"]
) -> Optional[Dict[str, Any]]:
    """The ``preset`` sub-dict for ``metrics.json``'s ``run_meta``. Returns
    ``None`` when no preset was used (caller should ``if block: meta["preset"] = block``)
    so absent presets don't show up as ``"preset": null``."""
    if preset is None:
        return None
    return {
        "spec": args.preset,
        "path": str(resolve_preset_path(args.preset)) if args.preset else None,
        "benchmark": preset.benchmark,
        "expected_score": preset.expected.score if preset.expected else None,
    }


def print_expected_vs_actual(result: Any, preset: Optional["Preset"]) -> None:
    """Print headline-metric comparison if the preset declared one.
    Informational only -- never affects exit code."""
    if preset is None or preset.expected is None or preset.expected.score is None:
        return
    expected_score = preset.expected.score
    if result.n_repeats > 1 and "pass@1" in result.aggregate:
        actual = result.aggregate["pass@1"]
    else:
        actual = result.aggregate.get("score", 0.0)
    delta = actual - expected_score
    sign = "+" if delta >= 0 else ""
    print(
        f"\nExpected: {expected_score * 100:.2f}%  "
        f"Got: {actual * 100:.2f}%  "
        f"(delta {sign}{delta * 100:.2f}%)"
    )
