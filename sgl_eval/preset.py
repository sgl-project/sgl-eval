"""Load presets and resolve CLI > preset > benchmark defaults.

Presets capture benchmark, endpoint, sampling, repeats, sample limits, and
an optional expected score. API keys and output locations are supplied
per invocation. This module also provides preset CLI commands and provenance.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import yaml

from sgl_eval.types import GenConfig

if TYPE_CHECKING:
    from sgl_eval.model_preset import ModelPreset

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
    reasoning_effort: Optional[Union[str, float]] = None
    # Model templates may read different keys, such as Qwen3 enable_thinking.
    chat_template_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class Expected:
    # Headline metric on [0, 1]; informational only, never gates exit code.
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
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{source}: unknown fields {sorted(unknown)}")


def resolve_preset_path(spec: str) -> Path:
    """Resolve names under PRESET_ROOT; slashes and YAML suffixes denote paths."""
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


def add_preset_run_flag(p_run: argparse.ArgumentParser) -> None:
    """Add ``--preset`` to ``sgl-eval run``."""
    p_run.add_argument(
        "--preset",
        default=None,
        help="preset name (under ~/.sgl_eval/presets/) or path to a preset .yaml; "
        "CLI flags always override preset values",
    )


def register_preset_subcommand(sub: Any) -> None:
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
    """Only None is unset; zero and False must survive override resolution."""
    for c in candidates:
        if c is not None:
            return c
    return None


_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def reasoning_effort(raw: str) -> Union[str, float]:
    """Reject unsupported effort values before they become failed requests."""
    if raw in _EFFORT_LEVELS:
        return raw
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{raw!r} is not one of {', '.join(_EFFORT_LEVELS)} or a float in [0, 0.99]"
        ) from None
    if not 0.0 <= value <= 0.99:
        raise argparse.ArgumentTypeError(f"float effort must be in [0, 0.99], got {value}")
    return value


def parse_chat_template_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Preserve JSON value types; unquoted non-JSON values remain strings."""
    parsed: Dict[str, Any] = {}
    for item in getattr(args, "chat_template_kwarg", None) or []:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--chat-template-kwarg expects K=V, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parsed[key.strip()] = value
    return parsed


def apply_to_gen(
    default: GenConfig,
    preset: Optional["Preset"],
    args: argparse.Namespace,
    model_preset: Optional["ModelPreset"] = None,
) -> GenConfig:
    """Resolve generation settings with CLI > user preset > model preset > benchmark default precedence.

    Seed has no preset field. Model-specific chat-template keys are retained
    because they change the prompt.
    """
    p = preset.sampling if preset else None
    mp = model_preset.sampling if model_preset else None
    cli = Sampling(thinking=args.thinking, chat_template_kwargs=parse_chat_template_kwargs(args))
    chat_template_kwargs = dict(default.chat_template_kwargs or {})
    # Lowest priority first; each later layer overwrites the previous one.
    for layer in (mp, p, cli):
        if layer is None:
            continue
        if layer.thinking is not None:
            chat_template_kwargs["thinking"] = layer.thinking
        # Within one source the nested key wins over the flat ``thinking`` alias.
        chat_template_kwargs.update(layer.chat_template_kwargs or {})
    return GenConfig(
        temperature=pick(
            args.temperature,
            p.temperature if p else None,
            mp.temperature if mp else None,
            default.temperature,
        ),
        top_p=pick(
            args.top_p,
            p.top_p if p else None,
            mp.top_p if mp else None,
            default.top_p,
        ),
        max_tokens=pick(
            args.max_tokens,
            p.max_tokens if p else None,
            mp.max_tokens if mp else None,
            default.max_tokens,
        ),
        min_p=default.min_p,
        repetition_penalty=default.repetition_penalty,
        reasoning_effort=pick(
            args.reasoning_effort,
            p.reasoning_effort if p else None,
            mp.reasoning_effort if mp else None,
            default.reasoning_effort,
        ),
        chat_template_kwargs=chat_template_kwargs or None,
        extra_body=default.extra_body,
        seed=pick(getattr(args, "seed", None), default.seed),
        system_message=default.system_message,
    )


@dataclass
class ResolvedRunInputs:
    """Run settings after CLI, preset, and benchmark-default resolution."""

    benchmark: str
    base_url: str
    model: Optional[str]
    n_repeats: int
    num_examples: Optional[int]
    gen: GenConfig
    preset: Optional["Preset"]
    model_preset: Optional["ModelPreset"]


def resolve_run_inputs(
    args: argparse.Namespace,
    spec_lookup: Callable[[str], Any],
) -> ResolvedRunInputs:
    """Resolve run settings; missing benchmark or base URL is a CLI error.

    The registry lookup is injected to avoid an import dependency on registry.
    """
    from sgl_eval.model_preset import UnsupportedModelPresetError, load_model_preset

    preset = load_preset(args.preset) if args.preset else None
    model_preset_id = getattr(args, "load_preset_from_model_id", None)
    try:
        model_preset = load_model_preset(model_preset_id) if model_preset_id else None
    except UnsupportedModelPresetError as exc:
        supported = "\n".join(f"  - {model_id}" for model_id in exc.supported_model_ids)
        sys.exit(
            f"error: no built-in model preset for {exc.model_id!r}.\n"
            f"Supported model IDs:\n{supported}\n"
            "Omit --load-preset-from-model-id and configure the benchmark manually with "
            "--model, --temperature, --top-p, --max-tokens, --thinking/--no-thinking, "
            "and --reasoning-effort."
        )
    benchmark = pick(args.name, preset.benchmark if preset else None)
    if not benchmark:
        sys.exit("error: benchmark name required (positional arg or --preset)")
    spec = spec_lookup(benchmark)

    base_url = pick(args.base_url, preset.endpoint.base_url if preset else None)
    if not base_url:
        detail = "; built-in model presets do not set endpoints" if model_preset is not None else ""
        sys.exit(
            "error: --base-url required "
            f"(pass it explicitly or set preset endpoint.base_url{detail})"
        )

    return ResolvedRunInputs(
        benchmark=spec.name,
        base_url=base_url,
        model=pick(
            args.model,
            preset.endpoint.model if preset else None,
            model_preset.model if model_preset else None,
        ),
        n_repeats=pick(
            args.n_repeats,
            preset.n_repeats if preset else None,
            spec.default_n_repeats,
        ),
        num_examples=pick(args.num_examples, preset.num_examples if preset else None),
        gen=apply_to_gen(spec.default_gen, preset, args, model_preset),
        preset=preset,
        model_preset=model_preset,
    )


# ---------- run-time integration ----------


def make_run_meta_block(
    args: argparse.Namespace, preset: Optional["Preset"]
) -> Optional[Dict[str, Any]]:
    """Return preset provenance, or None when the run does not use a preset."""
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
