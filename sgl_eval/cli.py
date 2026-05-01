"""``sgl-eval`` CLI entry point.

Subcommands:
  list                 enumerate registered benchmarks
  ping                 send one chat completion to the endpoint and print it
  run <name>           run a benchmark end-to-end
  preset list/show     manage saved (model, dataset, sampling) presets
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from sgl_eval import VENDORED_NS_ROOT
from sgl_eval import __version__ as _SGL_EVAL_VERSION
from sgl_eval.evals._predictions import PredictionsWriter
from sgl_eval.metrics import dump_run, format_summary
from sgl_eval.preset import (
    PRESET_ROOT,
    Preset,
    list_presets,
    load_preset,
    resolve_preset_path,
)
from sgl_eval.registry import get, list_evals
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import GenConfig


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="sgl-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list registered benchmarks")
    p_list.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show per-benchmark defaults (sampling params, n_repeats, thinking)",
    )
    p_list.set_defaults(func=cmd_list)

    p_ping = sub.add_parser("ping", help="send one request to the endpoint")
    _add_endpoint_args(p_ping)
    p_ping.add_argument("--prompt", default="Reply with the single word: pong.")
    p_ping.add_argument("--max-tokens", type=int, default=64)
    p_ping.add_argument("--temperature", type=float, default=0.0)
    p_ping.set_defaults(func=cmd_ping)

    p_run = sub.add_parser("run", help="run a benchmark")
    p_run.add_argument(
        "name",
        nargs="?",
        default=None,
        help="benchmark name (see `sgl-eval list`); optional if --preset provides one",
    )
    p_run.add_argument(
        "--preset",
        default=None,
        help="preset name (under ~/.sgl_eval/presets/) or path to a preset .yaml; "
        "CLI flags always override preset values",
    )
    _add_endpoint_args(p_run, base_url_required=False)
    p_run.add_argument("--num-examples", type=int, default=None)
    p_run.add_argument("--num-threads", type=int, default=64)
    p_run.add_argument("--n-repeats", type=int, default=None)
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--top-p", type=float, default=None)
    p_run.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override chat_template_kwargs.thinking (per-benchmark default applies otherwise)",
    )
    p_run.add_argument(
        "--out-dir",
        default="~/.sgl_eval",
        help="parent dir for run folders; each run writes into "
        "<out-dir>/sgl_eval_<name>_<stamp>/ (default: ~/.sgl_eval)",
    )
    p_run.add_argument(
        "--no-dump-predictions",
        dest="dump_predictions",
        action="store_false",
        help="skip streaming per-sample prediction JSONL (output-rs*.jsonl)",
    )
    p_run.set_defaults(func=cmd_run, dump_predictions=True)

    p_preset = sub.add_parser("preset", help="manage saved presets")
    preset_sub = p_preset.add_subparsers(dest="preset_cmd", required=True)
    p_preset_list = preset_sub.add_parser("list", help=f"list presets in {PRESET_ROOT}")
    p_preset_list.set_defaults(func=cmd_preset_list)
    p_preset_show = preset_sub.add_parser("show", help="print a preset's content")
    p_preset_show.add_argument("name", help="preset name (under PRESET_ROOT) or path")
    p_preset_show.set_defaults(func=cmd_preset_show)

    args = parser.parse_args(argv)
    return args.func(args)


def _add_endpoint_args(p: argparse.ArgumentParser, *, base_url_required: bool = True) -> None:
    p.add_argument(
        "--base-url",
        required=base_url_required,
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://host:30000/v1",
    )
    p.add_argument("--model", default=None, help="model id (defaults to first /v1/models entry)")
    p.add_argument("--api-key", default="EMPTY")


def cmd_list(args: argparse.Namespace) -> int:
    specs = list_evals()
    if not specs:
        print("(no benchmarks registered yet)")
        return 0
    if args.verbose:
        for s in specs:
            gen = s.default_gen
            ctk = gen.chat_template_kwargs or {}
            thinking = bool(ctk.get("thinking"))
            print(f"\n{s.name}  [{s.category}]")
            print(f"  description : {s.description}")
            print(f"  n_repeats   : {s.default_n_repeats}")
            print(f"  thinking    : {thinking}")
            print(f"  temperature : {gen.temperature}")
            print(f"  top_p       : {gen.top_p}")
            print(f"  max_tokens  : {gen.max_tokens}")
        return 0
    width = max(len(s.name) for s in specs)
    for s in specs:
        print(f"  {s.name:<{width}s}  [{s.category}]  {s.description}")
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    sampler = ChatCompletionSampler(base_url=args.base_url, model=args.model, api_key=args.api_key)
    gen = GenConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    sample = sampler([{"role": "user", "content": args.prompt}], gen)
    print(f"model            : {sampler.model}")
    print(f"finish_reason    : {sample.finish_reason}")
    print(f"completion_tokens: {sample.completion_tokens}")
    print(f"prompt_tokens    : {sample.prompt_tokens}")
    print("--- response ---")
    print(sample.text)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    preset = load_preset(args.preset) if args.preset else None

    # Resolve effective values. Priority: CLI flag > preset > spec default.
    benchmark = _pick(args.name, preset.benchmark if preset else None)
    if not benchmark:
        sys.exit("error: benchmark name required (positional arg or --preset)")
    spec = get(benchmark)

    base_url = _pick(args.base_url, preset.endpoint.base_url if preset else None)
    if not base_url:
        sys.exit("error: --base-url required (or set in preset endpoint.base_url)")

    model = _pick(args.model, preset.endpoint.model if preset else None)
    n_repeats = _pick(
        args.n_repeats,
        preset.n_repeats if preset else None,
        spec.default_n_repeats,
    )
    num_examples = _pick(args.num_examples, preset.num_examples if preset else None)
    gen = _resolve_gen(spec.default_gen, preset, args)

    sampler = ChatCompletionSampler(base_url=base_url, model=model, api_key=args.api_key)
    _warn_if_greedy_repeats(n_repeats, gen)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir).expanduser() / f"sgl_eval_{spec.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    writer = PredictionsWriter(run_dir, n_repeats) if args.dump_predictions else None
    try:
        result = spec.run(
            sampler=sampler,
            gen=gen,
            n_repeats=n_repeats,
            num_examples=num_examples,
            num_threads=args.num_threads,
            predictions_writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()

    print(format_summary(result))
    run_meta = _build_run_meta(
        args=args,
        sampler=sampler,
        gen=gen,
        stamp=stamp,
        base_url=base_url,
        preset=preset,
        preset_spec=args.preset,
    )
    metrics_path = dump_run(result, run_dir, run_meta=run_meta)
    print(f"\nMetrics: {metrics_path}")
    if writer is not None:
        print(f"Predictions: {run_dir}  ({n_repeats} jsonl file(s))")
    if preset and preset.expected and preset.expected.score is not None:
        _print_expected_vs_actual(result, preset.expected.score)
    return 0


def _pick(*candidates: Any) -> Any:
    """First non-``None`` candidate wins. Used for the
    ``CLI > preset > default`` resolution chain so that ``0`` /
    ``0.0`` / ``False`` aren't mistaken for "unset" the way ``or``
    would treat them."""
    for c in candidates:
        if c is not None:
            return c
    return None


def _resolve_gen(
    default: GenConfig, preset: Optional[Preset], args: argparse.Namespace
) -> GenConfig:
    p = preset.sampling if preset else None
    chat_template_kwargs = dict(default.chat_template_kwargs or {})
    thinking = _pick(args.thinking, p.thinking if p else None)
    if thinking is not None:
        chat_template_kwargs["thinking"] = thinking
    return GenConfig(
        temperature=_pick(args.temperature, p.temperature if p else None, default.temperature),
        top_p=_pick(args.top_p, p.top_p if p else None, default.top_p),
        max_tokens=_pick(args.max_tokens, p.max_tokens if p else None, default.max_tokens),
        reasoning_effort=default.reasoning_effort,
        chat_template_kwargs=chat_template_kwargs or None,
        extra_body=default.extra_body,
        seed=default.seed,
        system_message=default.system_message,
    )


def _print_expected_vs_actual(result: Any, expected_score: float) -> None:
    """Headline metric: ``pass@1`` for k>1, plain ``score`` for k==1.
    Informational only -- never affects exit code."""
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


def _build_run_meta(
    *,
    args: argparse.Namespace,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    stamp: str,
    base_url: str,
    preset: Optional[Preset] = None,
    preset_spec: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp": stamp,
        "model": sampler.model,
        "base_url": base_url,
        "num_threads": args.num_threads,
        "gen": dataclasses.asdict(gen),
        "sgl_eval_version": _SGL_EVAL_VERSION,
        "ns_commit_sha": _read_ns_commit_sha(),
    }
    if preset is not None:
        # ``preset_spec`` is the user-provided arg (name or path); resolved
        # path is more useful for provenance because it disambiguates same-
        # named presets across machines.
        meta["preset"] = {
            "spec": preset_spec,
            "path": str(resolve_preset_path(preset_spec)) if preset_spec else None,
            "benchmark": preset.benchmark,
            "expected_score": preset.expected.score if preset.expected else None,
        }
    return meta


def cmd_preset_list(args: argparse.Namespace) -> int:
    paths = list_presets()
    if not paths:
        print(f"(no presets in {PRESET_ROOT})")
        return 0
    width = max(len(p.stem) for p in paths)
    for p in paths:
        print(f"  {p.stem:<{width}s}  ({p})")
    return 0


def cmd_preset_show(args: argparse.Namespace) -> int:
    path = resolve_preset_path(args.name)
    if not path.exists():
        print(f"preset not found: {path}", file=sys.stderr)
        return 1
    sys.stdout.write(path.read_text(encoding="utf-8"))
    if not path.read_text(encoding="utf-8").endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _read_ns_commit_sha() -> Optional[str]:
    """Vendored slice's pinned upstream SHA (``SOURCES.yaml``). Returns
    ``None`` only when the manifest is absent (e.g. running from a non-
    standard checkout); a malformed manifest is a sync_vendored bug and
    should surface as a YAMLError, not be swallowed."""
    manifest = VENDORED_NS_ROOT / "SOURCES.yaml"
    try:
        text = manifest.read_text()
    except FileNotFoundError:
        return None
    return yaml.safe_load(text).get("synced_from_sha")


def _warn_if_greedy_repeats(n_repeats: int, gen: GenConfig) -> None:
    if n_repeats > 1 and gen.temperature == 0.0:
        print(
            f"WARNING: n_repeats={n_repeats} but temperature=0.0 (greedy). "
            f"All {n_repeats} samples per example will be identical -- pass "
            "--temperature N (e.g. 1.0 for DSv3.2/V4, 0.6 for R1) for stochastic sampling.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
