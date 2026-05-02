"""``sgl-eval`` CLI entry point.

Subcommands:
  list                 enumerate registered benchmarks
  ping                 send one chat completion to the endpoint and print it
  run <name>           run a benchmark end-to-end (orchestrated by ``pipeline``)
  preset list/show     manage saved (model, dataset, sampling) presets
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from sgl_eval.pipeline import cmd_run
from sgl_eval.preset import add_preset_run_flag, register_preset_subcommand
from sgl_eval.registry import list_evals
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
    add_preset_run_flag(p_run)
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

    register_preset_subcommand(sub)

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


if __name__ == "__main__":
    raise SystemExit(main())
