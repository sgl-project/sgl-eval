"""Harbor-format agentic benchmarks on the vendored pier trial runtime.

A Harbor task is a directory with ``task.toml``, ``instruction.md``, a
prebuilt task image, hidden ``tests/`` and a reference ``solution/``. DeepSWE
is the first dataset registered here (``_registry.py``).

Everything that decides a trial's outcome is the vendored pier runtime: the
``Trial`` state machine (environment start, agent, ``[[verifier.collect]]``,
artifact transfer, separate verifier, reward parsing), ``TrialQueue`` retries
and the ``Mean`` aggregation. This module selects tasks, builds
``TrialConfig`` objects, drives the queue, keeps a small ledger so an existing
``--run-dir`` can be continued, and projects the canonical ``result.json``
files onto sgl-eval's ``RunResult``.

The agent runs on the host (``sgl_eval.agents.mini_swe_agent``); the task
container only executes its commands, so tasks keep their ``no-network``
policy and no model traffic ever enters a sandbox. ``--<name>-oracle`` runs
the reference solution instead and needs no endpoint.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import openai
import yaml
from tqdm import tqdm

from sgl_eval.evals._loader import _download_verified_archive
from sgl_eval.predictions import PredSchema
from sgl_eval.registry import EvalSpec
from sgl_eval.runner._progress import _start_bar_refresher
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.sandbox import docker as sandbox
from sgl_eval.types import Example, ExampleResult, GenConfig, RunResult, Sample

_CACHE_ROOT = Path.home() / ".cache" / "sgl_eval" / "harbor"
_VENDORED_ROOT = Path(__file__).resolve().parent.parent / "_vendored"
_HOST_AGENT_IMPORT_PATH = "sgl_eval.agents.mini_swe_agent:HostMiniSweAgent"
LEDGER_FILENAME = "harbor_ledger.json"
TRIALS_DIRNAME = "trials"
# Agent trajectories on the reference runs reached 700K prompt tokens; below
# 128K almost every trial dies of context overflow.
_MIN_CONTEXT_TOKENS = 131072
_RECOMMENDED_CONTEXT_TOKENS = 393216
_SMOKE_MAX_TOKENS = 4096

# The prediction record's ``generation`` is the collected patch; scores are the
# verifier's reward (0/1 for DeepSWE, kept as a float).
PRED_SCHEMA = PredSchema(
    stringify_target=True, include_prompt=True, score_field="reward", binary_score=False
)


@dataclass(frozen=True)
class HarborConfig:
    """One Harbor dataset: where its tasks come from and the trial defaults."""

    name: str
    description: str
    archive_url: str
    archive_sha256: str
    tasks_subdir: str = "tasks"
    default_n_repeats: int = 1
    default_num_threads: int = 4
    thinking: bool = True
    # mini-swe-agent LocalEnvironment default; every action gets this long.
    per_action_timeout_sec: int = 30
    # The reference runs' mini-swe-agent overlay.
    max_consecutive_format_errors: int = 10
    # Transport budget per model call, matching the reference runs' litellm settings.
    request_timeout_sec: float = 2400.0
    sdk_max_retries: int = 8
    # pier CLI flags the reference runs used.
    default_agent_timeout_multiplier: float = 2.0
    default_max_retries: int = 1


class UnsupportedTaskError(ValueError):
    """The task uses a Harbor feature this runner does not implement."""


class ResumeError(RuntimeError):
    """A ledger entry claims a finished trial but its canonical result is unusable."""


def make_harbor_spec(cfg: HarborConfig) -> EvalSpec:
    return EvalSpec(
        name=cfg.name,
        category="harbor",
        description=cfg.description,
        default_gen=GenConfig(chat_template_kwargs={"thinking": True} if cfg.thinking else None),
        default_n_repeats=cfg.default_n_repeats,
        run=_make_run(cfg),
        default_num_threads=cfg.default_num_threads,
        pred_schema=PRED_SCHEMA,
        add_arguments=_make_add_arguments(cfg),
        resumable=True,
        requires_endpoint=lambda bench_args: not bench_args.get("oracle", False),
        fingerprint_exclude=frozenset({"retry_errored"}),
    )


# ---------- CLI ----------


@dataclass
class _Options:
    tasks: List[str]
    oracle: bool
    agent_timeout_multiplier: float
    max_retries: int
    retry_errored: bool


def _options(cfg: HarborConfig, bench_args: Optional[Dict[str, Any]]) -> _Options:
    b = bench_args or {}
    return _Options(
        tasks=list(b.get("task") or []),
        oracle=bool(b.get("oracle", False)),
        agent_timeout_multiplier=float(
            b.get("agent_timeout_multiplier", cfg.default_agent_timeout_multiplier)
        ),
        max_retries=int(b.get("max_retries", cfg.default_max_retries)),
        retry_errored=bool(b.get("retry_errored", False)),
    )


def _make_add_arguments(cfg: HarborConfig) -> Callable[[Any], None]:
    p = cfg.name

    def add_arguments(group: Any) -> None:
        group.add_argument(
            f"--{p}-task",
            action="append",
            metavar="TASK_ID",
            default=None,
            help="run only this task directory name; repeatable",
        )
        group.add_argument(
            f"--{p}-oracle",
            action="store_true",
            default=False,
            help="run each task's reference solution instead of a model (no endpoint needed); "
            "reward 1.0 everywhere means Docker and the verifiers work",
        )
        group.add_argument(
            f"--{p}-agent-timeout-multiplier",
            type=float,
            default=cfg.default_agent_timeout_multiplier,
            metavar="X",
            help="scale each task's agent timeout_sec; a timed-out agent is still collected "
            f"and verified (default {cfg.default_agent_timeout_multiplier})",
        )
        group.add_argument(
            f"--{p}-max-retries",
            type=int,
            default=cfg.default_max_retries,
            metavar="N",
            help="re-run a trial that errored, up to N times, keeping only the last attempt; "
            "agent and verifier timeouts and reward-file errors are never retried "
            f"(default {cfg.default_max_retries})",
        )
        group.add_argument(
            f"--{p}-retry-errored",
            action="store_true",
            default=False,
            help="with --run-dir on an existing run: schedule finished-but-errored trials again",
        )

    return add_arguments


# ---------- run ----------


def _make_run(cfg: HarborConfig) -> Callable[..., RunResult]:
    def run(
        *,
        sampler: Optional[ChatCompletionSampler],
        gen: GenConfig,
        n_repeats: int,
        num_examples: Optional[int],
        num_threads: int,
        predictions_writer: Optional[Callable] = None,
        load_examples: Optional[Callable] = None,
        bench_args: Optional[Dict[str, Any]] = None,
        prompt_yaml: Optional[Path] = None,
        run_dir: Optional[Path] = None,
        resume: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> RunResult:
        if prompt_yaml is not None:
            raise ValueError(
                f"{cfg.name} does not take --prompt: the agent prompt is mini-swe-agent's "
                "mini.yaml (see benchmarks.md)."
            )
        if load_examples is not None:
            raise ValueError(
                f"{cfg.name} does not take --from-dataset; pick tasks with --{cfg.name}-task."
            )
        if run_dir is None:
            raise ValueError(f"{cfg.name} needs a run directory for its trial state")
        return run_harbor(
            cfg,
            sampler=sampler,
            gen=gen,
            n_repeats=n_repeats,
            num_examples=num_examples,
            num_threads=num_threads,
            predictions_writer=predictions_writer,
            options=_options(cfg, bench_args),
            run_dir=run_dir,
            resume=resume,
            cancel_event=cancel_event or threading.Event(),
        )

    return run


@dataclass
class _PlannedTrial:
    task_dir: Path
    task: Any  # vendored pier Task
    rep: int

    @property
    def name(self) -> str:
        return trial_name(self.task_dir.name, self.rep)


def trial_name(task_id: str, rep: int) -> str:
    return f"{task_id}__rs{rep}"


def run_harbor(
    cfg: HarborConfig,
    *,
    sampler: Optional[ChatCompletionSampler],
    gen: GenConfig,
    n_repeats: int,
    num_examples: Optional[int],
    num_threads: int,
    predictions_writer: Optional[Callable],
    options: _Options,
    run_dir: Path,
    resume: bool,
    cancel_event: threading.Event,
) -> RunResult:
    from sgl_eval.agents.mini_swe_agent import (
        AGENT_NAME,
        HostAgentSettings,
        register_run,
        unregister_run,
    )

    if not options.oracle and sampler is None:
        raise ValueError(f"{cfg.name} needs a model endpoint unless --{cfg.name}-oracle is set")

    tasks_root = ensure_tasks_root(cfg)
    task_dirs = select_task_dirs(tasks_root, options.tasks, num_examples)
    tasks = [load_supported_task(task_dir) for task_dir in task_dirs]
    plan = [
        _PlannedTrial(task_dir, task, rep)
        for task_dir, task in zip(task_dirs, tasks)
        for rep in range(n_repeats)
    ]

    for warning in _docker_preflight(num_threads, tasks):
        print(f"WARNING: {warning}", file=sys.stderr)
    if not options.oracle:
        assert sampler is not None
        preflight_endpoint(sampler, gen)

    trials_dir = run_dir / TRIALS_DIRNAME
    trials_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / LEDGER_FILENAME)
    replayed, pending = _partition(plan, ledger, trials_dir, resume, options.retry_errored)
    if resume:
        print(
            f"Resuming: {len(replayed)} finished trial(s) kept, {len(pending)} to run"
            + (" (including errored trials scheduled again)" if options.retry_errored else "")
        )

    run_id = uuid.uuid4().hex
    model_name = sampler.model if sampler is not None else None
    if not options.oracle:
        assert sampler is not None
        register_run(
            run_id,
            HostAgentSettings(
                sampler=sampler,
                gen=gen,
                cancel_event=cancel_event,
                per_action_timeout_sec=cfg.per_action_timeout_sec,
                max_consecutive_format_errors=cfg.max_consecutive_format_errors,
                request_timeout_sec=cfg.request_timeout_sec,
                sdk_max_retries=cfg.sdk_max_retries,
            ),
        )

    tic = time.perf_counter()
    try:
        fresh = (
            asyncio.run(
                _execute(
                    cfg,
                    options,
                    pending,
                    trials_dir=trials_dir,
                    ledger=ledger,
                    num_threads=num_threads,
                    run_id=run_id,
                    model_name=model_name,
                    cancel_event=cancel_event,
                )
            )
            if pending and not cancel_event.is_set()
            else {}
        )
    finally:
        unregister_run(run_id)
    latency = time.perf_counter() - tic

    return _assemble(
        cfg,
        plan,
        {**replayed, **fresh},
        trials_dir=trials_dir,
        n_repeats=n_repeats,
        latency=latency,
        predictions_writer=predictions_writer,
        provenance={
            "agent": "oracle" if options.oracle else AGENT_NAME,
            "model": model_name,
            "agent_timeout_multiplier": options.agent_timeout_multiplier,
            "max_retries": options.max_retries,
            "resume": {
                "resumed": resume,
                "replayed_trials": len(replayed),
                "retry_errored": options.retry_errored,
            },
        },
    )


# ---------- task data ----------


def ensure_tasks_root(cfg: HarborConfig) -> Path:
    """Download and verify the pinned task archive once; return its tasks directory."""
    dest = _CACHE_ROOT / cfg.name / cfg.archive_sha256[:16]
    tasks_dir = dest / cfg.tasks_subdir
    if (dest / ".complete").exists() and tasks_dir.is_dir():
        return tasks_dir

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {cfg.name} tasks from {cfg.archive_url}")
    archive = _download_verified_archive(
        cfg.archive_url, expected_sha256=cfg.archive_sha256, directory=_CACHE_ROOT
    )
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=_CACHE_ROOT))
    try:
        with tarfile.open(archive) as tar:
            _safe_extractall(tar, staging)
        top_level = [p for p in staging.iterdir()]
        source = (
            top_level[0]
            if len(top_level) == 1
            and top_level[0].is_dir()
            and not (staging / cfg.tasks_subdir).exists()
            else staging
        )
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        (dest / ".complete").write_text(cfg.archive_sha256 + "\n")
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    if not tasks_dir.is_dir():
        raise RuntimeError(f"{cfg.archive_url} has no {cfg.tasks_subdir}/ directory")
    return tasks_dir


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing members or link targets that escape ``dest``.

    The stdlib ``filter="data"`` is not used: on 3.10.12 through 3.10.13 it
    resolves relative symlink targets against the destination root instead of
    the member's directory and rejects in-archive links such as
    ``tasks/README.md -> ../README.md``.
    """
    root = dest.resolve()
    members = []
    for member in tar.getmembers():
        if member.isdev():
            continue
        target = (root / member.name).resolve()
        if not _inside(target, root):
            raise ValueError(f"archive member {member.name!r} escapes the extraction directory")
        if member.issym() or member.islnk():
            base = target.parent if member.issym() else root
            if not _inside((base / member.linkname).resolve(), root):
                raise ValueError(f"archive link {member.name!r} points outside the archive")
        members.append(member)
    try:
        tar.extractall(dest, members=members, filter="fully_trusted")
    except TypeError:  # Python without the extraction filter argument
        tar.extractall(dest, members=members)


def _inside(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def select_task_dirs(
    tasks_root: Path, wanted: List[str], num_examples: Optional[int]
) -> List[Path]:
    """Tasks in alphabetical order; ``wanted`` narrows, ``num_examples`` takes the first N."""
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir() and (p / "task.toml").exists())
    if wanted:
        by_name = {p.name: p for p in task_dirs}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ValueError(
                f"unknown task(s) {missing}; task ids are the directory names under {tasks_root}"
            )
        task_dirs = [by_name[name] for name in wanted]
    if num_examples is not None:
        task_dirs = task_dirs[:num_examples]
    if not task_dirs:
        raise ValueError(f"no tasks selected under {tasks_root}")
    return task_dirs


def load_supported_task(task_dir: Path):
    """Parse the task with the vendored pier model and refuse shapes this runner does not cover."""
    from sgl_eval._vendored.pier.models.task.config import TaskOS, VerifierEnvironmentMode
    from sgl_eval._vendored.pier.models.task.task import Task
    from sgl_eval._vendored.pier.models.task.verifier_mode import resolve_task_verifier_mode

    task = Task(task_dir)
    problems = []
    if task.has_steps:
        problems.append("multi-step tasks ([[steps]]) are not supported")
    if task.config.environment.os != TaskOS.LINUX:
        problems.append(f"environment.os = {task.config.environment.os.value!r}; only linux")
    if resolve_task_verifier_mode(task.config) != VerifierEnvironmentMode.SEPARATE:
        problems.append("verifier.environment_mode must be 'separate'")
    if not task.config.environment.docker_image:
        problems.append("environment.docker_image (a prebuilt image) is required")
    if task.config.environment.allow_internet:
        problems.append("the agent environment must resolve to network_mode = 'no-network'")
    if problems:
        raise UnsupportedTaskError(f"{task_dir.name}: " + "; ".join(problems))
    return task


# ---------- preflight ----------


def _docker_preflight(num_threads: int, tasks: List[Any]) -> List[str]:
    task_cpus = max((t.config.environment.cpus or 1 for t in tasks), default=1)
    task_memory_mb = max((t.config.environment.memory_mb or 1024 for t in tasks), default=1024)
    try:
        return sandbox.host_preflight(
            num_threads, task_cpus=task_cpus, task_memory_mb=task_memory_mb
        )
    except sandbox.HostPreflightError as exc:
        sys.exit(f"error: {exc}")


def preflight_endpoint(sampler: ChatCompletionSampler, gen: GenConfig) -> None:
    """Refuse endpoints that cannot run an agent trial: short context or no tool calls."""
    limit = _endpoint_context_length(sampler)
    if limit is None:
        print(
            "WARNING: could not read the endpoint's context length; agent trajectories need "
            f"at least {_MIN_CONTEXT_TOKENS} tokens ({_RECOMMENDED_CONTEXT_TOKENS} recommended).",
            file=sys.stderr,
        )
    elif limit < _MIN_CONTEXT_TOKENS:
        sys.exit(
            f"error: endpoint context length {limit} is below {_MIN_CONTEXT_TOKENS}; agent "
            "trajectories routinely exceed it, so most trials would die of context overflow. "
            "Serve with a larger --context-length."
        )
    elif limit < _RECOMMENDED_CONTEXT_TOKENS:
        print(
            f"WARNING: endpoint context length {limit} is below the recommended "
            f"{_RECOMMENDED_CONTEXT_TOKENS}; long trials may overflow.",
            file=sys.stderr,
        )
    else:
        print(f"Preflight: endpoint context length {limit}")
    _tools_smoke(sampler, gen)


def _endpoint_context_length(sampler: ChatCompletionSampler) -> Optional[int]:
    base = str(sampler.client.base_url).rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        info = httpx.get(f"{root}/get_model_info", timeout=10).json()
    except Exception:
        info = {}
    for key in ("max_context_length", "context_length", "max_model_len"):
        if info.get(key):
            return int(info[key])
    try:
        models = httpx.get(f"{base}/models", timeout=10).json().get("data", [])
    except Exception:
        return None
    for entry in models:
        if entry.get("id") == sampler.model or len(models) == 1:
            if entry.get("max_model_len"):
                return int(entry["max_model_len"])
    return None


def _tools_smoke(sampler: ChatCompletionSampler, gen: GenConfig) -> None:
    from sgl_eval._vendored.mini_swe_agent.models.utils.actions_toolcall import BASH_TOOL

    smoke_gen = dataclasses.replace(
        gen, max_tokens=min(gen.max_tokens or _SMOKE_MAX_TOKENS, _SMOKE_MAX_TOKENS)
    )
    messages = [
        {
            "role": "user",
            "content": "Use the bash tool to run exactly this command: echo sgl-eval-smoke",
        }
    ]
    try:
        response = sampler.complete_raw(
            messages,
            smoke_gen,
            tools=[BASH_TOOL],
            tool_choice={"type": "function", "function": {"name": "bash"}},
        )
    except openai.BadRequestError as exc:
        sys.exit(
            f"error: the endpoint rejected a tool-calling request: {exc}\n"
            "mini-swe-agent drives the model through OpenAI function calling; serve with a "
            "tool-call parser (sglang: --tool-call-parser <name>)."
        )
    choice = response.choices[0]
    if choice.message.tool_calls:
        print("Preflight: endpoint returned a bash tool call")
        return
    if choice.finish_reason == "length":
        sys.exit(
            "error: the tool-calling smoke request hit the output limit before producing a "
            "tool call; the model spent it on reasoning. Raise --max-tokens or lower "
            "--reasoning-effort."
        )
    sys.exit(
        "error: the endpoint answered the smoke request without tool_calls. mini-swe-agent "
        "needs OpenAI function calling: serve with a tool-call parser for this model "
        "(sglang: --tool-call-parser <name>) and check the model supports tools."
    )


# ---------- ledger + resume ----------


class Ledger:
    """Scheduling state per trial name; the canonical results live in ``trials/<name>/result.json``.

    States: ``running`` (an attempt is in flight), ``finalized`` (``submit`` returned and the
    canonical result loaded), ``cancelled`` (the run was interrupted while it ran).
    Only ``finalized`` trials are scored or replayed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = (
            json.loads(path.read_text()) if path.exists() else {}
        )

    def state(self, name: str) -> Optional[str]:
        return self.entries.get(name, {}).get("state")

    def mark(self, name: str, state: str, **fields: Any) -> None:
        entry = self.entries.setdefault(name, {})
        entry.update(fields)
        entry["state"] = state
        entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
        os.replace(tmp, self.path)


def _partition(
    plan: List[_PlannedTrial], ledger: Ledger, trials_dir: Path, resume: bool, retry_errored: bool
) -> Tuple[Dict[str, Any], List[_PlannedTrial]]:
    """Split the plan into finished trials to replay and trials to run now."""
    replayed: Dict[str, Any] = {}
    pending: List[_PlannedTrial] = []
    for trial in plan:
        if resume and ledger.state(trial.name) == "finalized":
            result = _load_result(trials_dir / trial.name, trial.name)
            if retry_errored and result.exception_info is not None:
                _clear_trial_dir(trials_dir / trial.name)
                ledger.mark(trial.name, "pending", note="retry_errored")
                pending.append(trial)
            else:
                replayed[trial.name] = result
            continue
        _clear_trial_dir(trials_dir / trial.name)
        pending.append(trial)
    return replayed, pending


def _clear_trial_dir(trial_dir: Path) -> None:
    if trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)


def _load_result(trial_dir: Path, name: str):
    from sgl_eval._vendored.pier.models.trial.result import TrialResult

    path = trial_dir / "result.json"
    if not path.exists():
        raise ResumeError(f"{name}: recorded as finished but {path} is missing")
    try:
        result = TrialResult.model_validate_json(path.read_text())
    except Exception as exc:
        raise ResumeError(f"{name}: {path} is not a valid trial result: {exc}") from exc
    if result.trial_name != name:
        raise ResumeError(f"{name}: {path} belongs to trial {result.trial_name!r}")
    return result


# ---------- execution ----------


class _Progress:
    """One bar over trials with live phase counts; finalized results move the counters.

    Without a terminal (a log file) the bar would be rewritten twice a second
    for hours, so it is replaced by one line per finished trial.
    """

    def __init__(self, name: str, total: int) -> None:
        self.name = name
        self.total = total
        self.tty = sys.stderr.isatty()
        self.bar = tqdm(total=total, desc=name, dynamic_ncols=True, disable=not self.tty)
        self.phase: Dict[str, str] = {}
        self.resolved = self.failed = self.errored = 0
        if self.tty:
            self._stop, self._thread = _start_bar_refresher([self.bar])
        else:
            self._stop, self._thread = None, None

    async def on_event(self, event: Any) -> None:
        from sgl_eval._vendored.pier.trial.hooks import TrialEvent

        if event.event == TrialEvent.START:
            self.phase[event.trial_id] = "starting"
        elif event.event == TrialEvent.AGENT_START:
            self.phase[event.trial_id] = "agent"
        elif event.event == TrialEvent.VERIFICATION_START:
            self.phase[event.trial_id] = "verifying"
        elif event.event in (TrialEvent.END, TrialEvent.CANCEL):
            self.phase.pop(event.trial_id, None)
        self._refresh()

    def finalized(self, trial: str, reward: float, errored: bool) -> None:
        if reward >= 1.0:
            self.resolved += 1
        else:
            self.failed += 1
        if errored:
            self.errored += 1
        self.bar.update(1)
        self._refresh()
        if not self.tty:
            done = self.resolved + self.failed
            print(
                f"[{self.name}] {done}/{self.total} finished: {trial} reward={reward:g}"
                f"{' errored' if errored else ''} | resolved={self.resolved} "
                f"failed={self.failed} errored={self.errored}",
                flush=True,
            )

    def _refresh(self) -> None:
        if not self.tty:
            return
        phases = list(self.phase.values())
        self.bar.set_postfix(
            {
                "resolved": self.resolved,
                "failed": self.failed,
                "errored": self.errored,
                "starting": phases.count("starting"),
                "agent": phases.count("agent"),
                "verifying": phases.count("verifying"),
            },
            refresh=False,
        )

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.bar.close()


async def _execute(
    cfg: HarborConfig,
    options: _Options,
    pending: List[_PlannedTrial],
    *,
    trials_dir: Path,
    ledger: Ledger,
    num_threads: int,
    run_id: str,
    model_name: Optional[str],
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    from sgl_eval._vendored.pier.models.job.config import RetryConfig
    from sgl_eval._vendored.pier.trial.hooks import TrialEvent
    from sgl_eval._vendored.pier.trial.queue import TrialQueue

    loop = asyncio.get_running_loop()
    # Every running trial parks one worker thread on the agent loop and may
    # need a second for cancellation; the default pool is too small for that.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=max(8, num_threads * 3)))

    progress = _Progress(cfg.name, len(pending))
    queue = TrialQueue(
        n_concurrent=num_threads,
        retry_config=RetryConfig(max_retries=options.max_retries),
        hooks={event: [progress.on_event] for event in TrialEvent},
    )
    results: Dict[str, Any] = {}

    async def run_one(trial: _PlannedTrial) -> None:
        config = _trial_config(cfg, options, trial, trials_dir, run_id, model_name)
        ledger.mark(trial.name, "running", task=trial.task_dir.name, repeat=trial.rep)
        try:
            await queue.submit(config)
        except asyncio.CancelledError:
            ledger.mark(trial.name, "cancelled")
            raise
        result = _load_result(trials_dir / trial.name, trial.name)
        ledger.mark(trial.name, "finalized")
        results[trial.name] = result
        progress.finalized(trial.name, _reward_of(result), result.exception_info is not None)

    tasks = [asyncio.create_task(run_one(trial)) for trial in pending]
    watcher = asyncio.create_task(_cancel_on_event(cancel_event, tasks))
    try:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        watcher.cancel()
        progress.close()
    for trial, outcome in zip(pending, outcomes):
        if isinstance(outcome, Exception):
            print(
                f"ERROR: trial {trial.name}: {type(outcome).__name__}: {outcome}", file=sys.stderr
            )
    return results


async def _cancel_on_event(cancel_event: threading.Event, tasks: List[asyncio.Task]) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.2)
    for task in tasks:
        if not task.done():
            task.cancel()


def _trial_config(
    cfg: HarborConfig,
    options: _Options,
    trial: _PlannedTrial,
    trials_dir: Path,
    run_id: str,
    model_name: Optional[str],
):
    from sgl_eval._vendored.pier.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )

    if options.oracle:
        agent = AgentConfig(name="oracle")
    else:
        agent = AgentConfig(
            import_path=_HOST_AGENT_IMPORT_PATH,
            model_name=model_name,
            kwargs={"run_id": run_id},
        )
    return TrialConfig(
        task=TaskConfig(path=trial.task_dir, source=cfg.name),
        trial_name=trial.name,
        trials_dir=trials_dir,
        agent_timeout_multiplier=options.agent_timeout_multiplier,
        agent=agent,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(),
    )


# ---------- results ----------


def _reward_of(result: Any) -> float:
    rewards = result.verifier_result.rewards if result.verifier_result is not None else None
    if not rewards:
        return 0.0
    return float(rewards.get("reward", next(iter(rewards.values()), 0.0)))


def _assemble(
    cfg: HarborConfig,
    plan: List[_PlannedTrial],
    results: Dict[str, Any],
    *,
    trials_dir: Path,
    n_repeats: int,
    latency: float,
    predictions_writer: Optional[Callable],
    provenance: Dict[str, Any],
) -> RunResult:
    from sgl_eval._vendored.pier.metrics.mean import Mean

    per_example: List[ExampleResult] = []
    reward_dicts: List[Optional[Dict[str, float]]] = []
    counts = {"planned": len(plan), "finalized": 0, "resolved": 0, "failed": 0, "errored": 0}
    usage = {"n_responses": 0, "usage_reported": 0, "prompt": 0, "completion": 0, "reasoning": 0}
    trials_meta: Dict[str, Any] = {}

    tasks_in_order: List[Tuple[Path, Any]] = []
    for trial in plan:
        if not tasks_in_order or tasks_in_order[-1][0] != trial.task_dir:
            tasks_in_order.append((trial.task_dir, trial.task))

    for task_dir, task in tasks_in_order:
        example = Example(
            id=task_dir.name,
            inputs={"problem": task.instruction},
            target=1.0,
            meta={"task_name": task.name, "docker_image": task.config.environment.docker_image},
        )
        samples: List[Sample] = []
        scores: List[float] = []
        extracted: List[Optional[str]] = []
        for rep in range(n_repeats):
            name = trial_name(task_dir.name, rep)
            result = results.get(name)
            if result is None:
                continue
            rewards = result.verifier_result.rewards if result.verifier_result is not None else None
            reward = _reward_of(result)
            errored = result.exception_info is not None
            reward_dicts.append(dict(rewards) if rewards else None)
            counts["finalized"] += 1
            counts["resolved" if reward >= 1.0 else "failed"] += 1
            counts["errored"] += int(errored)

            agent_meta = (result.agent_result.metadata if result.agent_result else None) or {}
            sample = _sample_from_result(result, trials_dir / name, agent_meta)
            status = agent_meta.get("exit_status") or (
                result.exception_info.exception_type if errored else None
            )
            samples.append(sample)
            scores.append(reward)
            extracted.append(status)
            if predictions_writer is not None:
                predictions_writer(example, rep, sample, reward, status)

            calls = int(agent_meta.get("n_model_calls") or 0)
            reported = int(agent_meta.get("usage_reported_calls") or 0)
            usage["n_responses"] += calls
            usage["usage_reported"] += reported
            usage["prompt"] += int(sample.prompt_tokens or 0)
            usage["completion"] += int(sample.completion_tokens or 0)
            usage["reasoning"] += int(sample.reasoning_tokens or 0)
            trials_meta[name] = {
                "task": task_dir.name,
                "repeat": rep,
                "reward": rewards,
                "errored": errored,
                "exception_type": result.exception_info.exception_type if errored else None,
                "exit_status": agent_meta.get("exit_status"),
                "n_model_calls": calls or None,
                "task_checksum": result.task_checksum,
                "docker_image": task.config.environment.docker_image,
                "trial_dir": str(trials_dir / name),
            }
        if samples:
            per_example.append(
                ExampleResult(example=example, samples=samples, scores=scores, extracted=extracted)
            )

    means = Mean().compute(reward_dicts) if reward_dicts else {}
    score = means.get("reward", means.get("mean", 0.0))
    aggregate: Dict[str, float] = {"score": float(score)}
    for key, value in means.items():
        aggregate[f"reward_{key}" if key != "mean" else "reward_mean"] = float(value)

    counts["not_run"] = counts["planned"] - counts["finalized"]
    summary_rows = [
        ("agent", provenance["agent"]),
        ("resolved", f"{counts['resolved']}/{counts['finalized']}"),
        ("failed", str(counts["failed"])),
        ("errored", str(counts["errored"])),
        ("not_run", str(counts["not_run"])),
    ]
    for key in ("f2p", "p2p", "partial"):
        if key in means:
            summary_rows.append((f"{key}_mean", f"{means[key] * 100:.2f}%"))

    reported = usage["usage_reported"]
    response_usage = None
    if usage["n_responses"]:
        response_usage = {
            "n_responses": usage["n_responses"],
            "usage_reported": reported,
            "mean_prompt_tokens": usage["prompt"] / reported if reported else None,
            "mean_completion_tokens": usage["completion"] / reported if reported else None,
            "mean_reasoning_tokens": usage["reasoning"] / reported if reported else None,
        }

    metadata = {
        "harbor": {
            "dataset": cfg.name,
            "archive_url": cfg.archive_url,
            "archive_sha256": cfg.archive_sha256,
            "pier_commit_sha": _vendored_sha("pier"),
            "mini_swe_agent_commit_sha": _vendored_sha("mini_swe_agent"),
            "per_action_timeout_sec": cfg.per_action_timeout_sec,
            "max_consecutive_format_errors": cfg.max_consecutive_format_errors,
            "request_timeout_sec": cfg.request_timeout_sec,
            "sdk_max_retries": cfg.sdk_max_retries,
            "retry_excluded_exceptions": sorted(_retry_exclusions()),
            "environment_delta": (
                "host-side agent: the task container is the pristine task image (pier's "
                "installed-agent image additionally installs curl, build-essential, git and a "
                "Python for mini-swe-agent) and shell commands do not see OPENAI_*/MSWEA_* "
                "variables"
            ),
            "counts": counts,
            "trials": trials_meta,
            **provenance,
        },
        "summary_rows": summary_rows,
    }

    return RunResult(
        name=cfg.name,
        per_example=per_example,
        aggregate=aggregate,
        latency=latency,
        num_examples=len(per_example),
        n_repeats=n_repeats,
        total_completion_tokens=usage["completion"],
        total_prompt_tokens=usage["prompt"],
        partial=counts["finalized"] < counts["planned"],
        planned_examples=len(tasks_in_order),
        metadata=metadata,
        response_usage=response_usage,
    )


def _sample_from_result(result: Any, trial_dir: Path, agent_meta: Dict[str, Any]) -> Sample:
    patch = trial_dir / "artifacts" / "model.patch"
    text = patch.read_text(errors="replace") if patch.exists() else ""
    agent_result = result.agent_result
    return Sample(
        text=text,
        completion_tokens=agent_result.n_output_tokens if agent_result else None,
        prompt_tokens=agent_result.n_input_tokens if agent_result else None,
        reasoning_tokens=agent_meta.get("reasoning_tokens"),
        finish_reason="error" if result.exception_info is not None else "stop",
        generation_start_time=(
            result.agent_execution.started_at.timestamp()
            if result.agent_execution and result.agent_execution.started_at
            else None
        ),
        generation_end_time=(
            result.agent_execution.finished_at.timestamp()
            if result.agent_execution and result.agent_execution.finished_at
            else None
        ),
    )


def _vendored_sha(package: str) -> Optional[str]:
    manifest = _VENDORED_ROOT / package / "SOURCES.yaml"
    if not manifest.exists():
        return None
    return yaml.safe_load(manifest.read_text()).get("synced_from_sha")


def _retry_exclusions() -> set:
    from sgl_eval._vendored.pier.models.job.config import RetryConfig

    return set(RetryConfig().exclude_exceptions or ())
