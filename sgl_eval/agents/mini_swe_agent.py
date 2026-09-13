"""mini-swe-agent driven from the host process, plugged into the pier trial runtime.

The vendored ``DefaultAgent`` runs unchanged with mini-swe-agent's own
``mini.yaml`` prompt and observation templates. What sgl-eval supplies is the
two ends that upstream delegates to litellm and to the local shell:

- ``SglEvalModel`` keeps ``LitellmModel.query`` (history preparation, tool-call
  parsing, FormatError retention, observation formatting) and provides the
  transport: one ``chat.completions`` call per turn through the sgl-eval
  sampler, with ``tools=[BASH_TOOL]``.
- ``SandboxEnvironment`` keeps ``LocalEnvironment.execute`` (observation
  normalization, the submit sentinel) and runs each action inside the task
  container under ``/bin/sh -c`` with the 30s per-action timeout.

``HostMiniSweAgent`` is the pier ``BaseAgent`` that wires them together. pier
creates it from ``AgentConfig.import_path``; its kwargs are written to
``config.json``, so the live objects (sampler, generation config, cancel
event) are looked up from a per-run registry by ``run_id``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import yaml

from sgl_eval._vendored.mini_swe_agent import __version__ as MINI_SWE_AGENT_VERSION
from sgl_eval._vendored.mini_swe_agent.agents.default import DefaultAgent
from sgl_eval._vendored.mini_swe_agent.environments.local import LocalEnvironment
from sgl_eval._vendored.mini_swe_agent.models.litellm_model import LitellmModel
from sgl_eval._vendored.mini_swe_agent.models.utils.actions_toolcall import BASH_TOOL
from sgl_eval._vendored.mini_swe_agent.models.utils.retry import retry as upstream_retry
from sgl_eval._vendored.mini_swe_agent.utils.serialize import recursive_merge
from sgl_eval._vendored.pier.agents.base import BaseAgent
from sgl_eval._vendored.pier.agents.installed.base import NonZeroAgentExitCodeError
from sgl_eval._vendored.pier.environments.base import BaseEnvironment
from sgl_eval._vendored.pier.models.agent.context import AgentContext
from sgl_eval.runner import WorkerAborted
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.sandbox import docker as sandbox
from sgl_eval.sandbox.docker import ExecutionCancelled
from sgl_eval.types import GenConfig

LOG = logging.getLogger(__name__)

AGENT_NAME = "mini-swe-agent"
TRAJECTORY_FILENAME = "mini-swe-agent.trajectory.json"
MINI_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "_vendored" / "mini_swe_agent" / "config" / "mini.yaml"
)
_WORKER_EXIT_GRACE_SEC = 120
_SLEEP_SLICE_SEC = 0.25


@dataclass
class HostAgentSettings:
    """Per-run inputs a trial config cannot carry: live clients and events."""

    sampler: ChatCompletionSampler
    gen: GenConfig
    cancel_event: threading.Event
    per_action_timeout_sec: int = 30
    max_consecutive_format_errors: int = 10
    request_timeout_sec: float = 2400.0
    sdk_max_retries: int = 8


_RUNS: Dict[str, HostAgentSettings] = {}


def register_run(run_id: str, settings: HostAgentSettings) -> None:
    _RUNS[run_id] = settings


def unregister_run(run_id: str) -> None:
    _RUNS.pop(run_id, None)


def load_mini_config() -> Dict[str, Any]:
    """The vendored ``mini.yaml``: prompt, environment env and observation templates."""
    return yaml.safe_load(MINI_CONFIG_PATH.read_text())


def build_agent_config(settings: HostAgentSettings, output_path: Path) -> Dict[str, Any]:
    """``mini.yaml`` with the overlay the reference runs used.

    Same order as pier's flag assembly: ``mini.yaml``, then ``cost_limit=0``,
    then the user config (``max_consecutive_format_errors``), then the CLI
    ``output_path``. ``mode`` only matters to the interactive agent and is
    dropped.
    """
    config = recursive_merge(
        load_mini_config(),
        {
            "agent": {
                "cost_limit": 0,
                "max_consecutive_format_errors": settings.max_consecutive_format_errors,
                "output_path": str(output_path),
            }
        },
    )
    config["agent"].pop("mode", None)
    config["model"].pop("model_class", None)
    return config


class _CancelScope:
    """Run-wide and trial-wide cancel events checked by every hook."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def check(self) -> None:
        if self.is_set():
            raise ExecutionCancelled()

    def wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            if self.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(_SLEEP_SLICE_SEC, remaining))

    def as_event(self) -> threading.Event:
        """A single event view for code that polls one ``threading.Event``."""
        return _CompositeEvent(self)


class _CompositeEvent(threading.Event):
    def __init__(self, scope: _CancelScope) -> None:
        super().__init__()
        self._scope = scope

    def is_set(self) -> bool:  # type: ignore[override]
        return self._scope.is_set()


class SglEvalModel(LitellmModel):
    """Vendored query loop over the sgl-eval sampler.

    ``abort_exceptions`` mirror litellm's non-retryable set: a 4xx that will
    not change on retry (context overflow arrives as a 400) ends the trial
    instead of looping until the format-error limit.
    """

    abort_exceptions: list = [
        KeyboardInterrupt,
        ExecutionCancelled,
        WorkerAborted,
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.UnprocessableEntityError,
    ]

    def __init__(
        self,
        *,
        sampler: ChatCompletionSampler,
        gen: GenConfig,
        cancel: _CancelScope,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._sampler = sampler
        self._gen = gen
        self._cancel = cancel

    def _query(self, messages: List[dict], **kwargs: Any):
        self._cancel.check()
        wire = [_wire_message(message) for message in messages]
        return self._sampler.complete_raw(wire, self._gen, tools=[BASH_TOOL])

    def _calculate_cost(self, response) -> Dict[str, float]:
        return {"cost": 0.0}

    def _retry(self, *, logger: logging.Logger, abort_exceptions: list):
        retrying = upstream_retry(logger=logger, abort_exceptions=abort_exceptions)
        retrying.sleep = self._sleep
        return retrying

    def _sleep(self, seconds: float) -> None:
        if self._cancel.wait(seconds):
            raise ExecutionCancelled()


def _wire_message(message: dict) -> dict:
    """Drop the SDK's ``None`` placeholders (refusal, audio, ...) before sending history back.

    ``content`` stays even when ``None``: an assistant turn that only carried
    tool calls is still a valid message.
    """
    return {k: v for k, v in message.items() if v is not None or k == "content"}


class SandboxEnvironment(LocalEnvironment):
    """Vendored ``execute`` with the process launched inside the task container."""

    def __init__(
        self,
        *,
        container: str,
        cancel: _CancelScope,
        uname: Dict[str, str],
        container_env: Dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._container = container
        self._cancel = cancel
        self._uname_info = uname
        self._container_env = container_env

    def _run(
        self, command: str, cwd: str, env: Dict[str, str], timeout: int
    ) -> subprocess.CompletedProcess:
        return sandbox.exec_action(
            self._container,
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            cancel=self._cancel.as_event(),
        )

    def _uname(self) -> Dict[str, str]:
        return dict(self._uname_info)

    def _environ(self) -> Dict[str, str]:
        return {**self._container_env, **self.config.env}


class HostMiniSweAgent(BaseAgent):
    """pier agent whose loop runs on the host; the container only executes commands."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger, **kwargs)
        if run_id not in _RUNS:
            raise RuntimeError(f"no registered host agent settings for run {run_id!r}")
        self._settings = _RUNS[run_id]
        self._trial_cancel = threading.Event()
        self._cancel = _CancelScope(self._settings.cancel_event, self._trial_cancel)
        self._agent: Optional[DefaultAgent] = None
        self._sampler: Optional[ChatCompletionSampler] = None
        self._workdir: Optional[str] = None

    @staticmethod
    def name() -> str:
        return AGENT_NAME

    def version(self) -> str:
        return MINI_SWE_AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        loop = asyncio.get_running_loop()
        container = await sandbox.container_id(environment)
        done = threading.Event()
        outcome: Dict[str, Any] = {}

        def work() -> None:
            try:
                self._run_sync(container, instruction)
            except BaseException as exc:  # noqa: BLE001 - reported to the trial below
                outcome["error"] = exc
            finally:
                done.set()

        future = loop.run_in_executor(None, work)
        try:
            await future
        except asyncio.CancelledError:
            # pier's agent deadline or a run cancellation. Stop the model call
            # and the in-container command, then let the worker unwind before
            # upstream collects the workspace.
            self._trial_cancel.set()
            self._abort_transport()
            await loop.run_in_executor(None, done.wait, _WORKER_EXIT_GRACE_SEC)
            self._populate_context(context)
            raise
        finally:
            if done.is_set():
                self._close_transport()

        self._populate_context(context)
        error = outcome.get("error")
        if error is None or isinstance(error, ExecutionCancelled):
            return
        raise NonZeroAgentExitCodeError(
            f"mini-swe-agent loop ended with {type(error).__name__}: {error}"
        ) from error

    def _run_sync(self, container: str, instruction: str) -> None:
        settings = self._settings
        self._workdir = sandbox.workdir(container)
        uname = sandbox.uname_info(container)
        container_env = sandbox.environ(container)
        config = build_agent_config(settings, self.logs_dir / TRAJECTORY_FILENAME)

        self._sampler = settings.sampler.derive(
            request_timeout=settings.request_timeout_sec,
            sdk_max_retries=settings.sdk_max_retries,
        )
        model = SglEvalModel(
            sampler=self._sampler,
            gen=settings.gen,
            cancel=self._cancel,
            model_name=self.model_name or self._sampler.model,
            **config["model"],
        )
        env = SandboxEnvironment(
            container=container,
            cancel=self._cancel,
            uname=uname,
            container_env=container_env,
            cwd=self._workdir,
            env=dict(config.get("environment", {}).get("env", {})),
            timeout=settings.per_action_timeout_sec,
        )
        self._agent = DefaultAgent(model, env, **config["agent"])
        self._agent.run(instruction)

    def _abort_transport(self) -> None:
        if self._sampler is not None:
            self._sampler.abort()

    def _close_transport(self) -> None:
        if self._sampler is not None:
            self._sampler.close()

    def _populate_context(self, context: AgentContext) -> None:
        agent = self._agent
        settings = self._settings
        metadata: Dict[str, Any] = {
            "agent": AGENT_NAME,
            "mini_swe_agent_version": MINI_SWE_AGENT_VERSION,
            "per_action_timeout_sec": settings.per_action_timeout_sec,
            "request_timeout_sec": settings.request_timeout_sec,
            "sdk_max_retries": settings.sdk_max_retries,
            "model_retry_attempts": int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10")),
            "max_consecutive_format_errors": settings.max_consecutive_format_errors,
            "container_workdir": self._workdir,
        }
        if agent is None:
            context.metadata = metadata
            return

        usage = _sum_usage(agent.messages)
        last_extra = agent.messages[-1].get("extra", {}) if agent.messages else {}
        metadata.update(
            {
                "n_model_calls": agent.n_calls,
                "usage_reported_calls": usage["reported"],
                "reasoning_tokens": usage["reasoning"] if usage["reported"] else None,
                "exit_status": last_extra.get("exit_status"),
                "agent_config": agent.config.model_dump(mode="json"),
            }
        )
        context.n_input_tokens = usage["prompt"] if usage["reported"] else None
        context.n_output_tokens = usage["completion"] if usage["reported"] else None
        context.n_agent_steps = agent.n_calls
        context.metadata = metadata


def _sum_usage(messages: List[dict]) -> Dict[str, int]:
    """Token totals over every model response kept in the trajectory.

    Both regular turns (``extra.response``) and the responses retained on
    FormatError user messages carry the SDK response dump, so retried format
    errors count.
    """
    totals = {"prompt": 0, "completion": 0, "reasoning": 0, "reported": 0}
    for message in messages:
        response = (message.get("extra") or {}).get("response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage") or {}
        if not usage:
            continue
        totals["reported"] += 1
        totals["prompt"] += int(usage.get("prompt_tokens") or 0)
        totals["completion"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens")
        if reasoning is None:
            reasoning = usage.get("reasoning_tokens")
        totals["reasoning"] += int(reasoning or 0)
    return totals
