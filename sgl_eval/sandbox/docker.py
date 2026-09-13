"""Per-action command execution inside a pier-managed Docker container.

The vendored ``DockerEnvironment`` owns the container lifecycle, file transfer,
the verifier build and the ``[[verifier.collect]]`` hook. Its ``exec`` runs
``docker compose exec`` and, on timeout, only terminates that host-side client
and raises without the output (pier ``docker.py``,
``_run_docker_compose_command``). The agent loop needs the mini-swe-agent
``LocalEnvironment`` contract instead: ``/bin/sh -c``, a per-action timeout
that kills the whole process group inside the container, and the partial
output handed back to the model as the observation. This module is that path:
one plain ``docker exec`` per action, with the process id recorded inside the
container so a second ``docker exec`` can kill the group.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from typing import Dict, List

_PID_DIR = "/tmp/.sgl_eval"
_POLL_SEC = 0.5
_PROBE_TIMEOUT_SEC = 30
_KILL_TIMEOUT_SEC = 15
_GRACE_AFTER_KILL_SEC = 10
# Docker's default address pool holds ~31 /16 networks; pier gives every trial
# its own compose project and network.
MAX_CONCURRENT_TRIALS = 28
LOW_DISK_GB = 150


class ExecutionCancelled(BaseException):
    """Raised inside the agent worker when the trial or run is being cancelled.

    A ``BaseException`` so the vendored ``LocalEnvironment.execute`` (which
    turns any ``Exception`` into an observation) and the tenacity retry loop
    let it propagate; the agent adapter is the only place that catches it.
    """


class HostPreflightError(RuntimeError):
    """The host cannot run Docker trials; the message says what to fix."""


async def container_id(environment) -> str:
    """The id of the ``main`` service container of a started vendored DockerEnvironment."""
    result = await environment._run_docker_compose_command(["ps", "-q", "main"], check=True)
    ids = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if not ids:
        raise RuntimeError("docker compose reports no running `main` container for the trial")
    return ids[0]


def exec_action(
    container: str,
    command: str,
    *,
    cwd: str,
    env: Dict[str, str],
    timeout: float,
    cancel: threading.Event,
    docker: str = "docker",
) -> subprocess.CompletedProcess:
    """Run ``command`` under ``/bin/sh -c`` in the container like ``LocalEnvironment._run``.

    Returns a ``CompletedProcess`` with combined stdout/stderr. On timeout the
    in-container process group is killed and ``subprocess.TimeoutExpired`` is
    raised carrying the output produced so far, which the vendored ``execute``
    turns into the timeout observation. On cancellation the group is killed
    and ``ExecutionCancelled`` is raised.
    """
    action_id = uuid.uuid4().hex
    launcher = (
        f"mkdir -p {_PID_DIR} && echo $$ > {_PID_DIR}/{action_id}.pid && exec /bin/sh -c \"$1\""
    )
    argv: List[str] = [docker, "exec", "-w", cwd]
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    argv += [container, "/bin/sh", "-c", launcher, "sh", command]

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = time.monotonic() + timeout
    while True:
        if cancel.is_set():
            _kill_and_collect(proc, container, action_id, docker)
            raise ExecutionCancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            output = _kill_and_collect(proc, container, action_id, docker)
            raise subprocess.TimeoutExpired(command, timeout, output=output)
        try:
            output, _ = proc.communicate(timeout=min(remaining, _POLL_SEC))
        except subprocess.TimeoutExpired:
            continue
        return subprocess.CompletedProcess(command, proc.returncode, stdout=output)


def _kill_and_collect(proc: subprocess.Popen, container: str, action_id: str, docker: str) -> str:
    kill_process_group(container, action_id, docker=docker)
    try:
        output, _ = proc.communicate(timeout=_GRACE_AFTER_KILL_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
    return output or ""


def kill_process_group(container: str, action_id: str, *, docker: str = "docker") -> None:
    """Kill the action's process group (then its leader) inside the container."""
    pid_file = f"{_PID_DIR}/{action_id}.pid"
    script = (
        f"p=$(cat {pid_file} 2>/dev/null); "
        f'if [ -n "$p" ]; then kill -KILL -- -"$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; fi; '
        f"rm -f {pid_file}"
    )
    try:
        subprocess.run(
            [docker, "exec", container, "/bin/sh", "-c", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_KILL_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pass


def _probe(container: str, script: str, *, docker: str = "docker") -> str:
    result = subprocess.run(
        [docker, "exec", container, "/bin/sh", "-c", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_PROBE_TIMEOUT_SEC,
        check=True,
    )
    return result.stdout


def uname_info(container: str, *, docker: str = "docker") -> Dict[str, str]:
    """``platform.uname()._asdict()`` as seen from inside the container."""
    lines = _probe(container, "uname -s; uname -n; uname -r; uname -v; uname -m", docker=docker)
    fields = (lines.split("\n") + [""] * 5)[:5]
    system, node, release, version, machine = (f.strip() for f in fields)
    return {
        "system": system,
        "node": node,
        "release": release,
        "version": version,
        "machine": machine,
        "processor": "",
    }


def environ(container: str, *, docker: str = "docker") -> Dict[str, str]:
    """The container's default environment (what ``os.environ`` was for an in-container agent)."""
    out: Dict[str, str] = {}
    for line in _probe(container, "env", docker=docker).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def workdir(container: str, *, docker: str = "docker") -> str:
    """The image's ``WORKDIR``; the default cwd of every in-container command."""
    result = subprocess.run(
        [docker, "inspect", "--format", "{{.Config.WorkingDir}}", container],
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_SEC,
        check=True,
    )
    return result.stdout.strip() or "/"


def host_preflight(
    num_threads: int, *, task_cpus: int, task_memory_mb: int, docker: str = "docker"
) -> List[str]:
    """Fail fast when Docker trials cannot run; return non-fatal warnings."""
    if shutil.which(docker) is None:
        raise HostPreflightError("`docker` is not on PATH; install Docker Engine or Docker Desktop.")
    try:
        info = subprocess.run(
            [docker, "info", "--format", "{{.DockerRootDir}}\t{{.NCPU}}\t{{.MemTotal}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise HostPreflightError(
            "`docker info` failed; is the daemon running and does this user have access to "
            f"the socket?\n{(exc.stderr or exc.stdout or '').strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HostPreflightError("`docker info` did not answer within 30s.") from exc
    root_dir, ncpu, mem_total = (info.split("\t") + ["", "", ""])[:3]

    try:
        subprocess.run(
            [docker, "compose", "version"], capture_output=True, text=True, timeout=30, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise HostPreflightError(
            "`docker compose` (the v2 plugin) is required: the vendored pier runtime starts "
            "every task and verifier container through it."
        ) from exc

    networks = subprocess.run(
        [docker, "network", "ls", "-q"], capture_output=True, text=True, timeout=30, check=True
    ).stdout.split()
    if num_threads + len(networks) > MAX_CONCURRENT_TRIALS:
        raise HostPreflightError(
            f"--num-threads {num_threads} plus the {len(networks)} existing Docker networks "
            f"exceeds {MAX_CONCURRENT_TRIALS}: every trial gets its own compose network and "
            "Docker's default address pool holds about 31. Lower --num-threads or remove "
            "unused networks (`docker network prune`)."
        )

    warnings: List[str] = []
    try:
        free_gb = shutil.disk_usage(root_dir or "/").free / 1e9
        if free_gb < LOW_DISK_GB:
            warnings.append(
                f"{free_gb:.0f} GB free under {root_dir or '/'}; the full task image set "
                f"needs well over 100 GB."
            )
    except OSError:
        pass
    try:
        cpus = int(ncpu)
        if num_threads * task_cpus > cpus:
            warnings.append(
                f"--num-threads {num_threads} x {task_cpus} CPUs per task exceeds the "
                f"{cpus} CPUs Docker reports."
            )
        mem_gb = int(mem_total) / 1e9
        if num_threads * task_memory_mb / 1e3 > mem_gb:
            warnings.append(
                f"--num-threads {num_threads} x {task_memory_mb} MB per task exceeds the "
                f"{mem_gb:.0f} GB Docker reports."
            )
    except ValueError:
        pass
    if os.name == "nt":
        warnings.append("Windows hosts are untested; the runtime is exercised on Linux only.")
    return warnings
