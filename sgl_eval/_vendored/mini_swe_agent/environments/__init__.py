# Vendored from SWE-agent/mini-swe-agent@a83fcae82d2a08f0ee0c688f9d137b3566c097f8
# Source: src/minisweagent/environments/__init__.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

"""Environment implementations for mini-SWE-agent."""

import copy
import importlib

from sgl_eval._vendored.mini_swe_agent import Environment

_ENVIRONMENT_MAPPING = {
    "docker": "sgl_eval._vendored.mini_swe_agent.environments.docker.DockerEnvironment",
    "singularity": "sgl_eval._vendored.mini_swe_agent.environments.singularity.SingularityEnvironment",
    "local": "sgl_eval._vendored.mini_swe_agent.environments.local.LocalEnvironment",
    "swerex_docker": "sgl_eval._vendored.mini_swe_agent.environments.extra.swerex_docker.SwerexDockerEnvironment",
    "swerex_modal": "sgl_eval._vendored.mini_swe_agent.environments.extra.swerex_modal.SwerexModalEnvironment",
    "bubblewrap": "sgl_eval._vendored.mini_swe_agent.environments.extra.bubblewrap.BubblewrapEnvironment",
    "contree": "sgl_eval._vendored.mini_swe_agent.environments.extra.contree.ContreeEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    full_path = _ENVIRONMENT_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown environment type: {spec} (resolved to {full_path}, available: {_ENVIRONMENT_MAPPING})"
        raise ValueError(msg)


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)
