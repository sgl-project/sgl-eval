# Vendored from SWE-agent/mini-swe-agent@a83fcae82d2a08f0ee0c688f9d137b3566c097f8
# Source: src/minisweagent/agents/__init__.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

"""Agent implementations for mini-SWE-agent."""

import copy
import importlib

from sgl_eval._vendored.mini_swe_agent import Agent, Environment, Model

_AGENT_MAPPING = {
    "default": "sgl_eval._vendored.mini_swe_agent.agents.default.DefaultAgent",
    "interactive": "sgl_eval._vendored.mini_swe_agent.agents.interactive.InteractiveAgent",
}


def get_agent_class(spec: str) -> type[Agent]:
    full_path = _AGENT_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown agent type: {spec} (resolved to {full_path}, available: {_AGENT_MAPPING})"
        raise ValueError(msg)


def get_agent(model: Model, env: Environment, config: dict, *, default_type: str = "") -> Agent:
    config = copy.deepcopy(config)
    agent_class = get_agent_class(config.pop("agent_class", default_type))
    return agent_class(model, env, **config)
